# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import logging
import os
import pickle as pkl

import hydra
import numpy as np
import torch
from tqdm import tqdm

from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.scheduler import Cluster, ComponentPlacement, Worker

logger = logging.getLogger(__name__)


class DataCollector(Worker):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.num_data_episodes = cfg.runner.num_data_episodes
        self.total_cnt = 0
        self.env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )

        self.data_list = []
        
        # Initialize reward model if configured
        self.reward_model = None
        self.use_reward_model = cfg.get("reward", {}).get("use_reward_model", False)
        if self.use_reward_model:
            self._init_reward_model(cfg)
        
        # Gripper penalty configuration
        reward_cfg = cfg.get("reward", {})
        self.enable_gripper_penalty = reward_cfg.get("enable_gripper_penalty", False)
        self.gripper_penalty = reward_cfg.get("gripper_penalty", 0.1)
        self.binary_gripper_threshold = reward_cfg.get("binary_gripper_threshold", 0.5)
        self.gripper_state_index = reward_cfg.get("gripper_state_index", 0)
        self.gripper_action_index = reward_cfg.get("gripper_action_index", 6)
        self.gripper_open_threshold = reward_cfg.get("gripper_open_threshold", 0.04)
        
        # Time penalty configuration
        self.time_penalty = reward_cfg.get("time_penalty", 0.0)
        
        # Stage reward configuration
        self.stage_rewards = reward_cfg.get("stage_rewards", [-0.1, 0.0, 1.0])
        self.success_stage_index = reward_cfg.get("success_stage_index", 2)
        self.terminate_on_success = reward_cfg.get("terminate_on_success", True)
        
        # Track gripper logical state for penalty calculation
        self.gripper_is_open = {}

    def _init_reward_model(self, cfg):
        """Initialize the reward model from checkpoint."""
        reward_cfg = cfg.get("reward", {})
        model_cfg = reward_cfg.get("model", {})
        checkpoint_path = model_cfg.get("checkpoint_path")
        
        if checkpoint_path is None:
            logger.warning("No checkpoint_path specified for reward model, will use keyboard reward")
            return
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}, will use keyboard reward")
            return
        
        model_type = model_cfg.get("model_type", "resnet_reward")
        
        # Select device: prefer GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
        try:
            if model_type == "resnet_stage_classifier":
                from rlinf.models.embodiment.reward.resnet_stage_classifier_model import (
                    ResNetStageClassifierModel,
                )
                self.reward_model = ResNetStageClassifierModel(model_cfg)
            else:
                from rlinf.models.embodiment.reward import ResNetRewardModel
                self.reward_model = ResNetRewardModel(model_cfg)
            
            self.reward_model = self.reward_model.to(self.device)
            self.reward_model.eval()
            logger.info(f"[DataCollector] Loaded reward model from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load reward model: {e}")
            self.reward_model = None
            self.use_reward_model = False

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images to (B, C, H, W) in [0, 1]."""
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        
        # Handle different input formats
        if images.dim() == 4 and images.shape[-1] in [1, 3, 4]:
            # (B, H, W, C) -> (B, C, H, W)
            images = images.permute(0, 3, 1, 2)
        
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.dtype != torch.float32:
            images = images.float()
        
        return images

    def _compute_reward_model_output(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Compute reward using the reward model.
        
        Args:
            obs: Observation dict containing 'main_images'
            
        Returns:
            Tuple of (reward, stage_idx, is_success)
        """
        images = obs.get("main_images")
        if images is None:
            return torch.tensor([0.0]), torch.tensor([0]), False
        
        images = self._preprocess_images(images)
        images = images.to(self.device)
        
        with torch.no_grad():
            outputs = self.reward_model(images)
            probs = outputs["probabilities"]
            
            if probs.dim() == 2:
                # Multi-stage classifier
                stage_idx = torch.argmax(probs, dim=1)
                stage_rewards_tensor = torch.tensor(
                    self.stage_rewards, device=probs.device, dtype=probs.dtype
                )
                reward = stage_rewards_tensor[stage_idx]
                is_success = (stage_idx == int(self.success_stage_index)).item()
            else:
                # Binary reward model
                is_success = (probs > 0.5).item()
                reward = probs
                stage_idx = torch.tensor([int(is_success)])
        
        return reward.cpu(), stage_idx.cpu(), is_success

    def _compute_gripper_penalty(self, obs: dict, action: np.ndarray) -> float:
        """Compute gripper penalty for effective gripper actions.
        
        Args:
            obs: Observation dict containing 'states'
            action: Action array of shape (action_dim,)
            
        Returns:
            Penalty value (negative or zero)
        """
        if not self.enable_gripper_penalty:
            return 0.0
        
        states = obs.get("states")
        if states is None:
            return 0.0
        
        # Convert to numpy if tensor
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
        
        # Get gripper position from state (squeeze batch dimension if present)
        if states.ndim == 2:
            states = states[0]
        gripper_position = states[self.gripper_state_index]
        
        # Get gripper action
        gripper_action = action[self.gripper_action_index]
        
        # Determine gripper command
        command_open = gripper_action >= self.binary_gripper_threshold
        command_close = gripper_action <= -self.binary_gripper_threshold
        
        # Initialize gripper state if not tracked
        if 0 not in self.gripper_is_open:
            self.gripper_is_open[0] = gripper_position > self.gripper_open_threshold
        
        is_gripper_open = self.gripper_is_open[0]
        
        # Check if action is effective (changes state)
        is_effective_close = command_close and is_gripper_open
        is_effective_open = command_open and (not is_gripper_open)
        is_effective = is_effective_close or is_effective_open
        
        # Update tracked gripper state
        if is_effective_close:
            self.gripper_is_open[0] = False
        elif is_effective_open:
            self.gripper_is_open[0] = True
        
        # Return penalty
        if is_effective:
            return -self.gripper_penalty
        return 0.0

    def _compute_final_reward(self, obs: dict, action: np.ndarray) -> tuple[torch.Tensor, bool]:
        """Compute final reward combining reward model, gripper penalty, and time penalty.
        
        Args:
            obs: Observation dict
            action: Action array
            
        Returns:
            Tuple of (final_reward_tensor, is_terminated)
        """
        # Get base reward from reward model
        rm_reward, stage_idx, is_success = self._compute_reward_model_output(obs)
        
        # Compute gripper penalty
        gripper_pen = self._compute_gripper_penalty(obs, action)
        
        # Compute final reward
        final_reward = rm_reward.item() + gripper_pen + self.time_penalty
        
        # Log reward components
        logger.info(
            f"[Reward] stage={stage_idx.item()}, rm_reward={rm_reward.item():.3f}, "
            f"gripper_penalty={gripper_pen:.3f}, time_penalty={self.time_penalty:.3f}, "
            f"final_reward={final_reward:.3f}"
        )
        
        # Determine termination
        terminated = is_success if self.terminate_on_success else False
        
        return torch.tensor([final_reward]), terminated

    def _apply_penalties_to_keyboard_reward(self, env_reward: torch.Tensor, obs: dict, action: np.ndarray) -> torch.Tensor:
        """Apply gripper penalty and time penalty to keyboard reward.
        
        Args:
            env_reward: Base reward from keyboard input
            obs: Observation dict
            action: Action array
            
        Returns:
            Final reward tensor with penalties applied
        """
        # Get base reward value
        base_reward = env_reward[0].item() if env_reward.dim() > 0 else env_reward.item()
        
        # Compute gripper penalty
        gripper_pen = self._compute_gripper_penalty(obs, action)
        
        # Compute final reward
        final_reward = base_reward + gripper_pen + self.time_penalty
        
        # Log reward components
        logger.info(
            f"[Reward] keyboard_reward={base_reward:.3f}, "
            f"gripper_penalty={gripper_pen:.3f}, time_penalty={self.time_penalty:.3f}, "
            f"final_reward={final_reward:.3f}"
        )
        
        return torch.tensor([final_reward])

    def _reset_gripper_tracking(self):
        """Reset gripper tracking state for new episode."""
        self.gripper_is_open = {}

    def _extract_obs(self, obs):
        if not self.cfg.runner.record_task_description:
            obs.pop("task_descriptions", None)
        ret_obs = {}
        for key in obs:
            if key == "main_images":
                ret_obs[key] = obs[key].clone().permute(0, 3, 1, 2)[0].float() / 255.0
            elif key == "extra_view_images":
                ret_obs[key] = (
                    obs[key].clone().permute(0, 1, 4, 2, 3)[0].float() / 255.0
                )
            else:
                ret_obs[key] = obs[key][0]
        return ret_obs

    def run(self):
        obs, _ = self.env.reset()
        self._reset_gripper_tracking()
        success_cnt = 0
        progress_bar = tqdm(
            range(self.num_data_episodes), desc="Collecting Data Episodes:"
        )
        action_dim = 6 if self.cfg.env.eval.get("no_gripper", True) else 7
        while success_cnt < self.num_data_episodes:
            action = np.zeros((1, action_dim), dtype=np.float32)
            next_obs, env_reward, terminated, truncated, info = self.env.step(action)
            
            # Get intervene action if any
            if "intervene_action" in info:
                action = info["intervene_action"]
            
            # Convert action to numpy for processing
            if isinstance(action, torch.Tensor):
                action_np = action.cpu().numpy()
            else:
                action_np = action
            if action_np.ndim == 2:
                action_np = action_np[0]
            
            # Compute reward
            if self.use_reward_model and self.reward_model is not None:
                # Use reward model + penalties
                reward, model_terminated = self._compute_final_reward(next_obs, action_np)
                # Combine termination signals
                if model_terminated:
                    terminated = torch.tensor([True])
            else:
                # Use keyboard reward + auto penalties (gripper_penalty + time_penalty)
                reward = self._apply_penalties_to_keyboard_reward(env_reward, next_obs, action_np)
            
            if self.cfg.env.eval.get("enable_truncated", False):
                done = torch.logical_or(terminated, truncated)
            else:
                done = terminated.clone()
                truncated = torch.zeros_like(terminated)

            # Handle vector env
            single_obs = self._extract_obs(obs)
            single_next_obs = self._extract_obs(next_obs)
            single_action = action[0] if isinstance(action, (np.ndarray, torch.Tensor)) and len(action.shape) > 1 else action
            if isinstance(single_action, torch.Tensor):
                single_action = single_action.cpu()
            single_reward = reward[0] if reward.dim() > 0 else reward
            single_done = done[0] if done.dim() > 0 else done
            single_terminated = terminated[0] if terminated.dim() > 0 else terminated
            single_truncated = truncated[0] if truncated.dim() > 0 else truncated

            # Handle chunk
            chunk_done = single_done[None, ...]
            chunk_reward = single_reward[None, ...]
            chunk_terminated = single_terminated[None, ...]
            chunk_truncated = single_truncated[None, ...]

            transition = copy.deepcopy(
                {
                    "obs": single_obs,
                    "next_obs": single_next_obs,
                }
            )
            data = copy.deepcopy(
                {
                    "transitions": transition,
                    "action": single_action,
                    "rewards": chunk_reward,
                    "dones": chunk_done,
                    "terminations": chunk_terminated,
                    "truncations": chunk_truncated,
                }
            )
            self.data_list.append(data)

            obs = next_obs

            if done:
                success_cnt += 1
                self.total_cnt += 1
                self.log_info(
                    f"{reward}\tGot {success_cnt} successes of {self.total_cnt} trials. {self.num_data_episodes} successes needed."
                )
                obs, _ = self.env.reset()
                self._reset_gripper_tracking()
                progress_bar.update(1)
            else:
                self.log_info("Done is False, continue current episode.")

        save_file_path = os.path.join(self.cfg.runner.logger.log_path, "data.pkl")
        with open(save_file_path, "wb") as f:
            pkl.dump(self.data_list, f)
            self.log_info(
                f"Saved {self.num_data_episodes} demos with {len(self.data_list)} samples to {save_file_path}"
            )

        self.env.close()


@hydra.main(
    version_base="1.1", config_path="config", config_name="realworld_collect_data"
)
def main(cfg):
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")
    collector = DataCollector.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    collector.run().wait()


if __name__ == "__main__":
    main()
