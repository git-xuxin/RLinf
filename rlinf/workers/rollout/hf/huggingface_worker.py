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

import asyncio
import copy
import gc
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.data.io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.models import get_model
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.rollout.hf.utils import init_real_obs


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        actor_world_size = self.placement.get_world_size("actor")
        self.actor_weight_src_rank = self._rank % actor_world_size
        self.gather_num = self.placement.get_world_size(
            "rollout"
        ) // self.placement.get_world_size("env")

        self.train_queue = Channel.create(name="train_queue", local=True)
        self.eval_queue = Channel.create(name="eval_queue", local=True)

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )

        # Reward processor will be created in init_worker if configured
        self.reward_processor = None
        
        # Gripper penalty configuration (from SERL PR 65)
        # Penalizes effective gripper actions to prevent excessive gripper toggling
        reward_cfg = cfg.get("reward", {})
        self.enable_gripper_penalty = reward_cfg.get("enable_gripper_penalty", False)
        self.gripper_penalty = reward_cfg.get("gripper_penalty", 0.1)
        self.binary_gripper_threshold = reward_cfg.get("binary_gripper_threshold", 0.5)
        self.gripper_state_index = reward_cfg.get("gripper_state_index", 0)
        self.gripper_action_index = reward_cfg.get("gripper_action_index", 6)
        self.gripper_open_threshold = reward_cfg.get("gripper_open_threshold", 0.04)
        
        # Track gripper logical state per batch index (like franka_env.py's gripper_open boolean)
        # This ensures penalty is only applied when gripper state actually changes
        self.gripper_is_open: dict[int, bool] = {}

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.hf_model = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        self.hf_model.eval()

        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

        # Initialize reward processor if configured
        # Data flow: env -> send -> rollout (reward processing integrated here)
        self._init_reward_processor()

    def _init_reward_processor(self):
        """Initialize reward processor if configured in cfg.reward."""
        reward_cfg = self.cfg.get("reward", {})
        if not reward_cfg.get("use_reward_model", False):
            self.reward_processor = None
            return

        from rlinf.workers.reward.reward_worker import ImageRewardWorker

        # Create a local ImageRewardWorker instance (not a worker group)
        # This acts as a reward processor within the rollout worker
        self.reward_processor = ImageRewardWorker(self.cfg)
        self.reward_processor.init_worker()

        logger.info(
            f"[Rollout] Initialized reward processor with checkpoint: "
            f"{reward_cfg.get('model', {}).get('checkpoint_path', 'N/A')}"
        )

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"]
            if self._sampling_params["do_sample"]
            else 1.0,
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

        self._eval_sampling_params = {
            "do_sample": True
            if self._sampling_params.get("temperature_eval", -1) > 0
            else False,
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

    def predict(self, env_obs, mode="train"):
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.CNN_POLICY,
        ]:
            kwargs = {"mode": mode}

        kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs,
                **kwargs,
            )

        return actions, result

    def compute_gripper_penalty(
        self, env_output: dict[str, torch.Tensor], actions
    ):
        """Compute gripper penalty for effective gripper actions (from SERL PR 65).
        
        Effective gripper actions are those that would change the gripper state:
        - Command close (action < -threshold) when gripper is currently open
        - Command open (action > threshold) when gripper is currently closed
        
        This penalty prevents the policy from excessively opening and closing the gripper.
        
        IMPORTANT: This method tracks gripper logical state (like franka_env.py's gripper_open boolean)
        rather than using physical position. This ensures penalty is only applied when the gripper
        state actually changes, not when the same command is repeated.
        
        Args:
            env_output: Environment output containing 'obs' with states
            actions: Predicted actions (tensor or numpy array), shape (B, action_dim) or (B, num_chunks, action_dim)
            
        Returns:
            Tuple of (penalty, is_effective, is_effective_close, gripper_state, gripper_actions) or None
        """
        import numpy as np
        
        if not self.enable_gripper_penalty:
            return None
        
        # Get observation states
        obs = env_output.get("obs", {})
        states = obs.get("states") if isinstance(obs, dict) else None
        
        if states is None:
            return None
        
        # Convert states to torch tensor if needed
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        if isinstance(states, torch.Tensor):
            states = states.cpu()
        
        # Get gripper physical position from observation
        # States format after alphabetical sorting: [gripper_position, tcp_force(3), tcp_pose(7), tcp_torque(3), tcp_vel(6)]
        # gripper_position is at index 0
        gripper_position = states[:, self.gripper_state_index]  # [B,]
        batch_size = gripper_position.shape[0]
        
        # Convert actions to torch tensor if needed
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu()
        
        # Handle different action shapes
        if actions.dim() == 3:
            # [B, num_chunks, action_dim]
            gripper_actions = actions[:, :, self.gripper_action_index]  # [B, num_chunks]
        elif actions.dim() == 2:
            # [B, action_dim]
            gripper_actions = actions[:, self.gripper_action_index]  # [B,]
        else:
            return None
        
        # Determine gripper command (only use first chunk for state transition)
        # Positive action (> threshold) = open command
        # Negative action (< -threshold) = close command
        if gripper_actions.dim() == 2:
            # Use first chunk to determine command
            gripper_actions_for_cmd = gripper_actions[:, 0]  # [B,]
        else:
            gripper_actions_for_cmd = gripper_actions  # [B,]
        
        command_open = gripper_actions_for_cmd >= self.binary_gripper_threshold  # [B,]
        command_close = gripper_actions_for_cmd <= -self.binary_gripper_threshold  # [B,]
        
        # Build is_gripper_open tensor using tracked logical state
        # Initialize new batch indices based on physical position
        is_gripper_open_list = []
        for i in range(batch_size):
            if i not in self.gripper_is_open:
                # Initialize based on physical position for new environments
                self.gripper_is_open[i] = gripper_position[i].item() > self.gripper_open_threshold
            is_gripper_open_list.append(self.gripper_is_open[i])
        
        is_gripper_open = torch.tensor(is_gripper_open_list, dtype=torch.bool)  # [B,]
        
        # Effective action: command that changes gripper state
        # - Close command when gripper is logically open
        # - Open command when gripper is logically closed
        is_effective_close = command_close & is_gripper_open  # [B,]
        is_effective_open = command_open & (~is_gripper_open)  # [B,]
        is_effective = is_effective_close | is_effective_open  # [B,]
        
        # Update tracked gripper state for effective actions
        for i in range(batch_size):
            if is_effective_close[i].item():
                self.gripper_is_open[i] = False
            elif is_effective_open[i].item():
                self.gripper_is_open[i] = True
        
        # Expand is_effective to match gripper_actions shape if needed
        if gripper_actions.dim() == 2:
            # [B,] -> [B, num_chunks]
            is_effective_expanded = is_effective.unsqueeze(1).expand_as(gripper_actions)
        else:
            is_effective_expanded = is_effective
        
        # Compute penalty: -gripper_penalty for effective actions, 0 otherwise
        penalty = torch.where(
            is_effective_expanded,
            torch.full_like(gripper_actions, -self.gripper_penalty, dtype=torch.float32),
            torch.zeros_like(gripper_actions, dtype=torch.float32)
        )
        
        return penalty, is_effective, is_effective_close, gripper_position, gripper_actions

    def apply_gripper_penalty_to_rewards(
        self, env_output: dict[str, torch.Tensor], actions, rewards
    ):
        """Apply gripper penalty to rewards and log the results.
        
        This method computes the gripper penalty and adds it to the rewards.
        It also logs detailed information about the penalty application.
        
        Args:
            env_output: Environment output containing 'obs' with states
            actions: Predicted actions tensor
            rewards: Current rewards tensor, can be None (for first step)
            
        Returns:
            Modified rewards tensor with gripper penalty applied, or original rewards if penalty not applicable
        """
        if not self.enable_gripper_penalty:
            return rewards
        
        if rewards is None:
            # First step has no rewards, skip penalty
            return rewards
        
        result = self.compute_gripper_penalty(env_output, actions)
        if result is None:
            return rewards
        
        penalty, is_effective, is_effective_close, gripper_state, gripper_actions = result
        
        # Log for debugging (within Debug: log reward info section)
        # Note: is_effective is now [B,] shape (one value per batch item), not [B, num_chunks]
        if is_effective.any():
            batch_size = gripper_state.shape[0]
            for i in range(batch_size):
                if is_effective[i]:
                    action_type = "CLOSE" if is_effective_close[i] else "OPEN"
                    # Get the first action value for logging
                    if gripper_actions.dim() == 1:
                        action_val = gripper_actions[i].item()
                        penalty_val = penalty[i].item()
                    else:
                        action_val = gripper_actions[i, 0].item()
                        penalty_val = penalty[i, 0].item()
                    reward_val = rewards[i, 0].item() if rewards.dim() > 1 else rewards[i].item()
                    final_reward = reward_val + penalty_val
                    logger.info(
                        # f"Debug: log reward info - gripper_pos={gripper_state[i].item():.4f}, "
                        # f"gripper_is_open={self.gripper_is_open.get(i, 'N/A')}, "
                        # f"action={action_val:.4f}, type={action_type}, "
                        f"rm_reward={reward_val:.2f}, gripper_penalty={penalty_val:.2f}, "
                        f"final_reward={final_reward:.2f}"
                    )
        
        # Apply penalty to rewards
        # Ensure shapes match
        if rewards.dim() == 1 and penalty.dim() == 1:
            # Both are [B,]
            modified_rewards = rewards + penalty
        elif rewards.dim() == 2 and penalty.dim() == 1:
            # rewards is [B, num_chunks], penalty is [B,]
            # Expand penalty to match
            modified_rewards = rewards + penalty.unsqueeze(1)
        elif rewards.dim() == 2 and penalty.dim() == 2:
            # Both are [B, num_chunks]
            modified_rewards = rewards + penalty
        else:
            # Shape mismatch, just return original
            logger.warning(
                f"[Rollout] Gripper penalty shape mismatch: rewards={rewards.shape}, penalty={penalty.shape}"
            )
            return rewards
        
        # Reset gripper tracking state for environments that have terminated/truncated
        # This ensures the state is re-initialized on the next episode
        terminations = env_output.get("terminations")
        truncations = env_output.get("truncations")
        if terminations is not None or truncations is not None:
            batch_size = gripper_state.shape[0]
            for i in range(batch_size):
                should_reset = False
                if terminations is not None:
                    term_val = terminations[i].item() if hasattr(terminations[i], 'item') else terminations[i]
                    should_reset = should_reset or bool(term_val)
                if truncations is not None:
                    trunc_val = truncations[i].item() if hasattr(truncations[i], 'item') else truncations[i]
                    should_reset = should_reset or bool(trunc_val)
                if should_reset and i in self.gripper_is_open:
                    del self.gripper_is_open[i]
                    logger.debug(f"Reset gripper tracking state for env {i} due to episode end")
        
        return modified_rewards

    def get_dones_and_rewards(
        self, env_output: dict[str, torch.Tensor], extracted_obs: dict[str, Any]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None]:
        """
        Get dones and rewards from environment batch, handling auto_reset if needed.

        Args:
            env_output: Environment batch containing dones, rewards, and optionally final_obs

        Returns:
            Tuple of (dones, rewards, real_extracted_obs). dones and rewards are tensors.
        """
        # First step: no rewards yet, only dones
        real_extracted_obs = None
        if env_output["rewards"] is None:
            if hasattr(self.hf_model, "q_head"):
                real_extracted_obs = init_real_obs(extracted_obs)
            return (
                env_output["dones"].bool().cpu().contiguous(),
                None,
                real_extracted_obs,
            )

        dones = env_output["dones"].bool().cpu().contiguous()
        rewards = env_output["rewards"].cpu().contiguous()

        # Handle auto_reset: add bootstrap value to rewards for done episodes
        # Note: currently this is not correct for chunk-size>1 with partial reset
        if dones.any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head"):
                final_obs = env_output.get("final_obs")
                # Only compute bootstrap value if final_obs is available
                # (reward-based termination may not have final_obs)
                if final_obs is not None:
                    with torch.no_grad():
                        final_extracted_obs = self.hf_model.preprocess_env_obs(final_obs)
                        if hasattr(self.hf_model, "q_head"):
                            real_extracted_obs = init_real_obs(final_extracted_obs)
                        actions, result = self.predict(final_extracted_obs)
                        if "prev_values" in result:
                            _final_values = result["prev_values"]
                        else:
                            _final_values = torch.zeros_like(actions[:, 0])
                    final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                    last_step_dones = dones[:, -1]  # [bsz, ]

                    final_values[last_step_dones] = _final_values[:, 0][last_step_dones]

                    # Add bootstrap value to the last step of done episodes
                    rewards[:, -1] += self.cfg.algorithm.gamma * final_values.cpu()
                else:
                    # No final_obs (e.g., reward-based termination), use current obs for q_head
                    if hasattr(self.hf_model, "q_head"):
                        real_extracted_obs = init_real_obs(extracted_obs)

        if real_extracted_obs is None and hasattr(self.hf_model, "q_head"):
            real_extracted_obs = init_real_obs(extracted_obs)
        return dones, rewards, real_extracted_obs

    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""
        param_state_dict = await self.recv(
            self.actor_group_name,
            src_rank=self.actor_weight_src_rank,
            async_op=True,
            options=self._sync_weight_comm_options,
        ).async_wait()

        self.hf_model.load_state_dict(param_state_dict)
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    def update_intervene_actions(self, env_output, forward_inputs):
        intervene_actions = env_output["intervene_actions"]
        intervene_flags = env_output["intervene_flags"]
        if intervene_actions is not None:
            if "action" in forward_inputs:
                policy_action = forward_inputs["action"].to(intervene_actions.device)
                policy_action = policy_action.reshape(
                    policy_action.shape[0], self.hf_model.num_action_chunks, -1
                )
                intervene_actions = intervene_actions.reshape(
                    intervene_actions.shape[0], self.hf_model.num_action_chunks, -1
                )
                action = intervene_actions * intervene_flags[
                    ..., None
                ] + policy_action * (~intervene_flags[..., None])
                action = action.reshape(action.shape[0], -1)
                forward_inputs["action"] = action
            else:
                raise NotImplementedError(f"{forward_inputs.keys()=}")
        forward_inputs["intervene_flags"] = intervene_flags
        return forward_inputs

    async def generate(
        self, input_channel: Channel, output_channel: Channel, actor_channel: Channel
    ):
        if self.enable_offload:
            self.reload_model()

        self.buffer_list = [
            EmbodiedRolloutResult(rollout_epoch=self.cfg.algorithm.rollout_epoch)
            for _ in range(self.num_pipeline_stages)
        ]

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        for _ in tqdm(
            range(self.cfg.algorithm.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            last_extracted_obs = [None for i in range(self.num_pipeline_stages)]
            last_forward_inputs = [
                None for i in range(self.num_pipeline_stages)
            ]  # save actions

            for _ in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel)

                    if last_forward_inputs[stage_id] is not None:
                        last_forward_inputs[stage_id] = self.update_intervene_actions(
                            env_output, last_forward_inputs[stage_id]
                        )

                    extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                    dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                        env_output, extracted_obs
                    )
                    actions, result = self.predict(extracted_obs)
                    
                    # Apply gripper penalty to rewards (bypasses env reward since combine_mode="replace")
                    rewards = self.apply_gripper_penalty_to_rewards(env_output, actions, rewards)
                    
                    chunk_step_result = ChunkStepResult(
                        prev_logprobs=result["prev_logprobs"],
                        prev_values=result["prev_values"],
                        dones=dones,
                        truncations=env_output["truncations"],
                        terminations=env_output["terminations"],
                        rewards=rewards,  # the first step is reset step, reward is none, which will not be appended to the buffer
                        forward_inputs=last_forward_inputs[stage_id],
                    )
                    self.buffer_list[stage_id].append_result(chunk_step_result)
                    if last_extracted_obs[stage_id] is not None and hasattr(
                        self.hf_model, "q_head"
                    ):
                        self.buffer_list[stage_id].add_transition(
                            last_extracted_obs[stage_id], real_extracted_obs
                        )
                    last_extracted_obs[stage_id] = extracted_obs
                    last_forward_inputs[stage_id] = result["forward_inputs"]

                    # Get reward-based termination signal (if reward processor detected success)
                    reward_terminations = env_output.get("reward_terminations")
                    self.send_chunk_actions(output_channel, actions, reward_terminations=reward_terminations)

            for stage_id in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)
                last_forward_inputs[stage_id] = self.update_intervene_actions(
                    env_output, last_forward_inputs[stage_id]
                )

                extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                # Get dones and rewards from environment batch (final step of epoch)
                dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                    env_output, extracted_obs
                )
                
                with self.worker_timer():
                    actions, result = self.predict(extracted_obs)
                
                # Apply gripper penalty to rewards (bypasses env reward since combine_mode="replace")
                rewards = self.apply_gripper_penalty_to_rewards(env_output, actions, rewards)
                
                self.buffer_list[stage_id].dones.append(dones)
                self.buffer_list[stage_id].truncations.append(env_output["truncations"])
                self.buffer_list[stage_id].terminations.append(
                    env_output["terminations"]
                )
                self.buffer_list[stage_id].rewards.append(rewards)
                self.buffer_list[stage_id].forward_inputs.append(
                    put_tensor_device(last_forward_inputs[stage_id], "cpu")
                )
                # For the final step, we only need prev_values for bootstrapping
                # This is a special case that doesn't create a full ChunkStepResult
                if "prev_values" in result:
                    self.buffer_list[stage_id].prev_values.append(
                        result["prev_values"].cpu().contiguous()
                    )
                if hasattr(self.hf_model, "q_head"):
                    self.buffer_list[stage_id].add_transition(
                        last_extracted_obs[stage_id], real_extracted_obs
                    )

        for i in range(self.num_pipeline_stages):
            self.send_rollout_batch(actor_channel, i)

        if self.enable_offload:
            self.offload_model()

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()

        n_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(n_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel, mode="eval")
                    extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                    actions, _ = self.predict(extracted_obs, mode="eval")
                    self.send_chunk_actions(output_channel, actions, mode="eval")

        if self.enable_offload:
            self.offload_model()

    def offload_model(self):
        self.hf_model = self.hf_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def reload_model(self):
        self.hf_model = self.hf_model.to(self.device)

    async def recv_env_output(
        self, input_channel: Channel, mode="train"
    ) -> dict[str, torch.Tensor]:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        # Use asyncio so that it can run alongside async weight syncing
        src_rank_in_env = self._rank // self.gather_num
        work = self.recv(
            self.cfg.env.group_name, src_rank=src_rank_in_env, async_op=True
        )

        def _callback():
            env_mode, env_batch = work.wait()
            if env_mode == "train":
                self.train_queue.put_nowait(env_batch)
            elif env_mode == "eval":
                self.eval_queue.put_nowait(env_batch)

        work.then(_callback)

        if mode == "train":
            queue = self.train_queue
        elif mode == "eval":
            queue = self.eval_queue

        while queue.empty():
            await asyncio.sleep(0.001)
        batch = queue.get_nowait()

        # Process through reward processor if available (env -> reward -> rollout)
        if self.reward_processor is not None and mode == "train":
            batch = self.reward_processor.process_env_batch(batch)
            
            # Debug: log reward info (gripper penalty will be logged separately after predict())
            rewards = batch.get("rewards")
            probs = batch.get("probabilities")
            if rewards is not None and probs is not None:
                for i in range(len(rewards)):
                    reward_val = (
                        rewards[i, 0].item()
                        if rewards.dim() > 1
                        else rewards[i].item()
                    )
                    if probs.dim() > 1:
                        # prob_list = probs[i].tolist()
                        prob_list = [round(p, 2) for p in probs[i].tolist()]
                        stage_idx = int(torch.argmax(probs[i]).item())
                        logger.info(
                            "Debug: log reward info - stage=%d probs=%s rm_reward=%.1f",
                            stage_idx,
                            prob_list,
                            reward_val,
                        )
                    else:
                        prob_val = probs[i].item()
                        result = "SUCCESS" if reward_val > 0 else "FAIL"
                        logger.info(
                            "Debug: log reward info - prob=%.4f rm_reward=%.1f result=%s",
                            prob_val,
                            reward_val,
                            result,
                        )

        return batch

    def send_chunk_actions(
        self, output_channel: Channel, chunk_actions, mode="train", reward_terminations=None
    ):
        """Send actions (and optionally terminations) to env worker.
        
        Args:
            output_channel: Not used (legacy parameter)
            chunk_actions: Actions to send
            mode: "train" or "eval"
            reward_terminations: Optional termination signal from reward model (for early stopping on success)
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_rank_in_env = self._rank // self.gather_num
        
        # Pack data: if reward_terminations is provided, send as dict
        if reward_terminations is not None:
            data = {"actions": chunk_actions, "reward_terminations": reward_terminations}
        else:
            data = chunk_actions
            
        return self.send(
            (mode, data),
            self.cfg.env.group_name,
            dst_rank=dst_rank_in_env,
            async_op=True,
        )

    def send_rollout_batch(self, actor_channel: Channel, stage_id: int):
        # send rollout_batch to actor
        split_num = self.get_actor_split_num()
        splitted_rollout_result = self.buffer_list[stage_id].to_splitted_dict(split_num)
        for i in range(split_num):
            actor_channel.put(item=splitted_rollout_result[i], async_op=True)

    def get_actor_split_num(self):
        send_num = self.placement.get_world_size("rollout") * self.num_pipeline_stages
        recv_num = self.placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num

    def set_global_step(self, global_step):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
