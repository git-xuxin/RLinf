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
import logging

import torch
from tqdm import tqdm

from rlinf.data.io_struct import AsyncEmbodiedRolloutBuffer
from rlinf.scheduler import Channel
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

logger = logging.getLogger(__name__)


class AsyncMultiStepRolloutWorker(MultiStepRolloutWorker):
    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        replay_channel: Channel,
        demo_channel: Channel,
    ):
        self.buffer_list: list[AsyncEmbodiedRolloutBuffer] = [
            AsyncEmbodiedRolloutBuffer() for _ in range(self.num_pipeline_stages)
        ]

        self.buffer_tasks: list[asyncio.Task] = []
        for buffer in self.buffer_list:
            self.buffer_tasks.append(
                asyncio.create_task(
                    buffer.run(replay_channel, demo_channel, self.get_actor_split_num())
                )
            )

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        progress_bar = tqdm(
            total=None,
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        )

        is_first_step = True
        
        # For reward model termination handling:
        # - just_reset: marks the first frame after reset to force dones=False
        # - waiting_for_reset: handles frames between success detection and actual reset
        # These have no effect when not using reward model (is_reset_frame always False)
        just_reset = [False for _ in range(self.num_pipeline_stages)]
        waiting_for_reset = [False for _ in range(self.num_pipeline_stages)]
        
        # IMPORTANT: Initialize outside epoch loop to preserve state across epochs
        # and avoid index mismatch between dones and transitions
        last_extracted_obs = [None for _ in range(self.num_pipeline_stages)]
        last_results = [None for _ in range(self.num_pipeline_stages)]
        
        while not self.should_stop:

            for _ in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel)

                    if last_results[stage_id] is not None:
                        last_results[stage_id]["forward_inputs"] = (
                            self.update_intervene_actions(
                                env_output, last_results[stage_id]["forward_inputs"]
                            )
                        )

                    extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                    dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                        env_output, extracted_obs
                    )

                    actions, result = self.predict(extracted_obs)
                    
                    # Apply gripper penalty to rewards (bypasses env reward since combine_mode="replace")
                    rewards = self.apply_gripper_penalty_to_rewards(env_output, actions, rewards)

                    # Check if this is a reset frame (marked by env worker when using reward model)
                    is_reset_frame = env_output.get("is_reset_frame", False)
                    
                    # Get reward-based termination signal (only present when using reward model)
                    reward_terminations = env_output.get("reward_terminations")
                    has_reward_termination = reward_terminations is not None and reward_terminations.any().item()
                    
                    if not is_first_step:
                        if is_reset_frame:
                            # Skip buffer add for reset frame to avoid cross-episode mixing
                            logger.debug(f"[Rollout] Skipping reset frame for stage {stage_id}")
                            just_reset[stage_id] = True
                            waiting_for_reset[stage_id] = False
                        else:
                            # Determine if we need to force dones=False
                            is_first_after_reset = just_reset[stage_id]
                            is_waiting_for_reset = waiting_for_reset[stage_id]
                            
                            if is_first_after_reset or is_waiting_for_reset:
                                if is_first_after_reset:
                                    logger.debug(f"[Rollout] First frame after reset for stage {stage_id}, forcing dones=False")
                                    just_reset[stage_id] = False
                                    waiting_for_reset[stage_id] = False
                                else:
                                    logger.debug(f"[Rollout] Waiting for reset for stage {stage_id}, forcing dones=False")
                                # Force terminations and dones to False
                                forced_terminations = torch.zeros_like(env_output["terminations"])
                                forced_dones = torch.zeros_like(dones)
                            else:
                                forced_terminations = env_output["terminations"]
                                forced_dones = dones
                            
                            await self.buffer_list[stage_id].add(
                                "truncations",
                                env_output["truncations"].bool().cpu().contiguous(),
                            )
                            await self.buffer_list[stage_id].add(
                                "terminations",
                                forced_terminations.bool().cpu().contiguous(),
                            )
                            await self.buffer_list[stage_id].add("dones", forced_dones)
                            
                            if rewards is not None:
                                await self.buffer_list[stage_id].add("rewards", rewards)
                            
                            if last_results[stage_id] is not None:
                                await self.buffer_list[stage_id].add_result(
                                    last_results[stage_id]
                                )

                            # Add transition
                            if last_extracted_obs[stage_id] is not None and hasattr(
                                self.hf_model, "q_head"
                            ):
                                await self.buffer_list[stage_id].add_transition(
                                    last_extracted_obs[stage_id], real_extracted_obs
                                )
                    else:
                        is_first_step = False

                    # Mark waiting for reset when reward termination is detected
                    if has_reward_termination and not waiting_for_reset[stage_id]:
                        logger.debug(f"[Rollout] Reward termination detected for stage {stage_id}")
                        waiting_for_reset[stage_id] = True

                    last_extracted_obs[stage_id] = extracted_obs
                    last_results[stage_id] = result

                    self.send_chunk_actions(output_channel, actions, reward_terminations=reward_terminations)

            # Process remaining frames at epoch end
            for i in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)
                if last_results[i] is not None:
                    last_results[i]["forward_inputs"] = (
                        self.update_intervene_actions(
                            env_output, last_results[i]["forward_inputs"]
                        )
                    )
                
                extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                    env_output, extracted_obs
                )
                
                with self.worker_timer():
                    actions, result = self.predict(extracted_obs)
                
                # Apply gripper penalty to rewards
                rewards = self.apply_gripper_penalty_to_rewards(env_output, actions, rewards)
                
                is_reset_frame = env_output.get("is_reset_frame", False)
                
                if is_reset_frame:
                    logger.debug(f"[Rollout] Skipping reset frame at epoch end for stage {i}")
                    just_reset[i] = True
                    waiting_for_reset[i] = False
                else:
                    is_first_after_reset = just_reset[i]
                    is_waiting_for_reset = waiting_for_reset[i]
                    
                    if is_first_after_reset or is_waiting_for_reset:
                        if is_first_after_reset:
                            logger.debug(f"[Rollout] First frame after reset at epoch end for stage {i}")
                            just_reset[i] = False
                            waiting_for_reset[i] = False
                        forced_terminations = torch.zeros_like(env_output["terminations"])
                        forced_dones = torch.zeros_like(dones)
                    else:
                        forced_terminations = env_output["terminations"]
                        forced_dones = dones
                    
                    await self.buffer_list[i].add(
                        "truncations", env_output["truncations"].bool().cpu().contiguous()
                    )
                    await self.buffer_list[i].add(
                        "terminations", forced_terminations.bool().cpu().contiguous()
                    )
                    await self.buffer_list[i].add("dones", forced_dones)
                    if rewards is not None:
                        await self.buffer_list[i].add("rewards", rewards)
                    if last_results[i] is not None:
                        await self.buffer_list[i].add_result(
                            put_tensor_device(last_results[i], "cpu")
                        )
                    if "prev_values" in result:
                        await self.buffer_list[i].add(
                            "prev_values", result["prev_values"].cpu().contiguous()
                        )
                    if hasattr(self.hf_model, "q_head"):
                        await self.buffer_list[i].add_transition(
                            last_extracted_obs[i], real_extracted_obs
                        )
                
                # Mark waiting for reset when reward termination is detected
                reward_terminations = env_output.get("reward_terminations")
                if reward_terminations is not None and reward_terminations.any().item():
                    if not waiting_for_reset[i]:
                        logger.debug(f"[Rollout] Reward termination detected at epoch end for stage {i}")
                        waiting_for_reset[i] = True

            progress_bar.update(1)

    async def stop(self):
        self.should_stop = True
        for buffer in self.buffer_list:
            await buffer.stop()
        await asyncio.gather(*self.buffer_tasks)
