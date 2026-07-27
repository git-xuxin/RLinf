# Copyright 2026 The RLinf Authors.
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

"""FSDP worker for Residual Flow Policy Optimization."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
from rlinf.scheduler import Channel, Worker
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


def _align_transition_rows(
    value: Any,
    *,
    transition_count: int,
    epoch_count: int,
    drop_initial_bootstrap: bool,
) -> Any:
    """Remove RLinf's non-transition bootstrap row from each rollout epoch."""
    if isinstance(value, Mapping):
        return {
            key: _align_transition_rows(
                nested,
                transition_count=transition_count,
                epoch_count=epoch_count,
                drop_initial_bootstrap=drop_initial_bootstrap,
            )
            for key, nested in value.items()
        }
    if not isinstance(value, torch.Tensor):
        return value
    if value.ndim < 2:
        raise ValueError(
            "RFPO trajectory tensors must have time and batch dimensions."
        )
    if value.shape[0] == transition_count:
        return value
    if value.shape[0] != transition_count + epoch_count:
        raise ValueError(
            "RFPO trajectory field has an incompatible time dimension: "
            f"{value.shape[0]}."
        )
    if transition_count % epoch_count:
        raise ValueError(
            "RFPO transition count must be divisible by rollout epoch count."
        )
    epoch_length = transition_count // epoch_count
    per_epoch = value.reshape(epoch_count, epoch_length + 1, *value.shape[1:])
    per_epoch = per_epoch[:, 1:] if drop_initial_bootstrap else per_epoch[:, :-1]
    return per_epoch.reshape(transition_count, *value.shape[1:]).contiguous()


def _slice_transition_rows(
    value: Any,
    *,
    start: int,
    length: int,
    env_indices: torch.Tensor,
    transition_count: int,
    batch_size: int,
) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _slice_transition_rows(
                nested,
                start=start,
                length=length,
                env_indices=env_indices,
                transition_count=transition_count,
                batch_size=batch_size,
            )
            for key, nested in value.items()
        }
    if not isinstance(value, torch.Tensor):
        return value
    if value.shape[:2] != (transition_count, batch_size):
        raise ValueError("Aligned RFPO trajectory fields must share [T, B].")
    return value[start : start + length, env_indices].contiguous()


def _split_valid_trajectories(trajectory: Trajectory) -> list[Trajectory]:
    """Drop padded policy chunks recorded after an environment first finishes."""
    rewards = trajectory.rewards
    if not isinstance(rewards, torch.Tensor) or rewards.ndim < 2:
        raise ValueError("RFPO trajectories require rewards with shape [T, B, ...].")
    transition_count, batch_size = rewards.shape[:2]
    done_reference = trajectory.dones
    if not isinstance(done_reference, torch.Tensor) or done_reference.ndim < 2:
        raise ValueError("RFPO trajectories require chunk-level done tensors.")
    epoch_count = int(done_reference.shape[0] - transition_count)
    if epoch_count < 0:
        raise ValueError("RFPO done fields cannot be shorter than rewards.")
    epoch_count = max(epoch_count, 1)
    if transition_count % epoch_count:
        raise ValueError(
            "RFPO transition count must be divisible by rollout epoch count."
        )

    done_fields = {"dones", "terminations", "truncations"}
    aligned = {}
    for field_name in trajectory.__dataclass_fields__:
        value = getattr(trajectory, field_name)
        if value is None or isinstance(value, (int, str)):
            aligned[field_name] = value
        else:
            aligned[field_name] = _align_transition_rows(
                value,
                transition_count=transition_count,
                epoch_count=epoch_count,
                drop_initial_bootstrap=field_name in done_fields,
            )

    done_by_chunk = aligned["dones"].reshape(
        transition_count, batch_size, -1
    ).bool().any(dim=-1)
    epoch_length = transition_count // epoch_count
    filtered = []
    for epoch_index in range(epoch_count):
        start = epoch_index * epoch_length
        epoch_done = done_by_chunk[start : start + epoch_length]
        has_done = epoch_done.any(dim=0)
        first_done = epoch_done.to(torch.int64).argmax(dim=0) + 1
        valid_lengths = torch.where(
            has_done,
            first_done,
            torch.full_like(first_done, epoch_length),
        )
        for valid_length_tensor in torch.unique(valid_lengths, sorted=True):
            valid_length = int(valid_length_tensor.item())
            env_indices = torch.nonzero(
                valid_lengths == valid_length_tensor, as_tuple=False
            ).squeeze(1)
            values = {}
            for field_name, value in aligned.items():
                if value is None or isinstance(value, (int, str)):
                    values[field_name] = value
                else:
                    values[field_name] = _slice_transition_rows(
                        value,
                        start=start,
                        length=valid_length,
                        env_indices=env_indices,
                        transition_count=transition_count,
                        batch_size=batch_size,
                    )
            filtered.append(Trajectory(**values))
    return filtered


class EmbodiedRFPOFSDPPolicy(EmbodiedSACFSDPPolicy):
    """Synchronous trajectory actor-critic worker for RFPO."""

    def init_worker(self):
        if self.cfg.actor.get("enable_offload", False):
            raise ValueError("RFPO initial implementation does not support offload.")
        if self.cfg.actor.get("compile_model", False):
            raise ValueError(
                "RFPO initial implementation does not support compile_model."
            )
        if self.cfg.actor.fsdp_config.sharding_strategy != "no_shard":
            raise ValueError("RFPO initial implementation requires FSDP no_shard.")
        if not self.cfg.actor.fsdp_config.use_orig_params:
            raise ValueError(
                "RFPO initial implementation requires use_orig_params=True."
            )
        if self.cfg.actor.fsdp_config.get("disable", False):
            raise ValueError(
                "RFPO requires FSDP auto-wrap so the residual actor and critic "
                "remain separate from the mixed-dtype frozen pi0 parameters."
            )
        self.setup_model_and_optimizer()
        self.setup_sac_components()
        self.soft_update_target_critic(tau=1.0)

    def setup_model_and_optimizer(self, initialize_target=False) -> None:
        del initialize_target
        module = self.model_provider_func()
        if not hasattr(module, "residual_actor") or not hasattr(
            module, "online_critic"
        ):
            raise ValueError(
                "RFPO model must expose residual_actor and online_critic modules."
            )
        self.target_critic = module.online_critic.target_copy()
        trainable_names = [
            name for name, param in module.named_parameters() if param.requires_grad
        ]
        unexpected_trainable_names = [
            name
            for name in trainable_names
            if not name.startswith(("residual_actor.", "online_critic."))
        ]
        if unexpected_trainable_names:
            raise ValueError(
                "RFPO found trainable parameters outside residual actor and critic: "
                f"{unexpected_trainable_names[:8]}"
            )
        self.param_names_need_sync = [
            name
            for name in trainable_names
            if name.startswith("residual_actor.")
        ]
        if not self.param_names_need_sync:
            raise ValueError("RFPO residual actor has no trainable parameters.")
        if not any(name.startswith("online_critic.") for name in trainable_names):
            raise ValueError("RFPO online critic has no trainable parameters.")

        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        self.target_critic.to(device=self.device, dtype=torch.float32)
        self.target_critic.eval()
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype

        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters={"critic": ["online_critic"]},
            filtered_optim_config={"critic": self.cfg.actor.critic_optim},
        )
        self.optimizer, self.qf_optimizer = optimizers
        self.build_lr_schedulers()
        self.grad_scaler = self.build_grad_scaler(
            self.cfg.actor.fsdp_config.grad_scaler.get("enabled", False),
            **{
                key: value
                for key in ("init_scale", "growth_interval")
                if (
                    value := self.cfg.actor.fsdp_config.grad_scaler.get(key, None)
                )
                is not None
            },
        )
        entropy_config = self.cfg.algorithm.get("entropy_tuning", {})
        alpha_type = entropy_config.get("alpha_type", "fixed_alpha")
        if alpha_type != "fixed_alpha":
            raise ValueError(
                "RFPO initial implementation supports only fixed entropy alpha."
            )
        alpha = float(
            entropy_config.get(
                "initial_alpha", self.cfg.algorithm.get("entropy_alpha", 0.0)
            )
        )
        self.entropy_temp = EntropyTemperature(
            initial_alpha=alpha,
            alpha_type=alpha_type,
            device=self.device,
            dtype=torch.float32,
        )
        self.alpha_optimizer = None
        self.target_model_initialized = True
        self.use_dsrl = False

    def _unwrapped_model(self):
        return getattr(self.model, "module", self.model)

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """Receive rollout trajectories and keep only valid RFPO transitions."""
        clear_memory(sync=False)
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)
        trajectories = []
        for _ in range(split_num):
            trajectory = await input_channel.get(async_op=True).async_wait()
            trajectories.extend(_split_valid_trajectories(trajectory))
        self.replay_buffer.add_trajectories(trajectories)

    def _pi0_grad_norm(self) -> float:
        squared_norm = torch.zeros((), device=self.device, dtype=torch.float32)
        model = self._unwrapped_model()
        trainable_parameter_ids = {
            id(parameter)
            for parameter in (
                *model.residual_actor.parameters(),
                *model.online_critic.parameters(),
            )
        }
        for parameter in model.parameters():
            if id(parameter) in trainable_parameter_ids:
                continue
            if parameter.grad is not None:
                squared_norm = squared_norm + parameter.grad.float().pow(2).sum()
        grad_norm = squared_norm.sqrt().item()
        if grad_norm != 0.0:
            raise RuntimeError(f"Frozen pi0 received RFPO gradients: norm={grad_norm}.")
        return grad_norm

    @staticmethod
    def _prompt_inputs(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        forward_inputs = batch.get("forward_inputs", {})
        try:
            return (
                forward_inputs["tokenized_prompt"],
                forward_inputs["tokenized_prompt_mask"],
            )
        except KeyError as exc:
            raise ValueError(
                "RFPO replay requires cached tokenized_prompt and "
                "tokenized_prompt_mask."
            ) from exc

    def _reshape_actions(self, actions: torch.Tensor) -> torch.Tensor:
        action_chunk = int(self.cfg.actor.model.num_action_chunks)
        model_action_dim = int(self._unwrapped_model().config.action_dim)
        expected_flat_dim = action_chunk * model_action_dim
        if actions.ndim != 2 or actions.shape[1] != expected_flat_dim:
            raise ValueError(
                "RFPO replay actions must contain one flattened normalized model "
                f"chunk of size {expected_flat_dim}, got {tuple(actions.shape)}."
            )
        return actions.reshape(actions.shape[0], action_chunk, model_action_dim)

    def _chunk_target_terms(
        self, rewards: torch.Tensor, dones: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rewards = rewards.reshape(rewards.shape[0], -1).float()
        dones = dones.reshape(dones.shape[0], -1).to(dtype=torch.bool)
        if rewards.shape != dones.shape or rewards.shape[1] == 0:
            raise ValueError(
                "RFPO chunk rewards and dones must have the same non-empty shape."
            )
        horizon = rewards.shape[1]
        done_prefix = dones.cumsum(dim=1)
        valid_mask = torch.cat(
            [
                torch.ones_like(done_prefix[:, :1], dtype=torch.bool),
                done_prefix[:, :-1] == 0,
            ],
            dim=1,
        )
        gamma_powers = torch.arange(
            horizon, device=rewards.device, dtype=torch.float32
        )
        discounted_reward = (
            rewards
            * valid_mask.to(dtype=rewards.dtype)
            * (float(self.cfg.algorithm.gamma) ** gamma_powers)[None]
        ).sum(dim=1, keepdim=True)
        valid_steps = valid_mask.sum(dim=1, keepdim=True)
        bootstrap_discount = float(self.cfg.algorithm.gamma) ** valid_steps.float()
        return discounted_reward, bootstrap_discount, valid_mask

    @torch.no_grad()
    def soft_update_target_critic(self, tau=None):
        if tau is None:
            tau = float(self.cfg.algorithm.tau)
        online_critic = self._unwrapped_model().online_critic
        online_parameters = {
            name.replace("_fsdp_wrapped_module.", ""): parameter
            for name, parameter in online_critic.named_parameters()
        }
        target_parameters = dict(self.target_critic.named_parameters())
        if online_parameters.keys() != target_parameters.keys():
            raise RuntimeError(
                "RFPO online and target critic parameter structures do not match."
            )
        for name, target_parameter in target_parameters.items():
            target_parameter.data.lerp_(online_parameters[name].data.float(), tau)

    def forward_critic(self, batch):
        tokenized_prompt, tokenized_prompt_mask = self._prompt_inputs(batch)
        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = self._reshape_actions(batch["actions"])
        rewards, bootstrap_discount, valid_mask = self._chunk_target_terms(
            batch["rewards"], batch["dones"]
        )
        terminations = batch["terminations"].reshape(
            batch["terminations"].shape[0], -1
        ).to(dtype=torch.bool)
        if terminations.shape != valid_mask.shape:
            raise ValueError(
                "RFPO chunk terminations must match rewards and dones shape."
            )
        terminated = (terminations & valid_mask).any(dim=1, keepdim=True)

        with torch.no_grad():
            next_output = self.model(
                forward_type=ForwardType.RFPO_ACTOR,
                obs=next_obs,
                tokenized_prompt=tokenized_prompt,
                tokenized_prompt_mask=tokenized_prompt_mask,
            )
            target_q_values = self.target_critic(
                next_output["actions"],
                state=next_output["critic_state"],
                condition_tokens=next_output["critic_condition_tokens"],
                condition_mask=next_output["critic_condition_mask"],
            ).min(dim=-1, keepdim=True).values
            target = rewards + (
                ~terminated
            ) * bootstrap_discount * target_q_values.float()

        current_q_values = self.model(
            forward_type=ForwardType.RFPO_Q,
            obs=curr_obs,
            actions=actions,
            action_mask=valid_mask,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )
        critic_loss = F.mse_loss(
            current_q_values.float(), target.expand_as(current_q_values).float()
        )
        metrics = {
            "q1": current_q_values[:, 0].mean().item(),
            "q2": current_q_values[:, 1].mean().item(),
            "target_q": target.mean().item(),
        }
        return critic_loss, metrics

    def forward_actor(self, batch):
        tokenized_prompt, tokenized_prompt_mask = self._prompt_inputs(batch)
        output = self.model(
            forward_type=ForwardType.RFPO_ACTOR,
            obs=batch["curr_obs"],
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            retain_residual_grads=True,
            evaluate_q=True,
            compute_pi0_baseline=True,
        )
        q_min = output["q_values"].min(dim=-1, keepdim=True).values
        alpha = self.entropy_temp.compute_alpha().detach().float()
        actor_loss = (
            -q_min.float()
            + alpha * output["internal_log_prob"].float().unsqueeze(-1)
        ).mean()
        base_rms = output["base_velocity_rms"].float().clamp_min(1e-8)
        action_delta = output["actions"] - output["pi0_actions"]
        metrics = {
            "q_pi": q_min.mean().item(),
            "residual_rms": output["residual_rms"].mean().item(),
            "base_velocity_rms": output["base_velocity_rms"].mean().item(),
            "residual_to_base_ratio": (
                output["residual_rms"].float() / base_rms
            ).mean().item(),
            "internal_log_prob": output["internal_log_prob"].mean().item(),
            "action_delta_from_pi0_rms": (
                action_delta.float().pow(2).mean().sqrt().item()
            ),
        }
        return actor_loss, -output["internal_log_prob"].mean(), metrics, output[
            "active_residuals"
        ]

    def update_one_epoch(self, train_actor: bool = True):
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )
        global_batch = next(self.buffer_dataloader_iter)
        micro_batches = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )
        micro_batches = [
            put_tensor_device(batch, device=self.device) for batch in micro_batches
        ]

        self.optimizer.zero_grad()
        self.qf_optimizer.zero_grad()
        critic_losses = []
        critic_metrics = {}
        for batch in micro_batches:
            critic_loss, metrics = self.forward_critic(batch)
            critic_loss = critic_loss / self.gradient_accumulation
            critic_loss.backward()
            critic_losses.append(critic_loss.item() * self.gradient_accumulation)
            append_to_dict(critic_metrics, metrics)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._unwrapped_model().online_critic.parameters(),
            max_norm=self.cfg.actor.critic_optim.clip_grad,
        )
        self.qf_optimizer.step()
        self.qf_lr_scheduler.step()

        metrics_data = {
            "rfpo/critic_loss": np.mean(critic_losses),
            "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": critic_grad_norm,
            **{
                f"rfpo/{key}": np.mean(value)
                for key, value in critic_metrics.items()
            },
        }

        if self.update_step % self.critic_actor_ratio == 0 and train_actor:
            self.optimizer.zero_grad()
            self.qf_optimizer.zero_grad()
            actor_losses = []
            entropies = []
            actor_metrics = {}
            active_step_grad_norms: dict[int, list[float]] = {}
            for batch in micro_batches:
                actor_loss, entropy, metrics, active_residuals = self.forward_actor(
                    batch
                )
                actor_loss = actor_loss / self.gradient_accumulation
                actor_loss.backward()
                actor_losses.append(actor_loss.item() * self.gradient_accumulation)
                entropies.append(entropy.item())
                append_to_dict(actor_metrics, metrics)
                for step_idx, residual in active_residuals:
                    if residual.grad is not None:
                        active_step_grad_norms.setdefault(step_idx, []).append(
                            residual.grad.float().norm().item()
                        )
            self.qf_optimizer.zero_grad()
            pi0_grad_norm = self._pi0_grad_norm()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self._unwrapped_model().residual_actor.parameters(),
                max_norm=self.cfg.actor.optim.clip_grad,
            )
            self.optimizer.step()
            self.lr_scheduler.step()
            metrics_data.update(
                {
                    "rfpo/actor_loss": np.mean(actor_losses),
                    "rfpo/internal_residual_entropy": np.mean(entropies),
                    "rfpo/residual_actor_grad_norm": actor_grad_norm,
                    "rfpo/pi0_grad_norm": pi0_grad_norm,
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    **{
                        f"rfpo/{key}": np.mean(value)
                        for key, value in actor_metrics.items()
                    },
                    **{
                        f"rfpo/active_step_grad_norm/{step_idx}": np.mean(values)
                        for step_idx, values in active_step_grad_norms.items()
                    },
                }
            )

        if self.update_step % self.cfg.algorithm.get("target_update_freq", 1) == 0:
            self.soft_update_target_critic()
        return metrics_data

    def save_checkpoint(self, save_base_path, step):
        del step
        restore_weight_offload = self.is_weight_offloaded
        restore_optimizer_offload = self.is_optimizer_offloaded
        if restore_weight_offload:
            self.load_param_and_grad(self.device)
        if restore_optimizer_offload:
            self.load_optimizer(self.device)
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            save_path=save_base_path,
            checkpoint_format="local_shard",
        )
        component_dir = os.path.join(save_base_path, "rfpo_components")
        os.makedirs(component_dir, exist_ok=True)
        torch.save(
            self.target_critic.state_dict(),
            os.path.join(component_dir, f"target_critic_rank_{self._rank}.pt"),
        )
        torch.save(
            {"update_step": self.update_step},
            os.path.join(component_dir, f"worker_state_rank_{self._rank}.pt"),
        )
        self.replay_buffer.save_checkpoint(
            os.path.join(component_dir, f"replay_buffer/rank_{self._rank}")
        )
        if restore_weight_offload:
            self.offload_param_and_grad()
        if restore_optimizer_offload:
            self.offload_optimizer()

    def load_checkpoint(self, load_base_path):
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            load_path=load_base_path,
            checkpoint_format="local_shard",
        )
        component_dir = os.path.join(load_base_path, "rfpo_components")
        self.target_critic.load_state_dict(
            torch.load(
                os.path.join(component_dir, f"target_critic_rank_{self._rank}.pt"),
                map_location=self.device,
            )
        )
        worker_state_path = os.path.join(
            component_dir, f"worker_state_rank_{self._rank}.pt"
        )
        if os.path.exists(worker_state_path):
            worker_state = torch.load(worker_state_path, map_location="cpu")
            self.update_step = int(worker_state.get("update_step", 0))
        self.replay_buffer.load_checkpoint(
            os.path.join(component_dir, f"replay_buffer/rank_{self._rank}")
        )
