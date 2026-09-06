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

import math
import os
from collections.abc import Mapping
from numbers import Real

import numpy as np
import torch
import torch.nn.functional as F

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
from rlinf.models.embodiment.openpi_rfpo.rfpo_sampler import (
    RFPO_ACTION_GROUP_NAMES,
    compute_rfpo_raw_mean_l2,
    reduce_rfpo_action_groups,
)
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


def parse_raw_mean_l2_coefficients(
    config: Mapping | None,
) -> tuple[float, float, float]:
    """Validate RFPO raw-mean L2 coefficients in action-group order."""
    if config is None:
        return (0.0, 0.0, 0.0)
    if not isinstance(config, Mapping):
        raise ValueError("algorithm.raw_mean_l2_coefficients must be a mapping.")

    unknown_keys = set(config) - set(RFPO_ACTION_GROUP_NAMES)
    if unknown_keys:
        raise ValueError(
            f"Unsupported RFPO raw-mean L2 groups: {sorted(unknown_keys)}."
        )

    coefficients = []
    for group_name in RFPO_ACTION_GROUP_NAMES:
        value = config.get(group_name, 0.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError(
                "RFPO raw-mean L2 coefficient "
                f"'{group_name}' must be finite and non-negative."
            )
        coefficients.append(float(value))
    return tuple(coefficients)


def compute_rfpo_actor_loss(
    q_min: torch.Tensor,
    internal_log_prob: torch.Tensor,
    alpha: torch.Tensor,
    raw_mean_group_mse_per_step: torch.Tensor,
    active_step_mask: torch.Tensor,
    raw_mean_l2_coefficients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute RFPO actor loss and its non-redundant logged components."""
    actor_q_loss = -q_min.float().mean()
    actor_entropy_loss = (
        alpha.float() * internal_log_prob.float().unsqueeze(-1)
    ).mean()
    base_actor_loss = (
        -q_min.float() + alpha.float() * internal_log_prob.float().unsqueeze(-1)
    ).mean()
    raw_mean_l2_loss, raw_mean_group_mse, weighted_raw_mean_l2 = (
        compute_rfpo_raw_mean_l2(
            raw_mean_group_mse_per_step,
            active_step_mask,
            raw_mean_l2_coefficients,
        )
    )
    return (
        base_actor_loss + raw_mean_l2_loss,
        actor_q_loss,
        actor_entropy_loss,
        raw_mean_group_mse,
        weighted_raw_mean_l2,
    )


def compute_rfpo_alpha_loss(
    internal_log_prob: torch.Tensor,
    alpha: torch.Tensor,
    target_entropy: float,
) -> torch.Tensor:
    """Compute the RFPO temperature loss without policy gradients."""
    entropy_error = internal_log_prob.detach().float().mean() + float(target_entropy)
    return -alpha.float() * entropy_error


def get_rfpo_default_target_entropy(model_config) -> float:
    """Return target entropy in the sampler's internal log-prob units."""
    entropy_dim = int(model_config.rfpo_action_chunk) * int(
        model_config.rfpo_action_dim
    )
    reduction = model_config.internal_log_prob_reduction
    if reduction == "sum_active":
        entropy_dim *= len(model_config.active_step_indices)
    elif reduction != "mean_active":
        raise ValueError(
            "RFPO internal_log_prob_reduction must be 'mean_active' or 'sum_active'."
        )
    return -float(entropy_dim)


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
        self.raw_mean_l2_coefficients = torch.tensor(
            parse_raw_mean_l2_coefficients(
                self.cfg.algorithm.get("raw_mean_l2_coefficients", None)
            ),
            device=self.device,
            dtype=torch.float32,
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
            name for name in trainable_names if name.startswith("residual_actor.")
        ]
        if not self.param_names_need_sync:
            raise ValueError("RFPO residual actor has no trainable parameters.")
        if not any(name.startswith("online_critic.") for name in trainable_names):
            raise ValueError("RFPO online critic has no trainable parameters.")

        # Frozen pi0 must remain in the actor forward graph so value gradients
        # can flow through the denoising trajectory, but its parameters must
        # not enter FSDP flat handles. Frozen handles do not participate in the
        # critic backward pass and may otherwise retain unsharded parameter
        # views, causing a shape writeback failure on the next actor forward.
        module._fsdp_ignored_parameters = tuple(
            parameter
            for parameter in module.parameters()
            if not parameter.requires_grad
        )
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
        self.grad_scaler = self.build_grad_scaler(
            self.cfg.actor.fsdp_config.grad_scaler.get("enabled", False),
            **{
                key: value
                for key in ("init_scale", "growth_interval")
                if (value := self.cfg.actor.fsdp_config.grad_scaler.get(key, None))
                is not None
            },
        )
        entropy_config = self.cfg.algorithm.get("entropy_tuning", {})
        alpha_type = entropy_config.get("alpha_type", "fixed_alpha")
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
        if alpha_type != "fixed_alpha":
            self.target_entropy = float(
                entropy_config.get(
                    "target_entropy",
                    get_rfpo_default_target_entropy(module.config),
                )
            )
            if not math.isfinite(self.target_entropy):
                raise ValueError("RFPO target_entropy must be finite.")
            self.alpha_optimizer = torch.optim.Adam(
                self.entropy_temp.parameters(),
                lr=entropy_config.optim.lr,
            )
        self.build_lr_schedulers()
        self.target_model_initialized = True
        self.use_dsrl = False

    def _unwrapped_model(self):
        return getattr(self.model, "module", self.model)

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
        model_config = self._unwrapped_model().config
        action_chunk = int(model_config.rfpo_action_chunk)
        action_dim = int(model_config.rfpo_action_dim)
        expected_flat_dim = action_chunk * action_dim
        if actions.ndim != 2 or actions.shape[1] != expected_flat_dim:
            raise ValueError(
                "RFPO replay actions must contain one flattened normalized model "
                f"chunk of size {expected_flat_dim}, got {tuple(actions.shape)}."
            )
        return actions.reshape(actions.shape[0], action_chunk, action_dim)

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
        gamma_powers = torch.arange(horizon, device=rewards.device, dtype=torch.float32)
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
        terminations = (
            batch["terminations"]
            .reshape(batch["terminations"].shape[0], -1)
            .to(dtype=torch.bool)
        )
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
            target_q_values = (
                self.target_critic(
                    next_output["actions"],
                    state_embedding=next_output["critic_state_embedding"],
                    condition_tokens=next_output["critic_condition_tokens"],
                    condition_mask=next_output["critic_condition_mask"],
                )
                .min(dim=-1, keepdim=True)
                .values
            )
            if self.cfg.algorithm.get("backup_entropy", True):
                next_log_prob = next_output["internal_log_prob"].float().unsqueeze(-1)
                alpha = self.entropy_temp.compute_alpha().detach().float()
                target_q_values = target_q_values.float() - alpha * next_log_prob
            target = (
                rewards + (~terminated) * bootstrap_discount * target_q_values.float()
            )

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
            "q_data": current_q_values.mean().item(),
            "q_target": target.mean().item(),
            "q_disagreement": (current_q_values[:, 0] - current_q_values[:, 1])
            .abs()
            .mean()
            .item(),
        }
        return critic_loss, metrics

    def forward_actor(self, batch):
        tokenized_prompt, tokenized_prompt_mask = self._prompt_inputs(batch)
        output = self.model(
            forward_type=ForwardType.RFPO_ACTOR,
            obs=batch["curr_obs"],
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            evaluate_q=True,
            compute_pi0_baseline=True,
        )
        q_min = output["q_values"].min(dim=-1, keepdim=True).values
        alpha = self.entropy_temp.compute_alpha().detach().float()
        (
            actor_loss,
            actor_q_loss,
            actor_entropy_loss,
            raw_mean_group_mse,
            weighted_raw_mean_l2,
        ) = compute_rfpo_actor_loss(
            q_min,
            output["internal_log_prob"],
            alpha,
            output["raw_mean_group_mse_per_step"],
            output["active_step_mask"],
            self.raw_mean_l2_coefficients,
        )
        action_delta = output["actions"] - output["pi0_actions"]
        action_delta_group_rms = reduce_rfpo_action_groups(
            action_delta, "rms", preserve_leading_dims=0
        )
        metrics = {
            f"action_delta_from_pi0/{group_name}_rms": group_rms.item()
            for group_name, group_rms in zip(
                RFPO_ACTION_GROUP_NAMES, action_delta_group_rms, strict=True
            )
        }
        metrics.update(
            {
                "actor_loss/q": actor_q_loss.item(),
                "actor_loss/entropy": actor_entropy_loss.item(),
            }
        )
        for group_name, group_mse, weighted_loss in zip(
            RFPO_ACTION_GROUP_NAMES,
            raw_mean_group_mse,
            weighted_raw_mean_l2,
            strict=True,
        ):
            metrics[f"raw_mean_l2/{group_name}_mse"] = group_mse.item()
            metrics[f"actor_loss/raw_mean_l2/{group_name}"] = weighted_loss.item()
        base_group_rms_per_step = output["base_velocity_group_rms_per_step"]
        per_step_metrics = {
            ("base_velocity", "rms"): base_group_rms_per_step,
            ("delta_velocity", "rms"): output["delta_velocity_group_rms_per_step"],
            ("residual_mean", "abs_mean"): output["mean_group_abs_mean_per_step"],
            ("residual_log_std", "mean"): output["log_std_group_mean_per_step"],
            ("mean_tanh_saturation", "fraction"): output[
                "mean_tanh_group_saturation_fraction_per_step"
            ],
            ("log_std_tanh_saturation", "fraction"): output[
                "log_std_tanh_group_saturation_fraction_per_step"
            ],
        }
        expected_metric_shape = (
            base_group_rms_per_step.shape[0],
            base_group_rms_per_step.shape[1],
            len(RFPO_ACTION_GROUP_NAMES),
        )
        if base_group_rms_per_step.shape != expected_metric_shape:
            raise ValueError(
                "RFPO per-step metrics must have shape "
                "[batch, denoise_steps, action_groups]."
            )
        for (metric_name, _), metric_values in per_step_metrics.items():
            if metric_values.shape != expected_metric_shape:
                raise ValueError(
                    f"RFPO {metric_name} must match grouped base-velocity shape."
                )
        active_step_mask = output["active_step_mask"]
        if active_step_mask.shape != (base_group_rms_per_step.shape[1],):
            raise ValueError("RFPO active_step_mask must match denoise steps.")
        per_step_means = {
            metric_spec: values.mean(dim=0).detach().cpu()
            for metric_spec, values in per_step_metrics.items()
        }
        for step_idx, is_active in enumerate(active_step_mask.detach().cpu().tolist()):
            if not is_active:
                continue
            prefix = f"denoise_step_{step_idx}"
            for (metric_name, statistic), metric_values in per_step_means.items():
                for group_idx, group_name in enumerate(RFPO_ACTION_GROUP_NAMES):
                    metrics[f"{prefix}/{metric_name}/{group_name}_{statistic}"] = (
                        metric_values[step_idx, group_idx].item()
                    )
        entropy = -output["internal_log_prob"].detach().float().mean()
        return actor_loss, entropy, metrics

    def forward_alpha(self, batch):
        tokenized_prompt, tokenized_prompt_mask = self._prompt_inputs(batch)
        with torch.no_grad():
            output = self.model(
                forward_type=ForwardType.RFPO_ACTOR,
                obs=batch["curr_obs"],
                tokenized_prompt=tokenized_prompt,
                tokenized_prompt_mask=tokenized_prompt_mask,
            )
        return compute_rfpo_alpha_loss(
            output["internal_log_prob"],
            self.entropy_temp.compute_alpha(),
            self.target_entropy,
        )

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
        # Use FSDP-aware clipping for parameters managed through flat handles.
        # At this point only critic parameters have gradients, so root-level
        # clipping still clips exactly the intended parameter set.
        critic_grad_norm = self.model.clip_grad_norm_(
            max_norm=self.cfg.actor.critic_optim.clip_grad
        )
        self.qf_optimizer.step()
        self.qf_lr_scheduler.step()

        metrics_data = {
            "rfpo/critic_loss": np.mean(critic_losses),
            "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": critic_grad_norm,
            **{f"rfpo/{key}": np.mean(value) for key, value in critic_metrics.items()},
        }

        if self.update_step % self.critic_actor_ratio == 0 and train_actor:
            self.optimizer.zero_grad()
            self.qf_optimizer.zero_grad()
            actor_losses = []
            entropies = []
            actor_metrics = {}
            for batch in micro_batches:
                actor_loss, entropy, metrics = self.forward_actor(batch)
                actor_loss = actor_loss / self.gradient_accumulation
                actor_loss.backward()
                actor_losses.append(actor_loss.item() * self.gradient_accumulation)
                entropies.append(entropy.item())
                append_to_dict(actor_metrics, metrics)
            self.qf_optimizer.zero_grad()
            self._pi0_grad_norm()
            # Critic gradients were cleared above, hence FSDP root-level
            # clipping applies only to the residual actor in this phase.
            actor_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()
            self.lr_scheduler.step()

            alpha_losses = [0.0]
            alpha_grad_norm = 0.0
            if self.alpha_optimizer is not None:
                self.alpha_optimizer.zero_grad()
                alpha_losses = []
                for batch in micro_batches:
                    alpha_loss = self.forward_alpha(batch) / self.gradient_accumulation
                    alpha_loss.backward()
                    alpha_losses.append(alpha_loss.item() * self.gradient_accumulation)
                torch.distributed.all_reduce(
                    self.entropy_temp.base_alpha.grad,
                    op=torch.distributed.ReduceOp.AVG,
                )
                alpha_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.entropy_temp.base_alpha,
                    self.cfg.algorithm.entropy_tuning.optim.clip_grad,
                )
                self.alpha_optimizer.step()
                self.alpha_lr_scheduler.step()
            metrics_data.update(
                {
                    "rfpo/actor_loss": np.mean(actor_losses),
                    "rfpo/internal_residual_entropy": np.mean(entropies),
                    "rfpo/alpha_loss": np.mean(alpha_losses),
                    "rfpo/alpha": self.entropy_temp.alpha,
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "actor/grad_norm": actor_grad_norm,
                    "alpha/grad_norm": alpha_grad_norm,
                    **{
                        f"rfpo/{key}": np.mean(value)
                        for key, value in actor_metrics.items()
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
        if self.alpha_optimizer is not None:
            self._strategy.save_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                save_path=os.path.join(component_dir, "alpha"),
                save_full_model_weights=False,
            )
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
        if self.alpha_optimizer is not None:
            self._strategy.load_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                load_path=os.path.join(component_dir, "alpha"),
            )
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
