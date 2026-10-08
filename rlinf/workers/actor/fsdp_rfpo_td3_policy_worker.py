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
import math
from collections.abc import Mapping
from numbers import Real

import numpy as np
import torch
import torch.nn.functional as F

from rlinf.models import get_model
from rlinf.models.embodiment.openpi_rfpo import build_model_config
from rlinf.models.embodiment.openpi_rfpo.backbone import (
    RFPOBackboneAdapter,
    load_backbone_model,
)
from rlinf.models.embodiment.openpi_rfpo.metrics import (
    RFPO_ACTION_GROUP_NAMES,
    denoise_step_metrics,
)
from rlinf.models.embodiment.openpi_rfpo.rfpo_sampler import compute_raw_mean_l2
from rlinf.scheduler import Worker
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.workers.actor.fsdp_td3_policy_worker import EmbodiedTD3FSDPPolicy


def _parse_raw_mean_l2_coefficients(
    config: Mapping | None,
) -> tuple[float, float, float]:
    """Parse non-negative L2 weights; missing action groups default to zero."""
    if config is None:
        return (0.0, 0.0, 0.0)
    if not isinstance(config, Mapping):
        raise ValueError("raw_mean_l2_coefficients must be a mapping.")

    unknown_groups = set(config) - set(RFPO_ACTION_GROUP_NAMES)
    if unknown_groups:
        raise ValueError(
            f"Unsupported raw_mean_l2_coefficients groups: {sorted(unknown_groups)}."
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
                "RFPO raw_mean_l2_coefficients entry "
                f"'{group_name}' must be finite and non-negative."
            )
        coefficients.append(float(value))
    return tuple(coefficients)


class EmbodiedRFPOTD3FSDPPolicy(EmbodiedTD3FSDPPolicy):
    """Train the residual actor and critic while keeping pi frozen."""

    def init_worker(self):
        # Load pi before model construction; it stays outside the FSDP policy.
        self.rfpo_backbone_model = load_backbone_model(
            self.cfg.actor.rfpo_backbone_model, self.device
        )
        self.rfpo_adapter = RFPOBackboneAdapter(self.rfpo_backbone_model)
        super().init_worker()
        self.raw_mean_l2_coefficients = torch.tensor(
            _parse_raw_mean_l2_coefficients(
                self.cfg.algorithm.get("raw_mean_l2_coefficients", None)
            ),
            device=self.device,
            dtype=torch.float32,
        )

    def model_provider_func(self) -> torch.nn.Module:
        model_config = build_model_config(self.cfg.actor.model, self.rfpo_adapter)
        model = get_model(model_config)
        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        return model

    def setup_model_and_optimizer(self, initialize_target=False) -> None:
        """Set up the RFPO policy, target policy, optimizers, and schedulers."""
        if not self.cfg.actor.fsdp_config.use_orig_params:
            raise ValueError(
                "TD3 requires use_orig_params=True for separate optimizers and targets."
            )
        if not self.cfg.actor.model.actor.get("deterministic", False):
            raise ValueError("RFPO TD3 requires a deterministic residual actor.")
        module = self.model_provider_func()
        if initialize_target:
            target_module = copy.deepcopy(module)

        # Sync both online networks and their persistent buffers, excluding pi.
        self.param_names_need_sync = list(module.state_dict())

        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        if initialize_target:
            self.target_model = self._strategy.wrap_model(
                model=target_module, device_mesh=self._device_mesh
            )
            self.target_model.requires_grad_(False)
            self.target_model.eval()
            self.target_model_initialized = True

        param_filters = {"critic": ["critic."]}
        filtered_optim_config = {"critic": self.cfg.actor.critic_optim}
        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters=param_filters,
            filtered_optim_config=filtered_optim_config,
        )
        self.optimizer = optimizers[0]
        self.qf_optimizer = optimizers[1]

        self.build_lr_schedulers()

        self.grad_scaler = self.build_grad_scaler(
            **self.cfg.actor.fsdp_config.grad_scaler
        )

    @torch.no_grad()
    def soft_update_target_model(
        self, tau: float | None = None, component="all"
    ) -> None:
        """Polyak-average the selected target networks and copy their buffers."""
        if tau is None:
            tau = self.cfg.algorithm.tau
        if component not in ("all", "actor", "critic"):
            raise ValueError(f"Unsupported target component: {component}")
        components = ("actor", "critic") if component == "all" else (component,)
        for name in components:
            online_module = getattr(self.model, name)
            target_module = getattr(self.target_model, name)
            for online_param, target_param in zip(
                online_module.parameters(), target_module.parameters(), strict=True
            ):
                target_param.lerp_(online_param, tau)
            for online_buffer, target_buffer in zip(
                online_module.buffers(), target_module.buffers(), strict=True
            ):
                target_buffer.copy_(online_buffer)

    def prepare_batch(self, batch: dict) -> dict:
        """Encode current/next observations and reshape normalized replay actions."""
        chunk, action_dim = self.rfpo_adapter.env_action_shape
        return {
            **batch,
            "actions": batch["actions"].reshape(-1, chunk, action_dim),
            "curr_condition": self.rfpo_adapter.encode_condition(
                self.rfpo_adapter.preprocess_replay(batch["curr_obs"])
            ),
            "next_condition": self.rfpo_adapter.encode_condition(
                self.rfpo_adapter.preprocess_replay(batch["next_obs"])
            ),
        }

    def sample_actions(self, condition, *, mode="train", **kwargs):
        """Sample with the target actor only when building critic targets."""
        model = self.target_model if mode == "target" else self.model
        if mode == "target":
            kwargs["noise_std"] = self.cfg.algorithm.get("target_noise_std", 0.1)
            kwargs["noise_clip"] = self.cfg.algorithm.get("target_noise_clip", 0.2)
        return model(
            component="actor",
            adapter=self.rfpo_adapter,
            condition=condition,
            mode=mode,
            **kwargs,
        )

    @torch.no_grad()
    def compute_target(self, batch: dict) -> torch.Tensor:
        """Sample next actions with the target actor and bootstrap with target Qs."""
        condition = batch["next_condition"]
        output = self.sample_actions(condition, mode="target")
        all_qf_next_target = self.target_model(
            component="critic",
            actions=output["actions"],
            condition_tokens=condition.tokens,
            condition_mask=condition.mask,
            state_embedding=condition.state_embedding,
        )
        subsample_size = self.cfg.algorithm.get("critic_subsample_size", 2)
        num_q_heads = all_qf_next_target.shape[-1]
        if not 1 <= subsample_size <= num_q_heads:
            raise ValueError(
                "RFPO critic_subsample_size must be between 1 and num_q_heads."
            )
        sample_idx = torch.randperm(
            num_q_heads,
            device=all_qf_next_target.device,
            generator=self.critic_sample_generator,
        )[:subsample_size]
        qf_next_target = (
            all_qf_next_target.float()
            .index_select(-1, sample_idx)
            .min(dim=-1, keepdim=True)
            .values
        )

        # Discount the full executed chunk; only terminations suppress bootstrapping.
        rewards = batch["rewards"].float()
        chunk_size = rewards.shape[-1]
        gamma = self.cfg.algorithm.gamma
        discounts = gamma ** torch.arange(
            chunk_size, device=rewards.device, dtype=torch.float32
        )
        chunk_reward = (rewards * discounts).sum(dim=-1, keepdim=True)
        terminated = batch["terminations"].bool().any(dim=-1, keepdim=True)
        return chunk_reward + gamma**chunk_size * qf_next_target.masked_fill(
            terminated, 0.0
        )

    @Worker.timer("forward_critic")
    def forward_critic(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Fit every Q network to the same target for normalized replay actions."""
        target_q_values = self.compute_target(batch)
        condition = batch["curr_condition"]
        all_data_q_values = self.model(
            component="critic",
            actions=batch["actions"],
            condition_tokens=condition.tokens,
            condition_mask=condition.mask,
            state_embedding=condition.state_embedding,
        )
        critic_loss = F.mse_loss(
            all_data_q_values.float(),
            target_q_values.detach().float().expand_as(all_data_q_values),
        )
        return critic_loss, {
            "q_data": all_data_q_values.detach().float().mean().item(),
            "q_target": target_q_values.float().mean().item(),
        }

    @Worker.timer("forward_actor")
    def forward_actor(self, batch: dict) -> tuple[torch.Tensor, dict, dict]:
        """Minimize negative mean Q plus the residual-mean L2 penalty.

        L2 regularizes the mean only; gradients pass through frozen pi to the actor.
        """
        condition = batch["curr_condition"]
        output = self.sample_actions(condition, collect_step_stats=True)
        all_qf_pi = self.model(
            component="critic",
            actions=output["actions"],
            condition_tokens=condition.tokens,
            condition_mask=condition.mask,
            state_embedding=condition.state_embedding,
        )
        qf_pi = all_qf_pi.float().mean(dim=-1, keepdim=True)
        raw_mean_l2_loss, raw_mean_group_mse, weighted_raw_mean_l2 = (
            compute_raw_mean_l2(
                output["raw_mean_group_mse"], self.raw_mean_l2_coefficients
            )
        )
        actor_loss = -qf_pi.mean() + raw_mean_l2_loss
        rfpo_metrics = denoise_step_metrics(output["step_stats"])
        rfpo_metrics["actor_loss/q"] = -qf_pi.detach().mean().item()
        rfpo_metrics.update(
            {
                f"actor_loss/raw_mean_l2/{group_name}": weighted_term.item()
                for group_name, weighted_term in zip(
                    RFPO_ACTION_GROUP_NAMES, weighted_raw_mean_l2, strict=True
                )
            }
        )
        return (
            actor_loss,
            {"q_pi": qf_pi.detach().mean().item()},
            rfpo_metrics,
        )

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self, train_actor: bool = True):
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )

        with self.worker_timer("sample"):
            global_batch = next(self.buffer_dataloader_iter)

        train_micro_batch_list = split_dict_to_chunk(
            global_batch, global_batch_size_per_rank // self.cfg.actor.micro_batch_size
        )

        # Reuse frozen observation features across critic and actor updates.
        with self.worker_timer("prepare_batch"):
            for i, batch in enumerate(train_micro_batch_list):
                batch = put_tensor_device(batch, device=self.device)
                train_micro_batch_list[i] = self.prepare_batch(batch)

        self.optimizer.zero_grad()
        self.qf_optimizer.zero_grad()
        gbs_critic_loss = []
        all_critic_metrics = {}
        for idx, batch in enumerate(train_micro_batch_list):
            with self.before_micro_batch(
                self.model, is_last_micro_batch=idx + 1 == self.gradient_accumulation
            ):
                critic_loss, critic_metrics = self.forward_critic(batch)
                critic_loss = critic_loss / self.gradient_accumulation
                critic_loss.backward()
            gbs_critic_loss.append(critic_loss.item() * self.gradient_accumulation)
            append_to_dict(all_critic_metrics, critic_metrics)
        all_critic_metrics = {
            f"critic/{key}": np.mean(value) for key, value in all_critic_metrics.items()
        }
        # FSDP may materialize zero gradients for unused actor parameters.
        self.optimizer.zero_grad()
        qf_grad_norm = self.model.clip_grad_norm_(
            max_norm=self.cfg.actor.critic_optim.clip_grad
        )

        self.qf_optimizer.step()
        self.qf_lr_scheduler.step()
        self.qf_optimizer.zero_grad()

        metrics_data = {
            "critic/loss": np.mean(gbs_critic_loss),
            "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": qf_grad_norm,
            **all_critic_metrics,
        }

        if self.update_step % self.critic_actor_ratio == 0 and train_actor:
            gbs_actor_loss = []
            all_actor_metrics = {}
            all_rfpo_metrics = {}
            for idx, batch in enumerate(train_micro_batch_list):
                with self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=idx + 1 == self.gradient_accumulation,
                ):
                    actor_loss, actor_metrics, rfpo_metrics = self.forward_actor(batch)
                    actor_loss = actor_loss / self.gradient_accumulation
                    actor_loss.backward()
                gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)
                append_to_dict(all_actor_metrics, actor_metrics)
                append_to_dict(all_rfpo_metrics, rfpo_metrics)
            all_actor_metrics = {
                f"actor/{key}": np.mean(value)
                for key, value in all_actor_metrics.items()
            }
            all_rfpo_metrics = {
                f"rfpo/{key}": np.mean(value) for key, value in all_rfpo_metrics.items()
            }
            # Keep dQ/da; discard critic gradients before clipping only the actor.
            self.qf_optimizer.zero_grad()
            actor_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            metrics_data.update(
                {
                    "actor/loss": np.mean(gbs_actor_loss),
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "actor/grad_norm": actor_grad_norm,
                    **all_actor_metrics,
                    **all_rfpo_metrics,
                }
            )

        if self.update_step % self.target_update_freq == 0:
            self.soft_update_target_model(component="critic")
        if self.update_step % self.target_actor_update_freq == 0:
            self.soft_update_target_model(component="actor")

        return metrics_data

    @Worker.timer("run_training")
    def run_training(self):
        """Run RFPO updates when every actor rank has enough trajectories."""
        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        if not self._all_ranks_buffer_ready(min_buffer_size):
            self.log_on_first_rank(
                f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training"
            )
            return {}

        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)
        self.rfpo_backbone_model.to(self.device)

        # All ranks must take the same actor-update branch for FSDP collectives.
        train_actor_steps = self.cfg.algorithm.get("train_actor_steps", 0)
        train_actor_steps = max(min_buffer_size, train_actor_steps)
        train_actor = self._all_ranks_buffer_ready(train_actor_steps)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}

        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            metrics_data = self.update_one_epoch(train_actor=train_actor)
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        mean_metric_dict = self.process_train_metrics(metrics)

        self.optimizer.zero_grad()
        self.qf_optimizer.zero_grad()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_optimizer()
            self.offload_param_and_grad()
        Worker.torch_platform.synchronize()
        torch.distributed.barrier()
        Worker.torch_platform.empty_cache()
        return mean_metric_dict

    def offload_param_and_grad(self, offload_grad: bool = False) -> None:
        super().offload_param_and_grad(offload_grad)
        self.rfpo_backbone_model.to("cpu")
