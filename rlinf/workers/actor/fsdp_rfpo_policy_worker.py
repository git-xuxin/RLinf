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
import os
from collections.abc import Mapping
from numbers import Real

import numpy as np
import torch
import torch.nn.functional as F

from rlinf.models import get_model
from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
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
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


def _parse_raw_mean_l2_coefficients(
    config: Mapping | None,
) -> tuple[float, float, float]:
    """Validate ``algorithm.raw_mean_l2_coefficients`` in action-group order.

    A missing mapping disables the penalty; it must otherwise name only the
    action groups and give each a finite, non-negative weight.
    """
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


class EmbodiedRFPOFSDPPolicy(EmbodiedSACFSDPPolicy):
    """Train the residual actor and critic while keeping pi frozen."""

    def init_worker(self):
        # Load the frozen pi backbone; the small model reads it through the adapter.
        self.rfpo_backbone_model = load_backbone_model(
            self.cfg.actor.rfpo_backbone_model, self.device
        )
        self.rfpo_adapter = RFPOBackboneAdapter(self.rfpo_backbone_model)
        # Build model, replay buffer and SAC components on top of the adapter.
        super().init_worker()
        # Group-wise penalties on the actor raw mean; zeros disable the penalty.
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
        """Setup the small model, target critic, optimizers and schedulers."""
        module = self.model_provider_func()
        if initialize_target:
            target_module = copy.deepcopy(module.critic)

        # Sync the complete online small model, including persistent buffers.
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

        self.use_dsrl = False
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

        alpha_type = self.cfg.algorithm.entropy_tuning.alpha_type
        initial_alpha = self.cfg.algorithm.entropy_tuning.initial_alpha
        if alpha_type != "fixed_alpha" or initial_alpha != 0:
            raise ValueError(
                "RFPO currently requires fixed_alpha with initial_alpha=0."
            )
        self.entropy_temp = EntropyTemperature(
            initial_alpha=initial_alpha,
            alpha_type=alpha_type,
            device=self.device,
            dtype=self.torch_dtype,
        )
        self.entropy_temp.requires_grad_(False)

        self.build_lr_schedulers()

        self.grad_scaler = self.build_grad_scaler(
            **self.cfg.actor.fsdp_config.grad_scaler
        )

    @torch.no_grad()
    def soft_update_target_model(self, tau: float | None = None) -> None:
        """Initialize or interpolate the target critic from the online critic."""
        if tau is None:
            tau = self.cfg.algorithm.tau
        for online_param, target_param in zip(
            self.model.critic.parameters(), self.target_model.parameters(), strict=True
        ):
            target_param.lerp_(online_param, tau)
        for online_buffer, target_buffer in zip(
            self.model.critic.buffers(), self.target_model.buffers(), strict=True
        ):
            target_buffer.copy_(online_buffer)

    def prepare_batch(self, batch: dict) -> dict:
        """Rebuild both pi inputs and keep one complete normalized action chunk."""
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
        """Use the online actor for both training and target action sampling."""
        return self.model(
            component="actor",
            adapter=self.rfpo_adapter,
            condition=condition,
            mode=mode,
            **kwargs,
        )

    @torch.no_grad()
    def compute_target(self, batch: dict) -> torch.Tensor:
        """Sample next actions with the online actor and bootstrap with target Qs."""
        condition = batch["next_condition"]
        output = self.sample_actions(condition, mode="target")
        all_qf_next_target = self.target_model(
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

        # Replay rewards exclude trajectory value bootstrap; all C steps executed.
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
        """Fit replay actions to one target; batch must come from prepare_batch."""
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

    def _all_ranks_buffer_ready(self, min_size: int) -> bool:
        """Return True only when every actor rank's replay buffer reaches min_size.
        """
        local_ready = torch.tensor(
            [int(self.replay_buffer.is_ready(min_size))],
            device=self.device,
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(local_ready, op=torch.distributed.ReduceOp.MIN)
        return local_ready.item() == 1

    @Worker.timer("forward_actor")
    def forward_actor(self, batch: dict) -> tuple[torch.Tensor, None, dict, dict]:
        """Maximize online Q and bound the raw residual mean on guided steps.

        The actor objective is the mean online Q of the current final
        normalized action chunk. ``raw_mean_l2`` adds the coefficient-weighted
        group MSE of the actor raw mean on every guided step, which limits how
        far each delta velocity drifts from zero and narrows exploration;
        all-zero coefficients add a zero term and keep the behavior unchanged.

        The fourth return value carries the sampler's per-denoise-step
        velocity diagnostics plus the raw-mean L2 group metrics, reduced to
        scalars; the caller adds the ``rfpo/`` namespace.
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
            None,
            {"q_pi": qf_pi.detach().mean().item()},
            rfpo_metrics,
        )

    def forward_alpha(self, batch):
        raise NotImplementedError("RFPO entropy tuning is disabled.")

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

        # move train_micro_batch_list to device and rebuild the pi conditions
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
                    actor_loss, _, actor_metrics, rfpo_metrics = self.forward_actor(
                        batch
                    )
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

        # Soft update target network
        if (
            self.target_model_initialized
            and self.update_step % self.cfg.algorithm.get("target_update_freq", 1) == 0
        ):
            self.soft_update_target_model()

        return metrics_data

    @Worker.timer("run_training")
    def run_training(self):
        """RFPO training using replay buffer"""
        # Check if replay buffer has enough samples
        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        if not self.replay_buffer.is_ready(min_buffer_size):
            self.log_on_first_rank(
                f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training"
            )
            return {}

        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)
        self.rfpo_backbone_model.to(self.device)

        # Delay actor training until every rank's buffer has enough samples
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

    def save_checkpoint(self, save_base_path: str, step: int) -> None:
        """Save SAC components and the state that controls RFPO updates."""
        restore_weight_offload = self.is_weight_offloaded
        restore_optimizer_offload = self.is_optimizer_offloaded

        super().save_checkpoint(save_base_path, step)

        # save the state that controls RFPO updates
        worker_state = {
            "update_step": self.update_step,
            "critic_sample_rng": self.critic_sample_generator.get_state(),
            "replay_rng": self.replay_buffer.random_generator.get_state(),
        }
        worker_state_save_path = os.path.join(
            save_base_path, f"rfpo_state_rank_{self._rank}.pt"
        )
        torch.save(worker_state, worker_state_save_path)

        if restore_optimizer_offload:
            self.offload_optimizer()
        if restore_weight_offload:
            self.offload_param_and_grad()

    def load_checkpoint(self, load_base_path: str) -> None:
        """Restore training state on device, then restore the offload state."""
        restore_weight_offload = self.is_weight_offloaded
        restore_optimizer_offload = self.is_optimizer_offloaded

        if restore_weight_offload:
            self.load_param_and_grad(self.device)
        if restore_optimizer_offload:
            self.load_optimizer(self.device)

        super().load_checkpoint(load_base_path)

        # load the state that controls RFPO updates
        worker_state_load_path = os.path.join(
            load_base_path, f"rfpo_state_rank_{self._rank}.pt"
        )
        worker_state = torch.load(
            worker_state_load_path,
            map_location="cpu",
            weights_only=True,
        )
        self.update_step = worker_state["update_step"]
        self.critic_sample_generator.set_state(worker_state["critic_sample_rng"])
        self.replay_buffer.random_generator.set_state(worker_state["replay_rng"])

        if restore_optimizer_offload:
            self.offload_optimizer()
        if restore_weight_offload:
            self.offload_param_and_grad()

    def offload_param_and_grad(self, offload_grad: bool = False) -> None:
        if not self.is_weight_offloaded:
            super().offload_param_and_grad(offload_grad)
            if self.target_model is not None:
                self._strategy.offload_param_and_grad(self.target_model, offload_grad)
        self.rfpo_backbone_model.to("cpu")

    def load_param_and_grad(self, device_id: int, load_grad: bool = False) -> None:
        if self.is_weight_offloaded:
            super().load_param_and_grad(device_id, load_grad)
            if self.target_model is not None:
                self._strategy.onload_param_and_grad(
                    self.target_model, device_id, load_grad
                )
        # Weight synchronization only needs the small model; pi loads in the learner.

    def offload_optimizer(self) -> None:
        if not self.is_optimizer_offloaded:
            super().offload_optimizer()
            self._strategy.offload_optimizer(self.qf_optimizer)

    def load_optimizer(self, device_id: int) -> None:
        if self.is_optimizer_offloaded:
            super().load_optimizer(device_id)
            self._strategy.onload_optimizer(self.qf_optimizer, device_id)
