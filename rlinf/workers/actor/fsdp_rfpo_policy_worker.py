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

import torch
from torch.nn import functional as F

from rlinf.models import get_model
from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
from rlinf.models.embodiment.openpi_rfpo import build_model_config
from rlinf.models.embodiment.openpi_rfpo.backbone import (
    RFPOBackboneAdapter,
    load_backbone_model,
)
from rlinf.scheduler import Worker
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class EmbodiedRFPOFSDPPolicy(EmbodiedSACFSDPPolicy):
    """RFPO loss computation; optimizer scheduling is handled separately."""

    def init_worker(self):
        self.rfpo_backbone_model = load_backbone_model(
            self.cfg.actor.rfpo_backbone_model, self.device
        )
        self.rfpo_adapter = RFPOBackboneAdapter(self.rfpo_backbone_model)
        super().init_worker()

    def model_provider_func(self):
        model_config = build_model_config(self.cfg.actor.model, self.rfpo_adapter)
        model = get_model(model_config)
        if self.cfg.runner.get("ckpt_path", None):
            model.load_state_dict(torch.load(self.cfg.runner.ckpt_path))
        return model

    def setup_model_and_optimizer(self, initialize_target=False) -> None:
        module = self.model_provider_func()
        if initialize_target:
            target_module = copy.deepcopy(module.critic)
        # Sync the complete online small model, including persistent buffers.
        self.param_names_need_sync = list(module.state_dict())
        self.model = self._strategy.wrap_model(module, self._device_mesh)
        if initialize_target:
            self.target_model = self._strategy.wrap_model(
                target_module, self._device_mesh
            )
            self.target_model.requires_grad_(False)
            self.target_model.eval()
            self.target_model_initialized = True

        self.use_dsrl = False
        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters={"critic": ["critic."]},
            filtered_optim_config={"critic": self.cfg.actor.critic_optim},
        )
        self.optimizer, self.qf_optimizer = optimizers
        entropy_cfg = self.cfg.algorithm.entropy_tuning
        if entropy_cfg.alpha_type != "fixed_alpha" or entropy_cfg.initial_alpha != 0:
            raise ValueError(
                "RFPO currently requires fixed_alpha with initial_alpha=0."
            )
        self.entropy_temp = EntropyTemperature(
            initial_alpha=0.0,
            alpha_type="fixed_alpha",
            device=self.device,
            dtype=self.torch_dtype,
        )
        self.entropy_temp.requires_grad_(False)
        self.build_lr_schedulers()
        self.grad_scaler = self.build_grad_scaler(
            self.cfg.actor.fsdp_config.grad_scaler
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
        if self.cfg.algorithm.entropy_tuning.initial_alpha != 0.0:
            raise NotImplementedError(
                "RFPO entropy requires a future log_prob implementation."
            )
        condition = batch["next_condition"]
        output = self.sample_actions(condition, mode="target")
        next_q_values = self.target_model(
            actions=output["actions"],
            condition_tokens=condition.tokens,
            condition_mask=condition.mask,
            state=condition.observation.state,
        )
        subsample_size = self.cfg.algorithm.get("critic_subsample_size", 2)
        num_q_heads = next_q_values.shape[-1]
        if not 1 <= subsample_size <= num_q_heads:
            raise ValueError(
                "RFPO critic_subsample_size must be between 1 and num_q_heads."
            )
        sample_idx = torch.randperm(
            num_q_heads,
            device=next_q_values.device,
            generator=self.critic_sample_generator,
        )[:subsample_size]
        next_q = (
            next_q_values.float()
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
        return chunk_reward + gamma**chunk_size * next_q.masked_fill(terminated, 0.0)

    @torch.no_grad()
    def consume_batch(self, batch: dict) -> torch.Tensor:
        """Exercise learner inputs without gradients, losses, or optimizer steps."""
        prepared = self.prepare_batch(batch)
        condition = prepared["curr_condition"]
        return self.model(
            component="critic",
            actions=prepared["actions"],
            condition_tokens=condition.tokens,
            condition_mask=condition.mask,
            state=condition.observation.state,
        )

    @Worker.timer("run_training")
    def run_training(self):
        if not self.replay_buffer.is_ready(
            self.cfg.algorithm.replay_buffer.min_buffer_size
        ):
            return {}
        if self.enable_offload:
            self.load_param_and_grad(self.device)
        self.rfpo_backbone_model.to(self.device)
        self.model.eval()
        try:
            batch = next(self.buffer_dataloader_iter)
            for micro_batch in split_dict_to_chunk(batch, self.gradient_accumulation):
                self.consume_batch(put_tensor_device(micro_batch, self.device))
            return self.process_train_metrics({})
        finally:
            if self.enable_offload:
                self.offload_param_and_grad()
            Worker.torch_platform.synchronize()
            torch.distributed.barrier()
            Worker.torch_platform.empty_cache()

    @Worker.timer("forward_critic")
    def forward_critic(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Fit replay actions to one target; batch must come from prepare_batch."""
        target_q_values = self.compute_target(batch)
        condition = batch["curr_condition"]
        q_values = self.model(
            component="critic",
            actions=batch["actions"],
            condition_tokens=condition.tokens,
            condition_mask=condition.mask,
            state=condition.observation.state,
        )
        critic_loss = F.mse_loss(
            q_values.float(), target_q_values.detach().float().expand_as(q_values)
        )
        return critic_loss, {}

    @Worker.timer("forward_actor")
    def forward_actor(self, batch: dict) -> tuple[torch.Tensor, None, dict]:
        """Maximize online Q on the current actor's final normalized action chunk."""
        if self.cfg.algorithm.entropy_tuning.initial_alpha != 0.0:
            raise NotImplementedError(
                "RFPO entropy requires a future log_prob implementation."
            )
        condition = batch["curr_condition"]
        output = self.sample_actions(condition)
        q_values = self.model(
            component="critic",
            actions=output["actions"],
            condition_tokens=condition.tokens,
            condition_mask=condition.mask,
            state=condition.observation.state,
        )
        qf_pi = q_values.float().mean(dim=-1, keepdim=True)
        actor_loss = -qf_pi.mean()
        return actor_loss, None, {}

    def forward_alpha(self, batch):
        raise NotImplementedError("RFPO entropy tuning is disabled.")
