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


import torch

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
    """RFPO framework integration; consume replay without RL updates for now."""

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
        # Sync the complete online small model, including persistent buffers.
        self.param_names_need_sync = list(module.state_dict())
        self.model = self._strategy.wrap_model(module, self._device_mesh)
        if initialize_target:
            self.target_model = self._strategy.wrap_model(
                self.model_provider_func(), self._device_mesh
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

    def forward_critic(self, batch):
        raise NotImplementedError(
            "RFPO loss belongs to the algorithm-computation stage."
        )

    def forward_actor(self, batch):
        raise NotImplementedError(
            "RFPO loss belongs to the algorithm-computation stage."
        )

    def forward_alpha(self, batch):
        raise NotImplementedError("RFPO entropy tuning is disabled.")
