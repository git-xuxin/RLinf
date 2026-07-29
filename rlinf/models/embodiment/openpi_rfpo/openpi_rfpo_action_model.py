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

"""Frozen pi0 with an RFPO residual actor and trajectory critic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from openpi.models import model as _model

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)
from rlinf.utils.nested_dict_process import copy_dict_tensor

from .rfpo_actor import RFPOResidualActor
from .rfpo_critic import RFPODoubleQCritic
from .rfpo_sampler import RFPOGuidedSampler


@dataclass(frozen=True)
class OpenPiRFPOConfig(OpenPi0Config):
    """Configuration for pi0 residual-flow adaptation."""

    residual_ratio: float = 0.1
    num_denoise_steps: int = 4
    active_step_indices: tuple[int, ...] = field(default_factory=lambda: (2, 3))
    differentiate_base_velocity: bool = True
    initial_log_std: float = -4.0
    min_log_std: float = -8.0
    max_log_std: float = 0.0
    internal_log_prob_reduction: str = "mean_active"
    actor_hidden_size: int = 256
    actor_num_layers: int = 2
    actor_num_heads: int = 8
    actor_mlp_ratio: float = 4.0
    critic_hidden_size: int = 256
    critic_num_layers: int = 2
    critic_num_heads: int = 8
    critic_mlp_ratio: float = 4.0
    dropout: float = 0.0
    context_dim: int = 2048

    def __post_init__(self) -> None:
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        if self.config_name != "pi0_libero":
            raise ValueError(
                "OpenPi RFPO initially supports only config_name='pi0_libero'."
            )
        if self.noise_method != "flow_ode":
            raise ValueError("OpenPi RFPO requires noise_method='flow_ode'.")
        if self.use_dsrl or self.use_rlt or self.is_nft:
            raise ValueError("OpenPi RFPO cannot be combined with DSRL, RLT, or NFT.")
        if self.num_denoise_steps <= 0:
            raise ValueError("num_denoise_steps must be positive.")
        if self.num_denoise_steps != self.num_steps:
            raise ValueError(
                "RFPO num_denoise_steps must match pi0 num_steps to preserve the "
                "pretrained Euler schedule."
            )
        if self.residual_ratio < 0:
            raise ValueError("residual_ratio must be non-negative.")
        if self.action_chunk <= 0 or self.action_chunk > self.action_horizon:
            raise ValueError("action_chunk must lie within the pi0 action horizon.")
        if self.internal_log_prob_reduction not in {"mean_active", "sum_active"}:
            raise ValueError(
                "internal_log_prob_reduction must be 'mean_active' or 'sum_active'."
            )
        if self.context_dim != 2048:
            raise ValueError(
                "OpenPi RFPO pi0_libero prefix tokens have width 2048; "
                f"got context_dim={self.context_dim}."
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie within [0, 1).")
        active_steps = tuple(int(i) for i in self.active_step_indices)
        if not active_steps:
            raise ValueError("active_step_indices must contain at least one step.")
        if len(set(active_steps)) != len(active_steps):
            raise ValueError("active_step_indices must not contain duplicates.")
        if any(i < 0 or i >= self.num_denoise_steps for i in active_steps):
            raise ValueError(
                "active_step_indices must lie within the denoising schedule."
            )
        for hidden_size, num_heads, name in (
            (self.actor_hidden_size, self.actor_num_heads, "actor"),
            (self.critic_hidden_size, self.critic_num_heads, "critic"),
        ):
            if hidden_size <= 0 or num_heads <= 0:
                raise ValueError(
                    f"RFPO {name} hidden size and head count must be positive."
                )
            if hidden_size % num_heads != 0:
                raise ValueError(
                    f"RFPO {name} hidden size must be divisible by its head count."
                )


class OpenPiRFPOActionModel(OpenPi0ForRLActionPrediction):
    """pi0 action model with a constrained residual-flow policy."""

    config: OpenPiRFPOConfig

    def __init__(self, config: OpenPiRFPOConfig) -> None:
        super().__init__(config)
        self.requires_grad_(False)
        network_kwargs = {
            "action_dim": config.action_dim,
            "state_dim": config.action_dim,
            "context_dim": config.context_dim,
            "dropout": config.dropout,
        }
        self.residual_actor = RFPOResidualActor(
            **network_kwargs,
            hidden_size=config.actor_hidden_size,
            num_layers=config.actor_num_layers,
            num_heads=config.actor_num_heads,
            mlp_ratio=config.actor_mlp_ratio,
            initial_log_std=config.initial_log_std,
            min_log_std=config.min_log_std,
            max_log_std=config.max_log_std,
        ).to(dtype=torch.bfloat16)
        self.online_critic = RFPODoubleQCritic(
            **network_kwargs,
            hidden_size=config.critic_hidden_size,
            num_layers=config.critic_num_layers,
            num_heads=config.critic_num_heads,
            mlp_ratio=config.critic_mlp_ratio,
        ).to(dtype=torch.bfloat16)
        self.rfpo_sampler = RFPOGuidedSampler(
            num_denoise_steps=config.num_denoise_steps,
            action_chunk=config.action_chunk,
            active_step_indices=tuple(config.active_step_indices),
            residual_ratio=config.residual_ratio,
            differentiate_base_velocity=config.differentiate_base_velocity,
            log_prob_reduction=config.internal_log_prob_reduction,
        )
        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    @property
    def _no_split_modules(self) -> list[str]:
        return [
            *super()._no_split_modules,
            "RFPOResidualActor",
            "RFPODoubleQCritic",
        ]

    def train(self, mode: bool = True):
        super().train(mode)
        self.paligemma_with_expert.eval()
        self.residual_actor.train(mode)
        self.online_critic.train(mode)
        return self

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.RFPO_ACTOR:
            return self.rfpo_actor_forward(**kwargs)
        if forward_type == ForwardType.RFPO_Q:
            return self.rfpo_q_forward(**kwargs)
        return super().forward(forward_type=forward_type, **kwargs)

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "OpenPiRFPOActionModel uses RFPO_ACTOR and RFPO_Q forward routes."
        )

    def _processed_observation(
        self,
        obs: dict[str, Any],
        *,
        tokenized_prompt: torch.Tensor | None = None,
        tokenized_prompt_mask: torch.Tensor | None = None,
    ) -> _model.Observation:
        if "task_descriptions" in obs:
            to_process_obs = self.obs_processor(obs)
        else:
            to_process_obs = {
                "observation/image": obs["main_images"],
                "observation/state": obs["states"],
            }
            wrist_images = obs.get("wrist_images")
            if wrist_images is not None:
                to_process_obs["observation/wrist_image"] = wrist_images
            extra_view_images = obs.get("extra_view_images")
            if extra_view_images is not None:
                to_process_obs["observation/extra_view_image"] = extra_view_images
            if tokenized_prompt is None or tokenized_prompt_mask is None:
                raise ValueError(
                    "Replay RFPO observations require cached tokenized prompt tensors."
                )
            to_process_obs["tokenized_prompt"] = tokenized_prompt
            to_process_obs["tokenized_prompt_mask"] = tokenized_prompt_mask
        processed_obs = self.input_transform(to_process_obs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)
        return _model.Observation.from_dict(processed_obs)

    def _condition_from_observation(
        self, observation: _model.Observation
    ) -> dict[str, Any]:
        with torch.no_grad():
            images, img_masks, lang_tokens, lang_masks, state = (
                self._preprocess_observation(observation, train=False)
            )
            prefix_output, prefix_pad_masks, past_key_values = self._build_prefix_cache(
                images, img_masks, lang_tokens, lang_masks
            )
            if prefix_output.shape[-1] != self.config.context_dim:
                raise ValueError(
                    "RFPO condition token width does not match context_dim: "
                    f"{prefix_output.shape[-1]} != {self.config.context_dim}."
                )
        return {
            "state": state.detach(),
            "prefix_output": prefix_output.detach(),
            "condition_mask": prefix_pad_masks.detach().to(dtype=torch.bool),
            "prefix_pad_masks": prefix_pad_masks.detach(),
            "past_key_values": past_key_values,
        }

    def _sample_guided(
        self,
        observation: _model.Observation,
        *,
        noise: torch.Tensor | None = None,
        deterministic: bool = False,
        retain_residual_grads: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        condition = self._condition_from_observation(observation)
        if noise is None:
            noise = self.sample_noise(
                (
                    observation.state.shape[0],
                    self.config.action_horizon,
                    self.config.action_dim,
                ),
                observation.state.device,
            )
        sampler_output = self._sample_actions_with_prefix_cache(
            state=condition["state"],
            prefix_output=condition["prefix_output"],
            prefix_pad_masks=condition["prefix_pad_masks"],
            past_key_values=condition["past_key_values"],
            noise=noise.to(dtype=self.action_in_proj.weight.dtype),
            mode="eval" if deterministic else "train",
            compute_values=False,
            deterministic=deterministic,
            retain_residual_grads=retain_residual_grads,
        )
        return sampler_output, condition

    @torch.no_grad()
    def sample_actions(
        self,
        observation: _model.Observation,
        noise=None,
        mode="train",
        compute_values=True,
    ) -> dict[str, Any]:
        """Copy OpenPI's inference entry point and use the RFPO step body."""
        bsize = observation.state.shape[0]
        device = observation.state.device
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        else:
            noise = noise.to(self.action_in_proj.weight.dtype)

        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        prefix_output, prefix_pad_masks, past_key_values = self._build_prefix_cache(
            images, img_masks, lang_tokens, lang_masks
        )

        return self._sample_actions_with_prefix_cache(
            state,
            prefix_output,
            prefix_pad_masks,
            past_key_values,
            noise=noise,
            mode=mode,
            compute_values=compute_values,
        )

    def _sample_actions_with_prefix_cache(
        self,
        state,
        prefix_output,
        prefix_pad_masks,
        past_key_values,
        noise=None,
        mode="train",
        compute_values=True,
        *,
        deterministic: bool | None = None,
        retain_residual_grads: bool = False,
    ) -> dict[str, Any]:
        return self.rfpo_sampler.sample(
            self,
            state=state,
            prefix_output=prefix_output,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            noise=noise,
            mode=mode,
            compute_values=compute_values,
            deterministic=deterministic,
            retain_residual_grads=retain_residual_grads,
        )

    def _sample_frozen_pi0(
        self,
        *,
        state: torch.Tensor,
        prefix_output: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = OpenPi0ForRLActionPrediction._sample_actions_with_prefix_cache(
                self,
                state,
                prefix_output,
                prefix_pad_masks,
                past_key_values,
                noise=noise,
                mode="eval",
                compute_values=False,
            )
        return outputs["actions"]

    def rfpo_actor_forward(
        self,
        obs: dict[str, Any],
        *,
        tokenized_prompt: torch.Tensor,
        tokenized_prompt_mask: torch.Tensor,
        deterministic: bool = False,
        retain_residual_grads: bool = False,
        evaluate_q: bool = False,
        compute_pi0_baseline: bool = False,
    ) -> dict[str, Any]:
        observation = self._processed_observation(
            obs,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )
        noise = self.sample_noise(
            (
                observation.state.shape[0],
                self.config.action_horizon,
                self.config.action_dim,
            ),
            observation.state.device,
        ).to(dtype=self.action_in_proj.weight.dtype)
        output, condition = self._sample_guided(
            observation,
            noise=noise,
            deterministic=deterministic,
            retain_residual_grads=retain_residual_grads,
        )
        result = {
            "actions": output["actions"][:, : self.config.action_chunk],
            "model_action_horizon": output["actions"],
            "internal_log_prob": output["internal_log_prob"],
            "residual_rms": output["residual_rms"],
            "base_velocity_rms": output["base_velocity_rms"],
            "active_step_mask": output["active_step_mask"],
            "active_residuals": output["active_residuals"],
            "critic_state": condition["state"],
            "critic_condition_tokens": condition["prefix_output"],
            "critic_condition_mask": condition["condition_mask"],
        }
        if compute_pi0_baseline:
            result["pi0_actions"] = self._sample_frozen_pi0(
                state=condition["state"],
                prefix_output=condition["prefix_output"],
                prefix_pad_masks=condition["prefix_pad_masks"],
                past_key_values=condition["past_key_values"],
                noise=noise,
            )[:, : self.config.action_chunk]
        if evaluate_q:
            result["q_values"] = self.online_critic(
                result["actions"],
                state=condition["state"],
                condition_tokens=condition["prefix_output"],
                condition_mask=condition["condition_mask"],
            )
        return result

    def rfpo_q_forward(
        self,
        obs: dict[str, Any],
        actions: torch.Tensor,
        *,
        tokenized_prompt: torch.Tensor,
        tokenized_prompt_mask: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        observation = self._processed_observation(
            obs,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )
        condition = self._condition_from_observation(observation)
        return self.online_critic(
            actions,
            state=condition["state"],
            condition_tokens=condition["prefix_output"],
            condition_mask=condition["condition_mask"],
            action_mask=action_mask,
        )

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del kwargs
        to_process_obs = self.obs_processor(env_obs)
        processed_obs = self.input_transform(to_process_obs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)
        observation = _model.Observation.from_dict(processed_obs)

        outputs = self.sample_actions(
            observation, mode=mode, compute_values=compute_values
        )
        actions = self.output_transform(
            {"actions": outputs["actions"], "state": observation.state}
        )["actions"]
        prev_logprobs = outputs["prev_logprobs"]
        prev_values = outputs["prev_values"]
        model_chunk = outputs["actions"][:, : self.config.action_chunk]
        forward_inputs = {
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
            # RFPO's critic consumes normalized model actions, so replay keeps
            # the model chunk here instead of OpenPI's transformed env action.
            "action": model_chunk.reshape(model_chunk.shape[0], -1).contiguous(),
            "model_action": outputs["actions"]
            .reshape(outputs["actions"].shape[0], -1)
            .contiguous(),
            "rfpo_residual_rms": outputs["residual_rms"],
            "rfpo_base_velocity_rms": outputs["base_velocity_rms"],
        }

        # Clone observations to avoid cross-step reference issues.
        forward_inputs.update(
            copy_dict_tensor({k: v for k, v in to_process_obs.items() if k != "prompt"})
        )
        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }
        return actions, result
