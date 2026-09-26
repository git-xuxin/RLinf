# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frozen OpenPI loading and RFPO's boundary to the pi denoising model."""

from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.openpi import get_model
from rlinf.models.embodiment.openpi.modules.model import (
    Observation,
    preprocess_observation,
)
from rlinf.models.embodiment.openpi.tasks.eval import Pi0Eval


def load_backbone_model(cfg: DictConfig, device: torch.device | str) -> Pi0Eval:
    """Load one worker's pi with the existing eval loader and freeze every parameter.

    Pass ``actor.rfpo_backbone_model`` or ``rollout.rfpo_backbone_model``.
    Device placement preserves the loader's selective bf16/fp32 parameter dtypes.
    """
    if cfg.openpi.task != "eval":
        raise ValueError("RFPO backbone loading requires openpi.task='eval' (ODE).")
    model = get_model(cfg)
    model.requires_grad_(False)
    model.eval()
    return model.to(device=device)


@dataclass
class RFPOCondition:
    """Prepared observation and frozen prefix features/cache for one observation batch."""

    observation: Observation
    tokens: torch.Tensor
    mask: torch.Tensor
    kv_cache: tuple


@dataclass
class RFPOVelocity:
    """Full model-space velocity and action-only expert features for one ODE step."""

    velocity: torch.Tensor
    action_features: torch.Tensor


class RFPOBackboneAdapter:
    """Expose pi preprocessing, condition encoding, velocity, and action decoding.

    The worker owns this adapter and its model separately from ``RFPOPolicy``.
    Rollout callers use ``torch.no_grad()`` around velocity evaluation; actor
    callers may differentiate it with respect to the current noisy actions.
    """

    def __init__(self, model: Pi0Eval):
        self.model = model

    @property
    def action_shape(self) -> tuple[int, int]:
        """Full denoising horizon and action width, before environment decoding."""
        return self.model.action_horizon, self.model.action_dim

    @property
    def num_steps(self) -> int:
        """Number of Euler steps, shared with native pi sampling."""
        return self.model.num_steps

    @property
    def env_action_shape(self) -> tuple[int, int]:
        """Executed action chunk and environment action width."""
        return self.model.action_chunk, self.model.action_env_dim

    @property
    def condition_dim(self) -> int:
        """Prefix feature width for small model construction."""
        return self.model.llm.configs[0].width

    @property
    def action_feature_dim(self) -> int:
        """Action expert feature width for small model construction."""
        return self.model.action_out_proj.in_features

    @property
    def state_dim(self) -> int:
        """Normalized, padded state width."""
        return self.model.action_dim

    @torch.no_grad()
    def preprocess(self, env_obs: dict[str, Any]) -> Observation:
        """Tokenize and normalize an env batch, then prepare images on the pi device."""
        observation = self.model.env_obs_to_observation(env_obs)
        return preprocess_observation(observation, train=False)

    @torch.no_grad()
    def encode_condition(self, observation: Observation) -> RFPOCondition:
        """Encode a preprocessed observation once and retain its actual prefix mask."""
        tokens, mask, kv_cache = self.model.build_prefix_cache(observation)
        return RFPOCondition(observation, tokens, mask, kv_cache)

    def velocity(
        self,
        condition: RFPOCondition,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> RFPOVelocity:
        """Evaluate v(x_t, t) without a sampler or an input-gradient barrier.

        Inputs have shapes ``[B, *action_shape]`` and ``[B]``. OpenPI owns suffix
        layout/masking, including whether it contains a state token. Integration
        uses ``x_next = x_t + dt * (velocity + residual_velocity)``, with dt < 0.
        """
        device = condition.observation.state.device
        action_features = self.model.run_suffix(
            condition.observation,
            noisy_actions.to(device=device, dtype=torch.float32),
            timestep.to(device=device, dtype=torch.float32),
            condition.kv_cache,
            condition.mask,
        )
        return RFPOVelocity(
            self.model.velocity_from_suffix(action_features), action_features
        )

    @torch.no_grad()
    def decode_actions(
        self, model_actions: torch.Tensor, condition: RFPOCondition
    ) -> torch.Tensor:
        """Unnormalize and decode a full denoised chunk into environment actions.

        This environment boundary uses the loader's output transforms and does
        not preserve action gradients. Critic training uses model-space actions.
        """
        return self.model.decode_actions(model_actions, condition.observation.state)
