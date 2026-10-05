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

"""Frozen OpenPI features and velocities for RFPO denoising."""

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
from rlinf.models.embodiment.openpi.transforms.env import repack_env_obs


def load_backbone_model(cfg: DictConfig, device: torch.device | str) -> Pi0Eval:
    """Load frozen pi while preserving OpenPI's parameter dtypes."""
    if cfg.openpi.task != "eval":
        raise ValueError("RFPO backbone loading requires openpi.task='eval' (ODE).")
    model = get_model(cfg)
    model.requires_grad_(False)
    model.eval()
    return model.to(device=device)


@dataclass
class RFPOCondition:
    """Observation, frozen prefix features, and KV cache for one batch.

    ``state_embedding`` is Pi0's projected state [B, 1, S], or None for Pi05.
    """

    observation: Observation
    tokens: torch.Tensor
    mask: torch.Tensor
    kv_cache: tuple
    state_embedding: torch.Tensor | None = None


class RFPOBackboneAdapter:
    """Worker-owned frozen pi adapter with action gradients for actor training."""

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
    def suffix_dim(self) -> int:
        """Pi action-expert token width for the residual actor."""
        return self.model.action_in_proj.out_features

    @property
    def suffix_has_state(self) -> bool:
        """Whether pi prepends a continuous state token to its action tokens."""
        return not self.model.pi05

    @property
    def state_dim(self) -> int | None:
        """Continuous state embedding width, absent for Pi05."""
        return self.model.state_proj.out_features if self.suffix_has_state else None

    @torch.no_grad()
    def preprocess(self, env_obs: dict[str, Any]) -> Observation:
        """Tokenize and normalize an env batch, then prepare images on the pi device."""
        observation = self.model.env_obs_to_observation(env_obs)
        return preprocess_observation(observation, train=False)

    @torch.no_grad()
    def prepare_observation(
        self, env_obs: dict[str, Any]
    ) -> tuple[Observation, dict[str, torch.Tensor]]:
        """Return pi input and raw CPU replay observations with prompt tokens."""
        repacked = repack_env_obs(
            self.model.config_name,
            env_obs,
            select_state=self.model._select_configured_state,
        )
        replay_obs = {
            key: torch.as_tensor(value).detach().cpu().clone().contiguous()
            for key, value in repacked.items()
            if key != "prompt"
        }
        processed = self.model.input_transform(repacked, transpose=False)
        for key in ("tokenized_prompt", "tokenized_prompt_mask"):
            replay_obs[key] = processed[key].detach().cpu().clone().contiguous()
        observation = self.model._observation_dict_to_device(processed)
        return preprocess_observation(observation, train=False), replay_obs

    @torch.no_grad()
    def preprocess_replay(self, replay_obs: dict[str, torch.Tensor]) -> Observation:
        """Normalize replay observations while preserving stored prompt tokens."""
        processed = self.model.input_transform(replay_obs, transpose=False)
        observation = self.model._observation_dict_to_device(processed)
        return preprocess_observation(observation, train=False)

    @torch.no_grad()
    def encode_condition(self, observation: Observation) -> RFPOCondition:
        """Cache prefix features, their validity mask, and optional Pi0 state."""
        tokens, mask, kv_cache = self.model.build_prefix_cache(observation)
        state_embedding = None
        if self.suffix_has_state:
            state_dtype = self.model.state_proj.weight.dtype
            state_embedding = self.model.state_proj(
                observation.state.to(dtype=state_dtype)
            )[:, None]
        return RFPOCondition(observation, tokens, mask, kv_cache, state_embedding)

    def embed_suffix(
        self,
        condition: RFPOCondition,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Return pre-attention tokens: Pi0 state/action-time or Pi05 actions."""
        device = condition.tokens.device
        return self.model.embed_suffix(
            condition.observation,
            noisy_actions.to(device=device, dtype=torch.float32),
            timestep.to(device=device, dtype=torch.float32),
        )[0]

    def velocity(
        self,
        condition: RFPOCondition,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate frozen pi velocity while retaining noisy-action gradients."""
        device = condition.tokens.device
        action_features = self.model.run_suffix(
            condition.observation,
            noisy_actions.to(device=device, dtype=torch.float32),
            timestep.to(device=device, dtype=torch.float32),
            condition.kv_cache,
            condition.mask,
        )
        return self.model.velocity_from_suffix(action_features)

    @torch.no_grad()
    def decode_actions(
        self, model_actions: torch.Tensor, condition: RFPOCondition
    ) -> torch.Tensor:
        """Decode model-space actions for the environment without gradients."""
        return self.model.decode_actions(model_actions, condition.observation.state)
