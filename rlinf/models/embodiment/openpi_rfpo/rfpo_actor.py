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

"""Residual velocity actor used by RFPO."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from rlinf.models.embodiment.modules.dit import DiTBackbone
from rlinf.models.embodiment.modules.transformer import (
    RMSNorm,
    sinusoidal_position_embedding,
)


def project_residual_velocity(
    candidate: torch.Tensor,
    max_residual_velocity_rms: float,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project each residual sample onto a fixed RMS ball."""
    if not math.isfinite(max_residual_velocity_rms):
        raise ValueError("RFPO max_residual_velocity_rms must be finite.")
    if max_residual_velocity_rms < 0:
        raise ValueError("RFPO max_residual_velocity_rms must be non-negative.")
    if candidate.ndim != 3:
        raise ValueError(
            "RFPO residual velocity must have shape [batch, horizon, dim]."
        )

    candidate_rms = candidate.float().pow(2).mean(dim=(1, 2)).sqrt()
    projection_scale = torch.clamp(
        max_residual_velocity_rms / candidate_rms.clamp_min(eps),
        max=1.0,
    )
    projected = candidate * projection_scale[:, None, None]
    return projected, projection_scale


class RFPOResidualActor(nn.Module):
    """Gaussian DiT policy over an active residual-velocity region."""

    def __init__(
        self,
        *,
        action_horizon: int,
        pi0_action_dim: int,
        rfpo_action_chunk: int,
        rfpo_action_dim: int,
        state_dim: int,
        prefix_dim: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        initial_log_std: float,
        min_log_std: float,
        max_log_std: float,
    ) -> None:
        super().__init__()
        if min_log_std > initial_log_std or initial_log_std > max_log_std:
            raise ValueError(
                "initial_log_std must lie within [min_log_std, max_log_std]."
            )
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.action_horizon = int(action_horizon)
        self.pi0_action_dim = int(pi0_action_dim)
        self.rfpo_action_chunk = int(rfpo_action_chunk)
        self.rfpo_action_dim = int(rfpo_action_dim)
        self.action_input = nn.Linear(rfpo_action_dim * 2, hidden_size)
        self.action_input_norm = RMSNorm(hidden_size)
        self.prefix_proj = nn.Linear(prefix_dim, hidden_size)
        self.state_proj = nn.Linear(state_dim, hidden_size)
        self.action_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.prefix_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.state_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.backbone = DiTBackbone(
            hidden_size,
            num_layers,
            num_heads,
            context_dim=hidden_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.mean_head = nn.Linear(hidden_size, rfpo_action_dim)
        self.log_std_head = nn.Linear(hidden_size, rfpo_action_dim)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, initial_log_std)

    def forward(
        self,
        noisy_action: torch.Tensor,
        base_velocity: torch.Tensor,
        timestep: torch.Tensor,
        *,
        state_embedding: torch.Tensor,
        prefix_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None,
        deterministic: bool = False,
        max_residual_velocity_rms: float = 0.2,
    ) -> dict[str, torch.Tensor]:
        if noisy_action.shape != base_velocity.shape or noisy_action.ndim != 3:
            raise ValueError(
                "RFPO actor noisy_action and base_velocity must share [B, H, D]."
            )
        if noisy_action.shape[1:] != (self.action_horizon, self.pi0_action_dim):
            raise ValueError(
                "RFPO actor input must match the configured pi0 action shape."
            )
        if state_embedding.shape != (
            noisy_action.shape[0],
            1,
            self.state_proj.in_features,
        ):
            raise ValueError("RFPO actor state_embedding must have shape [B, 1, S].")
        if prefix_tokens.ndim != 3 or prefix_tokens.shape[0] != noisy_action.shape[0]:
            raise ValueError("RFPO actor prefix tokens must have shape [B, P, E].")

        active_noisy_action = noisy_action[
            :, : self.rfpo_action_chunk, : self.rfpo_action_dim
        ]
        active_base_velocity = base_velocity[
            :, : self.rfpo_action_chunk, : self.rfpo_action_dim
        ].detach()
        tokens = torch.cat([active_noisy_action, active_base_velocity], dim=-1)
        actor_dtype = self.action_input.weight.dtype
        tokens = self.action_input_norm(
            self.action_input(tokens.to(dtype=actor_dtype))
        )
        tokens = tokens + self.action_type_embedding
        tokens = (
            tokens
            + sinusoidal_position_embedding(
                tokens.shape[1],
                tokens.shape[2],
                device=tokens.device,
                dtype=tokens.dtype,
            )[None]
        )
        prefix_condition = self.prefix_proj(
            prefix_tokens.detach().to(dtype=actor_dtype)
        ) + self.prefix_type_embedding
        state_condition = self.state_proj(
            state_embedding.detach().to(dtype=actor_dtype)
        ) + self.state_type_embedding
        context = torch.cat([prefix_condition, state_condition], dim=1)
        if condition_mask is None:
            context_mask = None
        else:
            if condition_mask.shape != prefix_tokens.shape[:2]:
                raise ValueError(
                    "RFPO actor condition_mask must match the prefix token shape."
                )
            state_mask = torch.ones(
                (condition_mask.shape[0], 1),
                dtype=torch.bool,
                device=condition_mask.device,
            )
            context_mask = torch.cat(
                [condition_mask.to(dtype=torch.bool), state_mask], dim=1
            )
        hidden = self.backbone(
            tokens,
            timestep,
            padding_mask=torch.ones(
                tokens.shape[:2], dtype=torch.bool, device=tokens.device
            ),
            context=context,
            context_mask=context_mask,
        )
        mean = self.mean_head(hidden).to(dtype=torch.float32)
        log_std = (
            self.log_std_head(hidden).float().clamp(self.min_log_std, self.max_log_std)
        )
        std = log_std.exp()
        raw_delta_velocity = (
            mean if deterministic else mean + std * torch.randn_like(mean)
        )
        log_prob = -0.5 * (
            ((raw_delta_velocity - mean) / std).pow(2)
            + 2 * log_std
            + math.log(2 * math.pi)
        )

        active_delta_velocity, projection_scale = project_residual_velocity(
            raw_delta_velocity,
            max_residual_velocity_rms,
        )
        delta_velocity = F.pad(
            active_delta_velocity,
            (
                0,
                self.pi0_action_dim - self.rfpo_action_dim,
                0,
                self.action_horizon - self.rfpo_action_chunk,
            ),
        )
        return {
            "delta_velocity": delta_velocity,
            "active_delta_velocity": active_delta_velocity,
            "raw_delta_velocity": raw_delta_velocity,
            "mean": mean,
            "log_std": log_std,
            "log_prob": log_prob,
            "projection_scale": projection_scale,
        }
