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

from rlinf.models.embodiment.modules.dit import DiTBackbone
from rlinf.models.embodiment.modules.transformer import sinusoidal_position_embedding


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
    """Gaussian DiT policy over full-horizon residual velocities."""

    def __init__(
        self,
        *,
        action_dim: int,
        state_dim: int,
        context_dim: int,
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
        self.action_input = nn.Linear(action_dim * 2 + 1, hidden_size)
        self.state_proj = nn.Linear(state_dim, hidden_size)
        self.backbone = DiTBackbone(
            hidden_size,
            num_layers,
            num_heads,
            context_dim=context_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.log_std_head = nn.Linear(hidden_size, action_dim)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, initial_log_std)

    def forward(
        self,
        noisy_action: torch.Tensor,
        base_velocity: torch.Tensor,
        timestep: torch.Tensor,
        step_size: torch.Tensor,
        *,
        state: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None,
        deterministic: bool = False,
        max_residual_velocity_rms: float = 0.2,
    ) -> dict[str, torch.Tensor]:
        step_feature = step_size.reshape(-1, 1, 1).expand(
            noisy_action.shape[0], noisy_action.shape[1], 1
        )
        tokens = torch.cat([noisy_action, base_velocity.detach(), step_feature], dim=-1)
        actor_dtype = self.action_input.weight.dtype
        tokens = self.action_input(tokens.to(dtype=actor_dtype))
        tokens = (
            tokens
            + sinusoidal_position_embedding(
                tokens.shape[1],
                tokens.shape[2],
                device=tokens.device,
                dtype=tokens.dtype,
            )[None]
        )
        state_condition = self.state_proj(state.to(dtype=actor_dtype))
        hidden = self.backbone(
            tokens,
            timestep,
            context=condition_tokens.to(dtype=actor_dtype),
            context_mask=condition_mask,
            extra_condition=state_condition,
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

        delta_velocity, projection_scale = project_residual_velocity(
            raw_delta_velocity,
            max_residual_velocity_rms,
        )
        return {
            "delta_velocity": delta_velocity,
            "raw_delta_velocity": raw_delta_velocity,
            "mean": mean,
            "log_std": log_std,
            "log_prob": log_prob,
            "projection_scale": projection_scale,
        }
