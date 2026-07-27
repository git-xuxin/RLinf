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

"""Diffusion Transformer components for continuous-time action models."""

from __future__ import annotations

import math

import torch
from torch import nn

from .transformer import GatedFeedForward, RMSNorm


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10_000
) -> torch.Tensor:
    """Create sinusoidal embeddings for continuous timesteps."""
    half = embedding_dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = timesteps.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if embedding_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class DiTBlock(nn.Module):
    """Adaptive-normalization Transformer block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cross_attention: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}."
            )
        self.self_norm = RMSNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = cross_attention
        if cross_attention:
            self.cross_norm = RMSNorm(hidden_size)
            self.cross_attn = nn.MultiheadAttention(
                hidden_size, num_heads, dropout=dropout, batch_first=True
            )
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = GatedFeedForward(hidden_size, mlp_ratio, dropout)
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _modulate(
        hidden_states: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states * (1 + scale[:, None]) + shift[:, None]

    @staticmethod
    def _key_padding_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
        return None if mask is None else ~mask.to(dtype=torch.bool)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
            self.modulation(conditioning).chunk(6, dim=-1)
        )
        normed = self._modulate(self.self_norm(hidden_states), shift_attn, scale_attn)
        attn_out, _ = self.self_attn(
            normed,
            normed,
            normed,
            key_padding_mask=self._key_padding_mask(padding_mask),
            need_weights=False,
        )
        hidden_states = hidden_states + gate_attn[:, None] * self.dropout(attn_out)

        if self.cross_attention and context is not None:
            query = self.cross_norm(hidden_states)
            cross_out, _ = self.cross_attn(
                query,
                context,
                context,
                key_padding_mask=self._key_padding_mask(context_mask),
                need_weights=False,
            )
            hidden_states = hidden_states + self.dropout(cross_out)

        normed = self._modulate(self.ffn_norm(hidden_states), shift_ffn, scale_ffn)
        return hidden_states + gate_ffn[:, None] * self.ffn(normed)


class DiTBackbone(nn.Module):
    """Continuous-time DiT backbone returning action-token hidden states."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        *,
        context_dim: int | None = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        self.hidden_size = hidden_size
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.context_proj = (
            nn.Linear(context_dim, hidden_size)
            if context_dim is not None and context_dim != hidden_size
            else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    cross_attention=context_dim is not None,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        extra_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        time_embedding = sinusoidal_timestep_embedding(timesteps, self.hidden_size)
        conditioning = self.time_mlp(time_embedding.to(dtype=hidden_states.dtype))
        if extra_condition is not None:
            conditioning = conditioning + extra_condition
        if context is not None:
            context = self.context_proj(context)
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                conditioning,
                padding_mask=padding_mask,
                context=context,
                context_mask=context_mask,
            )
        return self.final_norm(hidden_states)
