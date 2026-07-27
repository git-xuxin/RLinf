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

"""Small reusable Transformer components for embodied policies."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return normalized * self.weight


def sinusoidal_position_embedding(
    length: int,
    embedding_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Create a fixed sinusoidal embedding for token positions."""
    half = embedding_dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=device, dtype=torch.float32)
        / max(half, 1)
    )
    positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    angles = positions * frequencies[None]
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if embedding_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.to(dtype=dtype)


class GatedFeedForward(nn.Module):
    """SwiGLU feed-forward layer."""

    def __init__(
        self,
        hidden_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        intermediate_size = int(hidden_size * mlp_ratio)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(hidden))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with optional cross-attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cross_attention: bool = False,
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
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _key_padding_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        return ~mask.to(dtype=torch.bool)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normed = self.self_norm(hidden_states)
        attn_out, _ = self.self_attn(
            normed,
            normed,
            normed,
            key_padding_mask=self._key_padding_mask(padding_mask),
            need_weights=False,
        )
        hidden_states = hidden_states + self.dropout(attn_out)

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

        return hidden_states + self.ffn(self.ffn_norm(hidden_states))


class TransformerBackbone(nn.Module):
    """Stackable Transformer backbone returning hidden tokens."""

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
        self.context_proj = (
            nn.Linear(context_dim, hidden_size)
            if context_dim is not None and context_dim != hidden_size
            else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
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
        *,
        padding_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context is not None:
            context = self.context_proj(context)
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                padding_mask=padding_mask,
                context=context,
                context_mask=context_mask,
            )
        return self.final_norm(hidden_states)
