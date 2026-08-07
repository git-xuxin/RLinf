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

"""
    Official DiT building blocks adapted to RFPO action tokens.
"""

from __future__ import annotations

import functools
import math

import torch
from timm.models.vision_transformer import Attention, Mlp
from torch import nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply the affine modulation used by DiT's adaLN blocks."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embed scalar diffusion timesteps as hidden-size vectors."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        if hidden_size <= 0 or frequency_embedding_size <= 0:
            raise ValueError("Timestep embedding dimensions must be positive.")
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(
        timesteps: torch.Tensor,
        embedding_dim: int,
        max_period: int = 10_000,
    ) -> torch.Tensor:
        """Create the fractional sinusoidal embedding from the official DiT."""
        half = embedding_dim // 2
        if half == 0:
            raise ValueError("Timestep embedding dimension must be at least 2.")
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / half
        )
        angles = timesteps[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if embedding_dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError(
                f"Timestep input must have shape [B], got {tuple(timesteps.shape)}."
            )
        frequency_embedding = self.timestep_embedding(
            timesteps, self.frequency_embedding_size
        )
        return self.mlp(frequency_embedding.to(dtype=self.mlp[0].weight.dtype))


def get_1d_sincos_pos_embed(
    length: int,
    embed_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a fixed 1-D sin-cos position embedding for action tokens."""
    if length <= 0 or embed_dim <= 0 or embed_dim % 2:
        raise ValueError("Position embedding length must be positive and dim even.")
    positions = torch.arange(length, device=device, dtype=torch.float32)
    frequencies = torch.arange(embed_dim // 2, device=device, dtype=torch.float32)
    frequencies = 1.0 / (10_000 ** (frequencies / (embed_dim // 2)))
    angles = positions[:, None] * frequencies[None]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).to(dtype=dtype)


class DiTBlock(nn.Module):
    """DiT transformer block with adaLN-Zero conditioning."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}."
            )
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=functools.partial(nn.GELU, approximate="tanh"),
            drop=dropout,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Run self-attention and MLP with DiT's six adaLN parameters."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(conditioning).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """RFPO adaptation of the official DiT final layer.

    The official image model expands each token to patch pixels. RFPO instead
    emits two action parameters per token: mean and log standard deviation.
    """

    def __init__(self, hidden_size: int, output_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Apply final adaLN and emit token-wise action parameters."""
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiTBackbone(nn.Module):
    """Official DiT trunk operating on pre-embedded action tokens."""

    def __init__(
        self,
        hidden_size: int,
        depth: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("DiT depth must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("DiT dropout must lie within [0, 1).")
        self.hidden_size = hidden_size
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, output_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Match the official DiT basic linear initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward_features(
        self, x: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        """Return hidden action tokens before the final action projection."""
        for block in self.blocks:
            x = block(x, conditioning)
        return x

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Return token-wise RFPO output parameters."""
        return self.final_layer(self.forward_features(x, conditioning), conditioning)
