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

"""DiT blocks modified for RFPO residual velocities and OpenPI conditioning."""

import math

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from .config import RFPOActorConfig


def _position_embedding(length: int, width: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32)
    frequencies = 10_000 ** (
        -torch.arange(width // 2, device=device, dtype=torch.float32) / (width // 2)
    )
    angles = positions[:, None] * frequencies[None]
    return F.pad(torch.cat([angles.sin(), angles.cos()], dim=-1), (0, width % 2))


def _modulate(
    tokens: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return tokens * (1 + scale[:, None]) + shift[:, None]


class RFPOTimestepEmbedder(nn.Module):
    """Embed flow time with log-spaced periods and a DiT MLP."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """Encode [B] timesteps in [0, 1], computing phases in float32."""
        fractions = torch.linspace(0, 1, 128, device=timestep.device)
        periods = 4e-3 * (4.0 / 4e-3) ** fractions
        angles = timestep.float()[:, None] * (2 * math.pi / periods)[None]
        features = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return self.mlp(features.to(dtype=self.mlp[0].weight.dtype))


class RFPODiTBlock(nn.Module):
    """Bidirectional MHA and MLP with adaLN-Zero conditioning."""

    def __init__(self, cfg: RFPOActorConfig) -> None:
        super().__init__()
        width = cfg.hidden_size
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(width, cfg.num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(width, cfg.mlp_hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(cfg.mlp_hidden_size, width),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Apply the six adaLN modulation parameters to action tokens."""
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(condition).chunk(6, dim=-1)
        )
        normalized = _modulate(self.norm1(tokens), shift_attn, scale_attn)
        attended = self.attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        tokens = tokens + gate_attn[:, None] * attended
        return tokens + gate_mlp[:, None] * self.mlp(
            _modulate(self.norm2(tokens), shift_mlp, scale_mlp)
        )


class RFPOActor(nn.Module):
    """Condition decoder and DiT over a caller-selected action region."""

    def __init__(
        self,
        cfg: DictConfig,
        *,
        action_dim: int,
        action_feature_dim: int,
        condition_dim: int,
        state_dim: int,
    ) -> None:
        super().__init__()
        self.cfg = RFPOActorConfig(**cfg)
        width = self.cfg.hidden_size
        self.velocity_input = nn.Linear(action_dim, width)
        self.action_feature_input = nn.Linear(action_feature_dim, width)
        self.state_input = nn.Linear(state_dim, width)
        self.condition_input = nn.Linear(condition_dim, width)
        self.cls_token = nn.Parameter(torch.empty(1, 1, width))
        self.timestep_embedder = RFPOTimestepEmbedder(width)
        self.condition_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=width,
                nhead=self.cfg.num_heads,
                dim_feedforward=self.cfg.mlp_hidden_size,
                dropout=0.0,
                activation=nn.GELU(approximate="tanh"),
                batch_first=True,
                norm_first=False,
            ),
            num_layers=1,
            norm=nn.LayerNorm(width, eps=1e-6),
        )
        self.blocks = nn.ModuleList(
            RFPODiTBlock(self.cfg) for _ in range(self.cfg.num_dit_blocks)
        )
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.mean_head = nn.Linear(width, action_dim)
        # Independent of DiT; shared across states and action positions.
        self.log_std = nn.Parameter(
            torch.full((1, 1, action_dim), self.cfg.init_log_std, dtype=torch.float32)
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        # Small, nonzero outputs let adaLN start learning on the first update.
        nn.init.normal_(self.mean_head.weight, std=1e-4)

    def _predict_mean(
        self,
        base_velocity: torch.Tensor,
        timestep: torch.Tensor,
        *,
        action_features: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, action_chunk = base_velocity.shape[:2]
        dtype = self.velocity_input.weight.dtype
        velocity_tokens = self.velocity_input(base_velocity.to(dtype=dtype))
        velocity_tokens = (
            velocity_tokens
            + _position_embedding(
                action_chunk, self.cfg.hidden_size, base_velocity.device
            ).to(dtype=dtype)[None]
        )

        # State is explicit: the adapter's expert features contain actions only.
        queries = torch.cat(
            [
                self.state_input(state.detach().to(dtype=dtype))[:, None],
                self.action_feature_input(action_features.to(dtype=dtype)),
            ],
            dim=1,
        )
        queries = (
            queries
            + _position_embedding(
                queries.shape[1], self.cfg.hidden_size, queries.device
            ).to(dtype=dtype)[None]
        )
        queries = torch.cat([queries, self.cls_token.expand(batch_size, -1, -1)], dim=1)
        memory = self.condition_input(condition_tokens.detach().to(dtype=dtype))
        decoded = self.condition_decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=~condition_mask.to(
                device=memory.device, dtype=torch.bool
            ),
        )
        condition = decoded[:, -1] + self.timestep_embedder(timestep)
        for block in self.blocks:
            velocity_tokens = block(velocity_tokens, condition)
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        return self.mean_head(
            _modulate(self.final_norm(velocity_tokens), shift, scale)
        ).float()

    def forward(
        self,
        base_velocity: torch.Tensor,
        timestep: torch.Tensor,
        *,
        action_features: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor,
        state: torch.Tensor,
        deterministic: bool = False,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return a reparameterized residual and elementwise Gaussian log probability.

        Velocity and outputs have shape [B, C, A]; action features are [B, C, E].
        State is normalized [B, S]. Prefix tokens/mask use their actual length.
        All distribution calculations use float32; log probability is unreduced.
        Optional standard-normal noise has shape [B, C, A]; eval ignores it.
        """
        mean = self._predict_mean(
            base_velocity,
            timestep,
            action_features=action_features,
            condition_tokens=condition_tokens,
            condition_mask=condition_mask,
            state=state,
        )
        log_std = self.log_std.float().expand_as(mean)
        std = log_std.exp()
        if deterministic:
            delta_velocity = mean
        else:
            noise = torch.randn_like(mean) if noise is None else noise.to(mean)
            delta_velocity = mean + std * noise
        log_prob = -0.5 * (
            ((delta_velocity - mean) / std).square()
            + 2 * log_std
            + math.log(2 * math.pi)
        )
        return {
            "delta_velocity": delta_velocity,
            "mean": mean,
            "log_std": log_std,
            "log_prob": log_prob,
        }
