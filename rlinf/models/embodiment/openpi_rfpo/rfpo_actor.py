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

"""Official-DiT residual velocity actor used by RFPO."""

from __future__ import annotations

import math

import torch
from torch import nn

from rlinf.models.embodiment.modules.dit import (
    DiTBackbone,
    TimestepEmbedder,
    get_1d_sincos_pos_embed,
)


class RFPOResidualActor(nn.Module):
    """Condition-decoder plus official DiT policy over active residual velocity."""

    suffix_dim = 1024

    def __init__(
        self,
        *,
        rfpo_action_chunk: int,
        rfpo_action_dim: int,
        prefix_dim: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        condition_decoder_num_layers: int,
        condition_decoder_num_heads: int,
        condition_decoder_mlp_ratio: float,
        timestep_frequency_embedding_size: int,
        dropout: float,
        mean_scale: float,
        min_log_std: float,
        max_log_std: float,
    ) -> None:
        super().__init__()
        for name, value in (
            ("mean_scale", mean_scale),
            ("min_log_std", min_log_std),
            ("max_log_std", max_log_std),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"RFPO {name} must be finite.")
        if mean_scale <= 0:
            raise ValueError("RFPO mean_scale must be positive.")
        if min_log_std >= max_log_std:
            raise ValueError("RFPO min_log_std must be less than max_log_std.")
        if rfpo_action_chunk <= 0 or rfpo_action_dim <= 0:
            raise ValueError("RFPO action chunk and dimension must be positive.")
        if hidden_size <= 0 or hidden_size % num_heads != 0:
            raise ValueError("DiT hidden_size must be positive and divisible by heads.")
        if hidden_size % condition_decoder_num_heads != 0:
            raise ValueError(
                "Condition decoder hidden_size must be divisible by its heads."
            )
        if condition_decoder_num_layers <= 0:
            raise ValueError("Condition decoder layer count must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("RFPO actor dropout must lie within [0, 1).")

        self.mean_scale = float(mean_scale)
        self.init_log_std = 0.5 * (float(min_log_std) + float(max_log_std))
        self.log_std_scale = 0.5 * (float(max_log_std) - float(min_log_std))
        self.rfpo_action_chunk = int(rfpo_action_chunk)
        self.rfpo_action_dim = int(rfpo_action_dim)
        self.hidden_size = int(hidden_size)

        # These projections are intentionally separate: RFPO treats the three
        # sources as distinct token types and trains each adapter independently.
        self.base_velocity_input = nn.Linear(rfpo_action_dim, hidden_size)
        self.suffix_input = nn.Linear(self.suffix_dim, hidden_size)
        self.prefix_input = nn.Linear(prefix_dim, hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.timestep_embedder = TimestepEmbedder(
            hidden_size, timestep_frequency_embedding_size
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=condition_decoder_num_heads,
            dim_feedforward=int(hidden_size * condition_decoder_mlp_ratio),
            dropout=dropout,
            activation=nn.GELU(approximate="tanh"),
            batch_first=True,
            norm_first=False,
        )
        self.condition_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=condition_decoder_num_layers,
            norm=nn.LayerNorm(hidden_size, eps=1e-6),
        )
        self.dit = DiTBackbone(
            hidden_size,
            depth,
            num_heads,
            mlp_ratio=mlp_ratio,
            output_dim=2 * rfpo_action_dim,
            dropout=dropout,
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize actor adapters without resetting the DiT trunk."""
        for name, module in self.named_modules():
            if name == "dit" or name.startswith("dit."):
                continue
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.dit.final_layer.linear.weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.dit.final_layer.linear.bias)

    def forward(
        self,
        base_velocity: torch.Tensor,
        timestep: torch.Tensor,
        *,
        suffix_embedding: torch.Tensor,
        prefix_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Predict and sample active residual velocities."""
        if base_velocity.ndim != 3:
            raise ValueError("RFPO base_velocity must have shape [B, C, A].")
        batch_size, action_chunk, action_dim = base_velocity.shape
        if (action_chunk, action_dim) != (
            self.rfpo_action_chunk,
            self.rfpo_action_dim,
        ):
            raise ValueError(
                "RFPO base_velocity must match [rfpo_action_chunk, rfpo_action_dim]."
            )
        if suffix_embedding.shape != (
            batch_size,
            action_chunk + 1,
            self.suffix_dim,
        ):
            raise ValueError(
                "RFPO suffix_embedding must have shape [B, C+1, 1024], got "
                f"{tuple(suffix_embedding.shape)}."
            )
        if prefix_tokens.ndim != 3 or prefix_tokens.shape[0] != batch_size:
            raise ValueError("RFPO prefix tokens must have shape [B, P, E].")
        if (
            condition_mask is not None
            and condition_mask.shape != prefix_tokens.shape[:2]
        ):
            raise ValueError("RFPO condition_mask must match prefix token shape.")

        actor_dtype = self.base_velocity_input.weight.dtype
        action_tokens = self.base_velocity_input(base_velocity.to(dtype=actor_dtype))
        action_tokens = (
            action_tokens
            + get_1d_sincos_pos_embed(
                action_chunk,
                self.hidden_size,
                device=action_tokens.device,
                dtype=action_tokens.dtype,
            )[None]
        )

        suffix_tokens = self.suffix_input(suffix_embedding.to(dtype=actor_dtype))
        suffix_tokens = (
            suffix_tokens
            + get_1d_sincos_pos_embed(
                action_chunk + 1,
                self.hidden_size,
                device=suffix_tokens.device,
                dtype=suffix_tokens.dtype,
            )[None]
        )
        timestep_token = self.timestep_embedder(timestep).to(dtype=actor_dtype)
        cls_token = self.cls_token.expand(batch_size, -1, -1) + timestep_token[:, None]
        queries = torch.cat([suffix_tokens, cls_token], dim=1)

        prefix_memory = self.prefix_input(prefix_tokens.detach().to(dtype=actor_dtype))
        memory_key_padding_mask = None
        if condition_mask is not None:
            memory_key_padding_mask = ~condition_mask.to(
                device=prefix_memory.device, dtype=torch.bool
            )
        decoded_queries = self.condition_decoder(
            tgt=queries,
            memory=prefix_memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        conditioning = decoded_queries[:, -1]

        raw_mean, raw_log_std = (
            self.dit(action_tokens, conditioning).float().chunk(2, dim=-1)
        )
        mean_tanh = torch.tanh(raw_mean)
        log_std_tanh = torch.tanh(raw_log_std)
        mean = self.mean_scale * mean_tanh
        log_std = self.init_log_std + self.log_std_scale * log_std_tanh
        std = log_std.exp()
        delta_velocity = mean if deterministic else mean + std * torch.randn_like(mean)
        log_prob = -0.5 * (
            ((delta_velocity - mean) / std).pow(2) + 2 * log_std + math.log(2 * math.pi)
        )
        return {
            "delta_velocity": delta_velocity,
            "raw_mean": raw_mean,
            "raw_log_std": raw_log_std,
            "mean_tanh": mean_tanh,
            "log_std_tanh": log_std_tanh,
            "mean": mean,
            "log_std": log_std,
            "std": std,
            "log_prob": log_prob,
        }
