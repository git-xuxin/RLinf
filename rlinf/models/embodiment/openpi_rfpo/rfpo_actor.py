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
    """Condition-decoder plus official DiT policy over active residual velocity."""

    suffix_dim = 1024

    def __init__(
        self,
        *,
        rfpo_action_chunk: int,
        rfpo_action_dim: int,
        prefix_dim: int,
        hidden_size: int,
        dit_depth: int,
        dit_num_heads: int,
        dit_mlp_ratio: float,
        condition_decoder_num_layers: int,
        condition_decoder_num_heads: int,
        condition_decoder_mlp_ratio: float,
        timestep_frequency_embedding_size: int,
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
        if rfpo_action_chunk <= 0 or rfpo_action_dim <= 0:
            raise ValueError("RFPO action chunk and dimension must be positive.")
        if hidden_size <= 0 or hidden_size % dit_num_heads != 0:
            raise ValueError("DiT hidden_size must be positive and divisible by heads.")
        if hidden_size % condition_decoder_num_heads != 0:
            raise ValueError(
                "Condition decoder hidden_size must be divisible by its heads."
            )
        if condition_decoder_num_layers <= 0:
            raise ValueError("Condition decoder layer count must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("RFPO actor dropout must lie within [0, 1).")

        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
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
            dit_depth,
            dit_num_heads,
            mlp_ratio=dit_mlp_ratio,
            output_dim=2 * rfpo_action_dim,
            dropout=dropout,
        )
        self._initialize_weights(initial_log_std)

    def _initialize_weights(self, initial_log_std: float) -> None:
        """Initialize adapters like the official DiT and set Gaussian prior."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)
        for block in self.dit.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.dit.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.dit.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.dit.final_layer.linear.weight)
        nn.init.zeros_(self.dit.final_layer.linear.bias)
        with torch.no_grad():
            self.dit.final_layer.linear.bias[self.rfpo_action_dim :].fill_(
                initial_log_std
            )

    def forward(
        self,
        base_velocity: torch.Tensor,
        timestep: torch.Tensor,
        *,
        suffix_embedding: torch.Tensor,
        prefix_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None,
        deterministic: bool = False,
        max_residual_velocity_rms: float = 0.2,
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
        if condition_mask is not None and condition_mask.shape != prefix_tokens.shape[:2]:
            raise ValueError("RFPO condition_mask must match prefix token shape.")

        actor_dtype = self.base_velocity_input.weight.dtype
        action_tokens = self.base_velocity_input(base_velocity.to(dtype=actor_dtype))
        action_tokens = action_tokens + get_1d_sincos_pos_embed(
            action_chunk,
            self.hidden_size,
            device=action_tokens.device,
            dtype=action_tokens.dtype,
        )[None]

        suffix_tokens = self.suffix_input(suffix_embedding.to(dtype=actor_dtype))
        suffix_tokens = suffix_tokens + get_1d_sincos_pos_embed(
            action_chunk + 1,
            self.hidden_size,
            device=suffix_tokens.device,
            dtype=suffix_tokens.dtype,
        )[None]
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

        hidden = self.dit.forward_features(action_tokens, conditioning)
        final_output = self.dit.final_layer(hidden, conditioning).float()
        mean, log_std = final_output.chunk(2, dim=-1)
        log_std = log_std.clamp(self.min_log_std, self.max_log_std)
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
            "active_delta_velocity": delta_velocity,
            "raw_delta_velocity": raw_delta_velocity,
            "mean": mean,
            "log_std": log_std,
            "std": std,
            "log_prob": log_prob,
            "projection_scale": projection_scale,
        }
