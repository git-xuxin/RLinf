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

"""SAC double-Q critic over OpenPI's normalized action and prefix spaces."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence

import torch
from torch import nn


def _sinusoidal_position_embedding(
    length: int,
    embedding_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Return the fixed positional encoding from the original Transformer."""
    positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(0, embedding_dim, 2, device=device, dtype=torch.float32)
        / embedding_dim
    )
    angles = positions * frequencies[None]
    embedding = torch.zeros((length, embedding_dim), device=device, dtype=torch.float32)
    embedding[:, 0::2] = torch.sin(angles)
    embedding[:, 1::2] = torch.cos(angles[:, : embedding[:, 1::2].shape[1]])
    return embedding.to(dtype=dtype)


class RFPOQNetwork(nn.Module):
    """Evaluate an action chunk with a standard encoder-decoder Transformer."""

    def __init__(
        self,
        *,
        action_dim: int,
        state_dim: int,
        context_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        mlp_hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        self.action_proj = nn.Linear(action_dim, d_model)
        self.prefix_proj = nn.Linear(context_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.value_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        q_head_dims = (d_model, *mlp_hidden_dims, 1)
        q_head_layers: list[nn.Module] = []
        for layer_index, (input_dim, output_dim) in enumerate(
            zip(q_head_dims[:-1], q_head_dims[1:])
        ):
            q_head_layers.append(nn.Linear(input_dim, output_dim))
            if layer_index < len(q_head_dims) - 2:
                q_head_layers.append(nn.GELU())
        self.q_head = nn.Sequential(*q_head_layers)
        nn.init.normal_(self.value_token, mean=0.0, std=0.02)

    @staticmethod
    def _padding_mask(valid_mask: torch.Tensor) -> torch.Tensor:
        return ~valid_mask.to(dtype=torch.bool)

    def forward(
        self,
        actions: torch.Tensor,
        *,
        state_embedding: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return Q values while preserving gradients through ``actions``."""
        if actions.ndim != 3 or actions.shape[2] != self.action_proj.in_features:
            raise ValueError("RFPO critic actions must have shape [B, H, A].")
        batch_size, action_chunk, _ = actions.shape
        if action_mask is None:
            action_mask = torch.ones(
                (batch_size, action_chunk),
                dtype=torch.bool,
                device=actions.device,
            )
        elif action_mask.shape != (batch_size, action_chunk):
            raise ValueError(
                "RFPO critic action_mask must match the action chunk shape: "
                f"{tuple(action_mask.shape)} != {(batch_size, action_chunk)}."
            )
        action_mask = action_mask.to(device=actions.device, dtype=torch.bool)
        if not torch.all(action_mask.any(dim=1)):
            raise ValueError("Every RFPO critic sample must contain a valid action.")

        if state_embedding.shape != (
            batch_size,
            1,
            self.state_proj.in_features,
        ):
            raise ValueError("RFPO critic state_embedding must have shape [B, 1, S].")
        if (
            condition_tokens.ndim != 3
            or condition_tokens.shape[0] != batch_size
            or condition_tokens.shape[2] != self.prefix_proj.in_features
        ):
            raise ValueError("RFPO critic prefix tokens must have shape [B, P, E].")

        network_dtype = self.action_proj.weight.dtype
        action_tokens = self.action_proj(actions.to(dtype=network_dtype))
        value_token = self.value_token.expand(batch_size, -1, -1)
        target_tokens = torch.cat([action_tokens, value_token], dim=1)
        target_tokens = (
            target_tokens
            + _sinusoidal_position_embedding(
                target_tokens.shape[1],
                target_tokens.shape[2],
                device=target_tokens.device,
                dtype=target_tokens.dtype,
            )[None]
        )
        target_valid_mask = torch.cat(
            [
                action_mask,
                torch.ones(
                    (batch_size, 1), dtype=torch.bool, device=action_mask.device
                ),
            ],
            dim=1,
        )

        prefix_context = self.prefix_proj(
            condition_tokens.detach().to(dtype=network_dtype)
        )
        state_context = self.state_proj(
            state_embedding.detach().to(dtype=network_dtype)
        )
        source_tokens = torch.cat([prefix_context, state_context], dim=1)
        source_tokens = (
            source_tokens
            + _sinusoidal_position_embedding(
                source_tokens.shape[1],
                source_tokens.shape[2],
                device=source_tokens.device,
                dtype=source_tokens.dtype,
            )[None]
        )

        source_padding_mask = None
        if condition_mask is not None:
            if condition_mask.shape != condition_tokens.shape[:2]:
                raise ValueError(
                    "RFPO critic condition_mask must match the prefix token shape."
                )
            source_valid_mask = torch.cat(
                [
                    condition_mask.to(device=condition_tokens.device, dtype=torch.bool),
                    torch.ones(
                        (batch_size, 1),
                        dtype=torch.bool,
                        device=condition_tokens.device,
                    ),
                ],
                dim=1,
            )
            source_padding_mask = self._padding_mask(source_valid_mask)

        hidden = self.transformer(
            src=source_tokens,
            tgt=target_tokens,
            src_key_padding_mask=source_padding_mask,
            tgt_key_padding_mask=self._padding_mask(target_valid_mask),
            memory_key_padding_mask=source_padding_mask,
            src_is_causal=False,
            tgt_is_causal=False,
            memory_is_causal=False,
        )
        return self.q_head(hidden[:, -1]).float()


class RFPODoubleQCritic(nn.Module):
    """Two fully independent Q networks."""

    def __init__(self, **network_kwargs) -> None:
        super().__init__()
        self.q1 = RFPOQNetwork(**network_kwargs)
        self.q2 = RFPOQNetwork(**network_kwargs)

    def forward(
        self,
        actions: torch.Tensor,
        *,
        state_embedding: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q1 = self.q1(
            actions,
            state_embedding=state_embedding,
            condition_tokens=condition_tokens,
            condition_mask=condition_mask,
            action_mask=action_mask,
        )
        q2 = self.q2(
            actions,
            state_embedding=state_embedding,
            condition_tokens=condition_tokens,
            condition_mask=condition_mask,
            action_mask=action_mask,
        )
        return torch.cat([q1, q2], dim=-1)

    def target_copy(self) -> "RFPODoubleQCritic":
        target = copy.deepcopy(self)
        target.requires_grad_(False)
        return target
