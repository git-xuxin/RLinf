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

import torch
from torch import nn

from rlinf.models.embodiment.modules.transformer import (
    TransformerBackbone,
    sinusoidal_position_embedding,
)


class RFPOQNetwork(nn.Module):
    """Evaluate a chunk using OpenPI's preprocessed state and prefix output."""

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
    ) -> None:
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_size)
        self.prefix_proj = nn.Linear(context_dim, hidden_size)
        self.state_context_proj = nn.Linear(state_dim, hidden_size)
        self.action_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.q_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.prefix_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.state_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.backbone = TransformerBackbone(
            hidden_size,
            num_layers,
            num_heads,
            context_dim=hidden_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        actions: torch.Tensor,
        *,
        state_embedding: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return Q values without changing OpenPI state or prefix features."""
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

        network_dtype = self.action_proj.weight.dtype
        action_tokens = (
            self.action_proj(actions.to(dtype=network_dtype))
            + self.action_type_embedding
        )
        action_tokens = (
            action_tokens
            + sinusoidal_position_embedding(
                action_tokens.shape[1],
                action_tokens.shape[2],
                device=action_tokens.device,
                dtype=action_tokens.dtype,
            )[None]
        )
        if state_embedding.shape != (
            batch_size,
            1,
            self.state_context_proj.in_features,
        ):
            raise ValueError("RFPO critic state_embedding must have shape [B, 1, S].")
        if condition_tokens.ndim != 3 or condition_tokens.shape[0] != batch_size:
            raise ValueError("RFPO critic prefix tokens must have shape [B, P, E].")
        q_token = self.q_token.expand(batch_size, -1, -1)
        input_tokens = torch.cat([action_tokens, q_token], dim=1)
        input_mask = torch.cat(
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
        ) + self.prefix_type_embedding
        state_token = self.state_context_proj(
            state_embedding.detach().to(dtype=network_dtype)
        ) + self.state_type_embedding
        context = torch.cat([prefix_context, state_token], dim=1)
        if condition_mask is not None:
            if condition_mask.shape != condition_tokens.shape[:2]:
                raise ValueError(
                    "RFPO critic condition_mask must match the prefix token shape."
                )
            state_mask = torch.ones(
                (condition_mask.shape[0], 1),
                dtype=torch.bool,
                device=condition_mask.device,
            )
            condition_mask = torch.cat(
                [condition_mask.to(dtype=torch.bool), state_mask], dim=1
            )
        hidden = self.backbone(
            input_tokens,
            padding_mask=input_mask,
            context=context,
            context_mask=condition_mask,
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
