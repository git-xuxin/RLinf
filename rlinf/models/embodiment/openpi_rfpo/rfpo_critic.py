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

"""Gemma3 decoder-only double-Q critic for RFPO."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch import nn
from transformers import Gemma3TextConfig, Gemma3TextModel


class RFPOQNetwork(nn.Module):
    """Evaluate an action chunk with a bidirectional Gemma3 text backbone."""

    def __init__(
        self,
        *,
        action_dim: int,
        state_dim: int,
        context_dim: int,
        prefix_length: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        attention_dropout: float,
        hidden_activation: str,
        initializer_range: float,
        rms_norm_eps: float,
        rope_theta: float,
        mlp_hidden_dims: Sequence[int],
        q_output_init_std: float,
    ) -> None:
        super().__init__()
        if len(mlp_hidden_dims) != 2:
            raise ValueError(
                "RFPO critic Q-head requires exactly two hidden dimensions."
            )

        self.prefix_length = int(prefix_length)
        self.max_position_embeddings = int(max_position_embeddings)
        self.action_proj = nn.Linear(action_dim, hidden_size)
        self.prefix_proj = nn.Linear(context_dim, hidden_size)
        self.state_proj = nn.Linear(state_dim, hidden_size)
        self.value_token = nn.Parameter(torch.empty(1, 1, hidden_size))

        gemma3_config = Gemma3TextConfig(
            vocab_size=1,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_activation=hidden_activation,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=False,
            pad_token_id=0,
            eos_token_id=0,
            bos_token_id=0,
            tie_word_embeddings=False,
            rope_theta=rope_theta,
            attention_bias=False,
            attention_dropout=attention_dropout,
            query_pre_attn_scalar=head_dim,
            sliding_window=max_position_embeddings,
            layer_types=["full_attention"] * num_hidden_layers,
        )
        self.backbone = Gemma3TextModel(gemma3_config)
        # RFPO always supplies continuous embeddings, so retaining even a
        # one-token vocabulary would leave an unused trainable parameter.
        self.backbone.embed_tokens = None
        self.backbone.config._attn_implementation = "sdpa"  # noqa: SLF001

        q_head_dims = (hidden_size, *mlp_hidden_dims, 1)
        q_head_layers: list[nn.Module] = []
        for layer_index, (input_dim, output_dim) in enumerate(
            zip(q_head_dims[:-1], q_head_dims[1:], strict=True)
        ):
            q_head_layers.append(nn.Linear(input_dim, output_dim))
            if layer_index < len(q_head_dims) - 2:
                q_head_layers.append(nn.GELU())
        self.q_head = nn.Sequential(*q_head_layers)

        nn.init.normal_(self.value_token, mean=0.0, std=initializer_range)
        self._init_q_head_weights(q_output_init_std)

    def _init_q_head_weights(self, output_std: float) -> None:
        """Initialize hidden layers normally and keep initial Q values small."""
        linear_layers = [
            module for module in self.q_head if isinstance(module, nn.Linear)
        ]
        for module in linear_layers[:-1]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        output_layer = linear_layers[-1]
        nn.init.normal_(output_layer.weight, mean=0.0, std=output_std)
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)

    @staticmethod
    def _position_ids(valid_mask: torch.Tensor) -> torch.Tensor:
        """Generate RoPE positions while excluding padding from the count."""
        return valid_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)

    @staticmethod
    def _bidirectional_attention_mask(
        valid_mask: torch.Tensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return a broadcastable key-padding mask without a causal triangle."""
        attention_mask = torch.zeros(
            (valid_mask.shape[0], 1, 1, valid_mask.shape[1]),
            dtype=dtype,
            device=valid_mask.device,
        )
        return attention_mask.masked_fill(
            ~valid_mask[:, None, None, :], torch.finfo(dtype).min
        )

    def _encode(
        self,
        actions: torch.Tensor,
        *,
        state_embedding: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor | None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the full RFPO critic hidden sequence."""
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

        expected_state_shape = (
            batch_size,
            1,
            self.state_proj.in_features,
        )
        if state_embedding.shape != expected_state_shape:
            raise ValueError(
                "RFPO critic state_embedding must have shape "
                f"{expected_state_shape}, got {tuple(state_embedding.shape)}."
            )
        expected_prefix_shape = (
            batch_size,
            self.prefix_length,
            self.prefix_proj.in_features,
        )
        if condition_tokens.shape != expected_prefix_shape:
            raise ValueError(
                "RFPO critic prefix tokens must have shape "
                f"{expected_prefix_shape}, got {tuple(condition_tokens.shape)}."
            )

        if condition_mask is None:
            condition_mask = torch.ones(
                (batch_size, self.prefix_length),
                dtype=torch.bool,
                device=actions.device,
            )
        elif condition_mask.shape != condition_tokens.shape[:2]:
            raise ValueError(
                "RFPO critic condition_mask must match the prefix token shape."
            )
        condition_mask = condition_mask.to(device=actions.device, dtype=torch.bool)

        network_dtype = self.action_proj.weight.dtype
        action_tokens = self.action_proj(actions.to(dtype=network_dtype))
        prefix_tokens = self.prefix_proj(
            condition_tokens.detach().to(dtype=network_dtype)
        )
        state_token = self.state_proj(state_embedding.detach().to(dtype=network_dtype))
        value_token = self.value_token.expand(batch_size, -1, -1)
        input_tokens = torch.cat(
            [action_tokens, prefix_tokens, state_token, value_token], dim=1
        )
        if input_tokens.shape[1] > self.max_position_embeddings:
            raise ValueError(
                "RFPO critic sequence length exceeds max_position_embeddings: "
                f"{input_tokens.shape[1]} > {self.max_position_embeddings}."
            )

        always_valid = torch.ones(
            (batch_size, 2), dtype=torch.bool, device=actions.device
        )
        valid_mask = torch.cat([action_mask, condition_mask, always_valid], dim=1)
        attention_mask = self._bidirectional_attention_mask(
            valid_mask, dtype=input_tokens.dtype
        )
        mask_mapping = {
            "full_attention": attention_mask,
            "sliding_attention": attention_mask,
        }
        hidden = self.backbone(
            inputs_embeds=input_tokens,
            attention_mask=mask_mapping,
            position_ids=self._position_ids(valid_mask),
            use_cache=False,
        ).last_hidden_state
        if hidden.shape != input_tokens.shape:
            raise RuntimeError(
                "RFPO critic backbone returned an unexpected hidden shape: "
                f"{tuple(hidden.shape)} != {tuple(input_tokens.shape)}."
            )
        return hidden

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
        hidden = self._encode(
            actions,
            state_embedding=state_embedding,
            condition_tokens=condition_tokens,
            condition_mask=condition_mask,
            action_mask=action_mask,
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
