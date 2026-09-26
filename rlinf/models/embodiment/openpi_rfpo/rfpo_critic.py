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

"""RFPO-specific Gemma3 critic with continuous inputs and bidirectional MHA."""

import torch
from omegaconf import DictConfig
from torch import nn
from transformers import Gemma3TextConfig, Gemma3TextModel

from .config import RFPOCriticConfig


class RFPOQNetwork(nn.Module):
    """Score an action chunk using a learned value token and one Q head."""

    def __init__(
        self,
        cfg: RFPOCriticConfig,
        *,
        action_dim: int,
        condition_dim: int,
        state_dim: int,
    ) -> None:
        super().__init__()
        self.action_proj = nn.Linear(action_dim, cfg.hidden_size)
        self.condition_proj = nn.Linear(condition_dim, cfg.hidden_size)
        self.state_proj = nn.Linear(state_dim, cfg.hidden_size)
        self.value_token = nn.Parameter(torch.empty(1, 1, cfg.hidden_size))

        config = Gemma3TextConfig(
            vocab_size=1,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.mlp_hidden_size,
            num_hidden_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            query_pre_attn_scalar=cfg.head_dim,
            hidden_activation="gelu_pytorch_tanh",
            rms_norm_eps=1e-6,
            rope_theta=1_000_000.0,
            attention_bias=False,
            attention_dropout=0.0,
            layer_types=["full_attention"] * cfg.num_hidden_layers,
            use_cache=False,
            pad_token_id=0,
            eos_token_id=0,
            bos_token_id=0,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "sdpa"
        self.transformer = Gemma3TextModel(config)
        # RFPO supplies continuous embeddings; no vocabulary is trained.
        self.transformer.embed_tokens = None
        self.q_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.normal_(self.value_token, std=0.02)
        for layer in self.q_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.normal_(self.q_head[-1].weight, std=1e-3)

    def forward(
        self,
        actions: torch.Tensor,
        *,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor,
        state: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [B, 1] Q values; retain gradients through normalized actions."""
        batch_size, action_chunk = actions.shape[:2]
        dtype = self.action_proj.weight.dtype
        tokens = torch.cat(
            [
                self.action_proj(actions.to(dtype=dtype)),
                self.condition_proj(condition_tokens.detach().to(dtype=dtype)),
                self.state_proj(state.detach().to(dtype=dtype))[:, None],
                self.value_token.expand(batch_size, -1, -1),
            ],
            dim=1,
        )
        if action_mask is None:
            action_mask = torch.ones(
                (batch_size, action_chunk), dtype=torch.bool, device=actions.device
            )
        valid_mask = torch.cat(
            [
                action_mask.to(device=actions.device, dtype=torch.bool),
                condition_mask.to(device=actions.device, dtype=torch.bool),
                torch.ones((batch_size, 2), dtype=torch.bool, device=actions.device),
            ],
            dim=1,
        )
        position_ids = valid_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)
        attention_mask = torch.zeros(
            (batch_size, 1, 1, tokens.shape[1]), dtype=dtype, device=actions.device
        ).masked_fill(~valid_mask[:, None, None, :], torch.finfo(dtype).min)
        hidden = self.transformer(
            inputs_embeds=tokens,
            # A mask mapping bypasses Gemma3's causal-mask construction.
            attention_mask={"full_attention": attention_mask},
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state
        return self.q_head(hidden[:, -1]).float()


class RFPOCritic(nn.Module):
    """Ensemble of independent Gemma3-style Q networks for RFPO."""

    def __init__(
        self,
        cfg: DictConfig,
        *,
        action_dim: int,
        condition_dim: int,
        state_dim: int,
    ) -> None:
        super().__init__()
        self.cfg = RFPOCriticConfig(**cfg)
        self.q_networks = nn.ModuleList(
            RFPOQNetwork(
                self.cfg,
                action_dim=action_dim,
                condition_dim=condition_dim,
                state_dim=state_dim,
            )
            for _ in range(self.cfg.num_q_heads)
        )

    def forward(
        self,
        actions: torch.Tensor,
        *,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor,
        state: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [B, num_q_heads] without reducing the independent Q estimates."""
        return torch.cat(
            [
                network(
                    actions,
                    condition_tokens=condition_tokens,
                    condition_mask=condition_mask,
                    state=state,
                    action_mask=action_mask,
                )
                for network in self.q_networks
            ],
            dim=-1,
        )
