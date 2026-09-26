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

"""Interfaces for the RFPO-specific DiT actor and Gemma3-style critic."""

from typing import Literal

import torch
from omegaconf import DictConfig
from torch import nn

from .config import RFPOActorConfig, RFPOCriticConfig


class RFPOActor(nn.Module):
    """Residual velocity actor interface; network implementation is pending."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = RFPOActorConfig(**cfg)

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
    ) -> dict[str, torch.Tensor]:
        """Return residual velocity and log probability for normalized actions.

        ``action_features`` contains only action positions from the pi expert.
        ``state`` is the normalized observation state, separate from these tokens.
        Prefix length and feature widths come from the backbone adapter.
        """
        raise NotImplementedError("The RFPO DiT actor will be implemented separately.")


class RFPOCritic(nn.Module):
    """Action-chunk Q critic interface; network implementation is pending."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = RFPOCriticConfig(**cfg)

    def forward(
        self,
        actions: torch.Tensor,
        *,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor,
        state: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ensemble Q values, preserving gradients through model-space actions."""
        raise NotImplementedError(
            "The RFPO Gemma3 critic will be implemented separately."
        )


class RFPOPolicy(nn.Module):
    """Own only the small actor and critic, including both in weight synchronization."""

    def __init__(self, actor: nn.Module, critic: nn.Module):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(
        self, *, component: Literal["actor", "critic"], **kwargs
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Dispatch to a small network without accessing the frozen backbone."""
        if component == "actor":
            return self.actor(**kwargs)
        if component == "critic":
            return self.critic(**kwargs)
        raise ValueError(f"Unknown RFPO policy component: {component!r}")
