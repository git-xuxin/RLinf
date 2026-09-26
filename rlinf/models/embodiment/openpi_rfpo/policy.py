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

"""Container for RFPO trainable networks, owned separately from pi."""

from typing import Literal

import torch
from torch import nn

from .rfpo_actor import RFPOActor
from .rfpo_critic import RFPOCritic


class RFPOPolicy(nn.Module):
    """Own only the small actor and critic, including both in weight synchronization."""

    def __init__(self, actor: RFPOActor, critic: RFPOCritic) -> None:
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
