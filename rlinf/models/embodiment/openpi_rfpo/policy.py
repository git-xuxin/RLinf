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

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from .rfpo_actor import RFPOActor
from .rfpo_critic import RFPOCritic
from .rfpo_sampler import RFPOSampler

if TYPE_CHECKING:
    from .backbone import RFPOBackboneAdapter, RFPOCondition


class RFPOPolicy(nn.Module):
    """Own only the small actor and critic, including both in weight synchronization."""

    def __init__(
        self, actor: RFPOActor, critic: RFPOCritic, sampler: RFPOSampler
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.sampler = sampler

    def forward(
        self, *, component: Literal["actor", "critic"], **kwargs
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if component == "actor":
            return self._sample_actions(**kwargs)
        if component == "critic":
            return self.critic(**kwargs)
        raise ValueError(f"Unknown RFPO policy component: {component!r}")

    def _sample_actions(
        self,
        *,
        adapter: "RFPOBackboneAdapter",
        condition: "RFPOCondition",
        mode: Literal["train", "target", "rollout", "eval"] = "train",
        noise: torch.Tensor | None = None,
        residual_noise: torch.Tensor | None = None,
        force_zero_residual: bool = False,
    ) -> dict[str, torch.Tensor]:
        if mode not in ("train", "target", "rollout", "eval"):
            raise ValueError(f"Unknown RFPO sampling mode: {mode!r}")
        # Target actions come from the online actor, as in SAC/RLPD.
        with torch.set_grad_enabled(mode == "train" and torch.is_grad_enabled()):
            model_actions = self.sampler.sample(
                self.actor,
                adapter=adapter,
                condition=condition,
                noise=noise,
                residual_noise=residual_noise,
                deterministic=mode == "eval",
                force_zero_residual=force_zero_residual,
            )
            chunk, action_dim = adapter.env_action_shape
            return {
                "model_actions": model_actions,
                "actions": model_actions[:, :chunk, :action_dim],
            }
