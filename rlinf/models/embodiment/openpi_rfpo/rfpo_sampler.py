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

"""Shared residual-velocity Euler sampler, called inside RFPOPolicy.forward."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

if TYPE_CHECKING:
    from .backbone import RFPOBackboneAdapter, RFPOCondition
    from .rfpo_actor import RFPOActor


@dataclass(frozen=True)
class RFPOSampler:
    """Integrate full pi actions while controlling a selected environment region."""

    rfpo_action_chunk: int = 20
    active_step_indices: tuple[int, ...] = (0, 1, 2)

    def sample(
        self,
        actor: "RFPOActor",
        *,
        adapter: "RFPOBackboneAdapter",
        condition: "RFPOCondition",
        noise: torch.Tensor | None = None,
        residual_noise: torch.Tensor | None = None,
        deterministic: bool = False,
        force_zero_residual: bool = False,
    ) -> torch.Tensor:
        """Return final model-space actions [B, H, D], preserving input gradients.

        ``noise`` is the initial [B, H, D] standard-normal sample. Optional
        ``residual_noise`` is [N, B, C, A], indexed by the absolute denoising
        step, including inactive steps. C is rfpo_action_chunk and A is the
        adapter's environment action width. Neither noise tensor is modified.
        The caller controls autograd and whether to use residual means.
        """
        horizon, model_dim = adapter.action_shape
        env_chunk, action_dim = adapter.env_action_shape
        chunk = self.rfpo_action_chunk
        num_steps = adapter.num_steps
        if not (0 < env_chunk <= horizon and 0 < chunk <= horizon):
            raise ValueError("Action chunks must be positive and fit action_horizon.")
        if not 0 < action_dim <= model_dim:
            raise ValueError("Environment action width must fit the pi action width.")
        if num_steps <= 0 or any(
            step < 0 or step >= num_steps for step in self.active_step_indices
        ):
            raise ValueError(
                "Residual step indices must fit the positive pi step count."
            )

        batch_size = condition.observation.state.shape[0]
        device = condition.observation.state.device
        shape = (batch_size, horizon, model_dim)
        if noise is None:
            noise = torch.randn(shape, device=device, dtype=torch.float32)
        elif noise.shape != shape:
            raise ValueError(f"Initial noise must have shape {shape}.")
        if residual_noise is not None and residual_noise.shape != (
            num_steps,
            batch_size,
            chunk,
            action_dim,
        ):
            raise ValueError("Residual noise must have shape [N, B, C, A].")

        x_t = noise.to(device=device, dtype=torch.float32)
        dt = -1.0 / num_steps
        t = 1.0
        for step in range(num_steps):
            timestep = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            base = adapter.velocity(condition, x_t, timestep)
            velocity = base.velocity
            if not force_zero_residual and step in self.active_step_indices:
                output = actor(
                    velocity[:, :chunk, :action_dim],
                    timestep,
                    action_features=base.action_features[:, :chunk],
                    condition_tokens=condition.tokens,
                    condition_mask=condition.mask,
                    state=condition.observation.state,
                    deterministic=deterministic,
                    noise=None if residual_noise is None else residual_noise[step],
                )
                residual = F.pad(
                    output["delta_velocity"],
                    (0, model_dim - action_dim, 0, horizon - chunk),
                )
                velocity = velocity + residual
            x_t = x_t + dt * velocity
            t += dt
        return x_t
