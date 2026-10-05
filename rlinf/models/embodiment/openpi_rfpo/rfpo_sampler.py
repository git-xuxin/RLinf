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

"""Euler denoising with learned residual velocities."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from .metrics import (
    RFPO_ACTION_GROUPS,
    RFPOStepStats,
    group_mean_square,
    group_rms,
    parallel_vertical_group_rms,
)

if TYPE_CHECKING:
    from .backbone import RFPOBackboneAdapter, RFPOCondition
    from .rfpo_actor import RFPOActor


def compute_raw_mean_l2(
    raw_mean_group_mse: torch.Tensor, coefficients: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted L2 loss, group MSE, and weighted group terms.

    Average [guided_steps, batch, 3] over steps and batch before weighting.
    """
    group_count = len(RFPO_ACTION_GROUPS)
    if raw_mean_group_mse.ndim != 3 or raw_mean_group_mse.shape[-1] != group_count:
        raise ValueError(
            f"RFPO raw-mean group MSE must have shape [steps, batch, {group_count}]."
        )
    if coefficients.shape != (group_count,):
        raise ValueError(
            f"RFPO raw_mean_l2 coefficients must have shape [{group_count}]."
        )
    if raw_mean_group_mse.shape[0] == 0:
        return (
            raw_mean_group_mse.new_zeros(()),
            raw_mean_group_mse.new_zeros((group_count,)),
            raw_mean_group_mse.new_zeros((group_count,)),
        )
    group_mse = raw_mean_group_mse.float().mean(dim=(0, 1))
    weighted_group_terms = group_mse * coefficients.to(
        device=group_mse.device, dtype=group_mse.dtype
    )
    return weighted_group_terms.sum(), group_mse, weighted_group_terms


@dataclass(frozen=True)
class RFPOSampler:
    """Add residual velocities on selected denoising steps and action positions."""

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
        collect_step_stats: bool = False,
    ) -> tuple[torch.Tensor, RFPOStepStats | None, torch.Tensor]:
        """Return model actions, optional step statistics, and residual-mean group MSE.

        Residual noise [steps, batch, chunk, action_dim] uses absolute step indices.
        Deterministic mode uses residual means; omitted initial noise is sampled.
        """
        horizon, model_dim = adapter.action_shape
        env_chunk, action_dim = adapter.env_action_shape
        chunk = self.rfpo_action_chunk
        num_steps = adapter.num_steps
        if not (0 < env_chunk <= horizon and 0 < chunk <= horizon):
            raise ValueError("Action chunks must be positive and fit action_horizon.")
        if not 0 < action_dim <= model_dim:
            raise ValueError("Environment action width must fit the pi action width.")
        if not self.active_step_indices:
            raise ValueError("Residual step indices must select at least one step.")
        if num_steps <= 0 or any(
            step < 0 or step >= num_steps for step in self.active_step_indices
        ):
            raise ValueError(
                "Residual step indices must fit the positive pi step count."
            )

        batch_size = condition.tokens.shape[0]
        device = condition.tokens.device
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
        # Integrate from t=1 (noise) to t=0 (actions).
        dt = -1.0 / num_steps
        t = 1.0
        base_velocity_steps: list[torch.Tensor] = []
        delta_velocity_steps: list[torch.Tensor] = []
        delta_log_std_steps: list[torch.Tensor] = []
        delta_parallel_steps: list[torch.Tensor] = []
        delta_vertical_steps: list[torch.Tensor] = []
        action_noise_steps: list[torch.Tensor] = []
        raw_mean_group_mse_steps: list[torch.Tensor] = []
        active_step_mask = torch.zeros(num_steps, dtype=torch.bool, device=device)
        for step in range(num_steps):
            timestep = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            base_velocity = adapter.velocity(condition, x_t, timestep)
            velocity = base_velocity
            output = None
            if not force_zero_residual and step in self.active_step_indices:
                suffix_embedding = adapter.embed_suffix(condition, x_t, timestep)
                output = actor(
                    velocity[:, :chunk, :action_dim],
                    timestep,
                    suffix_embedding=suffix_embedding[
                        :, : chunk + int(adapter.suffix_has_state)
                    ],
                    condition_tokens=condition.tokens,
                    condition_mask=condition.mask,
                    deterministic=deterministic,
                    noise=None if residual_noise is None else residual_noise[step],
                )
                # Add residuals only to the guided chunk's environment dimensions.
                residual = F.pad(
                    output["delta_velocity"],
                    (0, model_dim - action_dim, 0, horizon - chunk),
                )
                velocity = velocity + residual
                # Regularize the unsquashed mean, preserving its training graph.
                raw_mean_group_mse_steps.append(group_mean_square(output["mean"]))
            x_t = x_t + dt * velocity
            t += dt

            if collect_step_stats:
                with torch.no_grad():
                    active_base = base_velocity[:, :chunk, :action_dim]
                    base_velocity_steps.append(group_rms(active_base))
                    if output is None:
                        zero_group_rms = torch.zeros(
                            (batch_size, len(RFPO_ACTION_GROUPS)), device=device
                        )
                        delta_velocity_steps.append(zero_group_rms.clone())
                        delta_log_std_steps.append(zero_group_rms.clone())
                        delta_parallel_steps.append(zero_group_rms.clone())
                        delta_vertical_steps.append(zero_group_rms.clone())
                    else:
                        active_delta = output["delta_velocity"]
                        delta_velocity_steps.append(group_rms(active_delta))
                        delta_log_std_steps.append(group_rms(output["log_std"].float()))
                        parallel, vertical = parallel_vertical_group_rms(
                            active_delta, active_base
                        )
                        delta_parallel_steps.append(parallel)
                        delta_vertical_steps.append(vertical)
                    action_noise_steps.append(
                        x_t[:, :chunk, :action_dim].float().mean(dim=1)
                    )
                    active_step_mask[step] = output is not None
        step_stats = None
        if collect_step_stats:
            step_stats = RFPOStepStats(
                base_velocity_group_rms=torch.stack(base_velocity_steps),
                delta_velocity_group_rms=torch.stack(delta_velocity_steps),
                delta_log_std_group_rms=torch.stack(delta_log_std_steps),
                delta_parallel_group_rms=torch.stack(delta_parallel_steps),
                delta_vertical_group_rms=torch.stack(delta_vertical_steps),
                action_noise_dim_mean=torch.stack(action_noise_steps),
                active_step_mask=active_step_mask,
            )
        if raw_mean_group_mse_steps:
            raw_mean_group_mse = torch.stack(raw_mean_group_mse_steps)
        else:
            raw_mean_group_mse = torch.zeros(
                (0, batch_size, len(RFPO_ACTION_GROUPS)),
                device=device,
                dtype=torch.float32,
            )
        return x_t, step_stats, raw_mean_group_mse
