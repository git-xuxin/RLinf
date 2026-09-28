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

"""Per-sample denoising diagnostics reduced per LIBERO action group."""

from dataclasses import dataclass

import torch

# LIBERO end-effector layout: translation [0:3], rotation [3:6], gripper [6].
RFPO_ACTION_GROUPS: tuple[tuple[str, slice], ...] = (
    ("translation", slice(0, 3)),
    ("rotation", slice(3, 6)),
    ("gripper", slice(6, 7)),
)
RFPO_ACTION_GROUP_NAMES = tuple(name for name, _ in RFPO_ACTION_GROUPS)
_ACTION_WIDTH = RFPO_ACTION_GROUPS[-1][1].stop


@dataclass(frozen=True)
class RFPOStepStats:
    """Per-sample denoising diagnostics with the denoise-step axis first.

    Grouped tensors have shape [steps, batch, groups] in translation,
    rotation, gripper order; the action-noise mean has shape
    [steps, batch, action_dim]. Delta-family rows are zero on steps where the
    residual actor did not run; ``active_step_mask`` flags those steps.
    """

    base_velocity_group_rms: torch.Tensor
    delta_velocity_group_rms: torch.Tensor
    delta_log_std_group_rms: torch.Tensor
    delta_parallel_group_rms: torch.Tensor
    delta_vertical_group_rms: torch.Tensor
    action_noise_dim_mean: torch.Tensor
    active_step_mask: torch.Tensor


def _check_action_width(values: torch.Tensor) -> None:
    if values.ndim < 2 or values.shape[-1] != _ACTION_WIDTH:
        raise ValueError(
            f"RFPO action metrics expect [batch, ..., {_ACTION_WIDTH}] LIBERO "
            f"actions, got {tuple(values.shape)}."
        )


def group_mean_square(values: torch.Tensor) -> torch.Tensor:
    """Mean square per action group of the per-position group norm.

    ``values`` is [batch, ..., 7]; the result is [batch, 3]. Each group sums
    its member dimensions into one squared norm per position before the
    position mean, so the group width never divides the result; groups with
    different physical scales still never mix. The raw-mean L2 penalty uses
    this reduction on the actor mean.
    """
    _check_action_width(values)
    position_dims = tuple(range(1, values.ndim - 1))
    groups = []
    for _, group_slice in RFPO_ACTION_GROUPS:
        squared_norm = values[..., group_slice].float().square().sum(dim=-1)
        if position_dims:
            squared_norm = squared_norm.mean(dim=position_dims)
        groups.append(squared_norm)
    return torch.stack(groups, dim=-1)


def group_rms(values: torch.Tensor) -> torch.Tensor:
    """Root-mean-square per action group of the per-position group norm.

    Equivalent to ``group_mean_square(values).sqrt()``; see that function
    for the reduction convention.
    """
    return group_mean_square(values).sqrt()


def parallel_vertical_group_rms(
    delta_velocity: torch.Tensor, base_velocity: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group rms of delta's scalar projection on base and its remainder.

    Both inputs are [batch, ..., 7]; each result is [batch, 3]. Within each
    group and position, the parallel part is the scalar projection of delta
    onto the base unit vector and the perpendicular part is the norm of the
    remainder, so the two results satisfy
    ``group_rms(delta_velocity) ** 2 == parallel ** 2 + vertical ** 2`` per
    group. A zero base direction has no projection, so the whole delta counts
    as perpendicular.
    """
    _check_action_width(delta_velocity)
    _check_action_width(base_velocity)
    if delta_velocity.shape != base_velocity.shape:
        raise ValueError("RFPO parallel metrics need matching delta and base shapes.")
    position_dims = tuple(range(1, delta_velocity.ndim - 1))
    parallel_groups = []
    vertical_groups = []
    for _, group_slice in RFPO_ACTION_GROUPS:
        delta = delta_velocity[..., group_slice].float()
        base = base_velocity[..., group_slice].float()
        base_norm = base.norm(dim=-1, keepdim=True)
        # Clamping keeps a vanished base direction a zero unit vector instead
        # of producing NaNs; the delta then stays entirely perpendicular.
        unit_base = base / base_norm.clamp_min(torch.finfo(base.dtype).eps)
        parallel_scalar = (delta * unit_base).sum(dim=-1)
        vertical_norm = (delta - parallel_scalar.unsqueeze(-1) * unit_base).norm(dim=-1)
        parallel_square = parallel_scalar.square()
        vertical_square = vertical_norm.square()
        if position_dims:
            parallel_square = parallel_square.mean(dim=position_dims)
            vertical_square = vertical_square.mean(dim=position_dims)
        parallel_groups.append(parallel_square.sqrt())
        vertical_groups.append(vertical_square.sqrt())
    return torch.stack(parallel_groups, dim=-1), torch.stack(vertical_groups, dim=-1)


def denoise_step_metrics(stats: RFPOStepStats) -> dict[str, float]:
    """Average per-sample stats into ``denoise_step_*`` scalar metrics.

    Velocity-family keys carry an ``_rms`` suffix and use the
    ``denoise_step_{step}/{name}/{group}_rms`` layout. ``velocity/base`` and
    ``action_noise`` exist on every step; the delta family only exists on
    active steps. ``denoise_step_{step}/action_noise/dim_{dim}`` records the
    per-dimension mean noisy action after each step's update. Callers add the
    ``rfpo/`` namespace. Values move to the CPU once here, so the returned
    mapping holds plain floats.
    """
    step_count = int(stats.active_step_mask.shape[0])
    active_steps = stats.active_step_mask.tolist()
    grouped_specs = (
        ("velocity/base", stats.base_velocity_group_rms, False),
        ("velocity/delta", stats.delta_velocity_group_rms, True),
        ("velocity/delta_log_std", stats.delta_log_std_group_rms, True),
        ("velocity/parallel", stats.delta_parallel_group_rms, True),
        ("velocity/vertical", stats.delta_vertical_group_rms, True),
    )
    group_count = len(RFPO_ACTION_GROUP_NAMES)
    metrics: dict[str, float] = {}
    for name, values, active_only in grouped_specs:
        if values.ndim != 3 or values.shape[0] != step_count:
            raise ValueError(f"RFPO {name} stats must be [steps, batch, groups].")
        if values.shape[-1] != group_count:
            raise ValueError(f"RFPO {name} stats must contain the action groups.")
        step_means = values.detach().float().mean(dim=1).cpu().tolist()
        for step in range(step_count):
            if active_only and not active_steps[step]:
                continue
            for group_idx, group in enumerate(RFPO_ACTION_GROUP_NAMES):
                metrics[f"denoise_step_{step}/{name}/{group}_rms"] = step_means[step][
                    group_idx
                ]
    if stats.action_noise_dim_mean.ndim != 3 or (
        stats.action_noise_dim_mean.shape[0] != step_count
    ):
        raise ValueError("RFPO action-noise stats must be [steps, batch, action_dim].")
    noise_step_means = stats.action_noise_dim_mean.detach().float().mean(dim=1)
    noise_step_means = noise_step_means.cpu().tolist()
    for step in range(step_count):
        for dim, value in enumerate(noise_step_means[step]):
            metrics[f"denoise_step_{step}/action_noise/dim_{dim}"] = value
    return metrics
