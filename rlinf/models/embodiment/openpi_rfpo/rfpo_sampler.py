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

"""Residual-guided Euler sampler for frozen pi0."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RFPOSamplerOutput:
    """Outputs from one complete residual-guided denoising trajectory."""

    model_action_horizon: torch.Tensor
    executed_action_chunk: torch.Tensor
    internal_log_prob: torch.Tensor
    residual_rms: torch.Tensor
    base_velocity_rms: torch.Tensor
    active_step_mask: torch.Tensor
    active_residuals: list[tuple[int, torch.Tensor]]


class RFPOGuidedSampler:
    """Runs pi0 Euler integration with residual velocity interventions."""

    def __init__(
        self,
        *,
        num_denoise_steps: int,
        action_chunk: int,
        active_step_indices: tuple[int, ...],
        residual_ratio: float,
        differentiate_base_velocity: bool,
        log_prob_reduction: str,
    ) -> None:
        self.num_denoise_steps = num_denoise_steps
        self.action_chunk = action_chunk
        self.active_step_indices = frozenset(active_step_indices)
        self.residual_ratio = residual_ratio
        self.differentiate_base_velocity = differentiate_base_velocity
        self.log_prob_reduction = log_prob_reduction

    def _reduce_log_probs(self, log_probs: list[torch.Tensor], batch_size: int, device):
        if not log_probs:
            return torch.zeros(batch_size, device=device, dtype=torch.float32)
        stacked = torch.stack(log_probs, dim=1)
        per_step_log_prob = stacked.sum(dim=(2, 3))
        if self.log_prob_reduction == "mean_active":
            return per_step_log_prob.mean(dim=1)
        if self.log_prob_reduction == "sum_active":
            return per_step_log_prob.sum(dim=1)
        raise ValueError(
            "internal_log_prob_reduction must be 'mean_active' or 'sum_active'."
        )

    def sample(
        self,
        model,
        *,
        state: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        noise: torch.Tensor,
        deterministic: bool,
        retain_residual_grads: bool,
    ) -> RFPOSamplerOutput:
        device = noise.device
        batch_size = noise.shape[0]
        if noise.ndim != 3:
            raise ValueError(
                f"RFPO noise must have shape [B, H, D], got {tuple(noise.shape)}."
            )
        if self.action_chunk > noise.shape[1]:
            raise ValueError(
                "RFPO execution chunk cannot exceed the pi0 action horizon."
            )
        if state.shape[0] != batch_size or condition_tokens.shape[0] != batch_size:
            raise ValueError("RFPO condition tensors must share the noise batch size.")
        timesteps = model.get_rfpo_timesteps(device)
        if timesteps.shape != (self.num_denoise_steps + 1,):
            raise ValueError(
                "RFPO pi0 timestep schedule must contain one boundary per step."
            )
        x_t = noise
        log_probs = []
        residual_norms = []
        velocity_norms = []
        active_residuals: list[tuple[int, torch.Tensor]] = []
        active_mask = torch.zeros(
            self.num_denoise_steps, dtype=torch.bool, device=device
        )

        for step_idx in range(self.num_denoise_steps):
            timestep = timesteps[step_idx].expand(batch_size)
            step_size = (timesteps[step_idx + 1] - timesteps[step_idx]).expand(
                batch_size
            )
            base_velocity, _ = model.get_velocity(
                state, x_t, timestep, prefix_pad_masks, past_key_values
            )
            if not self.differentiate_base_velocity:
                base_velocity = base_velocity.detach()
            velocity_norms.append(base_velocity.float().pow(2).mean(dim=(1, 2)).sqrt())

            delta_velocity = torch.zeros_like(base_velocity)
            if step_idx in self.active_step_indices:
                active_mask[step_idx] = True
                actor_output = model.residual_actor(
                    x_t,
                    base_velocity.detach(),
                    timestep,
                    step_size,
                    state=state,
                    condition_tokens=condition_tokens,
                    condition_mask=condition_mask,
                    deterministic=deterministic,
                )
                if actor_output["sample"].shape != base_velocity.shape:
                    raise ValueError(
                        "RFPO residual actor output must match the pi0 velocity shape."
                    )
                delta_velocity = self.residual_ratio * actor_output["sample"]
                if retain_residual_grads and delta_velocity.requires_grad:
                    delta_velocity.retain_grad()
                active_residuals.append((step_idx, delta_velocity))
                log_probs.append(actor_output["log_prob"])
                residual_norms.append(
                    delta_velocity.float().pow(2).mean(dim=(1, 2)).sqrt()
                )
            x_t = x_t + step_size[:, None, None] * (
                base_velocity + delta_velocity
            )

        base_velocity_rms = torch.stack(velocity_norms, dim=1).mean(dim=1)
        if residual_norms:
            residual_rms = torch.stack(residual_norms, dim=1).mean(dim=1)
        else:
            residual_rms = torch.zeros_like(base_velocity_rms)
        return RFPOSamplerOutput(
            model_action_horizon=x_t,
            executed_action_chunk=x_t[:, : self.action_chunk],
            internal_log_prob=self._reduce_log_probs(log_probs, batch_size, device),
            residual_rms=residual_rms,
            base_velocity_rms=base_velocity_rms,
            active_step_mask=active_mask,
            active_residuals=active_residuals,
        )
