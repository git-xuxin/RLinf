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

"""OpenPI-aligned denoising sampler with an RFPO residual velocity."""

from __future__ import annotations

import random
from typing import Any

import torch


class RFPOGuidedSampler:
    """Run OpenPI denoising with SAC residual-velocity interventions."""

    def __init__(
        self,
        *,
        num_denoise_steps: int,
        rfpo_action_chunk: int,
        rfpo_action_dim: int,
        active_step_indices: tuple[int, ...],
        max_residual_velocity_rms: float,
        differentiate_base_velocity: bool,
        log_prob_reduction: str,
    ) -> None:
        self.num_denoise_steps = num_denoise_steps
        self.rfpo_action_chunk = rfpo_action_chunk
        self.rfpo_action_dim = rfpo_action_dim
        self.active_step_indices = frozenset(active_step_indices)
        self.max_residual_velocity_rms = max_residual_velocity_rms
        self.differentiate_base_velocity = differentiate_base_velocity
        self.log_prob_reduction = log_prob_reduction

    def _reduce_log_probs(
        self, log_probs: list[torch.Tensor], batch_size: int, device: torch.device
    ) -> torch.Tensor:
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

    def sample_mean_var_val(
        self,
        model,
        x_t,
        idx,
        state,
        state_embedding,
        prefix_output,
        prefix_pad_masks,
        past_key_values,
        sample_method,
        denoise_steps,
        compute_values=True,
        *,
        deterministic: bool,
        retain_residual_grads: bool,
    ):
        """Copy OpenPI's step body and add the SAC residual to ``v_t``."""
        # Keep this body aligned with
        # OpenPi0ForRLActionPrediction.sample_mean_var_val. RFPO-specific code
        # is confined to the block between base_velocity and v_t below.
        bsize = state.shape[0]
        device = state.device
        step_idx = idx
        if isinstance(idx, int):
            idx = torch.full((), idx, device=device).expand(bsize)
        noise_level = model._get_noise_level(device=device, dtype=x_t.dtype)
        timesteps = model._get_timesteps(denoise_steps, device)
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]

        base_velocity, suffix_out = model.get_velocity(
            state, x_t, t_input, prefix_pad_masks, past_key_values
        )
        if not self.differentiate_base_velocity:
            base_velocity = base_velocity.detach()
        active_base_velocity = base_velocity[
            :, : self.rfpo_action_chunk, : self.rfpo_action_dim
        ]
        base_velocity_rms = (
            active_base_velocity.float().pow(2).mean(dim=(1, 2)).sqrt()
        )

        delta_velocity = torch.zeros_like(base_velocity)
        actor_output = None
        if step_idx in self.active_step_indices:
            actor_output = model.residual_actor(
                x_t,
                base_velocity.detach(),
                t_input,
                state_embedding=state_embedding,
                prefix_tokens=prefix_output,
                condition_mask=prefix_pad_masks.to(dtype=torch.bool),
                deterministic=deterministic,
                max_residual_velocity_rms=self.max_residual_velocity_rms,
            )
            delta_velocity = actor_output["delta_velocity"]
            if delta_velocity.shape != base_velocity.shape:
                raise ValueError(
                    "RFPO SAC delta_velocity must match the pi0 velocity shape."
                )
            if retain_residual_grads and delta_velocity.requires_grad:
                delta_velocity.retain_grad()

        # This is the only change to OpenPI's denoising vector field.
        v_t = base_velocity + delta_velocity

        if (
            model.config.add_value_head
            and compute_values
            and not model.config.value_after_vlm
        ):
            value_t = model._compute_value_from_suffix(suffix_out)
        else:
            value_t = torch.zeros((bsize), device=device)

        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if sample_method == "flow_ode":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif sample_method == "flow_sde":
            denom_timesteps = torch.where(timesteps == 1, timesteps[1], timesteps)
            sigma_ratio = timesteps / (1 - denom_timesteps)
            sigmas = noise_level * torch.sqrt(sigma_ratio)[:-1]
            sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
            x0_weight = torch.ones_like(t_input) - (t_input - delta)
            x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
            x_t_std = torch.sqrt(delta) * sigma_i
        elif sample_method == "flow_cps":
            pi = torch.pi
            cos_term = torch.cos(pi * noise_level / 2).to(device)
            sin_term = torch.sin(pi * noise_level / 2).to(device)
            x0_weight = torch.ones_like(t_input) - (t_input - delta)
            x1_weight = (t_input - delta) * cos_term
            x_t_std = (t_input - delta) * sin_term
        elif sample_method == "flow_noise":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = model.noise_head(suffix_out)
        else:
            raise ValueError(f"Invalid noise method: {sample_method}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        step_output = {
            "actor_output": actor_output,
            "base_velocity_rms": base_velocity_rms,
            "delta_velocity": delta_velocity,
        }
        return x_t_mean, x_t_std, value_t, v_t, step_output

    def sample(
        self,
        model,
        *,
        state: torch.Tensor,
        state_embedding: torch.Tensor,
        prefix_output: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        noise: torch.Tensor | None = None,
        mode: str = "train",
        compute_values: bool = True,
        deterministic: bool | None = None,
        retain_residual_grads: bool = False,
    ) -> dict[str, Any]:
        """Copy OpenPI's cached-prefix sampler and collect RFPO SAC outputs."""
        bsize = state.shape[0]
        device = state.device
        num_steps = self.num_denoise_steps
        if noise is None:
            actions_shape = (
                bsize,
                model.config.action_horizon,
                model.config.action_dim,
            )
            noise = model.sample_noise(actions_shape, device)
        else:
            noise = noise.to(model.action_in_proj.weight.dtype)
        if noise.ndim != 3:
            raise ValueError(
                f"RFPO noise must have shape [B, H, D], got {tuple(noise.shape)}."
            )
        if self.rfpo_action_chunk > noise.shape[1]:
            raise ValueError(
                "RFPO execution chunk cannot exceed the pi0 action horizon."
            )
        if self.rfpo_action_dim > noise.shape[2]:
            raise ValueError(
                "RFPO action dimension cannot exceed the pi0 model action width."
            )
        if prefix_output.shape[0] != bsize:
            raise ValueError("RFPO prefix tensors must share the state batch size.")
        if state_embedding.ndim != 3 or state_embedding.shape[:2] != (bsize, 1):
            raise ValueError("RFPO state embedding must have shape [B, 1, S].")
        if deterministic is None:
            deterministic = mode == "eval"

        x_t = noise
        chains = []
        log_probs = []
        values = []
        chains.append(x_t)

        if model.use_vlm_value:
            values_vlm = model.get_value_from_vlm(prefix_output)
        if model.config.joint_logprob:
            initial_log_prob = model.get_logprob_norm(
                x_t, torch.zeros_like(noise), torch.ones_like(noise)
            )
            log_probs.append(initial_log_prob)

        if mode == "train":
            if model.config.joint_logprob:
                denoise_inds = torch.arange(num_steps)
            elif model.config.ignore_last:
                denoise_inds = torch.tensor(
                    [random.randint(0, num_steps - 2)] * num_steps
                )
            else:
                denoise_inds = torch.tensor(
                    [random.randint(0, num_steps - 1)] * num_steps
                )
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        internal_log_probs = []
        residual_norms = []
        residual_norms_per_step = []
        residual_ratios = []
        projection_scales = []
        velocity_norms = []
        active_residuals: list[tuple[int, torch.Tensor]] = []
        active_mask = torch.zeros(num_steps, dtype=torch.bool, device=device)

        for idx in range(num_steps):
            if idx == denoise_inds[0][idx]:
                sample_method = model.config.noise_method
            else:
                sample_method = "flow_ode"
            x_t_mean, x_t_std, value_t, _, step_output = self.sample_mean_var_val(
                model,
                x_t,
                idx,
                state,
                state_embedding,
                prefix_output,
                prefix_pad_masks,
                past_key_values,
                sample_method,
                num_steps,
                compute_values,
                deterministic=deterministic,
                retain_residual_grads=retain_residual_grads,
            )
            x_t = x_t_mean + model.sample_noise(x_t.shape, device) * x_t_std
            log_prob = model.get_logprob_norm(x_t, x_t_mean, x_t_std)
            values.append(value_t)
            chains.append(x_t)
            log_probs.append(log_prob)

            velocity_norms.append(step_output["base_velocity_rms"])
            actor_output = step_output["actor_output"]
            residual_rms = torch.zeros_like(step_output["base_velocity_rms"])
            if actor_output is not None:
                active_mask[idx] = True
                active_delta_velocity = actor_output["active_delta_velocity"]
                active_residuals.append((idx, active_delta_velocity))
                internal_log_probs.append(actor_output["log_prob"])
                residual_rms = (
                    active_delta_velocity.float().pow(2).mean(dim=(1, 2)).sqrt()
                )
                residual_norms.append(residual_rms)
                residual_ratios.append(
                    residual_rms / (step_output["base_velocity_rms"] + 1e-6)
                )
                projection_scales.append(actor_output["projection_scale"])
            residual_norms_per_step.append(residual_rms)

        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.rfpo_action_chunk, : self.rfpo_action_dim
        ]
        if model.config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[
                torch.arange(log_probs.shape[0], device=device),
                denoise_inds[:, 0],
            ]
        if model.use_vlm_value:
            values = values_vlm[:, None]
        else:
            values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)

        base_velocity_rms_per_step = torch.stack(velocity_norms, dim=1)
        residual_velocity_rms_per_step = torch.stack(
            residual_norms_per_step, dim=1
        )
        base_velocity_rms = base_velocity_rms_per_step.mean(dim=1)
        if residual_norms:
            residual_rms = torch.stack(residual_norms, dim=1).mean(dim=1)
            residual_to_base_ratio = torch.stack(residual_ratios, dim=1).mean(dim=1)
            residual_projection_scale = torch.stack(projection_scales, dim=1).mean(
                dim=1
            )
        else:
            residual_rms = torch.zeros_like(base_velocity_rms)
            residual_to_base_ratio = torch.zeros_like(base_velocity_rms)
            residual_projection_scale = torch.ones_like(base_velocity_rms)
        return {
            "actions": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
            "internal_log_prob": self._reduce_log_probs(
                internal_log_probs, bsize, device
            ),
            "residual_rms": residual_rms,
            "base_velocity_rms": base_velocity_rms,
            "base_velocity_rms_per_step": base_velocity_rms_per_step,
            "residual_velocity_rms_per_step": residual_velocity_rms_per_step,
            "residual_to_base_ratio": residual_to_base_ratio,
            "residual_projection_scale": residual_projection_scale,
            "active_step_mask": active_mask,
            "active_residuals": active_residuals,
        }
