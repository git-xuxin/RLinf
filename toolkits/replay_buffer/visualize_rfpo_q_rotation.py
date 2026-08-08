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

"""Compare RFPO Q values for angle-perturbed actions along one trajectory.

Every fifth replay observation is selected as a candidate by default. Each
candidate keeps its observation and replay action as ``a0``. The remaining
actions add the configured physical angle to one rotation component. Angles are
converted from radians into the normalized action space consumed by the critic.

The plot overlays ``min(Q1, Q2)`` for all action variants at every candidate.
The CSV retains both critic heads, their minimum, and the ranking against the
unperturbed replay action.

Example::

    # Reuse one episode selected from visualize_rfpo_q.py.
    python toolkits/replay_buffer/visualize_rfpo_q_rotation.py \
        --episode-ref /path/to/episode_0003_..._q.png
    
    python toolkits/replay_buffer/visualize_rfpo_q_rotation.py \
        --episode-ref /mnt/public2/xuxin/RFPO/RLinf/rfpo_q_visualizations_log_20260807-19-56-19_step_200/episode_0015_trajectory_465_0eab0f35-2da3-576a-a6ea-279529880cbd_env_002_t_0003_0040_q.png

    python toolkits/replay_buffer/visualize_rfpo_q_rotation.py \
        --ckpt /path/to/logs/20260806-03:30:52-run/.../global_step_80/actor \
        --trajectory-file /path/to/trajectory_12_weights.pt \
        --candidate-stride 5 --env-index 0
    
    python toolkits/replay_buffer/visualize_rfpo_q_rotation.py \
        --ckpt /mnt/public2/xuxin/RFPO/RLinf/logs/20260807-19:56:19-libero_object_async_rfpo_openpi/libero_object_async_rfpo_openpi/checkpoints/global_step_200/actor \
        --trajectory-file /mnt/public2/xuxin/RFPO/RLinf/logs/20260807-19:56:19-libero_object_async_rfpo_openpi/libero_object_async_rfpo_openpi/checkpoints/global_step_200/actor/rfpo_components/replay_buffer/rank_1/trajectory_464_0eab0f35-2da3-576a-a6ea-279529880cbd.pt \
        --candidate-stride 5 --env-index 0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__:
    from .visualize_rfpo_q import (
        _reshape_actions,
        _select_device,
        _torch_load,
        aligned_dones,
        build_rfpo_model,
        default_output_dir,
        find_train_config_near_checkpoint,
        load_actor_model_config,
        resolve_full_weights,
        rfpo_action_mask,
        trajectory_layout,
    )
else:
    from visualize_rfpo_q import (
        _reshape_actions,
        _select_device,
        _torch_load,
        aligned_dones,
        build_rfpo_model,
        default_output_dir,
        find_train_config_near_checkpoint,
        load_actor_model_config,
        resolve_full_weights,
        rfpo_action_mask,
        trajectory_layout,
    )


_DEFAULT_ROTATION_ANGLES = (0.0, 60.0, 180.0, 240.0, 300.0, 360.0)
_Q_Y_MIN = -0.1
_Q_Y_MAX = 1.0
_Q_Y_TICK_STEP = 0.1


@dataclass(frozen=True)
class ActionVariant:
    """Metadata for one angle-perturbed action variant."""

    index: int
    angle_degrees: float
    normalized_offset: float

    @property
    def name(self) -> str:
        """Return the short action-variant name used in outputs."""
        return f"a{self.index}"


@dataclass(frozen=True)
class EpisodeSelection:
    """Episode coordinates loaded from a Q-visualization manifest."""

    manifest_path: Path
    episode_index: int
    episode_id: str
    source_file: Path
    env_index: int
    start: int
    stop: int
    checkpoint: str | None
    training_config: str | None


def resolve_episode_selection(
    manifest_path: str | Path | None,
    *,
    episode_ref: str | Path | None = None,
    episode_index: int | None = None,
) -> EpisodeSelection:
    """Resolve an episode from ``visualize_rfpo_q.py`` manifest metadata."""
    if episode_ref is not None and episode_index is not None:
        raise ValueError("Use either episode_ref or episode_index, not both")
    if episode_ref is None and episode_index is None:
        raise ValueError("An episode_ref or episode_index is required")

    reference_path = (
        Path(episode_ref).expanduser() if episode_ref is not None else None
    )
    if manifest_path is None:
        if reference_path is None:
            raise ValueError("episode_index requires --episode-manifest")
        manifest = reference_path.parent / "manifest.json"
    else:
        manifest = Path(manifest_path).expanduser()
    manifest = manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Episode manifest does not exist: {manifest}")

    raw_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, Mapping):
        raise TypeError(f"Episode manifest is not a mapping: {manifest}")
    episodes = raw_manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"Episode manifest contains no episodes: {manifest}")

    if episode_index is not None:
        selected_index = int(episode_index)
        if selected_index < 0 or selected_index >= len(episodes):
            raise IndexError(
                f"episode_index must lie in [0, {len(episodes)}), got "
                f"{selected_index}"
            )
    else:
        assert episode_ref is not None
        reference_text = str(episode_ref)
        reference_name = Path(reference_text).name
        matches = []
        for fallback_index, entry in enumerate(episodes):
            if not isinstance(entry, Mapping):
                continue
            entry_index = int(entry.get("episode_index", fallback_index))
            identifiers = {
                str(entry_index),
                str(entry.get("episode_id", "")),
                *(
                    Path(str(entry.get(key, ""))).name
                    for key in ("curve", "csv", "video")
                ),
            }
            if reference_text in identifiers or reference_name in identifiers:
                matches.append(fallback_index)
        if len(matches) != 1:
            raise ValueError(
                f"Expected one episode matching {reference_text!r}, found "
                f"{len(matches)} in {manifest}"
            )
        selected_index = matches[0]

    entry = episodes[selected_index]
    if not isinstance(entry, Mapping):
        raise TypeError(f"Episode entry {selected_index} is not a mapping")
    source_value = entry.get("source_file")
    if not source_value:
        raise KeyError(f"Episode entry {selected_index} has no source_file")
    source_file = Path(str(source_value)).expanduser()
    if not source_file.is_absolute():
        trajectory_dir = raw_manifest.get("trajectory_dir")
        base_dir = (
            Path(str(trajectory_dir)).expanduser()
            if trajectory_dir
            else manifest.parent
        )
        source_file = base_dir / source_file
    source_file = source_file.resolve()

    start = int(entry["start"])
    stop = int(entry["stop"])
    if start < 0 or stop <= start:
        raise ValueError(
            f"Episode entry {selected_index} has invalid bounds [{start}, {stop})"
        )
    entry_index = int(entry.get("episode_index", selected_index))
    episode_id = str(entry.get("episode_id") or entry.get("curve") or entry_index)
    return EpisodeSelection(
        manifest_path=manifest,
        episode_index=entry_index,
        episode_id=episode_id,
        source_file=source_file,
        env_index=int(entry["env_index"]),
        start=start,
        stop=stop,
        checkpoint=(
            str(raw_manifest["checkpoint"])
            if raw_manifest.get("checkpoint")
            else None
        ),
        training_config=(
            str(raw_manifest["training_config"])
            if raw_manifest.get("training_config")
            else None
        ),
    )


def candidate_time_indices(
    time_size: int,
    candidate_stride: int = 5,
    candidate_start: int = 0,
    candidate_stop: int | None = None,
) -> list[int]:
    """Return source trajectory indices sampled at a fixed step interval."""
    if time_size <= 0:
        raise ValueError("time_size must be positive")
    if candidate_stride <= 0:
        raise ValueError("candidate_stride must be positive")
    if candidate_start < 0 or candidate_start >= time_size:
        raise ValueError(f"candidate_start must lie in [0, {time_size})")
    stop = time_size if candidate_stop is None else int(candidate_stop)
    if stop <= candidate_start or stop > time_size:
        raise ValueError(
            f"candidate_stop must lie in ({candidate_start}, {time_size}]"
        )
    return list(range(candidate_start, stop, candidate_stride))


def build_candidate_variants(
    base_actions: torch.Tensor,
    angles_degrees: Sequence[float],
    *,
    rotation_action_dim: int,
    normalized_units_per_radian: float,
    action_step: int = -1,
) -> tuple[torch.Tensor, list[ActionVariant]]:
    """Add angle offsets to one rotation component of candidate actions."""
    if base_actions.ndim != 3:
        raise ValueError(
            "base_actions must have shape [candidate, chunk, action_dim], got "
            f"{tuple(base_actions.shape)}"
        )
    if rotation_action_dim < 0 or rotation_action_dim >= base_actions.shape[2]:
        raise ValueError(
            f"rotation_action_dim {rotation_action_dim} exceeds action width "
            f"{base_actions.shape[2]}"
        )
    if action_step < -1 or action_step >= base_actions.shape[1]:
        raise ValueError(
            f"action_step must be -1 or lie in [0, {base_actions.shape[1]})"
        )
    if (
        not math.isfinite(normalized_units_per_radian)
        or normalized_units_per_radian <= 0
    ):
        raise ValueError("normalized_units_per_radian must be finite and positive")

    angles = [float(angle) for angle in angles_degrees]
    if not angles or angles[0] != 0.0:
        raise ValueError("angles_degrees must start at zero so a0 is unchanged")
    if any(not math.isfinite(angle) for angle in angles):
        raise ValueError("angles_degrees must be finite")
    if len(set(angles)) != len(angles):
        raise ValueError("angles_degrees must not contain duplicates")

    variants = []
    metadata = []
    for variant_index, angle_degrees in enumerate(angles):
        offset = math.radians(angle_degrees) * normalized_units_per_radian
        variant = base_actions.clone()
        if variant_index > 0:
            if action_step == -1:
                variant[:, :, rotation_action_dim] += offset
            else:
                variant[:, action_step, rotation_action_dim] += offset
        variants.append(variant)
        metadata.append(
            ActionVariant(
                index=variant_index,
                angle_degrees=angle_degrees,
                normalized_offset=0.0 if variant_index == 0 else offset,
            )
        )
    return torch.stack(variants, dim=1), metadata


def infer_normalized_units_per_radian(
    model: torch.nn.Module, rotation_action_dim: int
) -> float:
    """Infer the critic-action normalization scale from the output transform."""
    action_horizon = int(model.config.action_horizon)
    model_action_dim = int(model.config.action_dim)
    if rotation_action_dim < 0 or rotation_action_dim >= model_action_dim:
        raise ValueError(
            f"rotation_action_dim {rotation_action_dim} exceeds model action width "
            f"{model_action_dim}"
        )

    zero_actions = torch.zeros(1, action_horizon, model_action_dim)
    unit_actions = zero_actions.clone()
    unit_actions[:, :, rotation_action_dim] = 1.0
    # OpenPI's output transform unnormalizes every selector in the norm stats.
    # Even though only the action slope is used below, the transform therefore
    # requires a state entry. Pi0 pads state to the model action width.
    transform_state = torch.zeros(1, model_action_dim)
    try:
        zero_output = model.output_transform(
            {"actions": zero_actions, "state": transform_state}
        )["actions"]
        unit_output = model.output_transform(
            {"actions": unit_actions, "state": transform_state}
        )["actions"]
    except Exception as exc:
        raise RuntimeError(
            "Could not infer the action normalization scale through the model's "
            "output transform. Pass --normalized-units-per-radian explicitly."
        ) from exc

    physical_delta = (
        unit_output[..., rotation_action_dim]
        - zero_output[..., rotation_action_dim]
    ).float()
    finite_delta = physical_delta[torch.isfinite(physical_delta)]
    if finite_delta.numel() == 0:
        raise ValueError("The inferred rotation normalization scale is not finite")
    radians_per_normalized_unit = float(finite_delta.median())
    if radians_per_normalized_unit <= 0:
        raise ValueError(
            "The inferred rotation normalization scale must be positive; pass "
            "--normalized-units-per-radian explicitly."
        )
    if not torch.allclose(
        finite_delta,
        torch.full_like(finite_delta, radians_per_normalized_unit),
        rtol=1e-4,
        atol=1e-6,
    ):
        raise ValueError(
            "The model's rotation output transform is not affine across the action "
            "chunk; pass --normalized-units-per-radian explicitly."
        )
    return 1.0 / radians_per_normalized_unit


@torch.inference_mode()
def evaluate_candidate_variants(
    model: torch.nn.Module,
    payload: Mapping[str, Any],
    *,
    candidate_indices: Sequence[int],
    env_index: int,
    variants: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Evaluate every candidate/variant pair with its corresponding observation."""
    from rlinf.models.embodiment.base_policy import ForwardType

    time_size, env_count = trajectory_layout(payload)
    if env_index < 0 or env_index >= env_count:
        raise IndexError(f"env_index must lie in [0, {env_count}), got {env_index}")
    if not candidate_indices:
        raise ValueError("candidate_indices must not be empty")
    if min(candidate_indices) < 0 or max(candidate_indices) >= time_size:
        raise IndexError(f"candidate indices must lie in [0, {time_size})")
    if variants.ndim != 4 or variants.shape[0] != len(candidate_indices):
        raise ValueError(
            "variants must have shape [candidate, variant, chunk, action_dim]"
        )

    candidate_count, variant_count = variants.shape[:2]
    flat_actions = variants.reshape(
        candidate_count * variant_count, *variants.shape[2:]
    )
    candidate_slots = torch.arange(candidate_count).repeat_interleave(variant_count)
    source_times = torch.as_tensor(candidate_indices, dtype=torch.long)[candidate_slots]

    curr_obs = payload["curr_obs"]
    forward_inputs = payload["forward_inputs"]
    done_tensor = aligned_dones(payload, time_size)
    candidate_dones = done_tensor[
        torch.as_tensor(candidate_indices, dtype=torch.long), env_index
    ]
    action_masks = rfpo_action_mask(
        candidate_dones, int(model.config.rfpo_action_chunk)
    )

    all_q_values = []
    for batch_start in range(0, flat_actions.shape[0], batch_size):
        batch_stop = min(batch_start + batch_size, flat_actions.shape[0])
        batch_slots = candidate_slots[batch_start:batch_stop]
        batch_times = source_times[batch_start:batch_stop]
        obs_batch = {
            key: value[batch_times, env_index].to(device)
            for key, value in curr_obs.items()
            if torch.is_tensor(value)
        }
        q_values = model(
            forward_type=ForwardType.RFPO_Q,
            obs=obs_batch,
            actions=flat_actions[batch_start:batch_stop].to(device),
            action_mask=action_masks[batch_slots].to(device),
            tokenized_prompt=forward_inputs["tokenized_prompt"][
                batch_times, env_index
            ].to(device),
            tokenized_prompt_mask=forward_inputs["tokenized_prompt_mask"][
                batch_times, env_index
            ].to(device),
        )
        all_q_values.append(q_values.float().cpu())

    result = torch.cat(all_q_values, dim=0).numpy()
    expected_shape = (candidate_count * variant_count, 2)
    if result.shape != expected_shape:
        raise ValueError(
            f"RFPO critic returned {result.shape}, expected {expected_shape}"
        )
    return result.reshape(candidate_count, variant_count, 2)


def summarize_candidates(
    candidate_indices: Sequence[int],
    metadata: Sequence[ActionVariant],
    q_values: np.ndarray,
    comparison_tolerance: float,
) -> dict[str, Any]:
    """Rank the unchanged replay action against all angle variants."""
    expected_shape = (len(candidate_indices), len(metadata), 2)
    if q_values.shape != expected_shape:
        raise ValueError(f"q_values must have shape {expected_shape}")
    if not np.isfinite(q_values).all():
        raise ValueError("q_values must be finite")
    if comparison_tolerance < 0 or not math.isfinite(comparison_tolerance):
        raise ValueError("comparison_tolerance must be finite and non-negative")

    q_min = q_values.min(axis=-1)
    candidates = []
    for candidate_number, source_time_index in enumerate(candidate_indices):
        candidate_q = q_min[candidate_number]
        baseline_q = float(candidate_q[0])
        best_index = int(np.argmax(candidate_q))
        baseline_rank = 1 + int(
            np.sum(candidate_q > baseline_q + comparison_tolerance)
        )
        candidates.append(
            {
                "candidate_index": candidate_number,
                "source_time_index": int(source_time_index),
                "baseline_q_min": baseline_q,
                "baseline_rank": baseline_rank,
                "baseline_is_best": baseline_rank == 1,
                "best_variant": metadata[best_index].name,
                "best_angle_degrees": metadata[best_index].angle_degrees,
                "best_q_min": float(candidate_q[best_index]),
                "q_min_by_variant": {
                    item.name: float(candidate_q[item.index]) for item in metadata
                },
            }
        )

    baseline_best = np.asarray(
        [item["baseline_is_best"] for item in candidates], dtype=np.float64
    )
    baseline_ranks = np.asarray(
        [item["baseline_rank"] for item in candidates], dtype=np.float64
    )
    return {
        "candidate_count": len(candidates),
        "baseline_best_count": int(baseline_best.sum()),
        "baseline_best_fraction": float(baseline_best.mean()),
        "baseline_mean_rank": float(baseline_ranks.mean()),
        "comparison_tolerance": float(comparison_tolerance),
        "candidates": candidates,
    }


def _write_csv(
    path: Path,
    candidate_indices: Sequence[int],
    metadata: Sequence[ActionVariant],
    q_values: np.ndarray,
    summary: Mapping[str, Any],
) -> None:
    q_min = q_values.min(axis=-1)
    candidate_summaries = summary["candidates"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "candidate_index",
                "source_time_index",
                "action_variant",
                "rotation_angle_degrees",
                "normalized_action_offset",
                "q1",
                "q2",
                "q_min",
                "q_min_minus_a0",
                "is_best_for_candidate",
                "a0_rank",
            ]
        )
        for candidate_number, source_time_index in enumerate(candidate_indices):
            baseline_q = float(q_min[candidate_number, 0])
            best_q = float(q_min[candidate_number].max())
            tolerance = float(summary["comparison_tolerance"])
            for item in metadata:
                q_pair = q_values[candidate_number, item.index]
                current_min = float(q_min[candidate_number, item.index])
                writer.writerow(
                    [
                        candidate_number,
                        source_time_index,
                        item.name,
                        item.angle_degrees,
                        item.normalized_offset,
                        float(q_pair[0]),
                        float(q_pair[1]),
                        current_min,
                        current_min - baseline_q,
                        current_min >= best_q - tolerance,
                        candidate_summaries[candidate_number]["baseline_rank"],
                    ]
                )


def _render_plot(
    path: Path,
    candidate_indices: Sequence[int],
    metadata: Sequence[ActionVariant],
    q_values: np.ndarray,
    summary: Mapping[str, Any],
    *,
    source_name: str,
    env_index: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_min = q_values.min(axis=-1)
    source_steps = np.asarray(candidate_indices)
    figure_width = max(10.0, min(18.0, 8.0 + len(candidate_indices) * 0.08))
    fig, axis = plt.subplots(figsize=(figure_width, 5.6), constrained_layout=True)
    for item in metadata:
        is_baseline = item.index == 0
        axis.plot(
            source_steps,
            q_min[:, item.index],
            marker="o" if is_baseline else ".",
            markersize=4.5 if is_baseline else 3.0,
            linewidth=2.2 if is_baseline else 1.2,
            label=f"{item.name}: {item.angle_degrees:g}°",
            zorder=4 if is_baseline else 2,
        )

    axis.set(
        xlabel="Trajectory source step (candidate sampled every N steps)",
        ylabel="min(Q1, Q2)",
        title=(
            f"{source_name} | env={env_index} | "
            f"a0 best: {summary['baseline_best_count']}/"
            f"{summary['candidate_count']} "
            f"({summary['baseline_best_fraction']:.1%})"
        ),
    )
    if len(source_steps) == 1:
        axis.set_xlim(source_steps[0] - 1, source_steps[0] + 1)
    else:
        axis.set_xlim(source_steps[0], source_steps[-1])
    axis.set_ylim(_Q_Y_MIN, _Q_Y_MAX)
    axis.set_yticks(
        np.arange(_Q_Y_MIN, _Q_Y_MAX + _Q_Y_TICK_STEP / 2, _Q_Y_TICK_STEP)
    )
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=3, loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Sample one trajectory every N steps and compare RFPO Q for the "
            "unchanged action plus five angle-perturbed actions."
        )
    )
    parser.add_argument(
        "--ckpt",
        help="Checkpoint override; inferred from an episode manifest when omitted",
    )
    parser.add_argument(
        "--trajectory-file",
        "--trajectory_file",
        dest="trajectory_file",
        help="Direct trajectory input; omit when selecting an episode manifest entry",
    )
    parser.add_argument(
        "--episode-manifest",
        help="manifest.json produced by visualize_rfpo_q.py",
    )
    parser.add_argument(
        "--episode-ref",
        help=(
            "Episode id or curve/CSV/video path from visualize_rfpo_q.py; a full "
            "output path automatically locates its sibling manifest.json"
        ),
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        help="Zero-based episode index in --episode-manifest",
    )
    parser.add_argument(
        "--env-index",
        type=int,
        help="Environment stream; inferred from the selected episode by default",
    )
    parser.add_argument("--candidate-stride", type=int, default=5)
    parser.add_argument(
        "--candidate-start",
        type=int,
        help="Absolute trajectory start index; defaults to the episode start or 0",
    )
    parser.add_argument(
        "--candidate-stop",
        type=int,
        help="Exclusive absolute stop index; defaults to the episode or trajectory end",
    )
    parser.add_argument(
        "--rotation-angles",
        type=float,
        nargs="+",
        default=_DEFAULT_ROTATION_ANGLES,
        help="Angles for a0...aN in degrees; the first angle must be 0",
    )
    parser.add_argument(
        "--rotation-action-dim",
        type=int,
        default=5,
        help="Action component receiving the rotation offset (LIBERO rz: 5)",
    )
    parser.add_argument(
        "--action-step",
        type=int,
        default=-1,
        help="Action-chunk step to perturb; -1 perturbs the full chunk",
    )
    parser.add_argument(
        "--normalized-units-per-radian",
        type=float,
        help=(
            "Override angle conversion into normalized critic-action units; "
            "inferred from the model output transform by default"
        ),
    )
    parser.add_argument("--comparison-tolerance", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir")
    args = parser.parse_args(argv)

    episode_mode = (
        args.episode_manifest is not None
        or args.episode_ref is not None
        or args.episode_index is not None
    )
    if episode_mode:
        if args.trajectory_file is not None:
            parser.error(
                "--trajectory-file cannot be combined with episode manifest selection"
            )
        if args.episode_ref is not None and args.episode_index is not None:
            parser.error("Use either --episode-ref or --episode-index, not both")
        if args.episode_ref is None and args.episode_index is None:
            parser.error("Episode selection requires --episode-ref or --episode-index")
        if args.episode_index is not None and args.episode_manifest is None:
            parser.error("--episode-index requires --episode-manifest")
    elif args.ckpt is None or args.trajectory_file is None:
        parser.error(
            "Direct mode requires --ckpt and --trajectory-file; alternatively pass "
            "--episode-ref or --episode-manifest with --episode-index"
        )
    if args.env_index is not None and args.env_index < 0:
        parser.error("--env-index must be non-negative")
    if args.candidate_stride <= 0:
        parser.error("--candidate-stride must be positive")
    if args.candidate_start is not None and args.candidate_start < 0:
        parser.error("--candidate-start must be non-negative")
    if args.candidate_stop is not None and args.candidate_stop <= 0:
        parser.error("--candidate-stop must be positive")
    if (
        args.candidate_start is not None
        and args.candidate_stop is not None
        and args.candidate_stop <= args.candidate_start
    ):
        parser.error("--candidate-stop must be greater than --candidate-start")
    if args.rotation_action_dim < 0:
        parser.error("--rotation-action-dim must be non-negative")
    if args.action_step < -1:
        parser.error("--action-step must be -1 or non-negative")
    if not args.rotation_angles or args.rotation_angles[0] != 0.0:
        parser.error("--rotation-angles must start at 0 so a0 is unchanged")
    if any(not math.isfinite(angle) for angle in args.rotation_angles):
        parser.error("--rotation-angles must be finite")
    if len(set(args.rotation_angles)) != len(args.rotation_angles):
        parser.error("--rotation-angles must not contain duplicates")
    if args.normalized_units_per_radian is not None and (
        not math.isfinite(args.normalized_units_per_radian)
        or args.normalized_units_per_radian <= 0
    ):
        parser.error("--normalized-units-per-radian must be finite and positive")
    if (
        not math.isfinite(args.comparison_tolerance)
        or args.comparison_tolerance < 0
    ):
        parser.error("--comparison-tolerance must be finite and non-negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the trajectory-wide RFPO action-angle comparison."""
    args = parse_args(argv)
    episode_selection = None
    if (
        args.episode_manifest is not None
        or args.episode_ref is not None
        or args.episode_index is not None
    ):
        episode_selection = resolve_episode_selection(
            args.episode_manifest,
            episode_ref=args.episode_ref,
            episode_index=args.episode_index,
        )

    checkpoint_value = args.ckpt or (
        episode_selection.checkpoint if episode_selection is not None else None
    )
    if checkpoint_value is None:
        raise ValueError(
            "The selected manifest has no checkpoint; pass --ckpt explicitly."
        )
    weights_path = resolve_full_weights(checkpoint_value)
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
    elif args.ckpt is None and episode_selection is not None:
        config_path = (
            Path(episode_selection.training_config).expanduser().resolve()
            if episode_selection.training_config
            else None
        )
        if config_path is not None and not config_path.is_file():
            config_path = None
        if config_path is None:
            config_path = find_train_config_near_checkpoint(weights_path)
    else:
        config_path = find_train_config_near_checkpoint(weights_path)
    if config_path is None:
        raise FileNotFoundError(
            "Could not discover training config near checkpoint; pass --config."
        )

    trajectory_path = (
        episode_selection.source_file
        if episode_selection is not None
        else Path(args.trajectory_file).expanduser().resolve()
    )
    payload = _torch_load(trajectory_path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Trajectory is not a mapping: {trajectory_path}")

    device = _select_device(args.device)
    model_cfg = load_actor_model_config(config_path)
    model = build_rfpo_model(weights_path, model_cfg, device)
    time_size, env_count = trajectory_layout(payload)
    env_index = (
        args.env_index
        if args.env_index is not None
        else (
            episode_selection.env_index if episode_selection is not None else 0
        )
    )
    if (
        episode_selection is not None
        and args.env_index is not None
        and args.env_index != episode_selection.env_index
    ):
        raise ValueError(
            f"Selected episode belongs to env {episode_selection.env_index}; "
            f"received incompatible --env-index {args.env_index}."
        )
    if env_index >= env_count:
        raise IndexError(
            f"--env-index {env_index} exceeds environment count {env_count}"
        )
    candidate_start = (
        args.candidate_start
        if args.candidate_start is not None
        else (episode_selection.start if episode_selection is not None else 0)
    )
    candidate_stop = (
        args.candidate_stop
        if args.candidate_stop is not None
        else (episode_selection.stop if episode_selection is not None else time_size)
    )
    if episode_selection is not None and not (
        episode_selection.start
        <= candidate_start
        < candidate_stop
        <= episode_selection.stop
    ):
        raise ValueError(
            "Candidate bounds must remain inside the selected episode: "
            f"[{candidate_start}, {candidate_stop}) is outside "
            f"[{episode_selection.start}, {episode_selection.stop})"
        )
    candidate_indices = candidate_time_indices(
        time_size,
        candidate_stride=args.candidate_stride,
        candidate_start=candidate_start,
        candidate_stop=candidate_stop,
    )

    stored_actions = payload.get("actions")
    if not torch.is_tensor(stored_actions):
        stored_actions = payload["forward_inputs"]["action"]
    time_index_tensor = torch.as_tensor(candidate_indices, dtype=torch.long)
    base_actions = _reshape_actions(
        stored_actions[time_index_tensor, env_index],
        int(model.config.rfpo_action_chunk),
        int(model.config.rfpo_action_dim),
    )
    scale = (
        args.normalized_units_per_radian
        if args.normalized_units_per_radian is not None
        else infer_normalized_units_per_radian(model, args.rotation_action_dim)
    )
    variants, metadata = build_candidate_variants(
        base_actions,
        args.rotation_angles,
        rotation_action_dim=args.rotation_action_dim,
        normalized_units_per_radian=scale,
        action_step=args.action_step,
    )
    q_values = evaluate_candidate_variants(
        model,
        payload,
        candidate_indices=candidate_indices,
        env_index=env_index,
        variants=variants,
        device=device,
        batch_size=args.batch_size,
    )
    summary = summarize_candidates(
        candidate_indices,
        metadata,
        q_values,
        comparison_tolerance=args.comparison_tolerance,
    )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir(weights_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", trajectory_path.stem)
    episode_tag = (
        f"episode_{episode_selection.episode_index:04d}_"
        if episode_selection is not None
        else ""
    )
    output_stem = output_dir / (
        f"rotation_candidates_{episode_tag}{source_stem}_env_{env_index:03d}_"
        f"t_{candidate_start:04d}_{candidate_stop - 1:04d}_"
        f"stride_{args.candidate_stride}"
    )
    csv_path = output_stem.with_suffix(".csv")
    plot_path = output_stem.with_suffix(".png")
    summary_path = output_stem.with_suffix(".json")
    _write_csv(csv_path, candidate_indices, metadata, q_values, summary)
    _render_plot(
        plot_path,
        candidate_indices,
        metadata,
        q_values,
        summary,
        source_name=trajectory_path.name,
        env_index=env_index,
    )
    output_summary = {
        "checkpoint": str(weights_path),
        "training_config": str(config_path),
        "trajectory_file": str(trajectory_path),
        "env_index": env_index,
        "trajectory_time_size": time_size,
        "candidate_stride": args.candidate_stride,
        "candidate_start": candidate_start,
        "candidate_stop": candidate_stop,
        "candidate_source_indices": candidate_indices,
        "source_episode": (
            {
                "manifest": str(episode_selection.manifest_path),
                "episode_index": episode_selection.episode_index,
                "episode_id": episode_selection.episode_id,
                "start": episode_selection.start,
                "stop": episode_selection.stop,
            }
            if episode_selection is not None
            else None
        ),
        "rotation_action_dim": args.rotation_action_dim,
        "action_step": args.action_step,
        "normalized_units_per_radian": scale,
        "action_variants": [
            {
                "name": item.name,
                "angle_degrees": item.angle_degrees,
                "normalized_action_offset": item.normalized_offset,
            }
            for item in metadata
        ],
        "csv": csv_path.name,
        "plot": plot_path.name,
        **summary,
    }
    summary_path.write_text(
        json.dumps(output_summary, indent=2, allow_nan=False), encoding="utf-8"
    )

    if episode_selection is not None:
        print(
            f"Selected episode {episode_selection.episode_index}: "
            f"{trajectory_path.name}, env={env_index}, "
            f"t=[{episode_selection.start}, {episode_selection.stop})"
        )
    print(f"Candidates: {summary['candidate_count']}")
    print(
        "Unperturbed a0 has the highest min-Q for "
        f"{summary['baseline_best_count']}/{summary['candidate_count']} candidates "
        f"({summary['baseline_best_fraction']:.1%}); "
        f"mean rank={summary['baseline_mean_rank']:.3f}"
    )
    print(f"Normalized action units per radian: {scale:.8f}")
    print(f"Wrote plot, CSV, and summary to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
