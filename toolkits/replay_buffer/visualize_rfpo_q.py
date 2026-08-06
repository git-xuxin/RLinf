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

"""Render replay-buffer episodes with RFPO critic values.

Episodes enclosed by two ``done`` boundaries in one trajectory file are kept.
After removing RLinf's bootstrap done rows, consecutive boundaries ``left``
and ``right`` produce the observation slice ``(left, right]``. A file prefix
from observation zero through its first ``done`` is also kept when it is longer
than the configured minimum. Suffixes are discarded and files are never joined.

The plotted value is the replay-action value ``Q(s_t, a_t)``. The video and
static plot show both critic heads and ``min(Q1, Q2)``, which is the value used
by RFPO's actor objective.

Example: ``python toolkits/replay_buffer/visualize_rfpo_q.py --ckpt CKPT
--trajectory-dir TRAJ_DIR --num-episodes 10``

python toolkits/replay_buffer/visualize_rfpo_q.py \
--ckpt /mnt/public2/xuxin/RFPO/RLinf/logs/20260806-03:30:52-libero_object_async_rfpo_openpi/libero_object_async_rfpo_openpi/checkpoints/global_step_80/actor \
--trajectory-dir /mnt/public2/xuxin/RFPO/RLinf/logs/20260806-03:30:52-libero_object_async_rfpo_openpi/libero_object_async_rfpo_openpi/checkpoints/global_step_80/actor/rfpo_components/replay_buffer/rank_0 \
--num-episodes 10 \
--min-prefix-steps 20

"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


_WEIGHT_CANDIDATES = (
    Path("actor/model_state_dict/full_weights.pt"),
    Path("model_state_dict/full_weights.pt"),
    Path("full_weights.pt"),
)


@dataclass(frozen=True)
class EpisodeSpec:
    """An episode-like slice inside one trajectory environment stream."""

    source_file: Path
    env_index: int
    left_done_index: int
    right_done_index: int

    @property
    def start(self) -> int:
        """Inclusive observation index after the opening boundary."""
        return self.left_done_index + 1

    @property
    def stop(self) -> int:
        """Exclusive observation index after the closing boundary."""
        return self.right_done_index + 1

    @property
    def length(self) -> int:
        """Number of observations in the episode."""
        return self.stop - self.start

    @property
    def segment_type(self) -> str:
        """Describe whether the slice starts at a done or the file boundary."""
        return "file_prefix" if self.left_done_index < 0 else "between_dones"


def _torch_load(path: Path) -> Any:
    """Load tensor-only RLinf artifacts on CPU across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_full_weights(ckpt: str | Path) -> Path:
    """Resolve a checkpoint path to its consolidated ``full_weights.pt``."""
    path = Path(ckpt).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    checked = []
    for relative_path in _WEIGHT_CANDIDATES:
        candidate = path / relative_path
        checked.append(str(candidate))
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No full_weights.pt found under {path}. Checked: {checked}"
    )


def _has_actor_model_config(path: Path) -> bool:
    try:
        cfg = OmegaConf.load(path)
    except Exception:
        return False
    return OmegaConf.select(cfg, "actor.model") is not None


def find_train_config_near_checkpoint(ckpt: str | Path) -> Path | None:
    """Find the resolved training config saved by RLinf's metric logger."""
    path = Path(ckpt).expanduser().resolve()
    start = path if path.is_dir() else path.parent
    for directory in (start, *start.parents[:10]):
        for candidate in (
            directory / "config.yaml",
            directory / "tensorboard/config.yaml",
        ):
            if candidate.is_file() and _has_actor_model_config(candidate):
                return candidate
    return None


def load_actor_model_config(config_path: str | Path) -> DictConfig:
    """Load and detach ``actor.model`` from a resolved training config."""
    path = Path(config_path).expanduser().resolve()
    cfg = OmegaConf.load(path)
    model_cfg = OmegaConf.select(cfg, "actor.model")
    if model_cfg is None:
        raise KeyError(f"Could not find actor.model in training config: {path}")
    return OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))


def _unwrap_state_dict(raw: Any) -> dict[str, torch.Tensor]:
    """Extract and normalize a model state dict from common wrappers."""
    state = raw
    for key in ("state_dict", "model_state_dict", "model", "module"):
        if (
            isinstance(state, Mapping)
            and key in state
            and isinstance(state[key], Mapping)
            and not any(torch.is_tensor(value) for value in state.values())
        ):
            state = state[key]
            break
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) for key in state
    ):
        raise TypeError("Checkpoint does not contain a string-keyed model state dict.")

    normalized = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        parts = [part for part in key.split(".") if part != "_fsdp_wrapped_module"]
        if parts and parts[0] == "module":
            parts = parts[1:]
        normalized[".".join(parts)] = value
    return normalized


def build_rfpo_model(
    weights_path: Path, model_cfg: DictConfig, device: torch.device
) -> torch.nn.Module:
    """Build the configured RFPO model and load the trained online critic."""
    if str(model_cfg.get("model_type", "")) != "openpi_rfpo":
        raise ValueError(
            "Training config actor.model.model_type must be 'openpi_rfpo', got "
            f"{model_cfg.get('model_type')!r}."
        )

    # Import lazily so episode-discovery tests do not require OpenPI.
    from rlinf.models.embodiment.openpi_rfpo import get_model

    model = get_model(model_cfg)
    state_dict = _unwrap_state_dict(_torch_load(weights_path))
    critic_keys = {key for key in state_dict if key.startswith("online_critic.")}
    if not critic_keys:
        raise KeyError(
            f"Checkpoint {weights_path} contains no online_critic parameters."
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_critic = [
        key
        for key in incompatible.missing_keys
        if key.startswith("online_critic.")
    ]
    if missing_critic:
        raise RuntimeError(
            "Checkpoint is missing RFPO online critic parameters, including: "
            f"{missing_critic[:8]}"
        )
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


def _trajectory_files(directory: Path) -> list[Path]:
    def natural_key(path: Path) -> tuple[Any, ...]:
        return tuple(
            int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", str(path.relative_to(directory)))
        )

    return sorted(directory.rglob("trajectory_*.pt"), key=natural_key)


def _required_tensors(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    curr_obs = payload.get("curr_obs")
    forward_inputs = payload.get("forward_inputs")
    if not isinstance(curr_obs, Mapping):
        raise KeyError("trajectory has no curr_obs mapping")
    if not isinstance(forward_inputs, Mapping):
        raise KeyError("trajectory has no forward_inputs mapping")

    actions = payload.get("actions")
    if not torch.is_tensor(actions):
        actions = forward_inputs.get("action")
    required = {
        "dones": payload.get("dones"),
        "terminations": payload.get("terminations"),
        "truncations": payload.get("truncations"),
        "actions": actions,
        "curr_obs.main_images": curr_obs.get("main_images"),
        "curr_obs.states": curr_obs.get("states"),
        "forward_inputs.tokenized_prompt": forward_inputs.get("tokenized_prompt"),
        "forward_inputs.tokenized_prompt_mask": forward_inputs.get(
            "tokenized_prompt_mask"
        ),
    }
    missing = [name for name, value in required.items() if not torch.is_tensor(value)]
    if missing:
        raise KeyError(f"trajectory is missing required tensor fields: {missing}")
    return required


def trajectory_layout(payload: Mapping[str, Any]) -> tuple[int, int]:
    """Return the aligned ``(T, B)`` shared by critic input tensors."""
    tensors = _required_tensors(payload)
    bad_rank = [name for name, value in tensors.items() if value.ndim < 2]
    if bad_rank:
        raise ValueError(f"trajectory tensors must have [T, B, ...] axes: {bad_rank}")

    batch_sizes = {int(value.shape[1]) for value in tensors.values()}
    if len(batch_sizes) != 1:
        shapes = {name: tuple(value.shape) for name, value in tensors.items()}
        raise ValueError(f"trajectory batch axes do not match: {shapes}")
    aligned_time = min(int(value.shape[0]) for value in tensors.values())
    if aligned_time <= 0:
        raise ValueError("trajectory contains no aligned critic observations")
    return aligned_time, batch_sizes.pop()


def aligned_boundary_tensor(
    payload: Mapping[str, Any], time_size: int, key: str
) -> torch.Tensor:
    """Align a saved done-like tensor with replay observations."""
    values = payload.get(key)
    if not torch.is_tensor(values):
        raise KeyError(f"trajectory has no {key} tensor")
    if values.shape[0] == time_size:
        return values
    if values.shape[0] < time_size:
        raise ValueError(
            f"{key} time axis {values.shape[0]} is shorter than observations "
            f"{time_size}."
        )

    extra_rows = int(values.shape[0] - time_size)
    if time_size % extra_rows != 0:
        raise ValueError(
            f"Observation length {time_size} is not divisible by {extra_rows} "
            f"bootstrap {key} rows."
        )
    epoch_length = time_size // extra_rows
    return (
        values.reshape(extra_rows, epoch_length + 1, *values.shape[1:])[:, 1:]
        .reshape(time_size, *values.shape[1:])
        .contiguous()
    )


def aligned_dones(payload: Mapping[str, Any], time_size: int) -> torch.Tensor:
    """Remove per-rollout bootstrap done rows as replay-buffer sampling does."""
    return aligned_boundary_tensor(payload, time_size, "dones")


def closed_episode_specs(
    path: Path,
    payload: Mapping[str, Any],
    min_prefix_steps: int = 20,
) -> list[EpisodeSpec]:
    """Find eligible file prefixes and adjacent done-boundary pairs."""
    if min_prefix_steps < 0:
        raise ValueError("min_prefix_steps must be non-negative")
    time_size, batch_size = trajectory_layout(payload)
    dones = aligned_dones(payload, time_size).to(dtype=torch.bool)
    boundary_mask = dones.reshape(time_size, batch_size, -1).any(dim=-1)

    specs = []
    for env_index in range(batch_size):
        boundaries = torch.nonzero(boundary_mask[:, env_index], as_tuple=False)
        boundary_indices = boundaries.flatten().tolist()
        if boundary_indices:
            first_boundary = int(boundary_indices[0])
            prefix = EpisodeSpec(
                source_file=path,
                env_index=env_index,
                left_done_index=-1,
                right_done_index=first_boundary,
            )
            if prefix.length > min_prefix_steps:
                specs.append(prefix)
        for left, right in zip(boundary_indices, boundary_indices[1:]):
            specs.append(
                EpisodeSpec(
                    source_file=path,
                    env_index=env_index,
                    left_done_index=int(left),
                    right_done_index=int(right),
                )
            )
    return sorted(
        specs,
        key=lambda spec: (
            spec.segment_type != "file_prefix",
            spec.env_index,
            spec.right_done_index,
            spec.left_done_index,
        ),
    )


def discover_closed_episodes(
    trajectory_dir: str | Path,
    count: int,
    min_prefix_steps: int = 20,
    warn: Callable[[str], None] | None = None,
) -> tuple[list[EpisodeSpec], dict[str, int]]:
    """Scan trajectory files until ``count`` eligible segments are found."""
    if count <= 0:
        raise ValueError("count must be positive")
    if min_prefix_steps < 0:
        raise ValueError("min_prefix_steps must be non-negative")
    directory = Path(trajectory_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Trajectory directory does not exist: {directory}")
    files = _trajectory_files(directory)
    if not files:
        raise FileNotFoundError(f"No trajectory_*.pt files found under {directory}")

    selected: list[EpisodeSpec] = []
    stats = {"files_found": len(files), "files_scanned": 0, "files_skipped": 0}
    for path in files:
        stats["files_scanned"] += 1
        try:
            payload = _torch_load(path)
            if not isinstance(payload, Mapping):
                raise TypeError("top-level object is not a mapping")
            file_specs = closed_episode_specs(
                path, payload, min_prefix_steps=min_prefix_steps
            )
        except Exception as exc:
            stats["files_skipped"] += 1
            if warn is not None:
                warn(f"Skipping {path}: {exc}")
            continue
        if not file_specs:
            stats["files_skipped"] += 1
            if warn is not None:
                warn(f"Skipping {path}: no eligible done-delimited segment")
            continue
        selected.extend(file_specs[: count - len(selected)])
        if len(selected) >= count:
            break
    return selected, stats


def rfpo_action_mask(dones: torch.Tensor, action_chunk: int) -> torch.Tensor:
    """Match RFPO training's valid-action mask through the first done step."""
    flat_dones = dones.reshape(dones.shape[0], -1).to(dtype=torch.bool)
    if flat_dones.shape[1] != action_chunk:
        return torch.ones(
            (flat_dones.shape[0], action_chunk),
            dtype=torch.bool,
            device=dones.device,
        )
    done_prefix = flat_dones.cumsum(dim=1)
    return torch.cat(
        [
            torch.ones_like(done_prefix[:, :1], dtype=torch.bool),
            done_prefix[:, :-1] == 0,
        ],
        dim=1,
    )


def _slice_env_tensor(
    tensor: torch.Tensor, spec: EpisodeSpec, relative_slice: slice | None = None
) -> torch.Tensor:
    values = tensor[spec.start : spec.stop, spec.env_index]
    return values if relative_slice is None else values[relative_slice]


def _reshape_actions(
    actions: torch.Tensor, action_chunk: int, action_dim: int
) -> torch.Tensor:
    if actions.ndim == 2 and actions.shape[1] == action_chunk * action_dim:
        return actions.reshape(actions.shape[0], action_chunk, action_dim)
    if (
        actions.ndim == 3
        and actions.shape[1] >= action_chunk
        and actions.shape[2] >= action_dim
    ):
        return actions[:, :action_chunk, :action_dim]
    raise ValueError(
        "RFPO replay actions must be [N, chunk * dim] or [N, chunk, dim], "
        f"got {tuple(actions.shape)} for chunk={action_chunk}, dim={action_dim}."
    )


@torch.inference_mode()
def evaluate_episode_q(
    model: torch.nn.Module,
    payload: Mapping[str, Any],
    spec: EpisodeSpec,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Evaluate both online critic heads on replay observations and actions."""
    from rlinf.models.embodiment.base_policy import ForwardType

    curr_obs = payload["curr_obs"]
    forward_inputs = payload["forward_inputs"]
    actions = payload.get("actions")
    if not torch.is_tensor(actions):
        actions = forward_inputs["action"]
    action_chunk = int(model.config.rfpo_action_chunk)
    action_dim = int(model.config.rfpo_action_dim)
    time_size, _ = trajectory_layout(payload)
    done_tensor = aligned_dones(payload, time_size)
    all_q_values = []

    for batch_start in range(0, spec.length, batch_size):
        relative = slice(batch_start, min(batch_start + batch_size, spec.length))
        obs_batch = {
            key: _slice_env_tensor(value, spec, relative).to(device)
            for key, value in curr_obs.items()
            if torch.is_tensor(value)
        }
        action_batch = _reshape_actions(
            _slice_env_tensor(actions, spec, relative).to(device),
            action_chunk,
            action_dim,
        )
        done_batch = _slice_env_tensor(done_tensor, spec, relative).to(device)
        q_values = model(
            forward_type=ForwardType.RFPO_Q,
            obs=obs_batch,
            actions=action_batch,
            action_mask=rfpo_action_mask(done_batch, action_chunk),
            tokenized_prompt=_slice_env_tensor(
                forward_inputs["tokenized_prompt"], spec, relative
            ).to(device),
            tokenized_prompt_mask=_slice_env_tensor(
                forward_inputs["tokenized_prompt_mask"], spec, relative
            ).to(device),
        )
        all_q_values.append(q_values.float().cpu())
    return torch.cat(all_q_values, dim=0).numpy()


def _image_to_uint8(image: torch.Tensor | np.ndarray, camera_index: int) -> np.ndarray:
    array = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
    while array.ndim > 3:
        index = min(camera_index, array.shape[0] - 1)
        array = array[index]
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Camera frame must have 2 or 3 dimensions, got {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
            array = array * 255.0
    return np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0).clip(
        0, 255
    ).astype(np.uint8)


def _episode_scalar_series(
    payload: Mapping[str, Any], spec: EpisodeSpec, key: str
) -> np.ndarray:
    value = payload.get(key)
    if not torch.is_tensor(value):
        return np.full(spec.length, np.nan, dtype=np.float32)
    if key in {"dones", "terminations", "truncations"}:
        time_size, _ = trajectory_layout(payload)
        values = _slice_env_tensor(
            aligned_boundary_tensor(payload, time_size, key), spec
        )
        values = values.float().reshape(spec.length, -1)
        return values.bool().any(dim=1).float().numpy()
    # Bootstrap rows have reward=None and are not appended to the reward list,
    # so saved rewards already align with curr_obs/actions even though raw done
    # tensors retain the extra leading row.
    values = _slice_env_tensor(value, spec).float().reshape(spec.length, -1)
    if key == "rewards":
        time_size, _ = trajectory_layout(payload)
        dones = _slice_env_tensor(aligned_dones(payload, time_size), spec)
        mask = rfpo_action_mask(dones, values.shape[1]).float()
        if mask.shape == values.shape:
            values = values * mask
    return values.sum(dim=1).numpy()


def _write_episode_csv(
    path: Path,
    spec: EpisodeSpec,
    q_values: np.ndarray,
    rewards: np.ndarray,
    terminations: np.ndarray,
    truncations: np.ndarray,
    dones: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "source_time_index",
                "q1",
                "q2",
                "q_min",
                "reward",
                "termination",
                "truncation",
                "done",
            ]
        )
        for step, (q_pair, reward, termination, truncation, done) in enumerate(
            zip(
                q_values,
                rewards,
                terminations,
                truncations,
                dones,
                strict=True,
            )
        ):
            writer.writerow(
                [
                    step,
                    spec.start + step,
                    float(q_pair[0]),
                    float(q_pair[1]),
                    float(np.min(q_pair)),
                    float(reward),
                    bool(termination),
                    bool(truncation),
                    bool(done),
                ]
            )


def _curve_limits(q_values: np.ndarray) -> tuple[float, float]:
    finite = q_values[np.isfinite(q_values)]
    if finite.size == 0:
        return -1.0, 1.0
    lower, upper = float(finite.min()), float(finite.max())
    padding = max((upper - lower) * 0.12, 0.05)
    return lower - padding, upper + padding


def render_episode(
    output_stem: Path,
    frames: Sequence[np.ndarray],
    spec: EpisodeSpec,
    q_values: np.ndarray,
    rewards: np.ndarray,
    terminations: np.ndarray,
    truncations: np.ndarray,
    dones: np.ndarray,
    fps: float,
) -> tuple[Path, Path, Path]:
    """Write an MP4, a static Q curve, and per-step CSV values."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_min = q_values.min(axis=1)
    steps = np.arange(spec.length)
    y_lower, y_upper = _curve_limits(q_values)
    curve_path = output_stem.with_name(f"{output_stem.name}_q.png")
    csv_path = output_stem.with_name(f"{output_stem.name}_q.csv")
    video_path = output_stem.with_suffix(".mp4")

    fig, axis = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    axis.plot(steps, q_values[:, 0], color="#2563eb", label="Q1", linewidth=1.6)
    axis.plot(steps, q_values[:, 1], color="#ea580c", label="Q2", linewidth=1.6)
    axis.plot(steps, q_min, color="#15803d", label="min(Q1, Q2)", linewidth=2.2)
    axis.set(xlabel="Episode observation", ylabel="Q(s, replay action)")
    axis.set_xlim(0, max(spec.length - 1, 1))
    axis.set_ylim(y_lower, y_upper)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.savefig(curve_path, dpi=160)
    plt.close(fig)
    _write_episode_csv(
        csv_path, spec, q_values, rewards, terminations, truncations, dones
    )

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError(
            "Video export requires imageio with ffmpeg support: pip install 'imageio[ffmpeg]'"
        ) from exc

    fig, (image_axis, q_axis) = plt.subplots(
        1, 2, figsize=(12.0, 5.2), gridspec_kw={"width_ratios": (1.0, 1.35)}
    )
    image_artist = image_axis.imshow(frames[0])
    image_axis.axis("off")
    q_axis.plot(steps, q_values[:, 0], color="#2563eb", label="Q1", linewidth=1.5)
    q_axis.plot(steps, q_values[:, 1], color="#ea580c", label="Q2", linewidth=1.5)
    q_axis.plot(steps, q_min, color="#15803d", label="min(Q1, Q2)", linewidth=2.1)
    cursor = q_axis.axvline(0, color="#b91c1c", linewidth=1.5)
    point = q_axis.scatter([0], [q_min[0]], color="#b91c1c", s=36, zorder=5)
    q_axis.set(xlabel="Episode observation", ylabel="Q(s, replay action)")
    q_axis.set_xlim(0, max(spec.length - 1, 1))
    q_axis.set_ylim(y_lower, y_upper)
    q_axis.grid(True, alpha=0.25)
    q_axis.legend(loc="best")
    fig.tight_layout()

    with imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=2,
    ) as writer:
        for step, frame in enumerate(frames):
            image_artist.set_data(frame)
            image_axis.set_title(
                f"Observation {step + 1}/{spec.length}  |  "
                f"min Q {q_min[step]:.4f}  |  reward {rewards[step]:.3f}"
            )
            cursor.set_xdata([step, step])
            point.set_offsets(np.asarray([[step, q_min[step]]]))
            fig.canvas.draw()
            rendered = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            writer.append_data(rendered)
    plt.close(fig)
    return video_path, curve_path, csv_path


def _episode_frames(
    payload: Mapping[str, Any],
    spec: EpisodeSpec,
    camera_key: str,
    camera_index: int,
) -> list[np.ndarray]:
    curr_obs = payload["curr_obs"]
    camera = curr_obs.get(camera_key)
    if not torch.is_tensor(camera):
        available = [key for key, value in curr_obs.items() if torch.is_tensor(value)]
        raise KeyError(
            f"Camera key {camera_key!r} is unavailable. Tensor keys: {available}"
        )
    return [
        _image_to_uint8(frame, camera_index)
        for frame in _slice_env_tensor(camera, spec)
    ]


def _select_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RFPO Q(s, replay action) on done-delimited segments in saved "
            "trajectory files and render videos plus Q curves."
        )
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="full_weights.pt, model_state_dict dir, actor dir, or global-step dir",
    )
    parser.add_argument(
        "--trajectory-dir",
        "--trajectory_dir",
        required=True,
        dest="trajectory_dir",
        help="Directory recursively containing trajectory_*.pt files",
    )
    parser.add_argument(
        "--num-episodes",
        "--num_episodes",
        required=True,
        type=int,
        dest="num_episodes",
        help="Maximum number of done-delimited segments to produce",
    )
    parser.add_argument(
        "--min-prefix-steps",
        "--min_prefix_steps",
        type=int,
        default=20,
        dest="min_prefix_steps",
        help=(
            "Keep file-start-to-first-done prefixes only when longer than this "
            "many observation steps (default: 20)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        help="Output directory (default: <trajectory-dir>/rfpo_q_visualizations)",
    )
    parser.add_argument(
        "--config",
        help=(
            "Resolved RLinf training config.yaml. By default it is discovered "
            "from a tensorboard directory above the checkpoint."
        ),
    )
    parser.add_argument("--device", default="auto", help="Torch device (default: auto)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--camera-key", default="main_images")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args(argv)
    if args.num_episodes <= 0:
        parser.error("--num-episodes must be positive")
    if args.min_prefix_steps < 0:
        parser.error("--min-prefix-steps must be non-negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.camera_index < 0:
        parser.error("--camera-index must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run episode discovery, Q inference, and visualization export."""
    args = parse_args(argv)
    weights_path = resolve_full_weights(args.ckpt)
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else find_train_config_near_checkpoint(weights_path)
    )
    if config_path is None:
        raise FileNotFoundError(
            "Could not discover the resolved RLinf training config near the "
            "checkpoint. Pass --config /path/to/tensorboard/config.yaml."
        )

    episodes, scan_stats = discover_closed_episodes(
        args.trajectory_dir,
        args.num_episodes,
        min_prefix_steps=args.min_prefix_steps,
        warn=lambda message: print(message),
    )
    if not episodes:
        print(
            "No eligible segments were found: candidates need adjacent done "
            "boundaries or a sufficiently long file prefix ending in done."
        )
        return 2

    trajectory_dir = Path(args.trajectory_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else trajectory_dir / "rfpo_q_visualizations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _select_device(args.device)
    print(f"Using checkpoint: {weights_path}")
    print(f"Using training config: {config_path}")
    print(f"Found {len(episodes)} eligible segment(s); loading model on {device}.")
    model_cfg = load_actor_model_config(config_path)
    model = build_rfpo_model(weights_path, model_cfg, device)

    manifest_entries = []
    cached_path: Path | None = None
    cached_payload: Mapping[str, Any] | None = None
    for episode_index, spec in enumerate(episodes):
        if cached_path != spec.source_file:
            loaded = _torch_load(spec.source_file)
            if not isinstance(loaded, Mapping):
                raise TypeError(f"Trajectory is not a mapping: {spec.source_file}")
            cached_payload = loaded
            cached_path = spec.source_file
        assert cached_payload is not None

        q_values = evaluate_episode_q(
            model, cached_payload, spec, device, args.batch_size
        )
        if q_values.shape != (spec.length, 2):
            raise ValueError(
                f"RFPO critic returned {q_values.shape}, expected {(spec.length, 2)}"
            )
        rewards = _episode_scalar_series(cached_payload, spec, "rewards")
        terminations = _episode_scalar_series(
            cached_payload, spec, "terminations"
        ).astype(bool)
        truncations = _episode_scalar_series(
            cached_payload, spec, "truncations"
        ).astype(bool)
        dones = _episode_scalar_series(cached_payload, spec, "dones").astype(bool)
        frames = _episode_frames(
            cached_payload, spec, args.camera_key, args.camera_index
        )
        source_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.source_file.stem)
        output_stem = output_dir / (
            f"episode_{episode_index:04d}_{source_stem}_env_{spec.env_index:03d}_"
            f"t_{spec.start:04d}_{spec.stop - 1:04d}"
        )
        video_path, curve_path, csv_path = render_episode(
            output_stem,
            frames,
            spec,
            q_values,
            rewards,
            terminations,
            truncations,
            dones,
            args.fps,
        )
        terminated = bool(terminations[-1])
        truncated = bool(truncations[-1])
        if terminated and truncated:
            outcome = "termination+truncation"
        elif terminated:
            outcome = "termination"
        elif truncated:
            outcome = "truncation"
        else:
            outcome = "unknown"
        entry = {
            **asdict(spec),
            "source_file": str(spec.source_file),
            "start": spec.start,
            "stop": spec.stop,
            "length": spec.length,
            "segment_type": spec.segment_type,
            "reward_sum": float(rewards.sum()),
            "terminated": terminated,
            "truncated": truncated,
            "outcome": outcome,
            "q1_mean": float(q_values[:, 0].mean()),
            "q2_mean": float(q_values[:, 1].mean()),
            "q_min_mean": float(q_values.min(axis=1).mean()),
            "video": video_path.name,
            "curve": curve_path.name,
            "csv": csv_path.name,
        }
        manifest_entries.append(entry)
        print(
            f"[{episode_index + 1}/{len(episodes)}] {spec.source_file.name} "
            f"env={spec.env_index} type={spec.segment_type} "
            f"t=[{spec.start}, {spec.stop}) "
            f"outcome={outcome} reward={rewards.sum():.3f} -> {video_path.name}"
        )

    manifest = {
        "checkpoint": str(weights_path),
        "training_config": str(config_path),
        "trajectory_dir": str(trajectory_dir),
        "requested_episodes": args.num_episodes,
        "min_prefix_steps": args.min_prefix_steps,
        "produced_episodes": len(manifest_entries),
        "scan": scan_stats,
        "q_definition": "online min(Q1,Q2) on stored normalized replay action",
        "episodes": manifest_entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest_entries)} episode(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
