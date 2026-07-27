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

"""Model builder for OpenPI RFPO."""

import glob
import os
import pathlib

import torch
from omegaconf import DictConfig, OmegaConf


def get_model(cfg: DictConfig, torch_dtype=None):
    """Build a frozen pi0 model with RFPO actor and critic modules."""
    del torch_dtype
    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    from .openpi_rfpo_action_model import OpenPiRFPOActionModel, OpenPiRFPOConfig

    config_name = getattr(cfg.openpi, "config_name", None)
    data_kwargs = getattr(cfg, "openpi_data", None)
    actor_train_config = get_openpi_config(
        config_name, model_path=cfg.model_path, data_kwargs=data_kwargs
    )
    model_kwargs = dict(actor_train_config.model.__dict__)
    model_kwargs.update(OmegaConf.to_container(cfg.openpi, resolve=True))
    model_kwargs.update(OmegaConf.to_container(cfg.rfpo, resolve=True))
    model_kwargs["active_step_indices"] = tuple(model_kwargs["active_step_indices"])
    model_config = OpenPiRFPOConfig(**model_kwargs)

    checkpoint_dir = download.maybe_download(str(cfg.model_path))
    model = OpenPiRFPOActionModel(model_config)
    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )
    if os.path.exists(full_weights_path):
        model.load_state_dict(
            torch.load(full_weights_path, map_location="cpu"), strict=False
        )
    elif os.path.exists(actor_full_weights_path):
        model.load_state_dict(
            torch.load(actor_full_weights_path, map_location="cpu"), strict=False
        )
    else:
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        base_state_dict = {}
        for weight_path in weight_paths:
            base_state_dict.update(
                safetensors.torch.load_file(weight_path, device="cpu")
            )
        model.load_state_dict(base_state_dict, strict=False)

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, model_config
    )
    norm_stats_path = (
        data_kwargs.get("norm_stats_path") if data_kwargs is not None else None
    )
    if norm_stats_path is not None:
        norm_stats = data_config.norm_stats
        if norm_stats is None:
            norm_dir = pathlib.Path(norm_stats_path).expanduser()
            if norm_dir.is_file():
                norm_dir = norm_dir.parent
            norm_stats = _checkpoints.load_norm_stats(norm_dir.parent, norm_dir.name)
    else:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)

    repack_transforms = transforms.Group()
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )
    return model


__all__ = ["get_model"]
