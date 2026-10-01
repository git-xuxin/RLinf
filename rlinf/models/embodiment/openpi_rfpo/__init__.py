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

"""RFPO small policy factory; the worker loads OpenPI separately."""

import copy
from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig, open_dict

from rlinf.config import torch_dtype_from_precision

from .policy import RFPOActor, RFPOCritic, RFPOPolicy
from .rfpo_sampler import RFPOSampler

if TYPE_CHECKING:
    from .backbone import RFPOBackboneAdapter


def build_model_config(cfg: DictConfig, adapter: "RFPOBackboneAdapter") -> DictConfig:
    """Copy the small-model config with input dimensions from this worker's pi."""
    model_config = copy.deepcopy(cfg)
    with open_dict(model_config):
        model_config.input_dims = {
            "action_dim": adapter.env_action_shape[1],
            "suffix_dim": adapter.suffix_dim,
            "condition_dim": adapter.condition_dim,
            "state_dim": adapter.state_dim,
        }
    return model_config


def get_model(
    cfg: DictConfig,
    torch_dtype: torch.dtype | None = None,
) -> RFPOPolicy:
    """Build the small policy from a config prepared by ``build_model_config``."""
    dims = cfg.input_dims
    model = RFPOPolicy(
        RFPOActor(
            cfg.actor,
            action_dim=dims.action_dim,
            suffix_dim=dims.suffix_dim,
            condition_dim=dims.condition_dim,
        ),
        RFPOCritic(
            cfg.critic,
            action_dim=dims.action_dim,
            condition_dim=dims.condition_dim,
            state_dim=dims.state_dim,
        ),
        RFPOSampler(**cfg.get("sampler", {})),
    )
    dtype = (
        torch_dtype
        if torch_dtype is not None
        else torch_dtype_from_precision(cfg.precision)
    )
    return model.to(dtype=dtype) if dtype is not None else model
