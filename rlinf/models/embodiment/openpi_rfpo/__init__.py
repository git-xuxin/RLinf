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

import torch
from omegaconf import DictConfig

from rlinf.config import torch_dtype_from_precision

from .policy import RFPOActor, RFPOCritic, RFPOPolicy


def get_model(
    cfg: DictConfig,
    torch_dtype: torch.dtype | None = None,
    *,
    action_dim: int,
    action_feature_dim: int,
    condition_dim: int,
    state_dim: int,
) -> RFPOPolicy:
    """Build one side's small policy using dimensions supplied by its adapter.

    ``action_dim`` selects the controlled action width, normally
    ``adapter.env_action_shape[1]``. Feature/state widths come from the adapter's
    corresponding properties. No backbone object is retained by this factory.
    """
    model = RFPOPolicy(
        RFPOActor(
            cfg.actor,
            action_dim=action_dim,
            action_feature_dim=action_feature_dim,
            condition_dim=condition_dim,
            state_dim=state_dim,
        ),
        RFPOCritic(
            cfg.critic,
            action_dim=action_dim,
            condition_dim=condition_dim,
            state_dim=state_dim,
        ),
    )
    dtype = (
        torch_dtype
        if torch_dtype is not None
        else torch_dtype_from_precision(cfg.precision)
    )
    return model.to(dtype=dtype) if dtype is not None else model
