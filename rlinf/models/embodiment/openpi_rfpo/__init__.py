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


def get_model(cfg: DictConfig, torch_dtype: torch.dtype | None = None) -> RFPOPolicy:
    """Build the actor/critic interface container from one side's model config."""
    model = RFPOPolicy(RFPOActor(cfg.actor), RFPOCritic(cfg.critic))
    dtype = (
        torch_dtype
        if torch_dtype is not None
        else torch_dtype_from_precision(cfg.precision)
    )
    return model.to(dtype=dtype) if dtype is not None else model
