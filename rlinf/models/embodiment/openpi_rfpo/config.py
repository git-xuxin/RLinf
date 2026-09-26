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

"""Network and Gaussian settings for the RFPO small actor and critic."""

from dataclasses import dataclass


@dataclass(frozen=True)
class _RFPOTransformerConfig:
    hidden_size: int
    num_heads: int = 2
    mlp_ratio: float = 2.0

    def __post_init__(self) -> None:
        if (
            self.hidden_size <= 0
            or self.num_heads <= 0
            or self.hidden_size % self.num_heads != 0
        ):
            raise ValueError(
                "RFPO hidden_size must be positive and divisible by num_heads > 0."
            )

    @property
    def head_dim(self) -> int:
        """Split the hidden width evenly across all attention heads."""
        return self.hidden_size // self.num_heads

    @property
    def num_key_value_heads(self) -> int:
        """MHA uses the same number of query, key, and value heads."""
        return self.num_heads

    @property
    def mlp_hidden_size(self) -> int:
        """Feed-forward width derived from the configured expansion ratio."""
        return int(self.hidden_size * self.mlp_ratio)


@dataclass(frozen=True)
class RFPOActorConfig(_RFPOTransformerConfig):
    """DiT actor architecture; num_dit_blocks counts the residual DiT blocks."""

    hidden_size: int = 256
    num_dit_blocks: int = 3
    init_log_std: float = -6.0


@dataclass(frozen=True)
class RFPOCriticConfig(_RFPOTransformerConfig):
    """Gemma3-style critic architecture with multi-head attention."""

    hidden_size: int = 384
    num_hidden_layers: int = 3
    num_q_heads: int = 10
