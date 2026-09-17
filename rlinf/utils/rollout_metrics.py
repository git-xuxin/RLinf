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

"""Sample-weighted scalar metrics for asynchronous rollout workers."""

from collections import defaultdict

import torch


class RolloutMetricAccumulator:
    """Accumulate per-sample metrics without retaining trajectories or graphs."""

    def __init__(self) -> None:
        self._sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}

    def add(self, metrics: dict[str, torch.Tensor]) -> None:
        """Add one batch of per-sample scalar values on the worker device."""
        for name, values in metrics.items():
            if values.ndim != 1:
                raise ValueError(f"Rollout metric {name} must have shape [batch].")
            if values.numel() == 0:
                continue
            total = values.detach().float().sum()
            if name in self._sums:
                self._sums[name] += total
                self._counts[name] += values.numel()
            else:
                self._sums[name] = total
                self._counts[name] = values.numel()

    def pop(self) -> dict[str, tuple[float, int]]:
        """Return serializable (sum, count) pairs and clear the accumulator."""
        if not self._sums:
            return {}
        names = list(self._sums)
        totals = torch.stack([self._sums[name] for name in names]).cpu().tolist()
        metrics = {
            name: (total, self._counts[name])
            for name, total in zip(names, totals, strict=True)
        }
        self._sums.clear()
        self._counts.clear()
        return metrics


def aggregate_rollout_metrics(results: list[dict]) -> tuple[dict, list[dict]]:
    """Combine sum/count messages across batches and workers, and per rank."""
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    rank_totals = defaultdict(lambda: defaultdict(float))
    rank_counts = defaultdict(lambda: defaultdict(int))
    for result in results:
        for name, (total, count) in result.get("rollout", {}).items():
            if count <= 0:
                continue
            totals[name] += total
            counts[name] += count
            rank = result.get("rank")
            if rank is not None:
                rank = int(rank)
                rank_totals[rank][name] += total
                rank_counts[rank][name] += count

    metrics = {name: total / counts[name] for name, total in totals.items()}
    ranked_metrics = [{} for _ in range(max(rank_totals, default=-1) + 1)]
    for rank, sums in rank_totals.items():
        ranked_metrics[rank] = {
            name: total / rank_counts[rank][name] for name, total in sums.items()
        }
    return metrics, ranked_metrics
