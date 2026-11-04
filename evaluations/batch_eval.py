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

"""Run one LIBERO config repeatedly for each checkpoint, with CSV summaries.

Use run_rfpo_batch_eval.sh to configure the YAML, checkpoint list and repeat count.
Only checkpoint and log destination overrides are passed to run_eval.sh. Fresh
processes resample flow noise without changing the environment seed. TensorBoard
must remain enabled in the selected YAML. This does not provide seeded replay.

Each run writes eval.log and the original evaluation artifacts. runs.csv is
flushed after each run; summary.csv gets one row per completed checkpoint.
Failures stop the batch instead of silently averaging fewer than K results.
"""

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def read_result(run_dir: Path) -> tuple[float, int]:
    """Read the single final success rate and episode count from TensorBoard."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(
        str(run_dir / "tensorboard"), size_guidance={"scalars": 0}
    )
    events.Reload()
    values = {}
    for tag in ("eval/success_once", "eval/num_trajectories"):
        if tag not in events.Tags().get("scalars", []):
            raise ValueError(f"Missing {tag} in {run_dir}/tensorboard")
        entries = events.Scalars(tag)
        if len(entries) != 1 or entries[0].step != 0:
            raise ValueError(f"Expected one final step-0 value for {tag}: {run_dir}")
        values[tag] = float(entries[0].value)
    rate = values["eval/success_once"]
    count = values["eval/num_trajectories"]
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError(f"Invalid success rate: {rate}")
    if not math.isfinite(count) or count <= 0 or not count.is_integer():
        raise ValueError(f"Invalid episode count: {count}")
    return rate, int(count)


def run_eval(repo: Path, config: Path, checkpoint: Path, run_dir: Path) -> None:
    """Launch a fresh eval process and stream its output to console and disk."""
    command = [
        "bash",
        str(repo / "evaluations/run_eval.sh"),
        "libero",
        config.stem,
        # JSON strings are also Hydra quoted strings, including paths with spaces.
        "runner.ckpt_path=" + json.dumps(str(checkpoint)),
        "runner.logger.log_path=" + json.dumps(str(run_dir)),
    ]
    env = os.environ.copy()
    env.setdefault("LIBERO_TYPE", "standard")
    # run_eval.sh invokes python; use the same environment as this orchestrator.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    with (run_dir / "eval.log").open("w") as log:
        with subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        ) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                raise
        if returncode:
            raise RuntimeError(
                f"Evaluation exited with code {returncode}; see {run_dir}"
            )


def evaluate_batch(
    repo: Path,
    config: Path,
    checkpoints: list[Path],
    repeats: int,
    output_dir: Path,
) -> None:
    """Evaluate checkpoints serially, preserving completed results on failure."""
    with (
        (output_dir / "runs.csv").open("w", newline="") as runs_file,
        (output_dir / "summary.csv").open("w", newline="") as summary_file,
    ):
        runs = csv.writer(runs_file)
        summary = csv.writer(summary_file)
        runs.writerow(
            [
                "checkpoint",
                "repeat",
                "success_rate",
                "num_trajectories",
                "status",
                "error",
                "log_dir",
            ]
        )
        summary.writerow(
            ["checkpoint", "completed_repeats"]
            + [f"success_rate_{i}" for i in range(1, repeats + 1)]
            + ["mean_success_rate", "std_success_rate", "total_trajectories"]
        )
        runs_file.flush()
        summary_file.flush()
        for index, checkpoint in enumerate(checkpoints, 1):
            rates = []
            counts = []
            for repeat in range(1, repeats + 1):
                run_dir = (
                    output_dir / f"checkpoint_{index:03d}" / f"repeat_{repeat:03d}"
                )
                run_dir.mkdir(parents=True)
                print(
                    f"\nCheckpoint {index}/{len(checkpoints)}: {checkpoint}\n"
                    f"Repeat {repeat}/{repeats}; logs: {run_dir}",
                    flush=True,
                )
                try:
                    run_eval(repo, config, checkpoint, run_dir)
                    rate, count = read_result(run_dir)
                    if counts and count != counts[0]:
                        raise ValueError(
                            f"Episode count changed from {counts[0]} to {count}"
                        )
                except Exception as exc:
                    runs.writerow(
                        [
                            str(checkpoint),
                            repeat,
                            "",
                            "",
                            "failed",
                            str(exc),
                            str(run_dir),
                        ]
                    )
                    runs_file.flush()
                    raise
                rates.append(rate)
                counts.append(count)
                runs.writerow(
                    [str(checkpoint), repeat, rate, count, "ok", "", str(run_dir)]
                )
                runs_file.flush()
                print(f"Success rate: {rate:.6%} ({count} episodes)", flush=True)
            mean = statistics.mean(rates)
            std = statistics.stdev(rates) if repeats > 1 else 0.0
            summary.writerow(
                [str(checkpoint), repeats] + rates + [mean, std, sum(counts)]
            )
            summary_file.flush()
            print(
                f"\nCompleted checkpoint: {checkpoint}\n"
                f"Success rates: {', '.join(f'{rate:.6%}' for rate in rates)}\n"
                f"Mean: {mean:.6%}; sample std: {std:.6%}\n"
                f"Summary: {output_dir / 'summary.csv'}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="LIBERO YAML path or name")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    config = Path(args.config).expanduser()
    if config.suffix != ".yaml":
        config = config.with_suffix(".yaml")
    if not config.is_absolute():
        config = (
            repo / "evaluations/libero" / config
            if len(config.parts) == 1
            else repo / config
        )
    config = config.resolve()
    if not config.is_file() or config.parent != repo / "evaluations/libero":
        parser.error(f"Config must be a YAML under evaluations/libero: {config}")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    checkpoints = [Path(path).expanduser().resolve() for path in args.checkpoints]
    for path in checkpoints:
        if not path.is_file():
            parser.error(f"Checkpoint file does not exist: {path}")
    # Check dependencies before starting an expensive GPU evaluation.
    from tensorboard.backend.event_processing import event_accumulator  # noqa: F401

    batch_root = repo / "logs" / "batch_eval"
    batch_root.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix=config.stem + "-", dir=batch_root))
    (output_dir / "batch.json").write_text(
        json.dumps(
            {
                "config": str(config),
                "checkpoints": [str(p) for p in checkpoints],
                "repeats": args.repeats,
                "metric": "eval/success_once",
                "libero_type": os.environ.get("LIBERO_TYPE", "standard"),
            },
            indent=2,
        )
        + "\n"
    )
    (output_dir / config.name).write_text(config.read_text())
    print(f"Batch results: {output_dir}", flush=True)
    try:
        evaluate_batch(repo, config, checkpoints, args.repeats, output_dir)
    except Exception as exc:
        print(f"Batch stopped: {exc}\nPartial results: {output_dir}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
