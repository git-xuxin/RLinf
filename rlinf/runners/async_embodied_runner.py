# Copyright 2025 The RLinf Authors.
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

import asyncio
import time
from typing import TYPE_CHECKING, Optional

from omegaconf.dictconfig import DictConfig

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.metric_utils import (
    compute_env_metrics_per_env_worker,
    compute_evaluate_metrics,
)
from rlinf.utils.runner_utils import check_progress

if TYPE_CHECKING:
    from rlinf.data.replay_buffer import SACReplayBuffer
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )


class AsyncEmbodiedRunner(EmbodiedRunner):
    def __init__(
        self,
        cfg: DictConfig,
        actor: "AsyncEmbodiedSACFSDPPolicy",
        rollout: "AsyncMultiStepRolloutWorker",
        env: "AsyncEnvWorker",
        demo_buffer: Optional["SACReplayBuffer"] = None,
        critic=None,
        reward=None,
        run_timer=None,
    ):
        super().__init__(
            cfg, actor, rollout, env, demo_buffer, critic, reward, run_timer
        )

        # Data channels
        self.env_metric_channel = Channel.create("EnvMetric")
        self.replay_channel = Channel.create("ReplayBuffer")
        self.demo_channel = Channel.create("DemoBuffer")

    def get_env_metrics(self):
        try:
            result = self.env_metric_channel.get_nowait()
        except asyncio.QueueEmpty:
            return None

        rank_id = result.pop("rank_id")
        metric_keys = list(result.keys())
        if len(metric_keys) == 0:
            return None

        result_num = len(result[metric_keys[0]])
        for i in range(result_num):
            metric_time = result["time"][i] - self.start_time
            metric = {
                f"env/worker_{rank_id}/{key}": value[i].item() 
                for key, value in result.items() if key != "time"
            }
            self.metric_logger.log(metric, metric_time)

    def run(self):
        start_step = self.global_step
        self.update_rollout_weights()
        self.send_demo_buffer()

        self.start_time = int(10*time.time())
        env_handle: Handle = self.env.interact(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
            env_metric_channel=self.env_metric_channel,
        )
        rollout_handle: Handle = self.rollout.generate(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
            replay_channel=self.replay_channel,
            demo_channel=self.demo_channel,
        )
        self.actor.start_replay_buffer(self.replay_channel)
        self.actor.start_demo_buffer(self.demo_channel)

        train_step = start_step
        while train_step < self.max_steps:
            self.global_step = train_step
            log_time = int(10*time.time()) - self.start_time
            if (
                self.cfg.runner.val_check_interval > 0
                and train_step % self.cfg.runner.val_check_interval == 0
                and train_step > 0
            ):
                with self.timer("eval"):
                    self.update_rollout_weights()
                    eval_metrics = self.evaluate()
                    eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                    self.metric_logger.log(data=eval_metrics, step=log_time)

            with self.timer("actor_training"):
                actor_result = self.actor.run_training().wait()[0]
                has_trained, train_metrics = actor_result
            
            if has_trained:
                train_step += 1
                with self.timer("sync_weights"):
                    self.update_rollout_weights(enable_wait=True)

            training_metrics = {f"train/{k}": v for k, v in train_metrics.items()}
            self.metric_logger.log(training_metrics, log_time)

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            self.metric_logger.log(time_metrics, log_time)

            self.get_env_metrics()

            if not has_trained:
                time.sleep(1.0)
                continue
            
            _, save_model, _ = check_progress(
                self.global_step,
                self.max_steps,
                self.cfg.runner.val_check_interval,
                self.cfg.runner.save_interval,
                1.0,
                run_time_exceeded=False,
            )
            if save_model:
                self._save_checkpoint()

        self.env.stop().wait()
        self.rollout.stop().wait()
        env_handle.wait()
        rollout_handle.wait()

        self._save_checkpoint()
