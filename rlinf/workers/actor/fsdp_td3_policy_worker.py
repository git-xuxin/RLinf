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


import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay import (
    PreloadReplayBufferDataset,
    ReplayBufferDataset,
    TrajectoryReplayBuffer,
    replay_buffer_collate_fn,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.utils import drq
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.utils import clear_memory, collect_param_names_need_sync
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


class EmbodiedTD3FSDPPolicy(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # TD3-specific initialization
        self.replay_buffer = None
        self.target_model = None
        self.demo_buffer = None
        self.update_step = 0
        self.enable_drq = bool(getattr(self.cfg.actor, "enable_drq", False))

    def init_worker(self):
        self.setup_model_and_optimizer(initialize_target=True)
        self.setup_td3_components()
        self.soft_update_target_model(tau=1.0)
        self.target_model.eval()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        if self.cfg.actor.get("compile_model", False):
            self.model = torch.compile(self.model, mode="default")
            self.target_model = torch.compile(self.target_model, mode="default")

    def setup_model_and_optimizer(self, initialize_target=False) -> None:
        """Setup model, lr_scheduler, optimizer and grad_scaler."""
        """Add initializing target model logic."""
        if not self.cfg.actor.fsdp_config.use_orig_params:
            raise ValueError(
                "TD3 requires use_orig_params=True for separate optimizers and targets."
            )
        module = self.model_provider_func()
        if initialize_target:
            target_module = self.model_provider_func()

        # Enable gradient checkpointing if configured
        if self.cfg.actor.model.get("gradient_checkpointing", False):
            self.logger.info("[FSDP] Enabling gradient checkpointing")
            module.gradient_checkpointing_enable()
            if initialize_target:
                target_module.gradient_checkpointing_enable()
        else:
            self.logger.info("[FSDP] Gradient checkpointing is disabled")

        # Record the original trainable parameter names before FSDP wrapping.
        # Persistent buffer names are also recorded for selective weight syncing.
        self.param_names_need_sync = collect_param_names_need_sync(module)

        # build model, optimizer, lr_scheduler, grad_scaler
        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        # When precision is null (e.g. Pi0), detect actual dtype from wrapped model
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype
        if initialize_target:
            self.target_model = self._strategy.wrap_model(
                model=target_module, device_mesh=self._device_mesh
            )
            self.target_model.requires_grad_(False)
            self.target_model_initialized = True

        param_filters = {"critic": ["encoders", "encoder", "q_head", "state_proj"]}
        filtered_optim_config = {"critic": self.cfg.actor.critic_optim}
        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters=param_filters,
            filtered_optim_config=filtered_optim_config,
        )
        self.optimizer = optimizers[0]
        self.qf_optimizer = optimizers[1]

        self.build_lr_schedulers()

        self.grad_scaler = self.build_grad_scaler(
            **self.cfg.actor.fsdp_config.grad_scaler
        )

    def build_lr_schedulers(self):
        self.lr_scheduler = self.build_lr_scheduler(
            self.optimizer, self.cfg.actor.optim
        )
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.qf_optimizer, self.cfg.actor.critic_optim
        )

    def setup_td3_components(self):
        """Initialize TD3-specific components"""
        # Initialize replay buffer
        seed = self.cfg.actor.get("seed", 1234)
        auto_save_path = self.cfg.algorithm.replay_buffer.get("auto_save_path", None)
        if auto_save_path is None:
            auto_save_path = os.path.join(
                self.cfg.runner.logger.log_path, f"replay_buffer/rank_{self._rank}"
            )
        else:
            auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
        self.replay_buffer = TrajectoryReplayBuffer(
            seed=seed,
            enable_cache=self.cfg.algorithm.replay_buffer.enable_cache,
            cache_size=self.cfg.algorithm.replay_buffer.cache_size,
            sample_window_size=self.cfg.algorithm.replay_buffer.sample_window_size,
            auto_save=self.cfg.algorithm.replay_buffer.get("auto_save", False),
            auto_save_path=auto_save_path,
            trajectory_format=self.cfg.algorithm.replay_buffer.get(
                "trajectory_format", "pt"
            ),
        )

        min_demo_buffer_size = 0
        if self.cfg.algorithm.get("demo_buffer", None) is not None:
            auto_save_path = self.cfg.algorithm.demo_buffer.get("auto_save_path", None)
            if auto_save_path is None:
                auto_save_path = os.path.join(
                    self.cfg.runner.logger.log_path, f"demo_buffer/rank_{self._rank}"
                )
            else:
                auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
            self.demo_buffer = TrajectoryReplayBuffer(
                seed=seed,
                enable_cache=self.cfg.algorithm.demo_buffer.enable_cache,
                cache_size=self.cfg.algorithm.demo_buffer.cache_size,
                sample_window_size=self.cfg.algorithm.demo_buffer.sample_window_size,
                auto_save=self.cfg.algorithm.demo_buffer.get("auto_save", False),
                auto_save_path=auto_save_path,
                trajectory_format="pt",
            )
            min_demo_buffer_size = self.cfg.algorithm.demo_buffer.min_buffer_size
            if self.cfg.algorithm.demo_buffer.get("load_path", None) is not None:
                self.demo_buffer.load_checkpoint(
                    self.cfg.algorithm.demo_buffer.load_path,
                    is_distributed=True,
                    local_rank=self._rank,
                    world_size=self._world_size,
                )

        if self.cfg.algorithm.replay_buffer.get("enable_preload", False):
            buffer_dataset_cls = PreloadReplayBufferDataset
        else:
            buffer_dataset_cls = ReplayBufferDataset
        self.buffer_dataset = buffer_dataset_cls(
            replay_buffer=self.replay_buffer,
            demo_buffer=self.demo_buffer,
            batch_size=self.cfg.actor.global_batch_size // self._world_size,
            min_replay_buffer_size=self.cfg.algorithm.replay_buffer.min_buffer_size,
            min_demo_buffer_size=min_demo_buffer_size,
            prefetch_size=self.cfg.algorithm.replay_buffer.get("prefetch_size", 10),
        )
        self.buffer_dataloader = DataLoader(
            self.buffer_dataset,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            collate_fn=replay_buffer_collate_fn,
        )
        self.buffer_dataloader_iter = iter(self.buffer_dataloader)

        self.critic_actor_ratio = self.cfg.algorithm.get("critic_actor_ratio", 1)
        self.critic_subsample_size = self.cfg.algorithm.get("critic_subsample_size", 2)
        self.critic_sample_generator = torch.Generator(self.device)
        self.critic_sample_generator.manual_seed(seed)

        self.target_update_freq = self.cfg.algorithm.get("target_update_freq", 1)
        self.target_actor_update_freq = self.cfg.algorithm.get(
            "target_actor_update_freq", 1
        )
        for name in (
            "critic_actor_ratio",
            "target_update_freq",
            "target_actor_update_freq",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        num_q_heads = self.cfg.actor.model.get("num_q_heads", 2)
        if not 1 <= self.critic_subsample_size <= num_q_heads:
            raise ValueError("critic_subsample_size must be between 1 and num_q_heads.")
        if not 0.0 <= self.cfg.algorithm.tau <= 1.0:
            raise ValueError("tau must be between 0 and 1.")
        for name in ("target_noise_std", "target_noise_clip"):
            if self.cfg.algorithm.get(name, 0.0) < 0:
                raise ValueError(f"{name} must be nonnegative.")

    def soft_update_target_model(self, tau: Optional[float] = None, component="all"):
        """Update the selected target component and copy its persistent buffers."""
        if tau is None:
            tau = self.cfg.algorithm.tau
        if component not in ("all", "actor", "critic"):
            raise ValueError(f"Unsupported target component: {component}")
        assert self.target_model_initialized

        def selected(name):
            is_critic = any(
                key in name for key in ("encoders", "encoder", "q_head", "state_proj")
            )
            return component == "all" or is_critic == (component == "critic")

        with torch.no_grad():
            for (name1, online_param), (name2, target_param) in zip(
                self.model.named_parameters(),
                self.target_model.named_parameters(),
                strict=True,
            ):
                assert name1 == name2
                if selected(name1):
                    target_param.data.mul_(1.0 - tau)
                    target_param.data.add_(online_param.data * tau)
            online_buffers = dict(self.model.named_buffers())
            for name, target_buffer in self.target_model.named_buffers():
                original_name = name.replace("_fsdp_wrapped_module.", "").replace(
                    "_orig_mod.", ""
                )
                if selected(name) and original_name in self.param_names_need_sync:
                    target_buffer.copy_(online_buffers[name])

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []

        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.replay_buffer.add_trajectories(recv_list)

        if self.demo_buffer is not None:
            intervene_traj_list = []
            for traj in recv_list:
                assert isinstance(traj, Trajectory)
                intervene_trajs = traj.extract_intervene_traj()
                if intervene_trajs is not None:
                    intervene_traj_list.extend(intervene_trajs)

            if len(intervene_traj_list) > 0:
                self.demo_buffer.add_trajectories(intervene_traj_list)

    @torch.no_grad()
    def compute_target(self, batch: dict) -> torch.Tensor:
        """Bootstrap discounted chunk rewards with smoothed target-actor actions."""
        next_obs = batch["next_obs"]
        next_state_actions = self.target_model(
            forward_type=ForwardType.TD3,
            obs=next_obs,
            noise_std=self.cfg.algorithm.get("target_noise_std", 0.2),
            noise_clip=self.cfg.algorithm.get("target_noise_clip", 0.5),
        )
        all_qf_next_target = self.target_model(
            forward_type=ForwardType.TD3_Q, obs=next_obs, actions=next_state_actions
        )
        sample_idx = torch.randperm(
            all_qf_next_target.shape[-1],
            generator=self.critic_sample_generator,
            device=self.device,
        )[: self.critic_subsample_size]
        qf_next_target = (
            all_qf_next_target.float()
            .index_select(dim=-1, index=sample_idx)
            .min(dim=-1, keepdim=True)
            .values
        )

        rewards = batch["rewards"].float()
        chunk_size = rewards.shape[-1]
        gamma = self.cfg.algorithm.gamma
        discounts = gamma ** torch.arange(
            chunk_size, device=rewards.device, dtype=torch.float32
        )
        chunk_reward = (rewards * discounts).sum(dim=-1, keepdim=True)
        terminated = batch["terminations"].bool().any(dim=-1, keepdim=True)
        return chunk_reward + gamma**chunk_size * qf_next_target.masked_fill(
            terminated, 0.0
        )

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        target_q_values = self.compute_target(batch)
        all_data_q_values = self.model(
            forward_type=ForwardType.TD3_Q,
            obs=batch["curr_obs"],
            actions=batch["actions"],
        )
        critic_loss = F.mse_loss(
            all_data_q_values.float(), target_q_values.expand_as(all_data_q_values)
        )
        return critic_loss, {
            "q_data": all_data_q_values.detach().float().mean().item(),
            "q_target": target_q_values.mean().item(),
        }

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        curr_obs = batch["curr_obs"]
        pi = self.model(forward_type=ForwardType.TD3, obs=curr_obs)
        all_qf_pi = self.model(forward_type=ForwardType.TD3_Q, obs=curr_obs, actions=pi)
        metrics = {
            f"q_value_{q_id}": all_qf_pi[..., q_id].detach().mean().item()
            for q_id in range(all_qf_pi.shape[-1])
        }
        qf_pi = all_qf_pi.float().mean(dim=-1, keepdim=True)
        metrics["q_pi"] = qf_pi.detach().mean().item()
        actor_loss = -qf_pi.mean()
        return actor_loss, metrics

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self, train_actor: bool = True):
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )

        with self.worker_timer("sample"):
            global_batch = next(self.buffer_dataloader_iter)

        train_micro_batch_list = split_dict_to_chunk(
            global_batch, global_batch_size_per_rank // self.cfg.actor.micro_batch_size
        )

        for i, batch in enumerate(train_micro_batch_list):
            batch = put_tensor_device(batch, device=self.device)
            if self.enable_drq:
                drq.apply_drq(batch["curr_obs"], pad=4)
                drq.apply_drq(batch["next_obs"], pad=4)
            train_micro_batch_list[i] = batch

        self.optimizer.zero_grad()
        self.qf_optimizer.zero_grad()
        gbs_critic_loss = []
        all_critic_metrics = {}
        for idx, batch in enumerate(train_micro_batch_list):
            with self.before_micro_batch(
                self.model, is_last_micro_batch=idx + 1 == self.gradient_accumulation
            ):
                critic_loss, critic_metrics = self.forward_critic(batch)
                critic_loss = critic_loss / self.gradient_accumulation
                critic_loss.backward()
            gbs_critic_loss.append(critic_loss.item() * self.gradient_accumulation)
            append_to_dict(all_critic_metrics, critic_metrics)
        all_critic_metrics = {
            f"critic/{key}": np.mean(value) for key, value in all_critic_metrics.items()
        }
        # FSDP may materialize zero gradients for unused actor parameters.
        self.optimizer.zero_grad()
        qf_grad_norm = self.model.clip_grad_norm_(
            max_norm=self.cfg.actor.critic_optim.clip_grad
        )

        self.qf_optimizer.step()
        self.qf_lr_scheduler.step()
        self.qf_optimizer.zero_grad()

        metrics_data = {
            "critic/loss": np.mean(gbs_critic_loss),
            "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": qf_grad_norm,
            **all_critic_metrics,
        }

        if self.update_step % self.critic_actor_ratio == 0 and train_actor:
            gbs_actor_loss = []
            all_actor_metrics = {}
            for idx, batch in enumerate(train_micro_batch_list):
                with self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=idx + 1 == self.gradient_accumulation,
                ):
                    actor_loss, actor_metrics = self.forward_actor(batch)
                    actor_loss = actor_loss / self.gradient_accumulation
                    actor_loss.backward()
                gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)
                append_to_dict(all_actor_metrics, actor_metrics)
            all_actor_metrics = {
                f"actor/{key}": np.mean(value)
                for key, value in all_actor_metrics.items()
            }
            # Keep dQ/da; discard critic gradients before clipping only the actor.
            self.qf_optimizer.zero_grad()
            actor_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            metrics_data.update(
                {
                    "actor/loss": np.mean(gbs_actor_loss),
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "actor/grad_norm": actor_grad_norm,
                    **all_actor_metrics,
                }
            )

        if self.update_step % self.target_update_freq == 0:
            self.soft_update_target_model(component="critic")
        if self.update_step % self.target_actor_update_freq == 0:
            self.soft_update_target_model(component="actor")
        return metrics_data

    def process_train_metrics(self, metrics):
        replay_buffer_stats = self.replay_buffer.get_stats()
        replay_buffer_stats = {
            f"replay_buffer/{key}": value for key, value in replay_buffer_stats.items()
        }
        append_to_dict(metrics, replay_buffer_stats)

        if self.demo_buffer is not None:
            demo_buffer_stats = self.demo_buffer.get_stats()
            demo_buffer_stats = {
                f"demo_buffer/{key}": value for key, value in demo_buffer_stats.items()
            }
            append_to_dict(metrics, demo_buffer_stats)
        # Average metrics across updates
        mean_metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 0:
                # Convert tensor values to CPU and detach before computing mean
                cpu_values = []
                for v in value:
                    if isinstance(v, torch.Tensor):
                        cpu_values.append(v.detach().cpu().item())
                    else:
                        cpu_values.append(v)
                mean_metric_dict[key] = np.mean(cpu_values)
            else:
                # Handle single values
                if isinstance(value, torch.Tensor):
                    mean_metric_dict[key] = value.detach().cpu().item()
                else:
                    mean_metric_dict[key] = value

        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        return mean_metric_dict

    def _all_ranks_buffer_ready(self, min_size: int) -> bool:
        """Check that every actor rank has at least ``min_size`` trajectories."""
        ready = (
            self.replay_buffer.is_ready(min_size)
            and self.replay_buffer.total_samples > 0
        )
        if self.demo_buffer is not None:
            ready = (
                ready
                and self.demo_buffer.is_ready(
                    self.cfg.algorithm.demo_buffer.min_buffer_size
                )
                and self.demo_buffer.total_samples > 0
            )
        local_ready = torch.tensor(
            [int(ready)],
            device=self.device,
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(local_ready, op=torch.distributed.ReduceOp.MIN)
        return local_ready.item() == 1

    @Worker.timer("run_training")
    def run_training(self):
        """Run TD3 updates when every actor rank has enough trajectories."""
        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        if not self._all_ranks_buffer_ready(min_buffer_size):
            self.log_on_first_rank(
                f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training"
            )
            return {}

        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        # All ranks must take the same actor-update branch for FSDP collectives.
        train_actor_steps = self.cfg.algorithm.get("train_actor_steps", 0)
        train_actor_steps = max(min_buffer_size, train_actor_steps)
        train_actor = self._all_ranks_buffer_ready(train_actor_steps)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}

        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            metrics_data = self.update_one_epoch(train_actor=train_actor)
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        mean_metric_dict = self.process_train_metrics(metrics)

        self.optimizer.zero_grad()
        self.qf_optimizer.zero_grad()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_optimizer()
            self.offload_param_and_grad()
        Worker.torch_platform.synchronize()
        torch.distributed.barrier()
        Worker.torch_platform.empty_cache()
        return mean_metric_dict

    @Worker.timer("actor/compute_adv")
    def compute_advantages_and_returns(self):
        """TD3 uses replay targets and does not compute advantages or returns."""
        return {}

    def save_checkpoint(self, save_base_path, step):
        restore_weight_offload = self.is_weight_offloaded
        restore_optimizer_offload = self.is_optimizer_offloaded
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

        # Save model
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            save_path=save_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        # save target model
        target_model_save_path = os.path.join(
            save_base_path, "td3_components/target_model"
        )
        os.makedirs(target_model_save_path, exist_ok=True)
        target_model_state_dict = self._strategy.get_model_state_dict(
            self.target_model, cpu_offload=False, full_state_dict=True
        )
        torch.save(
            target_model_state_dict,
            os.path.join(target_model_save_path, f"checkpoint_rank_{self._rank}.pt"),
        )

        # save replay buffer
        buffer_save_path = os.path.join(
            save_base_path, f"td3_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.save_checkpoint(buffer_save_path)
        worker_state = {
            "update_step": self.update_step,
            "critic_sample_rng": self.critic_sample_generator.get_state(),
            "replay_rng": self.replay_buffer.random_generator.get_state(),
        }
        if self.demo_buffer is not None:
            worker_state["demo_rng"] = self.demo_buffer.random_generator.get_state()
        torch.save(
            worker_state,
            os.path.join(save_base_path, f"td3_components/state_rank_{self._rank}.pt"),
        )
        if restore_optimizer_offload:
            self.offload_optimizer()
        if restore_weight_offload:
            self.offload_param_and_grad()

    def load_checkpoint(self, load_base_path):
        restore_weight_offload = self.is_weight_offloaded
        restore_optimizer_offload = self.is_optimizer_offloaded
        if restore_weight_offload:
            self.load_param_and_grad(self.device)
        if restore_optimizer_offload:
            self.load_optimizer(self.device)

        # load model
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            load_path=load_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        # load target model
        target_model_load_path = os.path.join(
            load_base_path, "td3_components/target_model"
        )
        target_model_state_dict = torch.load(
            os.path.join(target_model_load_path, f"checkpoint_rank_{self._rank}.pt"),
            map_location="cpu",
            weights_only=True,
        )
        self._strategy.load_model_with_state_dict(
            self.target_model,
            target_model_state_dict,
            cpu_offload=False,
            full_state_dict=True,
        )

        # load replay buffer
        buffer_load_path = os.path.join(
            load_base_path, f"td3_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.load_checkpoint(buffer_load_path)
        worker_state = torch.load(
            os.path.join(load_base_path, f"td3_components/state_rank_{self._rank}.pt"),
            map_location="cpu",
            weights_only=True,
        )
        self.update_step = worker_state["update_step"]
        self.critic_sample_generator.set_state(worker_state["critic_sample_rng"])
        self.replay_buffer.random_generator.set_state(worker_state["replay_rng"])
        if self.demo_buffer is not None:
            self.demo_buffer.random_generator.set_state(worker_state["demo_rng"])
        if restore_optimizer_offload:
            self.offload_optimizer()
        if restore_weight_offload:
            self.offload_param_and_grad()

    def offload_param_and_grad(self, offload_grad: bool = False) -> None:
        if not self.is_weight_offloaded:
            super().offload_param_and_grad(offload_grad)
            if self.target_model is not None:
                self._strategy.offload_param_and_grad(self.target_model, offload_grad)

    def load_param_and_grad(self, device_id: int, load_grad: bool = False) -> None:
        """Reload online and target networks."""
        if self.is_weight_offloaded:
            super().load_param_and_grad(device_id, load_grad)
            if self.target_model is not None:
                self._strategy.onload_param_and_grad(
                    self.target_model, device_id, load_grad
                )

    def offload_optimizer(self) -> None:
        if not self.is_optimizer_offloaded:
            super().offload_optimizer()
            self._strategy.offload_optimizer(self.qf_optimizer)

    def load_optimizer(self, device_id: int) -> None:
        if self.is_optimizer_offloaded:
            super().load_optimizer(device_id)
            self._strategy.onload_optimizer(self.qf_optimizer, device_id)
