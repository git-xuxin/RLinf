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

from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


class BaseKeyboardRewardDoneWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.reward_modifier = 0
        self.listener = KeyboardListener()

    def _check_keypress(self) -> tuple[bool, bool, float]:
        raise NotImplementedError

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        updated_reward, updated_terminated = self.reward_terminated()
        return observation, updated_reward, updated_terminated, truncated, info

    def reward_terminated(
        self,
    ) -> tuple[float, bool]:
        last_intervened, terminated, keyboard_reward = self._check_keypress()
        return keyboard_reward, terminated


class KeyboardRewardDoneWrapper(BaseKeyboardRewardDoneWrapper):
    def _check_keypress(self) -> tuple[bool, bool, float]:
        last_intervened = False
        done = False
        reward = 0
        key = self.listener.get_key()
        print(f"Key pressed: {key}")
        if key not in ["a", "b", "c"]:
            return last_intervened, done, reward
        if key == "a":
            reward = -1
            done = True
        elif key == "b":
            reward = 0
        elif key == "c":
            reward = 1
            done = True
        return last_intervened, done, reward


class KeyboardRewardDoneMultiStageWrapper(BaseKeyboardRewardDoneWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.stage_rewards = [0, 1, 2]

    def reset(self, *, seed=None, options=None):
        self.reward_stage = 0
        return super().reset(seed=seed, options=options)

    def _check_keypress(self) -> tuple[bool, bool, float]:
        last_intervened = False
        done = False
        reward = 0
        key = self.listener.get_key()
        print(f"Key pressed: {key}")
        if key == "a":
            self.reward_stage = 0
        elif key == "b":
            self.reward_stage = 1
        elif key == "c":
            self.reward_stage = 2
            done = True

        reward = self.stage_rewards[self.reward_stage]
        if key == "q":
            reward = -1
            done = True
        return last_intervened, done, reward
    

class GripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"]["gripper_position"][0]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"]["gripper_position"][0]

        reward += self.penalty
        return observation, reward, terminated, truncated, info
