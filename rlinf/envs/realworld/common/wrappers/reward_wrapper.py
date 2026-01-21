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

import gymnasium as gym

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


class KeyboardBinaryRewardWrapper(gym.RewardWrapper):
    """
    Modify the reward based on keyboard input.
    Pressing 'u' increases the reward by 1.
    Pressing 'd' decreases the reward by 1.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.reward_modifier = 0
        self.listener = KeyboardListener()

    def _check_keypress(self) -> None:
        key = self.listener.get_key()
        if key not in ["a", "b", "c"]:
            return False, 0
        if key == "a":
            reward = -1
        elif key == "b":
            reward = 0
        elif key == "c":
            reward = 1
        return True, reward

    def reward(self, reward) -> float:
        last_intervened, keyboard_reward = self._check_keypress()
        return keyboard_reward
