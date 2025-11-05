"""Common wrappers for Tetris environments."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from gymnasium import ObservationWrapper, RewardWrapper, Wrapper
from gymnasium import spaces


class IdentityWrapper(Wrapper):
    """No-op wrapper retained for compatibility."""


class FloatBoardWrapper(ObservationWrapper):
    """Normalise the board channel to floats in [0, 1]."""

    def __init__(self, env, *, board_scale: float = 7.0):
        super().__init__(env)
        self.board_scale = board_scale
        board_space = self.observation_space["board"]
        self.observation_space = spaces.Dict(
            {
                **self.observation_space.spaces,
                "board": spaces.Box(
                    low=board_space.low.min() / board_scale,
                    high=board_space.high.max() / board_scale,
                    shape=board_space.shape,
                    dtype=np.float32,
                ),
            }
        )

    def observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        obs = dict(observation)
        obs["board"] = observation["board"].astype(np.float32) / self.board_scale
        return obs


class RewardScaleWrapper(RewardWrapper):
    """Scale rewards for stabilising value targets."""

    def __init__(self, env, scale: float = 1.0):
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, reward: float) -> float:
        return reward * self.scale
