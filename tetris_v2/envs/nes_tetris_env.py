"""NES-style Tetris Gymnasium environment (skeleton).

TODO:
- Implement 10x20 board, no hold, no ghost, no wall kicks
- Soft drop scoring, NES-like gravity, scoring table, level ups
- Uniform 7-piece RNG (not 7-bag) to mimic NES droughts
- `step`, `reset`, `render` (rgb_array + optional pygame human mode)
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class NesTetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=7, shape=(20,10), dtype=np.int8),
            "current": spaces.Box(low=0, high=10_000, shape=(4,), dtype=np.int16),
            "next": spaces.Discrete(7),
            "level": spaces.Box(low=0, high=1000, shape=(), dtype=np.int16),
            "lines": spaces.Box(low=0, high=10000, shape=(), dtype=np.int16),
            "score": spaces.Box(low=0, high=2**31-1, shape=(), dtype=np.int32),
        })
        self.action_space = spaces.Discrete(6)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        obs = {
            "board": np.zeros((20,10), dtype=np.int8),
            "current": np.array([0,0,-1,-1], dtype=np.int16),
            "next": np.int8(0),
            "level": np.int16(0),
            "lines": np.int16(0),
            "score": np.int32(0),
        }
        return obs, {}

    def step(self, action: int):
        obs, _ = self.reset()
        return obs, 0.0, True, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return np.zeros((100, 100, 3), dtype=np.uint8)

    def close(self):
        pass
