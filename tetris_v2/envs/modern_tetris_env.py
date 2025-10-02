"""Modern Tetris environment (skeleton)."""
from __future__ import annotations
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ModernTetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=8, shape=(20,10), dtype=np.int8),
        })
        self.action_space = spaces.Discrete(8)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        return {"board": np.zeros((20,10), dtype=np.int8)}, {}

    def step(self, action: int):
        obs, _ = self.reset()
        return obs, 0.0, True, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return np.zeros((100, 100, 3), dtype=np.uint8)

    def close(self):
        pass
