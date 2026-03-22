"""Gymnasium wrapper over the VersionTwo C++ `tetris_cc_*` runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # pragma: no cover - exercised in dependency-missing environments
    gym = None
    spaces = None

from TetrisVersionTwo.rl.actions import RL_ACTION_MAP
from TetrisVersionTwo.rl.runtime import (
    BOARD_COLS,
    BOARD_ROWS,
    EMPTY_CELL_ID,
    EnvCtypes,
    find_library,
)


ACTION_MAP = RL_ACTION_MAP


_GymEnvBase = gym.Env if gym is not None else object


class CCTetrisEnv(_GymEnvBase):
    """RL-friendly env wrapper with fixed 8-action space and flat observation."""

    metadata = {"render_modes": ["ansi"], "render_fps": 60}

    def __init__(
        self,
        *,
        seed: int = 1,
        lib_path: Optional[Path] = None,
        include_hidden_rows: bool = False,
        max_steps: int = 4000,
    ):
        if gym is None or spaces is None:
            raise ModuleNotFoundError(
                "gymnasium is required for CCTetrisEnv. Install dependencies with: pip install -r requirements.txt"
            )
        super().__init__()
        self.seed_value = int(seed)
        self.include_hidden_rows = bool(include_hidden_rows)
        self.max_steps = int(max_steps)

        lib = find_library(lib_path)
        self.runtime = EnvCtypes(lib, self.seed_value)

        obs_size = self.runtime.observation_size(include_hidden_rows=self.include_hidden_rows)
        self.action_space = spaces.Discrete(len(ACTION_MAP))
        self.observation_space = spaces.Box(
            low=-1_000_000.0,
            high=1_000_000.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self._steps = 0

    @property
    def action_map(self):
        return ACTION_MAP

    def _obs(self) -> np.ndarray:
        return self.runtime.observation(include_hidden_rows=self.include_hidden_rows)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed_value = int(seed)

        self.runtime.reset(self.seed_value)
        self._steps = 0
        obs = self._obs()
        info = self.runtime.meta()
        info["seed"] = self.seed_value
        return obs, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise AssertionError(f"Invalid RL action index: {action}")

        mapped = ACTION_MAP[int(action)]
        out = self.runtime.step_action(mapped)
        self._steps += 1

        terminated = bool(out["game_over"])
        truncated = self.max_steps > 0 and self._steps >= self.max_steps and not terminated

        obs = self._obs()
        reward = float(out["reward"])
        info = dict(out["meta"])
        info["action_index"] = int(action)
        info["action_code"] = int(mapped)
        info["success"] = bool(out["success"])
        if truncated:
            info["time_limit"] = True
        return obs, reward, terminated, truncated, info

    def render(self):
        board = self.runtime.board_piece_ids(include_active=True)
        lines = []
        for r in range(BOARD_ROWS):
            row = []
            for c in range(BOARD_COLS):
                pid = board[r][c]
                row.append("." if pid == EMPTY_CELL_ID else "#")
            lines.append("".join(row))
        return "\n".join(lines)

    def close(self):
        self.runtime.close()


__all__ = ["CCTetrisEnv", "ACTION_MAP"]
