"""Gymnasium wrapper over the C++ `tetris_cc_*` runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from tetris_v2.rl.actions import PLACEMENT_ACTION_DIM, POSE_ACTION_DIM, decode_action
from tetris_v2.rl.runtime import (
    BOARD_COLS,
    BOARD_ROWS,
    EMPTY_CELL_ID,
    EnvCtypes,
    find_library,
)


def board_potential(board: list[list[int]]) -> float:
    """Higher is better; based on compact stacks, few holes, and a flat surface."""

    cells = np.asarray(board, dtype=np.uint8)
    heights = np.zeros(BOARD_COLS, dtype=np.int32)
    holes = 0
    for column in range(BOARD_COLS):
        occupied = np.flatnonzero(cells[:, column])
        if occupied.size:
            top = int(occupied[0])
            heights[column] = BOARD_ROWS - top
            holes += int(np.count_nonzero(cells[top:, column] == 0))
    bumpiness = int(np.abs(np.diff(heights)).sum())
    return -(
        0.02 * float(heights.sum())
        + 0.12 * float(holes)
        + 0.01 * float(bumpiness)
    )


class CCTetrisEnv(gym.Env):
    """RL environment where every action locks one piece, optionally after Hold."""

    metadata = {"render_modes": ["ansi"], "render_fps": 60}

    def __init__(
        self,
        *,
        seed: int = 1,
        lib_path: Optional[Path] = None,
        include_hidden_rows: bool = False,
        max_steps: int = 4000,
        reward_mode: str = "shaped",
    ):
        super().__init__()
        self.seed_value = int(seed)
        self.include_hidden_rows = bool(include_hidden_rows)
        self.max_steps = int(max_steps)
        self.reward_mode = reward_mode
        if reward_mode not in {"shaped", "score"}:
            raise ValueError("reward_mode must be 'shaped' or 'score'")

        lib = find_library(lib_path)
        self.runtime = EnvCtypes(lib, self.seed_value)

        obs_size = self.runtime.observation_size(include_hidden_rows=self.include_hidden_rows)
        action_dim = self.runtime.decision_action_dim()
        if action_dim != PLACEMENT_ACTION_DIM:
            self.runtime.close()
            raise RuntimeError(
                f"RL action schema mismatch: Python={PLACEMENT_ACTION_DIM}, C++={action_dim}"
            )
        self.action_space = spaces.Discrete(action_dim)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self._steps = 0
        self._potential = 0.0
        self._mask_info: Dict[str, Any] = {}

    def _obs(self) -> np.ndarray:
        return self.runtime.observation(include_hidden_rows=self.include_hidden_rows)

    def _placement_mask_info(self) -> Dict[str, Any]:
        mask = self.runtime.decision_mask().astype(np.float32, copy=False)
        placement_count = int(np.count_nonzero(mask[:POSE_ACTION_DIM]))
        hold_placement_count = int(np.count_nonzero(mask[POSE_ACTION_DIM:]))

        return {
            "action_mask": mask,
            "legal_action_count": placement_count + hold_placement_count,
            "placement_count_raw": placement_count,
            "hold_placement_count": hold_placement_count,
            "placement_overflow": False,
            "hold_legal": hold_placement_count > 0,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.seed_value = (
            int(seed)
            if seed is not None
            else int(self.np_random.integers(0, 2**32, dtype=np.uint32))
        )

        self.runtime.reset(self.seed_value)
        self._steps = 0
        self._potential = board_potential(self.runtime.board())
        obs = self._obs()
        info = self.runtime.meta()
        mask_info = self._placement_mask_info()
        self._mask_info = mask_info
        info["seed"] = self.seed_value
        info["action_mask"] = mask_info["action_mask"]
        info["legal_action_count"] = mask_info["legal_action_count"]
        info["placement_count_raw"] = mask_info["placement_count_raw"]
        info["placement_overflow"] = mask_info["placement_overflow"]
        info["used_hold"] = False
        info["hold_placement_count"] = mask_info["hold_placement_count"]
        info["selected_placement_index"] = -1
        info["selected_is_hold"] = False
        info["placements"] = 0
        return obs, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid RL action index: {action}")

        mask_before = self._mask_info
        selected = int(action)
        if mask_before["action_mask"][selected] <= 0.5:
            raise ValueError(f"illegal RL action for the current state: {selected}")

        decision = decode_action(selected)
        outcome = self.runtime.apply_decision(selected)
        used_hold = bool(outcome["used_hold"])
        selected_placement_index = int(outcome["placement_index"])

        success = bool(outcome["success"])
        if not success:
            raise RuntimeError(f"C++ runtime rejected legal decision {selected}")
        raw_reward = float(outcome["reward"])

        self._steps += 1

        meta = self.runtime.meta()
        terminated = bool(meta["game_over"])
        truncated = self.max_steps > 0 and self._steps >= self.max_steps and not terminated

        next_potential = board_potential(self.runtime.board())
        shaping = next_potential - self._potential + 0.1
        if terminated:
            shaping -= 5.0
        self._potential = next_potential
        reward = raw_reward if self.reward_mode == "score" else raw_reward / 100.0 + shaping

        obs = self._obs()
        mask_after = self._placement_mask_info()
        self._mask_info = mask_after
        info = dict(meta)
        info["action_index"] = selected
        info["action_mask"] = mask_after["action_mask"]
        info["legal_action_count"] = mask_after["legal_action_count"]
        info["placement_count_raw"] = mask_after["placement_count_raw"]
        info["hold_placement_count"] = mask_after["hold_placement_count"]
        info["placement_overflow"] = mask_after["placement_overflow"]
        info["success"] = bool(success)
        info["used_hold"] = bool(used_hold)
        info["selected_placement_index"] = int(selected_placement_index)
        info["selected_is_hold"] = bool(decision["use_hold"])
        info["selected_x"] = int(decision["x"])
        info["selected_y"] = int(decision["y"])
        info["selected_rotation"] = int(decision["rotation"])
        info["placements"] = self._steps
        info["raw_reward"] = raw_reward
        info["reward_shaping"] = shaping
        info["board_potential"] = next_potential
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


__all__ = [
    "CCTetrisEnv",
    "PLACEMENT_ACTION_DIM",
    "board_potential",
]
