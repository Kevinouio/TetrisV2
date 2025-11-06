"""NES-style Tetris Gymnasium environment."""

from __future__ import annotations

from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from . import utils
from tetris_v2.rendering import PygameBoardRenderer

Action = int


class NesTetrisEnv(gym.Env):
    """Implements the classic NES ruleset (no bag, no hold, no kicks)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    MOVE_NONE = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    ROTATE_CW = 3
    SOFT_DROP = 4
    HARD_DROP = 5

    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        start_level: int = 0,
        max_steps: Optional[int] = None,
        reward_mode: str = "score",
        blitz_seconds: Optional[float] = 180.0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.start_level = start_level
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self._frame_limit = (
            None if blitz_seconds is None else int(blitz_seconds * self.metadata["render_fps"])
        )
        self._frames = 0

        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=7, shape=(20, 10), dtype=np.int8),
                "current": spaces.Box(low=-10_000, high=10_000, shape=(4,), dtype=np.int16),
                "next": spaces.Discrete(7),
                "level": spaces.Box(low=0, high=1000, shape=(), dtype=np.int16),
                "lines": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int16),
                "score": spaces.Box(low=0, high=2**31 - 1, shape=(), dtype=np.int32),
            }
        )
        self.action_space = spaces.Discrete(6)
        self._rng: np.random.Generator
        self._seed(seed)

        self._board = utils.create_board()
        self._current = utils.spawn_piece(utils.uniform_piece_id(self._rng))
        self._next_piece = utils.uniform_piece_id(self._rng)
        self._level = start_level
        self._lines = 0
        self._lines_until_level_up = utils.lines_to_next_level(self._level)
        self._score = 0
        self._steps = 0
        self._gravity_timer = 0
        self._top_out = False
        self._renderer: Optional[PygameBoardRenderer] = None

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def _seed(self, seed: Optional[int] = None) -> None:
        self._rng, _ = seeding.np_random(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._seed(seed)
        elif options and "seed" in options:
            self._seed(options["seed"])
        self._board = utils.create_board()
        self._level = int(options.get("start_level", self.start_level)) if options else self.start_level
        self._lines = 0
        self._lines_until_level_up = utils.lines_to_next_level(self._level)
        self._score = 0
        self._steps = 0
        self._gravity_timer = 0
        self._top_out = False
        self._current = utils.spawn_piece(utils.uniform_piece_id(self._rng))
        self._next_piece = utils.uniform_piece_id(self._rng)
        if utils.collides(self._board, self._current):
            self._top_out = True
        self._frames = 0
        return self._observe(), {}

    def step(self, action: Action):
        if self._top_out:
            return self._observe(), 0.0, True, False, {"top_out": True}

        assert self.action_space.contains(action), f"invalid action {action}"

        reward = 0.0
        info: Dict[str, float] = {}
        self._steps += 1

        locked = False
        distance_soft = 0

        # lateral movement
        if action == self.MOVE_LEFT:
            candidate = self._current.moved(d_col=-1)
            if not utils.collides(self._board, candidate):
                self._current = candidate
        elif action == self.MOVE_RIGHT:
            candidate = self._current.moved(d_col=1)
            if not utils.collides(self._board, candidate):
                self._current = candidate
        elif action == self.ROTATE_CW:
            candidate = self._current.rotated(delta=1)
            if not utils.collides(self._board, candidate):
                self._current = candidate

        # dropping logic
        if action == self.HARD_DROP:
            distance = utils.hard_drop_distance(self._board, self._current)
            self._current = self._current.moved(d_row=distance)
            if distance > 0:
                reward += utils.soft_drop_points(distance) * 2.0
            locked = True
        else:
            if action == self.SOFT_DROP:
                candidate = self._current.moved(d_row=1)
                if not utils.collides(self._board, candidate):
                    self._current = candidate
                    distance_soft += 1
                    self._gravity_timer = 0
                else:
                    locked = True
            if not locked:
                self._gravity_timer += 1
                if utils.tick_gravity(self._gravity_timer, self._level):
                    candidate = self._current.moved(d_row=1)
                    if utils.collides(self._board, candidate):
                        locked = True
                    else:
                        self._current = candidate
                    self._gravity_timer = 0

        if distance_soft:
            reward += utils.soft_drop_points(distance_soft)

        lines_cleared = 0
        score_delta = 0
        if locked:
            self._board = utils.lock_piece(self._board, self._current)
            self._board, lines_cleared, _ = utils.clear_lines(self._board)
            if lines_cleared:
                score_delta += utils.nes_score_for_lines(lines_cleared, self._level)
                self._lines += lines_cleared
                self._lines_until_level_up -= lines_cleared
                while self._lines_until_level_up <= 0:
                    self._level += 1
                    self._lines_until_level_up += utils.lines_to_next_level(self._level)
            self._score += score_delta
            self._spawn_next()
            self._gravity_timer = 0
            if utils.collides(self._board, self._current):
                self._top_out = True

        if score_delta:
            reward += score_delta / 100.0
            info["score_delta"] = float(score_delta)
        if lines_cleared:
            info["lines_cleared"] = int(lines_cleared)

        terminated = self._top_out
        truncated = bool(self.max_steps and self._steps >= self.max_steps)
        self._frames += 1
        if self._frame_limit is not None and self._frames >= self._frame_limit:
            truncated = True
            info["time_limit_reached"] = True

        obs = self._observe()
        info["level"] = int(self._level)
        info["score"] = int(self._score)
        info["lines"] = int(self._lines)
        info["top_out"] = terminated
        if self._frame_limit:
            info["time_remaining_frames"] = max(self._frame_limit - self._frames, 0)

        reward = self._select_reward(score_delta, lines_cleared, reward)
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _spawn_next(self) -> None:
        self._current = utils.spawn_piece(self._next_piece)
        self._next_piece = utils.uniform_piece_id(self._rng)

    def _observe(self) -> Dict[str, np.ndarray]:
        board = utils.board_visible(self._board)
        current = self._current
        if current is not None:
            for row, col in utils.iter_filled_cells(current):
                if row >= utils.HIDDEN_ROWS:
                    board[row - utils.HIDDEN_ROWS, col] = current.piece_id + 1
        return {
            "board": board.astype(np.int8),
            "current": np.array(
                [
                    current.piece_id,
                    current.rotation,
                    current.row - utils.HIDDEN_ROWS,
                    current.col,
                ],
                dtype=np.int16,
            ),
            "next": np.int8(self._next_piece),
            "level": np.int16(self._level),
            "lines": np.int16(self._lines),
            "score": np.int32(self._score),
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "rgb_array":
            return utils.render_board_rgb(self._board_with_current())
        if self.render_mode == "human":
            rgb = utils.render_board_rgb(self._board_with_current())
            hud = {
                "Score": self._score,
                "Lines": self._lines,
                "Level": self._level,
            }
            if self._frame_limit:
                remaining = max(self._frame_limit - self._frames, 0)
                seconds = remaining / self.metadata["render_fps"]
                hud["Time"] = f"{int(seconds // 60)}:{int(seconds % 60):02d}"
            if self._renderer is None:
                self._renderer = PygameBoardRenderer(title="NES Tetris", board_shape=rgb.shape[:2])
            self._renderer.draw(rgb, hud)
        return None

    def close(self):
        if self._renderer:
            self._renderer.close()
            self._renderer = None

    def _board_with_current(self) -> np.ndarray:
        board = self._board.copy()
        if not self._top_out and self._current is not None:
            board = utils.lock_piece(board, self._current)
        return board

    def _select_reward(self, score_delta: int, lines_cleared: int, default_reward: float) -> float:
        if self.reward_mode == "score":
            return score_delta / 100.0
        if self.reward_mode == "lines":
            return float(lines_cleared)
        return float(default_reward)
