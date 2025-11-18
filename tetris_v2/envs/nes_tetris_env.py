"""NES-style Tetris Gymnasium environment."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

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
        preview_pieces: int = 1,
        rotation_penalty: float = 0.0,
        line_clear_reward: Optional[Sequence[float]] = None,
        step_penalty: float = 0.0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.start_level = start_level
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self.rotation_penalty = max(0.0, float(rotation_penalty))
        self._line_clear_reward = self._build_line_clear_reward(line_clear_reward)
        self.step_penalty = max(0.0, float(step_penalty))
        self._frame_limit = (
            None if blitz_seconds is None else int(blitz_seconds * self.metadata["render_fps"])
        )
        self._frames = 0

        self.preview_pieces = max(1, int(preview_pieces))
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=7, shape=(20, 10), dtype=np.int8),
                "current": spaces.Box(low=-10_000, high=10_000, shape=(4,), dtype=np.int16),
                "next": spaces.Box(
                    low=0,
                    high=7,
                    shape=(self.preview_pieces,),
                    dtype=np.int8,
                ),
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
        self._preview_queue: List[int] = []
        self._refill_preview_queue()
        self._level = start_level
        self._lines = 0
        self._lines_until_level_up = utils.lines_to_next_level(self._level)
        self._score = 0
        self._steps = 0
        self._gravity_timer = 0
        self._top_out = False
        self._renderer: Optional[PygameBoardRenderer] = None
        self._consecutive_rotations = 0

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
        self._refill_preview_queue()
        if utils.collides(self._board, self._current):
            self._top_out = True
        self._frames = 0
        self._consecutive_rotations = 0
        return self._observe(), {}

    def step(self, action: Action):
        if self._top_out:
            return self._observe(), 0.0, True, False, {"top_out": True}

        assert self.action_space.contains(action), f"invalid action {action}"

        drop_reward = 0.0  # drop-based shaping in score units (scaled)
        extra_reward = 0.0  # other shaping (bonuses/penalties) in score units
        drop_score = 0  # integer points added to NES scoreboard for drops
        info: Dict[str, float] = {}
        self._steps += 1

        locked = False
        distance_soft = 0
        rotated_this_step = False

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
                rotated_this_step = True

        if rotated_this_step:
            self._consecutive_rotations += 1
            if self.rotation_penalty and self._consecutive_rotations > 1:
                extra_reward -= self.rotation_penalty / 100.0
                info["rotation_penalty"] = info.get("rotation_penalty", 0.0) - self.rotation_penalty
        else:
            self._consecutive_rotations = 0

        # dropping logic
        if action == self.HARD_DROP:
            distance = utils.hard_drop_distance(self._board, self._current)
            self._current = self._current.moved(d_row=distance)
            if distance > 0:
                drop_points = utils.soft_drop_points(distance) * 2
                drop_reward += drop_points / 100.0
                drop_score += drop_points
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
            drop_points = utils.soft_drop_points(distance_soft)
            drop_reward += drop_points / 100.0
            drop_score += drop_points

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

        if drop_score:
            self._score += drop_score
            info["drop_points"] = int(drop_score)
        if score_delta:
            info["score_delta"] = float(score_delta)
        line_bonus = self._line_clear_reward.get(lines_cleared, 0.0)
        if line_bonus:
            extra_reward += line_bonus / 100.0
            info["line_bonus"] = float(line_bonus)
        if lines_cleared:
            info["lines_cleared"] = int(lines_cleared)

        if self.step_penalty:
            extra_reward -= self.step_penalty / 100.0
            info["step_penalty"] = info.get("step_penalty", 0.0) - self.step_penalty

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

        reward = self._select_reward(score_delta, lines_cleared, drop_reward, extra_reward, drop_score)
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _spawn_next(self) -> None:
        next_piece = self._consume_preview()
        self._current = utils.spawn_piece(next_piece)

    def _refill_preview_queue(self) -> None:
        self._preview_queue = [utils.uniform_piece_id(self._rng) for _ in range(self.preview_pieces)]

    def _consume_preview(self) -> int:
        if not self._preview_queue:
            self._refill_preview_queue()
        next_piece = self._preview_queue.pop(0)
        self._preview_queue.append(utils.uniform_piece_id(self._rng))
        return next_piece

    def _observe(self) -> Dict[str, np.ndarray]:
        board = utils.board_visible(self._board)
        current = self._current
        if current is not None:
            for row, col in utils.iter_filled_cells(current):
                if row >= utils.HIDDEN_ROWS:
                    board[row - utils.HIDDEN_ROWS, col] = current.piece_id + 1
        preview = np.array(self._preview_queue, dtype=np.int8)
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
            "next": preview,
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
            queue_images = [
                utils.render_piece_preview(pid) for pid in self._preview_queue[: self.preview_pieces]
            ]
            self._renderer.draw(rgb, hud, queue_images=queue_images)
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

    def _select_reward(
        self,
        score_delta: int,
        lines_cleared: int,
        drop_reward: float,
        extra_reward: float,
        drop_score: int,
    ) -> float:
        base = (score_delta + drop_score) / 100.0
        if self.reward_mode == "score":
            return base + drop_reward + extra_reward
        if self.reward_mode == "lines":
            return float(lines_cleared)
        return float(drop_reward + extra_reward)

    def _build_line_clear_reward(self, values: Optional[Sequence[float]]) -> Dict[int, float]:
        rewards: Dict[int, float] = {i: 0.0 for i in range(1, 5)}
        if values is None:
            return rewards
        for idx, val in enumerate(values[:4], start=1):
            rewards[idx] = float(val)
        return rewards
