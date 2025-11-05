"""Modern Tetris environment with 7-bag RNG, hold, and SRS kicks."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from . import utils

Action = int


def _kick_table_jlstz() -> Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]]:
    # Offsets (x, y) with positive y meaning up.
    kicks = {
        (0, 1): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
        (1, 0): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
        (1, 2): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
        (2, 1): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
        (2, 3): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
        (3, 2): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
        (3, 0): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
        (0, 3): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    }
    return kicks


def _kick_table_i() -> Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]]:
    kicks = {
        (0, 1): ((0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)),
        (1, 0): ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)),
        (1, 2): ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
        (2, 1): ((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)),
        (2, 3): ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)),
        (3, 2): ((0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)),
        (3, 0): ((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)),
        (0, 3): ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
    }
    return kicks


KICKS_JLSTZ = _kick_table_jlstz()
KICKS_I = _kick_table_i()


class ModernTetrisEnv(gym.Env):
    """Core modern rules: 7-bag RNG, hold queue, hard drop, SRS rotation, combos."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    MOVE_NONE = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    ROTATE_CW = 3
    ROTATE_CCW = 4
    SOFT_DROP = 5
    HARD_DROP = 6
    HOLD = 7

    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        queue_size: int = 5,
        max_steps: Optional[int] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.queue_size = queue_size
        self.max_steps = max_steps

        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=8, shape=(20, 10), dtype=np.int8),
                "current": spaces.Box(low=-10_000, high=10_000, shape=(4,), dtype=np.int16),
                "queue": spaces.Box(low=0, high=7, shape=(queue_size,), dtype=np.int8),
                "hold": spaces.Box(low=-1, high=7, shape=(), dtype=np.int8),
                "combo": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int16),
                "back_to_back": spaces.Discrete(2),
                "level": spaces.Box(low=0, high=1000, shape=(), dtype=np.int16),
                "lines": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int16),
                "score": spaces.Box(low=0, high=2**31 - 1, shape=(), dtype=np.int32),
            }
        )
        self.action_space = spaces.Discrete(8)
        self._rng: np.random.Generator
        self._seed(seed)

        self._board = utils.create_board()
        self._queue: Deque[int] = deque()
        self._bag_iter = utils.bag_sequence(self._rng)
        self._current = utils.spawn_piece(next(self._bag_iter))
        self._hold_piece: Optional[int] = None
        self._hold_available = True
        self._lines = 0
        self._score = 0
        self._combo = 0
        self._back_to_back = False
        self._level = 0
        self._steps = 0
        self._top_out = False
        self._fill_queue()

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
        self._bag_iter = utils.bag_sequence(self._rng)
        self._queue.clear()
        self._fill_queue()
        self._current = utils.spawn_piece(self._queue.popleft())
        self._queue.append(next(self._bag_iter))
        self._hold_piece = None
        self._hold_available = True
        self._lines = 0
        self._score = 0
        self._combo = 0
        self._back_to_back = False
        self._level = 0
        self._steps = 0
        self._top_out = utils.collides(self._board, self._current)
        return self._observe(), {}

    def step(self, action: Action):
        if self._top_out:
            return self._observe(), 0.0, True, False, {"top_out": True}

        assert self.action_space.contains(action), f"invalid action {action}"
        self._steps += 1
        reward = 0.0
        info: Dict[str, float] = {}
        locked = False

        # movement
        if action == self.MOVE_LEFT:
            self._attempt_move(dx=-1)
        elif action == self.MOVE_RIGHT:
            self._attempt_move(dx=1)
        elif action == self.ROTATE_CW:
            self._attempt_rotate(delta=1)
        elif action == self.ROTATE_CCW:
            self._attempt_rotate(delta=-1)
        elif action == self.HOLD:
            self._attempt_hold()

        # dropping
        if action == self.HARD_DROP:
            distance = utils.hard_drop_distance(self._board, self._current)
            self._current = self._current.moved(d_row=distance)
            reward += distance * 2.0
            locked = True
        else:
            candidate = self._current.moved(d_row=1)
            if utils.collides(self._board, candidate):
                locked = True
            else:
                self._current = candidate
                if action == self.SOFT_DROP:
                    reward += utils.soft_drop_points(1)

        lines_cleared = 0
        score_delta = 0
        if locked:
            self._board = utils.lock_piece(self._board, self._current)
            self._hold_available = True
            self._board, lines_cleared, _ = utils.clear_lines(self._board)
            if lines_cleared:
                score_delta += self._modern_line_score(lines_cleared)
                self._lines += lines_cleared
                self._combo += 1
                score_delta += 50 * (self._combo - 1)
                if lines_cleared == 4:
                    if self._back_to_back:
                        score_delta += int(score_delta * 0.5)
                    self._back_to_back = True
                else:
                    self._back_to_back = False
            else:
                self._combo = 0
                self._back_to_back = False
            self._score += score_delta
            self._level = self._lines // 10
            self._spawn_next()
            if utils.collides(self._board, self._current):
                self._top_out = True

        if score_delta:
            reward += score_delta / 100.0
            info["score_delta"] = float(score_delta)
        if lines_cleared:
            info["lines_cleared"] = int(lines_cleared)
        if self._combo:
            info["combo"] = int(self._combo)
        info["back_to_back"] = int(self._back_to_back)

        terminated = self._top_out
        truncated = bool(self.max_steps and self._steps >= self.max_steps)
        obs = self._observe()
        info["score"] = int(self._score)
        info["lines"] = int(self._lines)
        info["level"] = int(self._level)
        info["top_out"] = bool(self._top_out)
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fill_queue(self) -> None:
        while len(self._queue) < self.queue_size:
            self._queue.append(next(self._bag_iter))

    def _spawn_next(self) -> None:
        if not self._queue:
            self._fill_queue()
        piece_id = self._queue.popleft()
        self._current = utils.spawn_piece(piece_id)
        self._queue.append(next(self._bag_iter))

    def _attempt_move(self, dx: int) -> None:
        candidate = self._current.moved(d_col=dx)
        if not utils.collides(self._board, candidate):
            self._current = candidate

    def _attempt_rotate(self, delta: int) -> None:
        piece_id = self._current.piece_id
        from_rotation = self._current.rotation % 4
        candidate = self._current.rotated(delta)
        to_rotation = candidate.rotation % 4
        if piece_id == utils.NAME_TO_ID["O"]:
            if not utils.collides(self._board, candidate):
                self._current = candidate
            return
        kicks = KICKS_I if piece_id == utils.NAME_TO_ID["I"] else KICKS_JLSTZ
        key = (from_rotation % 4, to_rotation % 4)
        for dx, dy in kicks.get(key, ((0, 0),)):
            shifted = candidate.moved(d_row=-dy, d_col=dx)
            if not utils.collides(self._board, shifted):
                self._current = shifted
                return

    def _attempt_hold(self) -> None:
        if not self._hold_available:
            return
        self._hold_available = False
        current_piece = self._current.piece_id
        if self._hold_piece is None:
            self._hold_piece = current_piece
            self._current = utils.spawn_piece(self._queue.popleft())
            self._queue.append(next(self._bag_iter))
        else:
            self._current = utils.spawn_piece(self._hold_piece)
            self._hold_piece = current_piece
        if utils.collides(self._board, self._current):
            self._top_out = True

    def _modern_line_score(self, lines: int) -> int:
        base = {1: 100, 2: 300, 3: 500, 4: 800}.get(lines, 0)
        return int(base * (1 + self._level * 0.2))

    def _observe(self) -> Dict[str, np.ndarray]:
        board = utils.board_visible(self._board)
        for row, col in utils.iter_filled_cells(self._current):
            if row >= utils.HIDDEN_ROWS:
                board[row - utils.HIDDEN_ROWS, col] = self._current.piece_id + 1
        queue = list(self._queue)[: self.queue_size]
        if len(queue) < self.queue_size:
            queue.extend([0] * (self.queue_size - len(queue)))
        hold_val = -1 if self._hold_piece is None else self._hold_piece
        return {
            "board": board.astype(np.int8),
            "current": np.array(
                [
                    self._current.piece_id,
                    self._current.rotation,
                    self._current.row - utils.HIDDEN_ROWS,
                    self._current.col,
                ],
                dtype=np.int16,
            ),
            "queue": np.array(queue, dtype=np.int8),
            "hold": np.int8(hold_val),
            "combo": np.int16(self._combo),
            "back_to_back": np.int8(int(self._back_to_back)),
            "level": np.int16(self._level),
            "lines": np.int16(self._lines),
            "score": np.int32(self._score),
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "rgb_array":
            board = utils.lock_piece(self._board.copy(), self._current)
            return utils.render_board_rgb(board)
        if self.render_mode == "human":
            board = self._observe()["board"]
            display = "\n".join("".join(" .:#%&@AB"[cell] for cell in row) for row in board[::-1])
            print(display)
        return None

    def close(self):
        pass
