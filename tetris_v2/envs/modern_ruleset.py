"""Shared modern Tetris logic for single-player and versus environments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from . import utils

MOVE_NONE = 0
MOVE_LEFT = 1
MOVE_RIGHT = 2
ROTATE_CW = 3
ROTATE_CCW = 4
SOFT_DROP = 5
HARD_DROP = 6
HOLD = 7

LINE_SCORE = {1: 100, 2: 300, 3: 500, 4: 800}
TSPIN_SCORE = {
    "mini": {0: 0, 1: 100, 2: 200},
    "tspin": {0: 0, 1: 400, 2: 800, 3: 1200},
}
ATTACK_TABLE = {1: 0, 2: 1, 3: 2, 4: 4}
TSPIN_ATTACK = {
    "mini": {0: 0, 1: 1, 2: 2},
    "tspin": {0: 0, 1: 2, 2: 4, 3: 6},
}


@dataclass
class StepResult:
    observation: Dict[str, np.ndarray]
    terminated: bool
    score_delta: int
    lines_cleared: int
    attack: int
    info: Dict[str, int | float | bool | str]


class ModernRuleset:
    """Implements modern Tetris gravity, scoring, and garbage logic."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        *,
        queue_size: int = 5,
        lock_delay_frames: int = 30,
        line_clear_delay_frames: int = 20,
        max_steps: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.queue_size = queue_size
        self.lock_delay_frames = lock_delay_frames
        self.line_clear_delay_frames = line_clear_delay_frames
        self.max_steps = max_steps
        self._rng = rng or np.random.default_rng()
        self._bag_iter = utils.bag_sequence(self._rng)
        self._queue: Deque[int] = deque()
        self._pending_garbage: Deque[int] = deque()
        self._board = utils.create_board()
        self._current = utils.spawn_piece(next(self._bag_iter))
        self._hold_piece: Optional[int] = None
        self._hold_available = True
        self._steps = 0
        self._lines = 0
        self._score = 0
        self._combo = 0
        self._back_to_back = False
        self._top_out = False
        self._level = 0
        self._line_clear_timer = 0
        self._ground_frames = 0
        self._gravity_timer = 0
        self._last_action = "none"
        self._fill_queue()
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._bag_iter = utils.bag_sequence(self._rng)
        self._queue.clear()
        self._fill_queue()
        self._current = utils.spawn_piece(self._queue.popleft())
        self._queue.append(next(self._bag_iter))
        self._board = utils.create_board()
        self._pending_garbage.clear()
        self._hold_piece = None
        self._hold_available = True
        self._steps = 0
        self._lines = 0
        self._score = 0
        self._combo = 0
        self._back_to_back = False
        self._top_out = False
        self._level = 0
        self._line_clear_timer = 0
        self._ground_frames = 0
        self._gravity_timer = 0
        self._last_action = "none"
        return self._observe()

    # ------------------------------------------------------------------
    def step(self, action: int) -> StepResult:
        if self._top_out:
            return StepResult(self._observe(), True, 0, 0, 0, {"top_out": True})

        if self.max_steps and self._steps >= self.max_steps:
            return StepResult(self._observe(), True, 0, 0, 0, {"top_out": False})

        self._steps += 1
        info: Dict[str, int | float | bool | str] = {}

        if self._line_clear_timer > 0:
            self._line_clear_timer -= 1
            if self._line_clear_timer == 0:
                self._inject_pending_garbage()
                self._spawn_next()
            obs = self._observe()
            info["line_clear_delay"] = self._line_clear_timer
            return StepResult(obs, self._top_out, 0, 0, 0, info)

        locked = False
        self._last_action = "none"
        def _reset_ground_if_touching(did_move: bool) -> None:
            if did_move and self._is_touching_ground():
                self._ground_frames = 0

        if action == MOVE_LEFT:
            moved = self._attempt_move(-1)
            if moved:
                self._last_action = "move"
            _reset_ground_if_touching(moved)
        elif action == MOVE_RIGHT:
            moved = self._attempt_move(1)
            if moved:
                self._last_action = "move"
            _reset_ground_if_touching(moved)
        elif action == ROTATE_CW:
            rotated = self._attempt_rotate(1)
            if rotated:
                self._last_action = "rotate_cw"
            _reset_ground_if_touching(rotated)
        elif action == ROTATE_CCW:
            rotated = self._attempt_rotate(-1)
            if rotated:
                self._last_action = "rotate_ccw"
            _reset_ground_if_touching(rotated)
        elif action == HOLD:
            if self._attempt_hold():
                self._last_action = "hold"
                self._ground_frames = 0

        score_delta = 0
        touching_ground = False
        if action == HARD_DROP:
            distance = utils.hard_drop_distance(self._board, self._current)
            self._current = self._current.moved(d_row=distance)
            info["hard_drop_distance"] = distance
            score_delta += distance * 2
            touching_ground = True
            locked = True
        else:
            moved_down = False
            if action == SOFT_DROP:
                if self._try_fall(soft_drop=True):
                    moved_down = True
                    self._last_action = "soft_drop"
                    score_delta += utils.soft_drop_points(1)
                    self._gravity_timer = 0
                else:
                    touching_ground = True
            if not moved_down:
                self._gravity_timer += 1
                if self._gravity_timer >= self._gravity_interval():
                    if self._try_fall(soft_drop=False):
                        moved_down = True
                    else:
                        touching_ground = True
                    self._gravity_timer = 0
            if moved_down:
                self._ground_frames = 0

        if not touching_ground and self._is_touching_ground():
            touching_ground = True

        if touching_ground and not locked:
            self._ground_frames += 1
            if self._ground_frames >= self.lock_delay_frames:
                locked = True
        elif not touching_ground:
            self._ground_frames = 0

        lines_cleared = 0
        attack = 0
        t_spin_result: Optional[str] = None
        perfect_clear = False

        if locked:
            placement = utils.lock_piece(self._board, self._current)
            t_spin_result = utils.detect_t_spin(placement, self._current, self._last_action)
            self._board = placement
            cleared, lines_cleared, _ = utils.clear_lines(self._board)
            perfect_clear = utils.is_perfect_clear(cleared)
            self._board = cleared
            if lines_cleared:
                score_delta += LINE_SCORE.get(lines_cleared, 0) * (1 + self._level * 0.2)
            if t_spin_result:
                score_delta += TSPIN_SCORE[t_spin_result].get(lines_cleared, 0)
            if perfect_clear:
                score_delta += 1000
            is_b2b_event = bool(lines_cleared == 4 or (t_spin_result and lines_cleared))
            if is_b2b_event:
                if self._back_to_back:
                    score_delta = int(score_delta * 1.5)
                self._back_to_back = True
            else:
                self._back_to_back = False
            self._lines += lines_cleared
            self._level = self._lines // 10
            attack = self._compute_attack(lines_cleared, t_spin_result, perfect_clear, is_b2b_event)
            if lines_cleared:
                self._combo += 1
                info["combo"] = self._combo
                self._line_clear_timer = self.line_clear_delay_frames
                self._current = None
            else:
                self._combo = 0
                self._line_clear_timer = 0
                self._inject_pending_garbage()
                self._spawn_next()
            self._score += int(score_delta)
            self._hold_available = True
            self._ground_frames = 0
            self._gravity_timer = 0
            info["lines_cleared"] = lines_cleared
            if t_spin_result:
                info["t_spin"] = t_spin_result
            if perfect_clear:
                info["perfect_clear"] = True
            info["attack"] = attack
            if self._top_out:
                info["top_out"] = True
        obs = self._observe()
        info.setdefault("attack", attack)
        info.setdefault("lines_cleared", lines_cleared)
        info["score"] = int(self._score)
        info["level"] = int(self._level)
        info["pending_garbage"] = len(self._pending_garbage)
        return StepResult(obs, self._top_out, int(score_delta), lines_cleared, attack, info)

    # ------------------------------------------------------------------
    def queue_garbage(self, lines: int) -> None:
        for _ in range(max(0, lines)):
            hole = int(self._rng.integers(0, utils.BOARD_WIDTH))
            self._pending_garbage.append(hole)

    # ------------------------------------------------------------------
    def _try_fall(self, *, soft_drop: bool) -> bool:
        candidate = self._current.moved(d_row=1)
        if utils.collides(self._board, candidate):
            return False
        self._current = candidate
        return True

    def _attempt_move(self, dx: int) -> bool:
        candidate = self._current.moved(d_col=dx)
        if utils.collides(self._board, candidate):
            return False
        self._current = candidate
        return True

    def _attempt_rotate(self, delta: int) -> bool:
        piece_id = self._current.piece_id
        from_rotation = self._current.rotation % 4
        candidate = self._current.rotated(delta)
        to_rotation = candidate.rotation % 4
        if piece_id == utils.NAME_TO_ID["O"]:
            if utils.collides(self._board, candidate):
                return False
            self._current = candidate
            return True
        kicks = ModernTetrisKicks.kicks_for(piece_id, from_rotation, to_rotation)
        for dx, dy in kicks:
            shifted = candidate.moved(d_row=-dy, d_col=dx)
            if not utils.collides(self._board, shifted):
                self._current = shifted
                return True
        return False

    def _attempt_hold(self) -> bool:
        if not self._hold_available:
            return False
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
        return True

    def _compute_attack(
        self,
        lines: int,
        t_spin: Optional[str],
        perfect_clear: bool,
        is_b2b_event: bool,
    ) -> int:
        attack = 0
        if t_spin:
            attack = TSPIN_ATTACK[t_spin].get(lines, 0)
        else:
            attack = ATTACK_TABLE.get(lines, 0)
        if perfect_clear:
            attack += 10
        if self._combo > 1:
            attack += min(4, self._combo - 1)
        if self._back_to_back and is_b2b_event:
            attack += 1
        return attack

    def _gravity_interval(self) -> int:
        return max(1, utils.gravity_frames(min(self._level, 29)))

    def _is_touching_ground(self) -> bool:
        candidate = self._current.moved(d_row=1)
        return utils.collides(self._board, candidate)

    def _fill_queue(self) -> None:
        while len(self._queue) < self.queue_size:
            self._queue.append(next(self._bag_iter))

    def _spawn_next(self) -> None:
        if not self._queue:
            self._fill_queue()
        piece_id = self._queue.popleft()
        self._current = utils.spawn_piece(piece_id)
        self._queue.append(next(self._bag_iter))
        if utils.collides(self._board, self._current):
            self._top_out = True

    def _inject_pending_garbage(self) -> None:
        while self._pending_garbage:
            hole = self._pending_garbage.popleft()
            garbage_row = np.ones(utils.BOARD_WIDTH, dtype=np.int8)
            garbage_row[hole] = 0
            self._board = np.vstack([self._board[1:], garbage_row])

    def _observe(self) -> Dict[str, np.ndarray]:
        board = utils.board_visible(self._board)
        if self._current is not None:
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
                    -1 if self._current is None else self._current.piece_id,
                    0 if self._current is None else self._current.rotation,
                    -20 if self._current is None else self._current.row - utils.HIDDEN_ROWS,
                    -10 if self._current is None else self._current.col,
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
            "pending_garbage": np.int16(len(self._pending_garbage)),
        }

    def snapshot(self) -> Dict[str, np.ndarray]:
        return self._observe()

    def board_matrix(self) -> np.ndarray:
        board = self._board.copy()
        if self._current is not None:
            board = utils.lock_piece(board, self._current)
        return board


class ModernTetrisKicks:
    """SRS wall kick data."""

    KICKS_JLSTZ = {
        (0, 1): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
        (1, 0): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
        (1, 2): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
        (2, 1): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
        (2, 3): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
        (3, 2): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
        (3, 0): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
        (0, 3): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    }
    KICKS_I = {
        (0, 1): ((0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)),
        (1, 0): ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)),
        (1, 2): ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
        (2, 1): ((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)),
        (2, 3): ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)),
        (3, 2): ((0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)),
        (3, 0): ((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)),
        (0, 3): ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
    }

    @classmethod
    def kicks_for(cls, piece_id: int, from_rotation: int, to_rotation: int) -> Tuple[Tuple[int, int], ...]:
        if utils.ID_TO_NAME[piece_id] == "I":
            return cls.KICKS_I.get((from_rotation, to_rotation), ((0, 0),))
        return cls.KICKS_JLSTZ.get((from_rotation, to_rotation), ((0, 0),))
