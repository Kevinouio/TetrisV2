"""Shared modern Tetris logic for single-player and versus environments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Sequence, Tuple

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
ROTATE_180 = 8

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
        line_clear_delay_frames: int = 0,
        soft_drop_factor: float = 6.0,
        max_steps: Optional[int] = None,
        das_frames: int = 10,  # Delayed Auto Shift
        arr_frames: int = 2,   # Auto Repeat Rate
        rng: Optional[np.random.Generator] = None,
        allowed_pieces: Optional[Sequence[int | str]] = None,
    ) -> None:
        self.queue_size = queue_size
        self.lock_delay_frames = lock_delay_frames
        self.line_clear_delay_frames = line_clear_delay_frames
        self.max_steps = max_steps
        self.soft_drop_factor = float(max(1.0, soft_drop_factor))
        self.das_frames = das_frames
        self.arr_frames = arr_frames
        self._rng = rng or np.random.default_rng()
        self._allowed_piece_ids: Optional[Tuple[int, ...]] = None
        if allowed_pieces is not None:
            ids: list[int] = []
            for entry in allowed_pieces:
                if isinstance(entry, str):
                    name = entry.strip().upper()
                    if name not in utils.NAME_TO_ID:
                        raise ValueError(f"Unknown piece '{entry}' in allowed_pieces")
                    ids.append(utils.NAME_TO_ID[name])
                else:
                    ids.append(int(entry))
            if not ids:
                raise ValueError("allowed_pieces must contain at least one piece name or id")
            self._allowed_piece_ids = tuple(ids)
        self._bag_iter = utils.bag_sequence(self._rng, self._allowed_piece_ids)
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
        self._level = 1  # Start at level 1 for modern
        self._line_clear_timer = 0
        self._ground_frames = 0
        self._gravity_progress = 0.0
        self._last_action = "none"
        self._rotation_streak = 0
        
        # Extended Placement
        self._manipulation_count = 0
        self._lock_reset_limit = 15
        
        # DAS/ARR
        self._das_timer = 0
        self._arr_timer = 0
        self._last_move_input = MOVE_NONE
        
        self._fill_queue()
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._bag_iter = utils.bag_sequence(self._rng, self._allowed_piece_ids)
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
        self._level = 1
        self._line_clear_timer = 0
        self._ground_frames = 0
        self._gravity_progress = 0.0
        self._last_action = "none"
        self._rotation_streak = 0
        self._manipulation_count = 0
        self._das_timer = 0
        self._arr_timer = 0
        self._last_move_input = MOVE_NONE
        return self._observe()

    # ------------------------------------------------------------------
    def step(self, action: int) -> StepResult:
        if self._top_out:
            return StepResult(self._observe(), True, 0, 0, 0, {"top_out": True})

        if self.max_steps and self._steps >= self.max_steps:
            # Treat max_steps as a timeout/truncation rather than a failure.
            return StepResult(self._observe(), False, 0, 0, 0, {"max_steps_reached": True})

        self._steps += 1
        info: Dict[str, int | float | bool | str] = {}
        rotation_performed = False

        if self._current is None:
            self._spawn_next()
            if self._current is None:
                obs = self._observe()
                info["top_out"] = self._top_out
                return StepResult(obs, self._top_out, 0, 0, 0, info)

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
        lock_reset = False

        def _reset_ground_if_touching(did_move: bool) -> None:
            nonlocal lock_reset
            if did_move:
                if self._is_touching_ground():
                    if self._manipulation_count < self._lock_reset_limit:
                        self._ground_frames = 0
                        self._manipulation_count += 1
                        lock_reset = True
                else:
                    # If we moved off ground, reset ground frames but NOT manipulation count
                    # (Standard: manipulation count only resets on step down)
                    self._ground_frames = 0

        # --- DAS/ARR Handling ---
        move_dir = 0
        if action == MOVE_LEFT:
            move_dir = -1
        elif action == MOVE_RIGHT:
            move_dir = 1
        
        if move_dir != 0:
            if action != self._last_move_input:
                # Initial Press
                self._das_timer = 0
                self._arr_timer = 0
                self._last_move_input = action
                moved = self._attempt_move(move_dir)
                if moved:
                    self._last_action = "move"
                _reset_ground_if_touching(moved)
            else:
                # Held
                self._das_timer += 1
                if self._das_timer >= self.das_frames:
                    self._arr_timer += 1
                    if self._arr_timer >= self.arr_frames:
                        # ARR Trigger
                        # If ARR is 0, we shift instantly to wall
                        if self.arr_frames == 0:
                            while self._attempt_move(move_dir):
                                self._last_action = "move"
                                _reset_ground_if_touching(True)
                        else:
                            moved = self._attempt_move(move_dir)
                            if moved:
                                self._last_action = "move"
                            _reset_ground_if_touching(moved)
                            self._arr_timer = 0
        else:
            # Released move input
            self._last_move_input = MOVE_NONE
            self._das_timer = 0
            self._arr_timer = 0

        if action == ROTATE_CW:
            rotated = self._attempt_rotate(1)
            if rotated:
                self._last_action = "rotate_cw"
                rotation_performed = True
            _reset_ground_if_touching(rotated)
        elif action == ROTATE_CCW:
            rotated = self._attempt_rotate(-1)
            if rotated:
                self._last_action = "rotate_ccw"
                rotation_performed = True
            _reset_ground_if_touching(rotated)
        elif action == HOLD:
            if self._attempt_hold():
                self._last_action = "hold"
                self._ground_frames = 0
                self._manipulation_count = 0 # New piece, reset
                lock_reset = True
        elif action == ROTATE_180:
            if self._attempt_rotate_180():
                self._last_action = "rotate_180"
                rotation_performed = True
                _reset_ground_if_touching(True)

        if rotation_performed:
            self._rotation_streak += 1
        else:
            self._rotation_streak = 0
        excess_rotations = max(0, self._rotation_streak - 1)
        if excess_rotations > 0:
            info["excess_rotations"] = excess_rotations

        score_delta = 0
        touching_ground = False
        gravity_per_frame = self._gravity_per_frame()
        moved_down = False
        rows_dropped = 0
        
        if action == HARD_DROP:
            distance = utils.hard_drop_distance(self._board, self._current)
            self._current = self._current.moved(d_row=distance)
            info["hard_drop_distance"] = distance
            touching_ground = True
            locked = True
            self._gravity_progress = 0.0
            self._manipulation_count = 0
        else:
            # Apply gravity (G per frame)
            # If soft dropping, we multiply the G force
            # Note: Standard soft drop is usually 20G or similar fixed speed, 
            # but factor * gravity is a reasonable approximation if gravity is low.
            # However, if gravity is very low, soft drop should still be fast.
            # Let's use max(gravity * factor, 1/2G) or similar?
            # For now, keeping existing factor logic but applied to G.
            
            # Actually, standard soft drop is 1G (1 row per frame) or faster.
            # If we use factor, we just add to progress.
            
            # Calculate rows to drop this frame
            rows_to_drop = 0.0
            if action == SOFT_DROP:
                rows_to_drop = gravity_per_frame * self.soft_drop_factor
                rows_to_drop = max(rows_to_drop, 0.5)
                self._last_action = "soft_drop"
            else:
                rows_to_drop = gravity_per_frame
            
            self._gravity_progress += rows_to_drop
            
            while self._gravity_progress >= 1.0:
                candidate = self._current.moved(d_row=1)
                if utils.collides(self._board, candidate):
                    touching_ground = True
                    self._gravity_progress = 0.0 # Stop accumulating if we hit ground
                    break
                self._current = candidate
                moved_down = True
                rows_dropped += 1
                lock_reset = True
                self._gravity_progress -= 1.0
                # Reset manipulation count on step down
                self._manipulation_count = 0
            
            if moved_down:
                self._ground_frames = 0

        if not touching_ground and self._is_touching_ground():
            touching_ground = True

        if touching_ground and not locked:
            if not lock_reset:
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
            self._level = 1 + self._lines // 10 # Level starts at 1
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
            self._gravity_progress = 0.0
            self._manipulation_count = 0
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
            # O piece doesn't rotate, but we still check collision just in case
            # (though it should be safe if it didn't move)
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

    def _attempt_rotate_180(self) -> bool:
        piece_id = self._current.piece_id
        if utils.ID_TO_NAME[piece_id] == "O":
            return True
        rotations = len(utils.PIECE_ROTATIONS[piece_id])
        from_rotation = self._current.rotation % rotations
        candidate = self._current.rotated(delta=2)
        to_rotation = candidate.rotation % rotations
        kicks = ModernTetrisKicks.kicks_180_for(piece_id, from_rotation, to_rotation)
        for dx, dy in kicks:
            shifted = candidate.moved(d_row=-dy, d_col=dx)
            if not utils.collides(self._board, shifted):
                self._current = shifted
                return True
        return False

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

    def _gravity_per_frame(self) -> float:
        # Modern Guideline Gravity
        # Seconds per row = (0.8 - ((Level-1)*0.007))^(Level-1)
        # Frames per row = Seconds * 60
        # Rows per frame = 1 / Frames
        
        lvl = max(1, self._level)
        if lvl >= 20:
            return 20.0 # 20G (instant drop)
            
        seconds_per_row = pow(0.8 - ((lvl - 1) * 0.007), lvl - 1)
        frames_per_row = seconds_per_row * 60.0
        if frames_per_row <= 0.0001:
             return 20.0
        return 1.0 / frames_per_row

    def _is_touching_ground(self) -> bool:
        candidate = self._current.moved(d_row=1)
        return utils.collides(self._board, candidate)

    def _fill_queue(self) -> None:
        while len(self._queue) < self.queue_size:
            self._queue.append(next(self._bag_iter))

    def _spawn_next(self) -> None:
        if self._pending_garbage:
            self._inject_pending_garbage()
        if not self._queue:
            self._fill_queue()
        piece_id = self._queue.popleft()
        self._current = utils.spawn_piece(piece_id)
        self._queue.append(next(self._bag_iter))
        self._manipulation_count = 0
        self._ground_frames = 0
        self._rotation_streak = 0
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

    def board_matrix(self, *, include_current: bool = True) -> np.ndarray:
        board = self._board.copy()
        if include_current and self._current is not None:
            board = utils.lock_piece(board, self._current)
        return board

    def current_piece(self) -> Optional[utils.PieceState]:
        return self._current

    def ghost_piece(self) -> Optional[utils.PieceState]:
        if self._current is None:
            return None
        return utils.project_lock_position(self._board, self._current)


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

    KICKS_180_JLSTZ = {
        0: ((0, 0), (-1, 0), (1, 0), (0, 1), (0, -1), (-2, 0), (2, 0), (-1, 1), (1, 1), (-1, -1), (1, -1)),
        1: ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (0, 2), (0, -2), (1, 1), (-1, 1), (1, -1), (-1, -1)),
        2: ((0, 0), (-1, 0), (1, 0), (0, 1), (0, -1), (-2, 0), (2, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)),
        3: ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (0, 2), (0, -2), (1, -1), (-1, -1), (1, 1), (-1, 1)),
    }

    KICKS_180_I = {
        0: ((0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, 1), (0, -1), (-3, 0), (3, 0), (0, 2), (0, -2)),
        1: ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (0, 2), (0, -2), (2, 0), (-2, 0), (0, 3), (0, -3)),
        2: ((0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, 1), (0, -1), (-3, 0), (3, 0), (0, 2), (0, -2)),
        3: ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (0, 2), (0, -2), (2, 0), (-2, 0), (0, 3), (0, -3)),
    }

    @classmethod
    def kicks_180_for(
        cls,
        piece_id: int,
        from_rotation: int,
        to_rotation: int,
    ) -> Tuple[Tuple[int, int], ...]:
        name = utils.ID_TO_NAME[piece_id]
        if name == "I":
            return cls.KICKS_180_I.get(from_rotation % 4, ((0, 0),))
        if name == "O":
            return ((0, 0),)
        return cls.KICKS_180_JLSTZ.get(from_rotation % 4, ((0, 0),))
