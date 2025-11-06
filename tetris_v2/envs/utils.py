"""Shared utilities for Tetris environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, List, Sequence, Tuple

import numpy as np

BOARD_HEIGHT = 20
BOARD_WIDTH = 10
VISIBLE_HEIGHT = 20
HIDDEN_ROWS = 2  # spawn buffer (kept for collision only)
TOTAL_ROWS = VISIBLE_HEIGHT + HIDDEN_ROWS

# Piece ids (match observation expectations: 0 == I, ..., 6 == L)
PIECE_NAMES = ("I", "O", "T", "S", "Z", "J", "L")
NAME_TO_ID = {name: idx for idx, name in enumerate(PIECE_NAMES)}
ID_TO_NAME = {idx: name for idx, name in enumerate(PIECE_NAMES)}


def _base_shapes() -> List[np.ndarray]:
    """Return canonical 4x4 matrices for the seven tetrominoes."""
    I = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    O = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    T = np.array(
        [
            [0, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    S = np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    Z = np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    J = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    L = np.array(
        [
            [0, 0, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    return [I, O, T, S, Z, J, L]


def _unique_rotations(shape: np.ndarray) -> List[np.ndarray]:
    rotations: List[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(shape, k=-k)
        if not rotations or not any(np.array_equal(rotated, existing) for existing in rotations):
            rotations.append(rotated)
    return rotations


PIECE_ROTATIONS: List[List[np.ndarray]] = [_unique_rotations(shape) for shape in _base_shapes()]

# NES spawn positions (row includes hidden rows)
NES_SPAWN_ROW = HIDDEN_ROWS - 2  # -2 relative to visible board
NES_SPAWN_COL = 3

# Modern spawn row/col can reuse NES defaults (with kicks handling externally).

NES_LINE_SCORES = {1: 40, 2: 100, 3: 300, 4: 1200}


@dataclass
class PieceState:
    """Runtime description of the falling piece."""

    piece_id: int
    rotation: int
    row: int
    col: int

    def rotated(self, delta: int) -> "PieceState":
        rotations = len(PIECE_ROTATIONS[self.piece_id])
        return PieceState(
            piece_id=self.piece_id,
            rotation=(self.rotation + delta) % rotations,
            row=self.row,
            col=self.col,
        )

    def moved(self, d_row: int = 0, d_col: int = 0) -> "PieceState":
        return PieceState(
            piece_id=self.piece_id,
            rotation=self.rotation,
            row=self.row + d_row,
            col=self.col + d_col,
        )


def get_piece_matrix(piece_state: PieceState) -> np.ndarray:
    rotations = PIECE_ROTATIONS[piece_state.piece_id]
    return rotations[piece_state.rotation % len(rotations)]


def iter_filled_cells(piece_state: PieceState) -> Generator[Tuple[int, int], None, None]:
    matrix = get_piece_matrix(piece_state)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if matrix[r, c]:
                yield piece_state.row + r, piece_state.col + c


def within_bounds(row: int, col: int) -> bool:
    return 0 <= col < BOARD_WIDTH and row < TOTAL_ROWS


def collides(board: np.ndarray, piece_state: PieceState) -> bool:
    for row, col in iter_filled_cells(piece_state):
        if col < 0 or col >= BOARD_WIDTH or row >= TOTAL_ROWS:
            return True
        if row >= 0 and board[row, col] != 0:
            return True
    return False


def lock_piece(board: np.ndarray, piece_state: PieceState) -> np.ndarray:
    locked = board.copy()
    for row, col in iter_filled_cells(piece_state):
        if row >= 0:
            locked[row, col] = piece_state.piece_id + 1
    return locked


def clear_lines(board: np.ndarray) -> Tuple[np.ndarray, int, Sequence[int]]:
    to_keep: List[np.ndarray] = []
    cleared_indices: List[int] = []
    for idx in range(TOTAL_ROWS):
        row = board[idx]
        if idx < HIDDEN_ROWS:
            to_keep.append(row)
            continue
        if np.all(row):
            cleared_indices.append(idx)
        else:
            to_keep.append(row)
    lines_cleared = len(cleared_indices)
    if lines_cleared == 0:
        return board, 0, ()
    new_rows = [np.zeros(BOARD_WIDTH, dtype=np.int8) for _ in range(lines_cleared)]
    new_board = np.vstack(new_rows + to_keep)  # type: ignore[arg-type]
    return new_board, lines_cleared, tuple(cleared_indices)


def board_visible(board: np.ndarray) -> np.ndarray:
    return board[HIDDEN_ROWS:].copy()


def spawn_piece(piece_id: int) -> PieceState:
    return PieceState(piece_id=piece_id, rotation=0, row=NES_SPAWN_ROW, col=NES_SPAWN_COL)


def nes_score_for_lines(lines_cleared: int, level: int) -> int:
    base = NES_LINE_SCORES.get(lines_cleared, 0)
    return base * (level + 1)


def soft_drop_points(dist: int) -> int:
    return dist  # NES gives 1 point per row soft-dropped


def hard_drop_distance(board: np.ndarray, piece_state: PieceState) -> int:
    distance = 0
    test = piece_state
    while True:
        nxt = test.moved(d_row=1)
        if collides(board, nxt):
            break
        test = nxt
        distance += 1
    return distance


def project_lock_position(board: np.ndarray, piece_state: PieceState) -> PieceState:
    drop = hard_drop_distance(board, piece_state)
    return piece_state.moved(d_row=drop)


def uniform_piece_id(rng: np.random.Generator) -> int:
    return rng.integers(0, len(PIECE_NAMES))


def bag_sequence(rng: np.random.Generator) -> Generator[int, None, None]:
    while True:
        bag = list(range(len(PIECE_NAMES)))
        rng.shuffle(bag)
        for piece in bag:
            yield piece


def lines_to_next_level(level: int) -> int:
    return 10


def is_perfect_clear(board: np.ndarray) -> bool:
    visible = board_visible(board)
    return not np.any(visible)


def _t_spin_center(piece_state: PieceState) -> Tuple[int, int]:
    return piece_state.row + 1, piece_state.col + 1


def detect_t_spin(board: np.ndarray, piece_state: PieceState, last_action: str) -> str | None:
    """Return 'tspin', 'mini', or None if the placement is not a T-Spin."""
    if piece_state.piece_id != NAME_TO_ID["T"]:
        return None
    if last_action not in {"rotate_cw", "rotate_ccw"}:
        return None
    center_row, center_col = _t_spin_center(piece_state)
    corner_offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    filled_corners = 0
    for dr, dc in corner_offsets:
        r, c = center_row + dr, center_col + dc
        if r < 0 or not within_bounds(r, c) or board[r, c] != 0:
            filled_corners += 1
    if filled_corners < 3:
        return None
    front_offsets = {
        0: [(1, -1), (1, 1)],
        1: [(-1, 1), (1, 1)],
        2: [(-1, -1), (-1, 1)],
        3: [(-1, -1), (1, -1)],
    }
    rotation = piece_state.rotation % 4
    filled_front = 0
    for dr, dc in front_offsets[rotation]:
        r, c = center_row + dr, center_col + dc
        if r < 0 or not within_bounds(r, c) or board[r, c] != 0:
            filled_front += 1
    is_mini = filled_front < 2
    return "mini" if is_mini else "tspin"


def create_board() -> np.ndarray:
    return np.zeros((TOTAL_ROWS, BOARD_WIDTH), dtype=np.int8)


def gravity_frames(level: int) -> int:
    # NES gravity approximation (frames per row)
    level = max(0, min(level, 29))
    table = [
        48,
        43,
        38,
        33,
        28,
        23,
        18,
        13,
        8,
        6,
        5,
        5,
        4,
        4,
        3,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
    ]
    return table[level]


def tick_gravity(rows_elapsed: int, level: int) -> bool:
    return rows_elapsed >= gravity_frames(level)


def render_board_rgb(board: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [0, 0, 0],
            [0, 255, 255],  # I
            [255, 255, 0],  # O
            [128, 0, 128],  # T
            [0, 255, 0],  # S
            [255, 0, 0],  # Z
            [0, 0, 255],  # J
            [255, 165, 0],  # L
        ],
        dtype=np.uint8,
    )
    pixels = palette[np.clip(board_visible(board), 0, len(palette) - 1)]
    return np.repeat(np.repeat(pixels, 20, axis=0), 20, axis=1)
