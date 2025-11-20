"""Shared utilities for Tetris environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, List, Optional, Sequence, Tuple

import numpy as np

BOARD_HEIGHT = 20
BOARD_WIDTH = 10
VISIBLE_HEIGHT = 20
HIDDEN_ROWS = 2  # spawn buffer (kept for collision only)
TOTAL_ROWS = VISIBLE_HEIGHT + HIDDEN_ROWS
NES_SPAWN_ROW = 0
NES_SPAWN_COL = 3

# Piece ids (match observation expectations: 0 == I, ..., 6 == L)
PIECE_NAMES = ("I", "O", "T", "S", "Z", "J", "L")
NAME_TO_ID = {name: idx for idx, name in enumerate(PIECE_NAMES)}
ID_TO_NAME = {idx: name for idx, name in enumerate(PIECE_NAMES)}

PALETTE = np.array(
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

GHOST_COLOR = np.array([200, 200, 200], dtype=np.uint8)


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


def _unique_rotations(shape: np.ndarray, size: int = 4) -> List[np.ndarray]:
    rotations: List[np.ndarray] = []
    base = shape
    for _ in range(4):
        rotations.append(base)
        if size == 3:
            # Rotate only the top-left 3x3
            sub = base[:3, :3]
            rotated_sub = np.rot90(sub, k=-1)
            new_shape = base.copy()
            new_shape[:3, :3] = rotated_sub
            base = new_shape
        else:
            # Rotate full 4x4
            base = np.rot90(base, k=-1)
            
    return rotations


# Define rotation sizes: I=4, O=4 (invariant), others=3
# Note: O is 4x4 invariant, so size=4 or 3 doesn't matter much if centered, 
# but O is centered at (1.5, 1.5) in 4x4, so size=4 is correct.
# Define rotation sizes: I=4, O=4 (invariant), others=3
# Note: O is 4x4 invariant, so size=4 or 3 doesn't matter much if centered, 
# but O is centered at (1.5, 1.5) in 4x4, so size=4 is correct.
PIECE_SIZES = {
    "I": 4,
    "O": 4,
    "T": 3,
    "S": 3,
    "Z": 3,
    "J": 3,
    "L": 3,
}

PIECE_ROTATIONS: Tuple[Tuple[np.ndarray, ...], ...] = tuple(
    tuple(_unique_rotations(shape, PIECE_SIZES[name]))
    for name, shape in zip(PIECE_NAMES, _base_shapes())
)


@dataclass(frozen=True)
class PieceState:
    piece_id: int
    rotation: int
    row: int
    col: int

    def __post_init__(self) -> None:
        if not (0 <= self.piece_id < len(PIECE_ROTATIONS)):
            raise ValueError(f"Invalid piece_id {self.piece_id}")

    @property
    def _rotation_count(self) -> int:
        return len(PIECE_ROTATIONS[self.piece_id])

    @property
    def matrix(self) -> np.ndarray:
        return PIECE_ROTATIONS[self.piece_id][self.rotation % self._rotation_count]

    def moved(self, *, d_row: int = 0, d_col: int = 0) -> "PieceState":
        return PieceState(
            piece_id=self.piece_id,
            rotation=self.rotation,
            row=self.row + d_row,
            col=self.col + d_col,
        )

    def rotated(self, delta: int) -> "PieceState":
        return PieceState(
            piece_id=self.piece_id,
            rotation=(self.rotation + delta) % self._rotation_count,
            row=self.row,
            col=self.col,
        )


def within_bounds(row: int, col: int) -> bool:
    return 0 <= row < TOTAL_ROWS and 0 <= col < BOARD_WIDTH


def board_visible(board: np.ndarray) -> np.ndarray:
    return board[HIDDEN_ROWS:, :].copy()


def iter_filled_cells(piece_state: PieceState) -> Iterable[Tuple[int, int]]:
    matrix = piece_state.matrix
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if matrix[r, c]:
                yield piece_state.row + r, piece_state.col + c


def spawn_piece(piece_id: int, *, row: int = NES_SPAWN_ROW, col: int = NES_SPAWN_COL) -> PieceState:
    if not (0 <= piece_id < len(PIECE_NAMES)):
        raise ValueError(f"Invalid piece id {piece_id}")
    return PieceState(piece_id=piece_id, rotation=0, row=row, col=col)


def collides(board: np.ndarray, piece_state: PieceState) -> bool:
    for row, col in iter_filled_cells(piece_state):
        if col < 0 or col >= BOARD_WIDTH or row >= TOTAL_ROWS:
            return True
        if row >= 0 and board[row, col] != 0:
            return True
    return False


def lock_piece(board: np.ndarray, piece_state: PieceState) -> np.ndarray:
    new_board = board.copy()
    for row, col in iter_filled_cells(piece_state):
        if within_bounds(row, col):
            new_board[row, col] = piece_state.piece_id + 1
    return new_board


def clear_lines(board: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    filled = np.all(board != 0, axis=1)
    lines_cleared = int(np.count_nonzero(filled))
    if not lines_cleared:
        return board, 0, np.array([], dtype=int)
    remaining = board[~filled]
    cleared = np.vstack(
        [np.zeros((lines_cleared, BOARD_WIDTH), dtype=board.dtype), remaining]
    )
    return cleared, lines_cleared, np.where(filled)[0]


def nes_score_for_lines(lines: int, level: int) -> int:
    table = {1: 40, 2: 100, 3: 300, 4: 1200}
    return table.get(lines, 0) * (level + 1)


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


def bag_sequence(
    rng: np.random.Generator,
    pieces: Optional[Sequence[int]] = None,
) -> Generator[int, None, None]:
    """Yield randomised 7-bag (or subset) piece ids indefinitely."""

    if pieces is None:
        population = list(range(len(PIECE_NAMES)))
    else:
        population = [int(pid) for pid in pieces]
        if not population:
            raise ValueError("'pieces' must contain at least one piece id.")

    while True:
        bag = list(population)
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
    if last_action not in {"rotate_cw", "rotate_ccw", "rotate_180"}:
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


def render_board_rgb(
    board: np.ndarray,
    *,
    current: Optional[PieceState] = None,
    ghost: Optional[PieceState] = None,
) -> np.ndarray:
    visible = board_visible(board)
    overlay = visible.copy()
    ghost_mask = np.zeros_like(overlay, dtype=bool)
    if ghost is not None:
        for row, col in iter_filled_cells(ghost):
            if row >= HIDDEN_ROWS:
                r = row - HIDDEN_ROWS
                if 0 <= col < BOARD_WIDTH and overlay[r, col] == 0:
                    ghost_mask[r, col] = True
    if current is not None:
        for row, col in iter_filled_cells(current):
            if row >= HIDDEN_ROWS:
                overlay[row - HIDDEN_ROWS, col] = current.piece_id + 1
    pixels = PALETTE[np.clip(overlay, 0, len(PALETTE) - 1)]
    if ghost is not None:
        pixels[ghost_mask] = GHOST_COLOR
    return np.repeat(np.repeat(pixels, 20, axis=0), 20, axis=1)


def render_piece_preview(piece_id: int, *, scale: int = 16) -> np.ndarray:
    canvas = np.zeros((4 * scale, 4 * scale, 3), dtype=np.uint8)
    if piece_id < 0:
        return canvas
    matrix = PIECE_ROTATIONS[piece_id][0]
    color = PALETTE[piece_id + 1]
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if matrix[r, c]:
                canvas[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale] = color
    border = 2
    canvas[:border, :, :] = 40
    canvas[-border:, :, :] = 40
    canvas[:, :border, :] = 40
    canvas[:, -border:, :] = 40
    return canvas
