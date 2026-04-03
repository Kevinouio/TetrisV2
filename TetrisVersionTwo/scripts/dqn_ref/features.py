from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class AfterstateFeatures:
    total_height: float
    bumpiness: float
    lines_removed: float
    holes: float
    y_pos: float
    pillar: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.total_height,
                self.bumpiness,
                self.lines_removed,
                self.holes,
                self.y_pos,
                self.pillar,
            ],
            dtype=np.float32,
        )


def compute_features_from_board(
    board_after: np.ndarray,
    y_pos: int,
    lines_removed: int,
) -> AfterstateFeatures:
    """Replicates Version2/settings.py:get_states behavior."""
    board = np.asarray(board_after, dtype=np.uint8)
    if board.shape != (20, 10):
        raise ValueError(f"Expected board shape (20, 10), got {board.shape}.")

    cols = [0] * 10
    holes = 0
    bumpiness = 0

    for c in range(10):
        block_seen = False
        for r in range(20):
            if board[r, c] > 0 and not block_seen:
                block_seen = True
                cols[c] = 20 - r
            if board[r, c] == 0 and block_seen:
                holes += 1
        if c > 0:
            bumpiness += abs(cols[c] - cols[c - 1])

    total_heights = int(sum(cols))
    pillar = 0

    for i in range(1, 9):
        if (cols[i - 1] - cols[i] >= 3) and (cols[i + 1] - cols[i] >= 3):
            pillar = 1
            break

    if pillar == 0:
        if cols[1] - cols[0] >= 3 or cols[-2] - cols[-1] >= 3:
            pillar = 1

    return AfterstateFeatures(
        total_height=float(total_heights),
        bumpiness=float(bumpiness),
        lines_removed=float(lines_removed),
        holes=float(holes),
        y_pos=float(y_pos),
        pillar=float(pillar),
    )


def as_feature_array(features: Sequence[float] | AfterstateFeatures) -> np.ndarray:
    if isinstance(features, AfterstateFeatures):
        return features.as_array()
    arr = np.asarray(features, dtype=np.float32)
    if arr.shape != (6,):
        raise ValueError(f"Expected feature vector shape (6,), got {arr.shape}.")
    return arr


def stack_feature_arrays(vectors: Iterable[Sequence[float] | AfterstateFeatures]) -> np.ndarray:
    rows: List[np.ndarray] = [as_feature_array(v) for v in vectors]
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32, copy=False)

