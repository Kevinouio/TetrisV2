"""Board and match statistics shared by battle training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from tetris_v2.rl.runtime import BOARD_COLS, BOARD_ROWS


@dataclass(frozen=True)
class BoardStats:
    aggregate_height: int = 0
    max_height: int = 0
    holes: int = 0
    bumpiness: int = 0
    wells: int = 0
    occupied_cells: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class PlayerStepStats:
    """Events resolved for one player during one atomic joint step."""

    lines_cleared: int = 0
    raw_reward: float = 0.0
    attack_generated: int = 0
    garbage_cancelled: int = 0
    garbage_sent: int = 0
    garbage_received: int = 0
    garbage_applied: int = 0
    incoming_garbage: int = 0
    next_garbage_delay: int = 0
    top_out: bool = False

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass
class PlayerBattleStats:
    placements: int = 0
    score: float = 0.0
    lines_cleared: int = 0
    attack_generated: int = 0
    garbage_cancelled: int = 0
    garbage_sent: int = 0
    garbage_received: int = 0
    garbage_applied: int = 0
    incoming_garbage: int = 0
    next_garbage_delay: int = 0
    top_out: bool = False
    height_sum: int = 0
    holes_sum: int = 0
    maximum_height: int = 0
    board: BoardStats = field(default_factory=BoardStats)
    last_step: PlayerStepStats = field(default_factory=PlayerStepStats)

    def record(self, step: PlayerStepStats, board: BoardStats) -> None:
        self.placements += 1
        self.score += float(step.raw_reward)
        self.lines_cleared += int(step.lines_cleared)
        self.attack_generated += int(step.attack_generated)
        self.garbage_cancelled += int(step.garbage_cancelled)
        self.garbage_sent += int(step.garbage_sent)
        self.garbage_received += int(step.garbage_received)
        self.garbage_applied += int(step.garbage_applied)
        self.incoming_garbage = int(step.incoming_garbage)
        self.next_garbage_delay = int(step.next_garbage_delay)
        self.top_out = bool(step.top_out)
        self.height_sum += int(board.max_height)
        self.holes_sum += int(board.holes)
        self.maximum_height = max(self.maximum_height, int(board.max_height))
        self.board = board
        self.last_step = step

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["average_height"] = self.height_sum / max(1, self.placements)
        value["average_holes"] = self.holes_sum / max(1, self.placements)
        return value


@dataclass
class BattleStats:
    steps: int = 0
    players: tuple[PlayerBattleStats, PlayerBattleStats] = field(
        default_factory=lambda: (PlayerBattleStats(), PlayerBattleStats())
    )
    winner: int | None = None
    terminated: bool = False
    truncated: bool = False

    @property
    def result(self) -> str:
        if not (self.terminated or self.truncated):
            return "ongoing"
        if self.winner is None:
            return "draw"
        return f"player_{self.winner}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": int(self.steps),
            "players": [player.to_dict() for player in self.players],
            "winner": self.winner,
            "result": self.result,
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
        }


def compute_board_stats(board: Sequence[Sequence[int]] | np.ndarray) -> BoardStats:
    """Measure a visible, top-to-bottom 20 by 10 locked-cell board."""

    cells = np.asarray(board, dtype=np.uint8)
    if cells.shape == (BOARD_ROWS * BOARD_COLS,):
        cells = cells.reshape(BOARD_ROWS, BOARD_COLS)
    if cells.shape != (BOARD_ROWS, BOARD_COLS):
        raise ValueError(f"visible board must have shape {(BOARD_ROWS, BOARD_COLS)}")
    occupied = cells != 0
    heights = np.zeros(BOARD_COLS, dtype=np.int32)
    holes = 0
    for column in range(BOARD_COLS):
        filled = np.flatnonzero(occupied[:, column])
        if filled.size:
            top = int(filled[0])
            heights[column] = BOARD_ROWS - top
            holes += int(np.count_nonzero(~occupied[top:, column]))

    wells = 0
    for column, height in enumerate(heights):
        left = BOARD_ROWS if column == 0 else int(heights[column - 1])
        right = BOARD_ROWS if column == BOARD_COLS - 1 else int(heights[column + 1])
        depth = max(0, min(left, right) - int(height))
        wells += depth * (depth + 1) // 2

    return BoardStats(
        aggregate_height=int(heights.sum()),
        max_height=int(heights.max(initial=0)),
        holes=int(holes),
        bumpiness=int(np.abs(np.diff(heights)).sum()),
        wells=int(wells),
        occupied_cells=int(np.count_nonzero(occupied)),
    )


def board_quality(stats: BoardStats) -> float:
    """Bounded board potential used only through temporal differences."""

    return -(
        0.20 * stats.aggregate_height / 200.0
        + 0.15 * stats.max_height / 20.0
        + 0.35 * stats.holes / 200.0
        + 0.20 * stats.bumpiness / 180.0
        + 0.10 * stats.wells / 420.0
    )


__all__ = [
    "BattleStats",
    "BoardStats",
    "PlayerBattleStats",
    "PlayerStepStats",
    "board_quality",
    "compute_board_stats",
]
