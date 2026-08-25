"""Cold Clear top-1 supervision helpers for TetrisV2 RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tetris_v2.rl.actions import POSE_ACTION_DIM
from tetris_v2.rl.runtime import EnvCtypes


@dataclass
class ExpertRank:
    action_mask: np.ndarray
    teacher_best_action: int
    nodes: int
    think_ms: float
    nps: float
    budget_miss: int
    placement_count_raw: int
    placement_overflow: bool
    unexpanded_count: int


class ExpertRanker:
    """Expose Cold Clear's best move as a masked top-1 supervision target."""

    def __init__(self, runtime: EnvCtypes, *, think_ms: int = 10):
        self.runtime = runtime
        self.default_think_ms = int(think_ms)

    def rank_current_state(self, think_ms: Optional[int] = None) -> ExpertRank:
        out = self.runtime.bot_choose(
            think_ms=self.default_think_ms if think_ms is None else int(think_ms)
        )
        if not out["success"]:
            raise RuntimeError("Cold Clear failed to choose an action.")

        mask = self.runtime.decision_mask()
        best = self.runtime.decision_for_choice(
            use_hold=bool(out["use_hold"]),
            placement_index=int(out["placement_index"]),
        )
        current_count = int(np.count_nonzero(mask[:POSE_ACTION_DIM]))
        legal_count = int(np.count_nonzero(mask))

        return ExpertRank(
            action_mask=mask,
            teacher_best_action=best,
            nodes=int(out["nodes"]),
            think_ms=float(out["think_ms"]),
            nps=float(out["nps"]),
            budget_miss=int(out["budget_miss"]),
            placement_count_raw=current_count,
            placement_overflow=False,
            unexpanded_count=max(0, legal_count - 1),
        )


__all__ = [
    "ExpertRank",
    "ExpertRanker",
]
