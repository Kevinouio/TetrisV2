"""Shared RL action-space definition for VersionTwo."""

from __future__ import annotations

from TetrisVersionTwo.rl.runtime import (
    ACTION_CCW,
    ACTION_CW,
    ACTION_HARD_DROP,
    ACTION_HOLD,
    ACTION_LEFT,
    ACTION_NONE,
    ACTION_RIGHT,
    ACTION_SOFT_DROP,
)

# Fixed RL policy action space (8 actions, rotate-180 excluded by design):
# 0=None, 1=Left, 2=Right, 3=SoftDrop, 4=HardDrop, 5=RotateCW, 6=RotateCCW, 7=Hold
RL_ACTION_MAP = (
    ACTION_NONE,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_SOFT_DROP,
    ACTION_HARD_DROP,
    ACTION_CW,
    ACTION_CCW,
    ACTION_HOLD,
)

__all__ = ["RL_ACTION_MAP"]
