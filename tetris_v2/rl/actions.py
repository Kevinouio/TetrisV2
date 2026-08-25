"""Stable one-piece placement decisions shared by every RL algorithm."""

from __future__ import annotations

BOARD_WIDTH = 10
BOARD_ROWS = 40
ROTATION_COUNT = 4

# An action encodes (use_hold, rotation, landing_y, x). Unlike a placement-list
# index, its physical meaning does not change when another tuck becomes legal.
POSE_ACTION_DIM = BOARD_WIDTH * BOARD_ROWS * ROTATION_COUNT
PLACEMENT_ACTION_DIM = 2 * POSE_ACTION_DIM


def encode_action(*, use_hold: bool, rotation: int, y: int, x: int) -> int:
    pose = (int(rotation) * BOARD_ROWS + int(y)) * BOARD_WIDTH + int(x)
    return (POSE_ACTION_DIM if use_hold else 0) + pose


def decode_action(action: int) -> dict[str, int | bool]:
    value = int(action)
    use_hold = value >= POSE_ACTION_DIM
    pose = value - POSE_ACTION_DIM if use_hold else value
    rotation, remainder = divmod(pose, BOARD_ROWS * BOARD_WIDTH)
    y, x = divmod(remainder, BOARD_WIDTH)
    return {"use_hold": use_hold, "rotation": rotation, "y": y, "x": x}

__all__ = [
    "BOARD_WIDTH",
    "BOARD_ROWS",
    "ROTATION_COUNT",
    "POSE_ACTION_DIM",
    "PLACEMENT_ACTION_DIM",
    "encode_action",
    "decode_action",
]
