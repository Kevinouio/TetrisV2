from __future__ import annotations

import numpy as np

from tetris_v2.rl.actions import (
    BOARD_ROWS,
    BOARD_WIDTH,
    PLACEMENT_ACTION_DIM,
    PLACEMENT_MAP_SHAPE,
    action_channel,
    decode_action,
    encode_action,
)


def test_structured_action_map_preserves_every_stable_action_id() -> None:
    action_map = np.arange(PLACEMENT_ACTION_DIM).reshape(PLACEMENT_MAP_SHAPE)

    for action in range(PLACEMENT_ACTION_DIM):
        decoded = decode_action(action)
        rebuilt = encode_action(**decoded)
        channel = action_channel(
            use_hold=bool(decoded["use_hold"]),
            rotation=int(decoded["rotation"]),
        )

        assert rebuilt == action
        assert action_map[channel, int(decoded["y"]), int(decoded["x"])] == action


def test_structured_action_map_shape_is_hold_rotation_row_column() -> None:
    assert PLACEMENT_MAP_SHAPE == (2 * 4, BOARD_ROWS, BOARD_WIDTH)

