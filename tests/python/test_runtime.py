from __future__ import annotations

import math

import numpy as np
import pytest

from tetris_v2.rl.actions import PLACEMENT_ACTION_DIM
from tetris_v2.rl.runtime import (
    ACTION_CW,
    BOARD_COLS,
    BOARD_ROWS,
    EnvCtypes,
    find_library,
)


@pytest.fixture
def runtime() -> EnvCtypes:
    instance = EnvCtypes(find_library(None), seed=123)
    yield instance
    instance.close()


def test_shared_ctypes_wrapper_exercises_environment_abi(runtime: EnvCtypes) -> None:
    assert runtime.bot_handle is None
    observation = runtime.observation(include_hidden_rows=False)
    board = runtime.board()
    piece_ids = runtime.board_piece_ids(include_active=True)

    assert observation.dtype == np.float32
    assert observation.shape == (runtime.observation_size(include_hidden_rows=False),)
    assert np.all(np.isfinite(observation))
    assert len(board) == BOARD_ROWS
    assert all(len(row) == BOARD_COLS for row in board)
    assert len(piece_ids) == BOARD_ROWS
    assert all(len(row) == BOARD_COLS for row in piece_ids)
    assert runtime.active() is not None
    assert runtime.queue()

    placements = runtime.placements()
    assert len(placements) == runtime.placement_count()
    assert placements
    placement_board = runtime.placement_board(0)
    assert len(placement_board) == BOARD_ROWS
    assert all(len(row) == BOARD_COLS for row in placement_board)

    trace = runtime.rotation_trace(ACTION_CW)
    assert {"success", "final_x", "final_y", "final_rotation", "tests"} <= trace.keys()

    outcome = runtime.apply_placement(0)
    assert outcome["success"]
    assert 0 <= int(outcome["lines"]) <= 4
    assert math.isfinite(float(outcome["reward"]))
    assert runtime.bot_handle is None


def test_shared_ctypes_wrapper_exercises_expert_abi(runtime: EnvCtypes) -> None:
    observation_before = runtime.observation()
    board_before = runtime.board()
    meta_before = runtime.meta()
    choice = runtime.bot_choose(think_ms=1)

    assert choice["success"]
    assert choice["use_hold"] or 0 <= choice["placement_index"] < runtime.placement_count()
    assert math.isfinite(float(choice["score"]))
    assert int(choice["nodes"]) >= 0
    assert float(choice["think_ms"]) >= 0.0
    assert float(choice["nps"]) >= 0.0
    assert int(choice["budget_miss"]) in (0, 1)
    np.testing.assert_array_equal(runtime.observation(), observation_before)
    assert runtime.board() == board_before
    assert runtime.meta() == meta_before

    assert runtime.bot_handle is not None
    action = runtime.decision_for_choice(
        use_hold=choice["use_hold"],
        placement_index=choice["placement_index"],
    )
    mask = runtime.decision_mask()
    decision = runtime.decision(action)
    assert mask.shape == (PLACEMENT_ACTION_DIM,)
    assert mask[action] == 1
    assert decision["use_hold"] == choice["use_hold"]

    outcome = runtime.apply_decision(action)
    assert outcome["success"]
    assert 0 <= int(outcome["lines"]) <= 4
    assert outcome["used_hold"] == choice["use_hold"]
