from __future__ import annotations

import math

import numpy as np
import pytest

from tetris_v2.rl.actions import PLACEMENT_ACTION_DIM
from tetris_v2.rl.runtime import (
    ACTION_CW,
    ACTION_HARD_DROP,
    ACTION_HOLD,
    ACTION_LEFT,
    ACTION_ROTATE_180,
    ACTION_RIGHT,
    ACTION_SOFT_DROP,
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


def test_play_runtime_exposes_rich_steps_180_rotation_and_ghost() -> None:
    runtime = EnvCtypes(find_library(None), seed=99, play_mode=True)
    try:
        active = runtime.active()
        ghost = runtime.ghost()
        assert runtime.play_mode
        assert active["y"] == 19
        assert ghost is not None
        assert ghost["piece"] == active["piece"]
        assert ghost["rotation"] == active["rotation"]
        assert ghost["x"] == active["x"]
        assert ghost["y"] <= active["y"]

        rotated = runtime.step(ACTION_ROTATE_180)
        assert rotated["action_succeeded"]
        assert not rotated["piece_locked"]
        assert runtime.active()["rotation"] == (active["rotation"] + 2) % 4

        dropped = runtime.step(ACTION_HARD_DROP)
        assert dropped["action_succeeded"]
        assert dropped["piece_locked"]
        assert not dropped["hold_used"]
        assert 0 <= dropped["lines_cleared"] <= 4
        assert 0 <= dropped["spin_type"] <= 2
        assert math.isfinite(dropped["reward"])
        assert not dropped["game_over"]
        assert not dropped["top_out"]
    finally:
        runtime.close()


def test_default_runtime_keeps_cold_clear_rotation_parity(runtime: EnvCtypes) -> None:
    active = runtime.active()
    outcome = runtime.step(ACTION_ROTATE_180)
    assert not runtime.play_mode
    assert not outcome["action_succeeded"]
    assert runtime.active()["rotation"] == active["rotation"]


def test_play_runtime_splits_zero_time_inputs_from_ticks() -> None:
    runtime = EnvCtypes(find_library(None), seed=2718, play_mode=True)
    try:
        spawn_y = runtime.active()["y"]
        assert runtime.input(ACTION_LEFT)["action_succeeded"]
        assert runtime.input(ACTION_RIGHT)["action_succeeded"]
        assert runtime.active()["y"] == spawn_y

        soft_drops = 0
        while runtime.input(ACTION_SOFT_DROP)["action_succeeded"]:
            soft_drops += 1
            assert soft_drops < 40
        assert soft_drops > 0
        assert runtime.meta()["lock_timer"] == 0
        assert runtime.meta()["lock_resets"] == 0

        assert runtime.input(ACTION_LEFT)["action_succeeded"]
        assert runtime.meta()["lock_timer"] == 0
        assert runtime.meta()["lock_resets"] == 1

        tick = runtime.tick()
        assert not tick["piece_locked"]
        assert runtime.meta()["lock_timer"] == 1

        hard_drop = runtime.input(ACTION_HARD_DROP)
        assert hard_drop["action_succeeded"]
        assert hard_drop["piece_locked"]
    finally:
        runtime.close()


def test_play_hold_rejection_is_a_zero_time_noop_until_ticks_advance_gravity() -> None:
    subject = EnvCtypes(find_library(None), seed=1618, play_mode=True)
    control = EnvCtypes(find_library(None), seed=1618, play_mode=True)
    try:
        initial = subject.active()
        expected_replacement = subject.queue()[0]

        for env in (subject, control):
            held = env.input(ACTION_HOLD)
            assert held["action_succeeded"]
            assert held["hold_used"]
            assert not held["piece_locked"]

        assert subject.active()["piece"] == expected_replacement
        assert subject.active()["y"] == initial["y"]
        assert subject.hold_info() == {
            "has_hold": True,
            "hold_piece": initial["piece"],
            "hold_available": False,
        }

        pose_before = subject.active()
        meta_before = subject.meta()
        hold_before = subject.hold_info()
        queue_before = subject.queue()
        rejected = subject.input(ACTION_HOLD)

        assert not rejected["action_succeeded"]
        assert not rejected["hold_used"]
        assert not rejected["piece_locked"]
        assert subject.active() == pose_before
        assert subject.meta() == meta_before
        assert subject.hold_info() == hold_before
        assert subject.queue() == queue_before

        # A rejected input must not perturb the hidden gravity phase. Both
        # environments move on the same (and only the) 60th simulation tick.
        for _ in range(59):
            assert subject.tick() == control.tick()
            assert subject.active() == control.active()
            assert subject.active()["y"] == pose_before["y"]
            assert subject.meta() == control.meta()

        assert subject.tick() == control.tick()
        assert subject.active() == control.active()
        assert subject.active()["y"] == pose_before["y"] - 1
        assert subject.meta() == control.meta()
    finally:
        subject.close()
        control.close()


@pytest.mark.parametrize(
    ("action", "result_flag"),
    ((ACTION_HOLD, "hold_used"), (ACTION_HARD_DROP, "piece_locked")),
)
def test_zero_time_piece_replacement_matches_step_semantics(
    action: int, result_flag: str
) -> None:
    zero_time = EnvCtypes(find_library(None), seed=5772, play_mode=True)
    stepped = EnvCtypes(find_library(None), seed=5772, play_mode=True)
    try:
        replacement_piece = zero_time.queue()[0]
        input_result = zero_time.input(action)
        step_result = stepped.step(action)

        assert input_result["action_succeeded"]
        assert input_result[result_flag]
        assert input_result == step_result
        assert zero_time.active()["piece"] == replacement_piece
        assert zero_time.active() == stepped.active()
        assert zero_time.board() == stepped.board()
        assert zero_time.hold_info() == stepped.hold_info()
        assert zero_time.queue() == stepped.queue()
        assert zero_time.meta() == stepped.meta()
    finally:
        zero_time.close()
        stepped.close()


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
