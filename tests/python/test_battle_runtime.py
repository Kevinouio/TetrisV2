from __future__ import annotations

import numpy as np
import pytest

from tetris_v2.rl.actions import PLACEMENT_ACTION_DIM
from tetris_v2.rl.battle.config import BattleRulesConfig
from tetris_v2.rl.battle.env import BATTLE_OBSERVATION_DIM, BattleEnv
from tetris_v2.rl.battle.runtime import BattleRuntime, _validate_native_observation_size


def _first_action(runtime: BattleRuntime, player: int) -> int:
    legal = np.flatnonzero(runtime.decision_mask(player))
    assert legal.size > 0
    return int(legal[0])


def test_native_battle_abi_observation_masks_meta_and_cold_clear() -> None:
    runtime = BattleRuntime(seed=101, rules=BattleRulesConfig(max_steps=20))
    try:
        assert runtime.action_dim() == PLACEMENT_ACTION_DIM
        assert runtime.observation_size() == BATTLE_OBSERVATION_DIM
        assert runtime.observation(0).shape == (BATTLE_OBSERVATION_DIM,)
        assert runtime.board(0).shape == (20, 10)
        assert runtime.board_piece_ids(1).shape == (20, 10)
        assert np.array_equal(runtime.observation(0), runtime.observation(1))
        assert np.array_equal(runtime.decision_mask(0), runtime.decision_mask(1))
        assert runtime.meta()["players"][0]["incoming_garbage"] == 0

        choices = [runtime.bot_choose(0, think_ms=0) for _ in range(6)]
        assert all(choice["success"] for choice in choices)
        assert all(runtime.decision_mask(0)[choice["action"]] == 1 for choice in choices)
        assert len({choice["action"] for choice in choices}) == 1
        assert len({choice["nodes"] for choice in choices}) == 1
        assert len({choice["score"] for choice in choices}) == 1
        assert all(choice["nodes"] > 0 for choice in choices)
        assert all(choice["think_ms"] >= 0.0 for choice in choices)
        assert all(choice["budget_miss"] == 0 for choice in choices)

        runtime.reset(101)
        reset_choice = runtime.bot_choose(0)
        assert reset_choice["action"] == choices[0]["action"]
        assert reset_choice["nodes"] == choices[0]["nodes"]
        assert reset_choice["score"] == choices[0]["score"]
    finally:
        runtime.close()


def test_native_observation_schema_rejects_size_drift() -> None:
    _validate_native_observation_size(BATTLE_OBSERVATION_DIM)
    with pytest.raises(RuntimeError, match="expected 470, got 469"):
        _validate_native_observation_size(BATTLE_OBSERVATION_DIM - 1)


def test_native_joint_step_rejects_illegal_action_atomically() -> None:
    runtime = BattleRuntime(seed=103)
    try:
        legal0 = _first_action(runtime, 0)
        mask1 = runtime.decision_mask(1)
        illegal1 = int(np.flatnonzero(mask1 == 0)[0])
        before_observations = (runtime.observation(0), runtime.observation(1))
        before_boards = (runtime.board(0), runtime.board(1))
        before_meta = runtime.meta()

        result = runtime.step((legal0, illegal1))
        assert not result["success"]
        np.testing.assert_array_equal(runtime.observation(0), before_observations[0])
        np.testing.assert_array_equal(runtime.observation(1), before_observations[1])
        np.testing.assert_array_equal(runtime.board(0), before_boards[0])
        np.testing.assert_array_equal(runtime.board(1), before_boards[1])
        assert runtime.meta() == before_meta
    finally:
        runtime.close()


def test_native_seeded_joint_matches_reproduce_state_and_results() -> None:
    first = BattleRuntime(seed=107)
    repeat = BattleRuntime(seed=107)
    try:
        for _ in range(8):
            actions_first = (_first_action(first, 0), _first_action(first, 1))
            actions_repeat = (_first_action(repeat, 0), _first_action(repeat, 1))
            assert actions_first == actions_repeat
            result_first = first.step(actions_first)
            result_repeat = repeat.step(actions_repeat)
            assert result_first == result_repeat
            for player in (0, 1):
                np.testing.assert_array_equal(first.board(player), repeat.board(player))
                np.testing.assert_array_equal(first.observation(player), repeat.observation(player))
            assert first.meta() == repeat.meta()
            if result_first["terminated"]:
                break
    finally:
        first.close()
        repeat.close()


def test_native_scripted_garbage_delay_and_application() -> None:
    runtime = BattleRuntime(seed=109)
    try:
        assert runtime.enqueue_garbage(0, [1, 2, 3], delay=0)
        queued = runtime.meta()
        assert queued["players"][0]["incoming_garbage"] == 3
        assert queued["players"][0]["next_garbage_delay"] == 1

        result = runtime.step((_first_action(runtime, 0), _first_action(runtime, 1)))
        assert result["success"]
        assert runtime.meta()["players"][0]["score"] == result["players"][0]["raw_reward"]
        assert result["players"][0]["garbage_applied"] == 3
        assert result["players"][0]["incoming_garbage"] == 0
        assert runtime.meta()["players"][0]["garbage_applied"] == 3
    finally:
        runtime.close()


def test_native_backed_environment_builds_470_features_and_truncation_masks() -> None:
    env = BattleEnv(seed=113, rules=BattleRulesConfig(max_steps=1))
    try:
        observations, masks, info = env.reset(seed=113)
        assert observations[0].shape == observations[1].shape == (BATTLE_OBSERVATION_DIM,)
        np.testing.assert_array_equal(observations[0], env.runtime.observation(0))
        np.testing.assert_array_equal(observations[1], env.runtime.observation(1))
        assert info["legal_action_counts"][0] > 0
        actions = (
            int(np.flatnonzero(masks[0])[0]),
            int(np.flatnonzero(masks[1])[0]),
        )
        _, rewards, terminated, truncated, next_info = env.step(actions)
        assert rewards[0] == -rewards[1]
        assert not terminated
        assert truncated
        assert next_info["result"] == "draw"
        assert next_info["legal_action_counts"][0] > 0
        assert next_info["legal_action_counts"][1] > 0
    finally:
        env.close()
