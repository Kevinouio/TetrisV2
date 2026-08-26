from __future__ import annotations

import math

import numpy as np
import pytest

from tetris_v2.rl.actions import (
    PLACEMENT_ACTION_DIM,
    POSE_ACTION_DIM,
    decode_action,
    encode_action,
)
from tetris_v2.rl.env import CCTetrisEnv, board_potential


def assert_mask_contract(info: dict) -> None:
    mask = np.asarray(info["action_mask"], dtype=np.float32)
    legal_count = int(info["legal_action_count"])
    placement_count = int(info["placement_count_raw"])
    hold_count = int(info["hold_placement_count"])

    assert mask.shape == (PLACEMENT_ACTION_DIM,)
    assert legal_count == int(np.count_nonzero(mask > 0.5))
    assert placement_count == int(np.count_nonzero(mask[:POSE_ACTION_DIM] > 0.5))
    assert hold_count == int(np.count_nonzero(mask[POSE_ACTION_DIM:] > 0.5))
    assert legal_count == placement_count + hold_count
    assert not info["placement_overflow"]


def first_legal_action(info: dict) -> int:
    legal = np.flatnonzero(np.asarray(info["action_mask"], dtype=np.float32) > 0.5)
    assert legal.size > 0
    return int(legal[0])


@pytest.fixture
def env() -> CCTetrisEnv:
    instance = CCTetrisEnv(seed=11, max_steps=200)
    yield instance
    instance.close()


def test_actions_have_stable_physical_encoding() -> None:
    action = encode_action(use_hold=True, rotation=3, y=17, x=8)
    assert action >= POSE_ACTION_DIM
    assert decode_action(action) == {
        "use_hold": True,
        "rotation": 3,
        "y": 17,
        "x": 8,
    }


def test_reset_observation_and_info_contract(env: CCTetrisEnv) -> None:
    observation, info = env.reset(seed=11)
    expected_size = env.runtime.observation_size(include_hidden_rows=False)

    assert observation.dtype == np.float32
    assert observation.shape == (expected_size,)
    assert env.observation_space.shape == (expected_size,)
    assert np.all((observation >= 0.0) & (observation <= 1.0))
    assert_mask_contract(info)


def test_active_piece_is_encoded_in_default_observation() -> None:
    env_a = CCTetrisEnv(seed=3)
    env_b = CCTetrisEnv(seed=115)
    try:
        obs_a, info_a = env_a.reset(seed=3)
        obs_b, info_b = env_b.reset(seed=115)

        assert env_a.runtime.active()["piece"] != env_b.runtime.active()["piece"]
        assert not np.array_equal(obs_a, obs_b)
    finally:
        env_a.close()
        env_b.close()


def test_hold_and_placement_are_one_decision(env: CCTetrisEnv) -> None:
    _, info = env.reset(seed=23)
    assert_mask_contract(info)
    hold_actions = np.flatnonzero(info["action_mask"][POSE_ACTION_DIM:] > 0.5)
    assert hold_actions.size > 0

    action = POSE_ACTION_DIM + int(hold_actions[0])
    _, _, terminated, truncated, next_info = env.step(action)
    assert_mask_contract(next_info)
    assert next_info["used_hold"]
    assert next_info["selected_is_hold"]
    assert next_info["placements"] == 1
    assert not (terminated or truncated)


def test_board_potential_penalizes_holes_and_height() -> None:
    empty = [[0] * 10 for _ in range(20)]
    clean = [row[:] for row in empty]
    clean[-1][0] = 1
    hole = [row[:] for row in empty]
    hole[-2][0] = 1

    assert board_potential(empty) > board_potential(clean) > board_potential(hole)


def test_masked_action_is_rejected(env: CCTetrisEnv) -> None:
    _, info = env.reset(seed=41)
    illegal = np.flatnonzero(np.asarray(info["action_mask"]) <= 0.5)
    assert illegal.size > 0

    with pytest.raises(ValueError, match="illegal|masked"):
        env.step(int(illegal[0]))


def test_seeded_environments_are_deterministic() -> None:
    env_a = CCTetrisEnv(seed=37, max_steps=80)
    env_b = CCTetrisEnv(seed=37, max_steps=80)
    try:
        obs_a, info_a = env_a.reset(seed=37)
        obs_b, info_b = env_b.reset(seed=37)
        np.testing.assert_allclose(obs_a, obs_b, atol=1e-6)

        for _ in range(40):
            action_a = first_legal_action(info_a)
            action_b = first_legal_action(info_b)
            assert action_a == action_b

            obs_a, reward_a, terminated_a, truncated_a, info_a = env_a.step(action_a)
            obs_b, reward_b, terminated_b, truncated_b, info_b = env_b.step(action_b)

            np.testing.assert_allclose(obs_a, obs_b, atol=1e-6)
            assert reward_a == pytest.approx(reward_b)
            assert terminated_a == terminated_b
            assert truncated_a == truncated_b
            assert info_a["lines"] == info_b["lines"]
            assert_mask_contract(info_a)
            assert_mask_contract(info_b)
            if terminated_a or truncated_a:
                break
    finally:
        env_a.close()
        env_b.close()


def test_implicit_resets_advance_a_deterministic_seed_stream() -> None:
    env_a = CCTetrisEnv(seed=37)
    env_b = CCTetrisEnv(seed=37)
    try:
        _, first_a = env_a.reset(seed=37)
        _, first_b = env_b.reset(seed=37)
        _, second_a = env_a.reset()
        _, second_b = env_b.reset()
        _, third_a = env_a.reset()
        _, third_b = env_b.reset()

        assert first_a["seed"] == first_b["seed"] == 37
        assert second_a["seed"] == second_b["seed"]
        assert third_a["seed"] == third_b["seed"]
        assert len({first_a["seed"], second_a["seed"], third_a["seed"]}) == 3
    finally:
        env_a.close()
        env_b.close()


def test_step_metadata_stays_consistent(env: CCTetrisEnv) -> None:
    _, info = env.reset(seed=101)
    previous_lines = 0

    for _ in range(100):
        _, reward, terminated, truncated, info = env.step(first_legal_action(info))
        assert math.isfinite(float(reward))
        assert int(info["lines"]) >= previous_lines
        assert_mask_contract(info)
        previous_lines = int(info["lines"])
        if terminated or truncated:
            if terminated:
                assert info["game_over"]
            break
