from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from tetris_v2.rl.actions import PLACEMENT_ACTION_DIM
from tetris_v2.rl.battle.config import BattleRewardConfig, BattleRulesConfig
from tetris_v2.rl.battle.env import (
    BATTLE_FEATURE_SLICE,
    BATTLE_OBSERVATION_DIM,
    BATTLE_OBSERVATION_SCHEMA,
    OPP_BOARD_SLICE,
    OWN_OBS_SLICE,
    BattleEnv,
    _public_features,
)
from tetris_v2.rl.battle.reward import compute_battle_rewards
from tetris_v2.rl.battle.stats import (
    BoardStats,
    PlayerBattleStats,
    PlayerStepStats,
    compute_board_stats,
)


class FakeBattleRuntime:
    def __init__(self, *, player_secrets: tuple[float, float] = (0.25, 0.75), swap: bool = False):
        self.player_secrets = player_secrets[::-1] if swap else player_secrets
        self.swap = swap
        self.closed = False
        self.step_calls = 0
        self.reset(1)

    def action_dim(self) -> int:
        return PLACEMENT_ACTION_DIM

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        self.joint_step = 0
        self.terminated = False
        self.winner = None
        base = np.zeros((2, 20, 10), dtype=np.uint8)
        base[0, -1, 0] = 1
        base[1, -1, 9] = 1
        self.boards = base[::-1].copy() if self.swap else base
        self.pending = [0, 0]
        self.delays = [-1, -1]
        self.received = [0, 0]

    def observation(self, player: int) -> np.ndarray:
        own = np.zeros(254, dtype=np.float32)
        own[200] = self.player_secrets[player]
        opponent_bottom_up = self.boards[1 - player][::-1].reshape(-1).astype(np.float32)
        players = tuple(
            PlayerBattleStats(
                incoming_garbage=self.pending[index],
                next_garbage_delay=self.delays[index],
                board=compute_board_stats(self.boards[index]),
            )
            for index in (0, 1)
        )
        public = _public_features(players, player, BattleRulesConfig())
        return np.concatenate((own, opponent_bottom_up, public))

    def decision_mask(self, player: int) -> np.ndarray:
        mask = np.zeros(PLACEMENT_ACTION_DIM, dtype=np.uint8)
        if not self.terminated:
            mask[0] = 1
        return mask

    def board(self, player: int) -> np.ndarray:
        return self.boards[player].copy()

    def meta(self) -> dict[str, Any]:
        return {
            "joint_step": self.joint_step,
            "terminated": self.terminated,
            "winner": self.winner,
            "players": [
                {
                    "incoming_garbage": self.pending[player],
                    "next_garbage_delay": self.delays[player],
                    "garbage_received": self.received[player],
                    "top_out": False,
                }
                for player in (0, 1)
            ],
        }

    def step(self, actions: tuple[int, int]) -> dict[str, Any]:
        self.step_calls += 1
        self.joint_step += 1
        return {
            "success": True,
            "terminated": False,
            "winner": None,
            "joint_step": self.joint_step,
            "players": [
                {
                    "lines_cleared": 0,
                    "attack_generated": 0,
                    "garbage_cancelled": 0,
                    "garbage_sent": 0,
                    "garbage_received": 0,
                    "garbage_applied": 0,
                    "incoming_garbage": self.pending[player],
                    "next_garbage_delay": self.delays[player],
                    "top_out": False,
                }
                for player in (0, 1)
            ],
        }

    def enqueue_garbage(self, player: int, holes: list[int], delay: int = 0) -> bool:
        if player not in (0, 1) or delay < 0 or any(hole < 0 or hole >= 10 for hole in holes):
            return False
        self.pending[player] += len(holes)
        self.received[player] += len(holes)
        self.delays[player] = delay + 1 if holes else self.delays[player]
        return True

    def bot_choose(self, player: int, think_ms: int = 0) -> dict[str, Any]:
        return {"success": True, "action": 0, "think_ms": float(think_ms)}

    def close(self) -> None:
        self.closed = True


def test_battle_rule_defaults_and_attack_table() -> None:
    rules = BattleRulesConfig()
    assert rules.attack_table == (0, 0, 1, 2, 4)
    assert rules.garbage_delay == 1
    assert rules.max_steps == 500
    assert rules.mirrored_piece_seeds
    assert [rules.attack_for_lines(lines) for lines in range(5)] == [0, 0, 1, 2, 4]

    with pytest.raises(ValueError, match="exactly five"):
        BattleRulesConfig(attack_table=(0, 1))
    with pytest.raises(ValueError, match="not implemented"):
        BattleRulesConfig(combo_attack_enabled=True)


def test_board_statistics_measure_height_holes_bumpiness_and_wells() -> None:
    board = np.zeros((20, 10), dtype=np.uint8)
    board[-3, 0] = 1
    board[-1, 0] = 1
    board[-1, 2] = 1
    stats = compute_board_stats(board)
    assert stats.aggregate_height == 4
    assert stats.max_height == 3
    assert stats.holes == 1
    assert stats.bumpiness > 0
    assert stats.wells > 0


def test_reward_is_terminal_heavy_potential_based_and_exactly_antisymmetric() -> None:
    previous = (BoardStats(), BoardStats())
    current = (BoardStats(max_height=2), BoardStats(max_height=5, holes=2))
    events = (
        PlayerStepStats(lines_cleared=2, garbage_sent=1, garbage_cancelled=1),
        PlayerStepStats(garbage_applied=2),
    )
    result = compute_battle_rewards(
        previous,
        current,
        events,
        winner=0,
        terminated=True,
        config=BattleRewardConfig(),
    )
    assert result.rewards[0] == -result.rewards[1]
    assert result.rewards[0] > 10.0
    for name, value in result.components[0].items():
        assert value == -result.components[1][name]

    unchanged = compute_battle_rewards(current, current, events=(PlayerStepStats(), PlayerStepStats()))
    assert unchanged.rewards == (0.0, -0.0)


def test_observation_layout_is_canonical_normalized_and_has_no_opponent_secret() -> None:
    env_a = BattleEnv(runtime=FakeBattleRuntime(player_secrets=(0.2, 0.7)), seed=13)
    env_b = BattleEnv(runtime=FakeBattleRuntime(player_secrets=(0.2, 0.9)), seed=13)
    try:
        observations_a, masks, info = env_a.reset(seed=13)
        observations_b, _, _ = env_b.reset(seed=13)
        assert info["observation_schema"] == BATTLE_OBSERVATION_SCHEMA
        assert observations_a[0].shape == (BATTLE_OBSERVATION_DIM,)
        assert observations_a[0][OWN_OBS_SLICE][200] == pytest.approx(0.2)
        np.testing.assert_array_equal(observations_a[0], observations_b[0])
        assert observations_a[1][OWN_OBS_SLICE][200] != observations_b[1][OWN_OBS_SLICE][200]
        np.testing.assert_array_equal(
            observations_a[0][OPP_BOARD_SLICE],
            env_a.runtime.boards[1][::-1].reshape(-1),  # type: ignore[attr-defined]
        )
        assert np.all((observations_a[0][BATTLE_FEATURE_SLICE] >= 0.0) & (observations_a[0][BATTLE_FEATURE_SLICE] <= 1.0))
        assert masks[0].shape == masks[1].shape == (PLACEMENT_ACTION_DIM,)
    finally:
        env_a.close()
        env_b.close()


def test_illegal_joint_action_is_rejected_without_native_mutation() -> None:
    runtime = FakeBattleRuntime()
    env = BattleEnv(runtime=runtime)
    try:
        _, _, info = env.reset(seed=3)
        boards_before = deepcopy(runtime.boards)
        with pytest.raises(ValueError, match="player 1"):
            env.step((0, 1))
        assert runtime.step_calls == 0
        np.testing.assert_array_equal(runtime.boards, boards_before)
        assert info["legal_action_counts"] == (1, 1)
    finally:
        env.close()


def test_fixed_seed_and_seat_swap_are_symmetric() -> None:
    first = BattleEnv(runtime=FakeBattleRuntime(), seed=31)
    repeat = BattleEnv(runtime=FakeBattleRuntime(), seed=31)
    swapped = BattleEnv(runtime=FakeBattleRuntime(swap=True), seed=31)
    try:
        obs_a, masks_a, _ = first.reset(seed=31)
        obs_b, masks_b, _ = repeat.reset(seed=31)
        obs_swap, _, _ = swapped.reset(seed=31)
        np.testing.assert_array_equal(obs_a, obs_b)
        np.testing.assert_array_equal(masks_a, masks_b)
        np.testing.assert_array_equal(obs_a[0], obs_swap[1])
        np.testing.assert_array_equal(obs_a[1], obs_swap[0])

        step_a = first.step((0, 0))
        step_b = repeat.step((0, 0))
        np.testing.assert_array_equal(step_a[0], step_b[0])
        assert step_a[1:4] == step_b[1:4]
        assert step_a[4]["players"] == step_b[4]["players"]
        assert step_a[4]["reward_components"] == step_b[4]["reward_components"]
        np.testing.assert_array_equal(step_a[4]["action_masks"], step_b[4]["action_masks"])
    finally:
        first.close()
        repeat.close()
        swapped.close()


def test_scripted_garbage_refreshes_public_features_and_info() -> None:
    env = BattleEnv(runtime=FakeBattleRuntime())
    try:
        _, _, initial = env.reset(seed=7)
        observations, masks, info = env.enqueue_garbage(0, [1, 2, 3], delay=2)
        assert initial["players"][0]["incoming_garbage"] == 0
        assert info["players"][0]["incoming_garbage"] == 3
        assert info["players"][0]["garbage_received"] == 3
        assert observations[0][BATTLE_FEATURE_SLICE][0] == pytest.approx(3.0 / 40.0)
        assert masks[0][0] == masks[1][0] == 1
    finally:
        env.close()


def test_match_limit_is_a_draw_truncation_with_bootstrap_masks() -> None:
    env = BattleEnv(runtime=FakeBattleRuntime(), rules=BattleRulesConfig(max_steps=1))
    try:
        env.reset(seed=5)
        _, rewards, terminated, truncated, info = env.step((0, 0))
        assert not terminated
        assert truncated
        assert info["winner"] is None
        assert info["result"] == "draw"
        assert rewards[0] == -rewards[1]
        assert info["legal_action_counts"] == (1, 1)
    finally:
        env.close()
