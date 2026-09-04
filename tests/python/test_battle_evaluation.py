from __future__ import annotations

import numpy as np
import pytest

from tetris_v2.rl.battle.config import BattleRulesConfig
from tetris_v2.rl.battle.env import BattleEnv
from tetris_v2.rl.battle.evaluation import (
    ScheduledBattle,
    compare_repeated_matches,
    evaluate_paired_battles,
    paired_seat_schedule,
    run_battle_match,
)
from tetris_v2.rl.battle.metrics import BattleMatchMetrics


def _match(inference: float = 0.1, *, winner: int | None = 0) -> BattleMatchMetrics:
    return BattleMatchMetrics(
        match=1,
        seed=12,
        learner_seat=0,
        opponent_type="random",
        opponent_checkpoint=None,
        result="draw" if winner is None else "win",
        winner=winner,
        match_steps=3,
        placements=(3, 3),
        lines_cleared=(1, 0),
        attack_generated=(0, 0),
        attack_sent=(0, 0),
        garbage_received=(0, 0),
        garbage_cancelled=(0, 0),
        top_out=(False, winner == 0),
        average_height=(2.0, 3.0),
        maximum_height=(3, 4),
        average_holes=(0.0, 1.0),
        illegal_actions=(0, 0),
        inference_ms=(inference, inference),
        inference_decisions=(3, 3),
        board_samples=(3, 3),
        returns=(20.0, -20.0),
    )


def test_paired_schedule_uses_same_seed_from_both_seats() -> None:
    schedule = paired_seat_schedule(matches=4, seed=1000)
    assert [(item.match, item.seed, item.learner_seat) for item in schedule] == [
        (1, 1000, 0),
        (2, 1000, 1),
        (3, 1001, 0),
        (4, 1001, 1),
    ]
    with pytest.raises(ValueError, match="even"):
        paired_seat_schedule(matches=3, seed=0)


def test_reproducibility_ignores_latency_but_not_game_result() -> None:
    assert compare_repeated_matches([_match(0.1)], [_match(9.9)])["passed"]
    report = compare_repeated_matches([_match()], [_match(winner=None)])
    assert not report["passed"]


def test_match_metrics_retain_generated_and_post_cancellation_attack() -> None:
    class FirstLegalPolicy:
        identifier = "first"
        kind = "test"

        def reset(self, seed: int) -> None:
            self.seed = seed

        def select_action(self, observation, action_mask, *, player, env) -> int:
            del observation, player, env
            return int(np.flatnonzero(np.asarray(action_mask) > 0.5)[0])

    class OneStepEnv:
        def __init__(self) -> None:
            self.mask = np.zeros(3200, dtype=np.float32)
            self.mask[0] = 1.0

        def _info(self, step: int) -> dict[str, object]:
            attack_generated = (0, 0) if step == 0 else (5, 4)
            garbage_cancelled = (0, 0) if step == 0 else (3, 3)
            garbage_sent = (0, 0) if step == 0 else (2, 1)
            players = tuple(
                {
                    "placements": step,
                    "lines_cleared": 4 - seat if step else 0,
                    "attack_generated": attack_generated[seat],
                    "garbage_sent": garbage_sent[seat],
                    "garbage_received": 0,
                    "garbage_applied": 0,
                    "garbage_cancelled": garbage_cancelled[seat],
                    "top_out": False,
                }
                for seat in (0, 1)
            )
            return {
                "step": step,
                "winner": None,
                "result": "draw",
                "players": players,
                "board_stats": (
                    {"max_height": 6, "holes": 2},
                    {"max_height": 8, "holes": 3},
                ),
                "step_stats": ({}, {}),
                "action_masks": (self.mask.copy(), self.mask.copy()),
                "reward_components": ({}, {}),
            }

        def reset(self, *, seed: int):
            del seed
            observations = (
                np.zeros(470, dtype=np.float32),
                np.zeros(470, dtype=np.float32),
            )
            return observations, (self.mask.copy(), self.mask.copy()), self._info(0)

        def step(self, actions):
            assert actions == (0, 0)
            observations = (
                np.zeros(470, dtype=np.float32),
                np.zeros(470, dtype=np.float32),
            )
            return observations, (0.0, 0.0), False, True, self._info(1)

    metrics = run_battle_match(
        OneStepEnv(),
        (FirstLegalPolicy(), FirstLegalPolicy()),
        scheduled=ScheduledBattle(match=1, seed=77, learner_seat=0),
        opponent_type="test",
    )

    assert metrics.attack_generated == (5, 4)
    assert metrics.attack_sent == (2, 1)
    assert metrics.garbage_cancelled == (3, 3)
    assert metrics.inference_decisions == (1, 1)
    assert metrics.board_samples == (1, 1)
    assert metrics.average_height == pytest.approx((6.0, 8.0))
    assert metrics.average_holes == pytest.approx((2.0, 3.0))


def test_real_native_distinct_policies_are_symmetric_when_seats_swap() -> None:
    class ExtremeLegalPolicy:
        def __init__(self, identifier: str, *, choose_last: bool) -> None:
            self.identifier = identifier
            self.kind = "test"
            self.choose_last = choose_last

        def reset(self, seed: int) -> None:
            self.seed = seed

        def select_action(self, observation, action_mask, *, player, env) -> int:
            del observation, player, env
            legal = np.flatnonzero(np.asarray(action_mask) > 0.5)
            return int(legal[-1] if self.choose_last else legal[0])

    rules = BattleRulesConfig(
        attack_table=(0, 0, 0, 0, 0),
        max_steps=8,
    )
    matches, summary = evaluate_paired_battles(
        ExtremeLegalPolicy("first", choose_last=False),
        ExtremeLegalPolicy("last", choose_last=True),
        env_factory=lambda: BattleEnv(seed=991, rules=rules),
        matches=2,
        seed=991,
    )
    first, swapped = matches
    assert (first.learner_seat, swapped.learner_seat) == (0, 1)
    assert first.result == swapped.result
    for name in (
        "placements",
        "lines_cleared",
        "attack_generated",
        "attack_sent",
        "garbage_received",
        "garbage_applied",
        "garbage_cancelled",
        "top_out",
        "maximum_height",
        "illegal_actions",
        "inference_decisions",
        "board_samples",
        "returns",
    ):
        values = getattr(first, name)
        swapped_values = getattr(swapped, name)
        assert values == (swapped_values[1], swapped_values[0])
    assert first.average_height == pytest.approx(
        (swapped.average_height[1], swapped.average_height[0])
    )
    assert first.average_holes == pytest.approx(
        (swapped.average_holes[1], swapped.average_holes[0])
    )
    assert summary["seat_win_rate_gap"] == 0.0
    assert summary["illegal_action_count"] == [0, 0]
