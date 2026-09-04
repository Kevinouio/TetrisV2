from __future__ import annotations

import numpy as np

from tetris_v2.rl.actions import PLACEMENT_ACTION_DIM
from tetris_v2.rl.battle.policies import ColdClearBattlePolicy, RandomBattlePolicy
from tetris_v2.rl.battle.train import parse_args


def test_seeded_random_policy_samples_only_legal_actions() -> None:
    mask = np.zeros(PLACEMENT_ACTION_DIM, dtype=np.uint8)
    mask[[3, 100, 999]] = 1
    observation = np.zeros(470, dtype=np.float32)
    first = RandomBattlePolicy(seed=12)
    second = RandomBattlePolicy(seed=12)

    actions_a = [
        first.select_action(observation, mask, player=0, env=None) for _ in range(20)
    ]
    actions_b = [
        second.select_action(observation, mask, player=1, env=None) for _ in range(20)
    ]
    assert actions_a == actions_b
    assert set(actions_a) <= {3, 100, 999}


def test_cold_clear_training_defaults_to_deterministic_fixed_work() -> None:
    class Runtime:
        think_ms: int | None = None

        def bot_choose(self, player: int, think_ms: int = 0):
            del player
            self.think_ms = think_ms
            return {"success": True, "action": 3}

    class Env:
        runtime = Runtime()

    mask = np.zeros(PLACEMENT_ACTION_DIM, dtype=np.uint8)
    mask[3] = 1
    policy = ColdClearBattlePolicy()
    assert policy.think_ms == 0
    assert policy.select_action(
        np.zeros(470, dtype=np.float32), mask, player=0, env=Env()
    ) == 3
    assert Env.runtime.think_ms == 0
    assert parse_args([]).cold_clear_think_ms == 0
