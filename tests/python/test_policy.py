from __future__ import annotations

from pathlib import Path

import numpy as np

from tetris_v2.rl.actions import PLACEMENT_ACTION_DIM
from tetris_v2.rl.flow_dqn.core import FlowDQNAgent, FlowDQNConfig
from tetris_v2.rl.battle.dqn import BattleDQNAgent, BattleDQNConfig
from tetris_v2.rl.policy import load_policy


def test_shared_policy_loads_flow_dqn_without_dqn_fallback(tmp_path: Path) -> None:
    checkpoint = tmp_path / "flow_dqn.pt"
    agent = FlowDQNAgent(
        FlowDQNConfig(
            obs_dim=254,
            action_dim=PLACEMENT_ACTION_DIM,
            channels=2,
            flow_steps=1,
            device="cpu",
        )
    )
    agent.save(checkpoint, metadata={"source": "test"})

    policy = load_policy("flow_dqn", checkpoint, device="cpu")
    mask = np.zeros(PLACEMENT_ACTION_DIM, dtype=np.float32)
    mask[1731] = 1.0

    assert policy.algo == "flow_dqn"
    assert policy.flow_dqn_agent is not None
    assert policy.dqn_agent is None
    assert policy.metadata == {"source": "test"}
    assert policy.act(
        np.zeros(254, dtype=np.float32),
        deterministic=True,
        action_mask=mask,
    ) == 1731


def test_battle_policy_has_explicit_blank_opponent_survival_adapter(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "battle_dqn.pt"
    BattleDQNAgent(BattleDQNConfig(device="cpu", seed=4)).save_frozen(checkpoint)
    policy = load_policy("battle_dqn", checkpoint, device="cpu")
    observation = np.zeros(254, dtype=np.float32)
    mask = np.zeros(3200, dtype=np.uint8)
    mask[317] = 1

    assert policy.algo == "battle_dqn"
    assert policy.obs_dim == 254
    assert policy.action_dim == 3200
    assert policy.metadata["survival_adapter"] == "blank_public_opponent_v1"
    assert policy.act(observation, action_mask=mask) == 317
