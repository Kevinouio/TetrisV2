from __future__ import annotations

from pathlib import Path

import numpy as np

from tetris_v2.rl.actions import PLACEMENT_ACTION_DIM
from tetris_v2.rl.flow_dqn.core import FlowDQNAgent, FlowDQNConfig
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

