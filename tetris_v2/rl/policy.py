"""Shared checkpoint loader + policy inference helpers for PPO/DQN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Literal, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from tetris_v2.rl.dqn.core import DQNAgent
    from tetris_v2.rl.ppo.core import PPOAgent


AlgoName = Literal["ppo", "dqn"]


@dataclass
class LoadedPolicy:
    algo: AlgoName
    obs_dim: int
    action_dim: int
    metadata: Dict[str, float]
    ppo_agent: Optional["PPOAgent"] = None
    dqn_agent: Optional["DQNAgent"] = None

    def act(
        self,
        obs: np.ndarray,
        *,
        deterministic: bool = True,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        if self.algo == "ppo":
            assert self.ppo_agent is not None
            action, _, _ = self.ppo_agent.act(
                obs,
                deterministic=deterministic,
                temperature=temperature,
                epsilon=epsilon,
                action_mask=action_mask,
            )
            return int(action)
        assert self.dqn_agent is not None
        dqn_epsilon = 0.0 if deterministic else max(0.0, float(epsilon))
        return int(
            self.dqn_agent.select_action(
                obs,
                epsilon=dqn_epsilon,
                deterministic=deterministic,
                action_mask=action_mask,
            )
        )


def load_policy(algo: AlgoName, checkpoint: Path, *, device: Optional[str] = None) -> LoadedPolicy:
    algo = str(algo).lower()  # type: ignore[assignment]
    if algo not in {"ppo", "dqn"}:
        raise ValueError("algo must be 'ppo' or 'dqn'")
    if algo == "ppo":
        from tetris_v2.rl.ppo.core import PPOAgent

        agent, metadata = PPOAgent.load(str(checkpoint), device=device)
        return LoadedPolicy(
            algo="ppo",
            obs_dim=int(agent.config.obs_dim),
            action_dim=int(agent.config.action_dim),
            metadata=dict(metadata),
            ppo_agent=agent,
        )
    from tetris_v2.rl.dqn.core import DQNAgent

    agent, metadata = DQNAgent.load(str(checkpoint), device=device)
    return LoadedPolicy(
        algo="dqn",
        obs_dim=int(agent.config.obs_dim),
        action_dim=int(agent.config.action_dim),
        metadata=dict(metadata),
        dqn_agent=agent,
    )


__all__ = ["AlgoName", "LoadedPolicy", "load_policy"]
