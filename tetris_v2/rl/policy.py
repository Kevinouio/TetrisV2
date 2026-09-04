"""Shared checkpoint loading and masked inference for every RL algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Literal, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from tetris_v2.rl.battle.dqn import BattleDQNAgent
    from tetris_v2.rl.dqn.core import DQNAgent
    from tetris_v2.rl.flow_dqn.core import FlowDQNAgent
    from tetris_v2.rl.ppo.core import PPOAgent


AlgoName = Literal["ppo", "dqn", "flow_dqn", "battle_dqn"]


def _single_player_battle_observation(observation: np.ndarray) -> np.ndarray:
    """Adapt public single-player state to the battle policy's no-opponent view."""

    from tetris_v2.rl.battle.dqn import BATTLE_OBSERVATION_DIM
    from tetris_v2.rl.battle.stats import compute_board_stats

    own = np.asarray(observation, dtype=np.float32)
    if own.shape != (254,):
        raise ValueError("Battle-DQN survival inference requires a 254-value observation.")
    value = np.zeros(BATTLE_OBSERVATION_DIM, dtype=np.float32)
    value[:254] = own
    board = own[:200].reshape(20, 10)[::-1]
    stats = compute_board_stats(board)
    value[454:470] = np.asarray(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            stats.aggregate_height / 200.0,
            stats.max_height / 20.0,
            stats.holes / 200.0,
            stats.bumpiness / 180.0,
            stats.wells / 420.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            (20.0 - stats.max_height) / 40.0,
            (200.0 - stats.holes) / 400.0,
        ],
        dtype=np.float32,
    )
    np.clip(value[454:470], 0.0, 1.0, out=value[454:470])
    return value


@dataclass
class LoadedPolicy:
    algo: AlgoName
    obs_dim: int
    action_dim: int
    metadata: Dict[str, object]
    ppo_agent: Optional["PPOAgent"] = None
    dqn_agent: Optional["DQNAgent"] = None
    flow_dqn_agent: Optional["FlowDQNAgent"] = None
    battle_dqn_agent: Optional["BattleDQNAgent"] = None

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
        if self.algo == "dqn":
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
        if self.algo == "battle_dqn":
            assert self.battle_dqn_agent is not None
            if action_mask is None:
                raise ValueError("Battle-DQN inference requires a legal-action mask.")
            battle_obs = _single_player_battle_observation(obs)
            battle_epsilon = 0.0 if deterministic else max(0.0, float(epsilon))
            return int(
                self.battle_dqn_agent.select_action(
                    battle_obs,
                    action_mask=np.asarray(action_mask),
                    epsilon=battle_epsilon,
                    deterministic=deterministic,
                )
            )
        assert self.flow_dqn_agent is not None
        return int(
            self.flow_dqn_agent.select_action(
                obs,
                deterministic=deterministic,
                temperature=temperature,
                action_mask=action_mask,
            )
        )


def load_policy(algo: AlgoName, checkpoint: Path, *, device: Optional[str] = None) -> LoadedPolicy:
    algo = str(algo).lower()  # type: ignore[assignment]
    if algo not in {"ppo", "dqn", "flow_dqn", "battle_dqn"}:
        raise ValueError("algo must be 'ppo', 'dqn', 'flow_dqn', or 'battle_dqn'")
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
    if algo == "dqn":
        from tetris_v2.rl.dqn.core import DQNAgent

        agent, metadata = DQNAgent.load(str(checkpoint), device=device)
        return LoadedPolicy(
            algo="dqn",
            obs_dim=int(agent.config.obs_dim),
            action_dim=int(agent.config.action_dim),
            metadata=dict(metadata),
            dqn_agent=agent,
        )
    if algo == "battle_dqn":
        from tetris_v2.rl.battle.dqn import BattleDQNAgent

        agent, metadata = BattleDQNAgent.load_frozen(
            checkpoint,
            device=device or "cpu",
        )
        adapted_metadata = dict(metadata)
        adapted_metadata["survival_adapter"] = "blank_public_opponent_v1"
        return LoadedPolicy(
            algo="battle_dqn",
            obs_dim=254,
            action_dim=int(agent.config.action_dim),
            metadata=adapted_metadata,
            battle_dqn_agent=agent,
        )
    from tetris_v2.rl.flow_dqn.core import FlowDQNAgent

    agent, metadata = FlowDQNAgent.load(str(checkpoint), device=device)
    return LoadedPolicy(
        algo="flow_dqn",
        obs_dim=int(agent.config.obs_dim),
        action_dim=int(agent.config.action_dim),
        metadata=dict(metadata),
        flow_dqn_agent=agent,
    )


__all__ = ["AlgoName", "LoadedPolicy", "load_policy"]
