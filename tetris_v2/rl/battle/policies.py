"""Seeded policy adapters for joint battle training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

import numpy as np

from tetris_v2.rl.battle.dqn import BattleDQNAgent


class BattlePolicy(Protocol):
    identifier: str
    kind: str

    def reset(self, seed: int) -> None: ...

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        player: int,
        env: Any,
    ) -> int: ...


def _legal_indices(action_mask: np.ndarray) -> np.ndarray:
    legal = np.flatnonzero(np.asarray(action_mask) > 0.5)
    if legal.size == 0:
        raise ValueError("Action mask contains no legal battle actions.")
    return legal


@dataclass
class RandomBattlePolicy:
    identifier: str = "random"
    kind: str = "random"
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        player: int,
        env: Any,
    ) -> int:
        del observation, player, env
        return int(self.rng.choice(_legal_indices(action_mask)))


@dataclass
class BattleDQNPolicy:
    agent: BattleDQNAgent
    identifier: str = "battle_dqn"
    kind: str = "checkpoint"
    epsilon: float = 0.0
    deterministic: bool = True
    seed: int = 0
    checkpoint: str | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        player: int,
        env: Any,
    ) -> int:
        del player, env
        return self.agent.select_action(
            observation,
            action_mask=action_mask,
            epsilon=self.epsilon,
            deterministic=self.deterministic,
            rng=self.rng,
        )


@dataclass
class ColdClearBattlePolicy:
    """Cold Clear adapted to one seat with deterministic fixed-work by default."""

    think_ms: int = 0
    identifier: str = "cold_clear"
    kind: str = "cold_clear"

    def reset(self, seed: int) -> None:
        del seed

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        player: int,
        env: Any,
    ) -> int:
        del observation
        runtime = getattr(env, "runtime", None)
        if runtime is None or not hasattr(runtime, "bot_choose"):
            raise RuntimeError("Battle runtime does not expose Cold Clear seat inference.")
        result = runtime.bot_choose(int(player), think_ms=int(self.think_ms))
        if isinstance(result, dict):
            if not bool(result.get("success", True)):
                raise RuntimeError("Cold Clear failed to choose a battle action.")
            action = int(result.get("action", result.get("action_index", -1)))
        else:
            action = int(result)
        legal = np.asarray(action_mask) > 0.5
        if not 0 <= action < legal.size or not legal[action]:
            raise RuntimeError(f"Cold Clear returned illegal battle action {action}.")
        return action


def load_battle_dqn_policy(
    checkpoint: str | Path,
    *,
    device: Optional[str] = "cpu",
    identifier: Optional[str] = None,
) -> BattleDQNPolicy:
    agent, metadata = BattleDQNAgent.load_frozen(checkpoint, device=device)
    policy_id = identifier or str(metadata.get("identifier", Path(checkpoint).stem))
    return BattleDQNPolicy(
        agent=agent,
        identifier=policy_id,
        deterministic=True,
        checkpoint=str(checkpoint),
    )


def load_embedded_battle_dqn_policy(
    payload: Mapping[str, object],
    *,
    device: Optional[str] = "cpu",
    identifier: str,
) -> BattleDQNPolicy:
    agent, metadata = BattleDQNAgent.from_frozen_payload(payload, device=device)
    policy = BattleDQNPolicy(
        agent=agent,
        identifier=identifier,
        deterministic=True,
        checkpoint=f"embedded:{identifier}",
    )
    normalized = dict(metadata)
    steps = normalized.get("training_steps", normalized.get("global_step"))
    wall_seconds = normalized.get(
        "wall_clock_training_time", normalized.get("wall_seconds")
    )
    normalized.update(
        {
            "checkpoint_type": "embedded_frozen_policy",
            "training_steps": None if steps is None else int(steps),
            "wall_clock_training_time": (
                None if wall_seconds is None else float(wall_seconds)
            ),
            "wall_clock_training_time_units": "seconds",
        }
    )
    setattr(policy, "checkpoint_metadata", normalized)
    return policy


__all__ = [
    "BattleDQNPolicy",
    "BattlePolicy",
    "ColdClearBattlePolicy",
    "RandomBattlePolicy",
    "load_embedded_battle_dqn_policy",
    "load_battle_dqn_policy",
]
