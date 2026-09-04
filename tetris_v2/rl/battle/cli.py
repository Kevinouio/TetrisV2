"""Shared helpers for the public battle evaluation command-line tools."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any, Callable, Mapping, Optional

import torch

from tetris_v2.rl.battle.checkpoint import BATTLE_TRAINING_CHECKPOINT_VERSION
from tetris_v2.rl.battle.config import BattleRewardConfig, BattleRulesConfig
from tetris_v2.rl.battle.dqn import (
    BATTLE_ACTION_ORDER,
    BATTLE_OBSERVATION_SCHEMA,
    BattleDQNAgent,
)
from tetris_v2.rl.battle.env import BattleEnv
from tetris_v2.rl.battle.opponents import OpponentDescriptor, OpponentPool
from tetris_v2.rl.battle.policies import (
    BattleDQNPolicy,
    BattlePolicy,
    ColdClearBattlePolicy,
    RandomBattlePolicy,
    load_battle_dqn_policy,
)


DEFAULT_BATTLE_GATE_WIN_RATES = {
    "random": 0.95,
    "cold_clear": 0.65,
}


@dataclass(frozen=True)
class TrainingCheckpointView:
    """Inference-relevant state from a trainer checkpoint, without replay."""

    agent: BattleDQNAgent
    opponent_pool: OpponentPool
    training_config: dict[str, object]
    extra: dict[str, object]
    global_step: int
    episode_index: int
    checkpoint_metadata: dict[str, object]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def positive_even_int(value: str) -> int:
    parsed = positive_int(value)
    if parsed % 2:
        raise argparse.ArgumentTypeError("must be even for paired-seat evaluation")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def write_json_report(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def make_env_factory(
    *,
    seed: int,
    lib_path: Optional[Path],
    rules: BattleRulesConfig,
    rewards: Optional[BattleRewardConfig] = None,
) -> Callable[[], BattleEnv]:
    return lambda: BattleEnv(
        seed=int(seed),
        lib_path=lib_path,
        rules=rules,
        reward_config=rewards,
    )


def _load_payload(path: Path, device: str) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except FileNotFoundError:
        raise
    except (EOFError, OSError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Could not read battle checkpoint {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Battle checkpoint payload is malformed: {path}")
    return payload


def _validate_training_payload(payload: Mapping[str, Any]) -> None:
    if (
        int(payload.get("format_version", -1)) != BATTLE_TRAINING_CHECKPOINT_VERSION
        or payload.get("algo") != "battle_dqn_training"
    ):
        raise ValueError("Unsupported battle training checkpoint.")
    if payload.get("observation_schema") != BATTLE_OBSERVATION_SCHEMA:
        raise ValueError("Battle training checkpoint observation schema is incompatible.")
    if payload.get("action_order") != BATTLE_ACTION_ORDER:
        raise ValueError("Battle training checkpoint action order is incompatible.")
    if payload.get("episode_boundary") is not True:
        raise ValueError("Battle training checkpoint is not an episode-boundary snapshot.")


def _normalized_checkpoint_metadata(
    payload: Mapping[str, Any],
    *,
    checkpoint_type: str,
) -> dict[str, object]:
    if checkpoint_type == "training":
        counters = payload.get("counters", {})
        extra = payload.get("extra", {})
        training_config = payload.get("training_config", {})
        if (
            not isinstance(counters, Mapping)
            or not isinstance(extra, Mapping)
            or not isinstance(training_config, Mapping)
        ):
            raise ValueError("Battle training checkpoint provenance is malformed.")
        metadata: dict[str, object] = {
            "checkpoint_type": "training",
            "training_steps": int(counters.get("global_step", 0)),
            "wall_clock_training_time": (
                None
                if extra.get("wall_seconds") is None
                else float(extra["wall_seconds"])
            ),
            "wall_clock_training_time_units": "seconds",
            "episode_index": int(counters.get("episode_index", 0)),
        }
        for name in ("rules", "rewards"):
            value = training_config.get(name)
            if value is not None:
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"Battle training checkpoint {name} configuration is malformed."
                    )
                metadata[name] = copy.deepcopy(dict(value))
        return metadata
    raw = payload.get("metadata", {})
    if not isinstance(raw, Mapping):
        raise ValueError("Frozen Battle-DQN checkpoint metadata is malformed.")
    metadata = copy.deepcopy(dict(raw))
    raw_steps = metadata.get("training_steps", metadata.get("global_step"))
    raw_wall_time = metadata.get(
        "wall_clock_training_time", metadata.get("wall_seconds")
    )
    metadata.update(
        {
            "checkpoint_type": "frozen_policy",
            "training_steps": None if raw_steps is None else int(raw_steps),
            "wall_clock_training_time": (
                None if raw_wall_time is None else float(raw_wall_time)
            ),
            "wall_clock_training_time_units": "seconds",
        }
    )
    return metadata


def policy_checkpoint_metadata(policy: BattlePolicy) -> dict[str, object]:
    value = getattr(policy, "checkpoint_metadata", {})
    return copy.deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def load_training_checkpoint_view(
    path: str | Path,
    *,
    device: str = "cpu",
) -> TrainingCheckpointView:
    """Load the current policy and pool without requiring the replay sidecar."""

    source = Path(path)
    payload = _load_payload(source, device)
    _validate_training_payload(payload)
    agent_state = payload.get("agent")
    pool_state = payload.get("opponent_pool")
    counters = payload.get("counters")
    training_config = payload.get("training_config", {})
    extra = payload.get("extra", {})
    if not isinstance(agent_state, Mapping) or not isinstance(pool_state, Mapping):
        raise ValueError("Battle training checkpoint policy or pool state is malformed.")
    if not isinstance(counters, Mapping):
        raise ValueError("Battle training checkpoint counters are malformed.")
    if not isinstance(training_config, Mapping) or not isinstance(extra, Mapping):
        raise ValueError("Battle training checkpoint metadata is malformed.")
    return TrainingCheckpointView(
        agent=BattleDQNAgent.from_state_dict(agent_state, device=device),
        opponent_pool=OpponentPool.from_state_dict(pool_state),
        training_config=copy.deepcopy(dict(training_config)),
        extra=copy.deepcopy(dict(extra)),
        global_step=int(counters.get("global_step", 0)),
        episode_index=int(counters.get("episode_index", 0)),
        checkpoint_metadata=_normalized_checkpoint_metadata(
            payload,
            checkpoint_type="training",
        ),
    )


def load_evaluation_policy(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    identifier: Optional[str] = None,
    kind: str = "checkpoint",
) -> BattleDQNPolicy:
    """Load either a compact frozen policy or a full trainer checkpoint."""

    source = Path(checkpoint)
    payload = _load_payload(source, device)
    if payload.get("algo") == "battle_dqn_training":
        _validate_training_payload(payload)
        state = payload.get("agent")
        counters = payload.get("counters", {})
        if not isinstance(state, Mapping) or not isinstance(counters, Mapping):
            raise ValueError("Battle training checkpoint policy state is malformed.")
        agent = BattleDQNAgent.from_state_dict(state, device=device)
        policy_id = identifier or f"battle_step_{int(counters.get('global_step', 0)):012d}"
        policy = BattleDQNPolicy(
            agent=agent,
            identifier=policy_id,
            kind=kind,
            deterministic=True,
            epsilon=0.0,
            checkpoint=str(source),
        )
        setattr(
            policy,
            "checkpoint_metadata",
            _normalized_checkpoint_metadata(payload, checkpoint_type="training"),
        )
        return policy
    policy = load_battle_dqn_policy(source, device=device, identifier=identifier)
    policy.kind = kind
    setattr(
        policy,
        "checkpoint_metadata",
        _normalized_checkpoint_metadata(payload, checkpoint_type="frozen_policy"),
    )
    return policy


def make_opponent_policy(
    mode: str,
    *,
    learner_checkpoint: Path,
    opponent_checkpoint: Optional[Path],
    device: str,
    cold_clear_think_ms: int,
) -> BattlePolicy:
    if mode == "random":
        return RandomBattlePolicy()
    if mode == "cold_clear":
        return ColdClearBattlePolicy(think_ms=int(cold_clear_think_ms))
    if mode == "self":
        return load_evaluation_policy(
            learner_checkpoint,
            device=device,
            identifier="self",
            kind="self",
        )
    if mode == "checkpoint":
        if opponent_checkpoint is None:
            raise ValueError("--opponent-checkpoint is required for checkpoint opponents.")
        return load_evaluation_policy(
            opponent_checkpoint,
            device=device,
            kind="checkpoint",
        )
    raise ValueError(f"Unsupported battle opponent mode: {mode!r}")


def rules_and_rewards_from_training_config(
    training_config: Mapping[str, object],
    *,
    max_steps: Optional[int] = None,
) -> tuple[BattleRulesConfig, BattleRewardConfig]:
    raw_rules = training_config.get("rules", {})
    raw_rewards = training_config.get("rewards", {})
    if not isinstance(raw_rules, Mapping) or not isinstance(raw_rewards, Mapping):
        raise ValueError("Training checkpoint rules or rewards are malformed.")
    rule_values = dict(raw_rules)
    if max_steps is not None:
        rule_values["max_steps"] = int(max_steps)
    rules = BattleRulesConfig(**rule_values) if rule_values else BattleRulesConfig(
        max_steps=500 if max_steps is None else int(max_steps)
    )
    rewards = (
        BattleRewardConfig(**dict(raw_rewards))
        if raw_rewards
        else BattleRewardConfig()
    )
    return rules, rewards


def resolve_pool_checkpoint(
    descriptor: OpponentDescriptor,
    *,
    training_checkpoint: str | Path,
) -> Path:
    """Resolve retained snapshots after a run directory has been moved."""

    if descriptor.checkpoint is None:
        raise ValueError(f"Pool entry {descriptor.identifier!r} has no checkpoint path.")
    source = Path(descriptor.checkpoint).expanduser()
    trainer_dir = Path(training_checkpoint).resolve().parent
    candidates = [source] if source.is_absolute() else [trainer_dir / source]
    candidates.extend(
        [
            trainer_dir / "opponent_pool" / source.name,
            trainer_dir / source.name,
        ]
    )
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and resolved.is_file():
            return resolved
        seen.add(resolved)
    attempted = ", ".join(str(value) for value in seen)
    raise FileNotFoundError(
        f"Retained pool snapshot {descriptor.identifier!r} is missing; checked: {attempted}"
    )


__all__ = [
    "DEFAULT_BATTLE_GATE_WIN_RATES",
    "TrainingCheckpointView",
    "load_evaluation_policy",
    "load_training_checkpoint_view",
    "make_env_factory",
    "make_opponent_policy",
    "nonnegative_float",
    "nonnegative_int",
    "positive_even_int",
    "positive_int",
    "policy_checkpoint_metadata",
    "resolve_pool_checkpoint",
    "rules_and_rewards_from_training_config",
    "unit_interval",
    "write_json_report",
]
