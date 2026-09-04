"""Episode-boundary training checkpoints for battle DQN."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import Dict, Mapping, Optional

import numpy as np
import torch

from tetris_v2.rl.battle.dqn import (
    BATTLE_ACTION_ORDER,
    BATTLE_OBSERVATION_SCHEMA,
    BattleDQNAgent,
    LinearSchedule,
)
from tetris_v2.rl.battle.opponents import OpponentPool
from tetris_v2.rl.battle.replay import PackedBattleReplayBuffer


BATTLE_TRAINING_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class BattleCheckpointPaths:
    checkpoint: Path
    replay_sidecar: Path


@dataclass
class BattleTrainingBundle:
    agent: BattleDQNAgent
    replay: PackedBattleReplayBuffer
    opponent_pool: OpponentPool
    global_step: int
    episode_index: int
    epsilon_schedule: LinearSchedule
    learning_rate_schedule: LinearSchedule
    training_config: Dict[str, object]
    extra: Dict[str, object]
    checkpoint_path: Path
    replay_sidecar: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rng_state() -> Dict[str, object]:
    cuda_states = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_states = torch.cuda.get_rng_state_all()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
    }


def _restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch_cpu"])  # type: ignore[arg-type]
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)  # type: ignore[arg-type]


def _pool_state_with_embedded_checkpoints(
    opponent_pool: OpponentPool,
) -> Dict[str, object]:
    """Make every retained opponent recoverable from this trainer checkpoint."""

    state = opponent_pool.state_dict()
    embedded: Dict[str, Mapping[str, object]] = {}
    for descriptor in opponent_pool.frozen:
        payload = opponent_pool.embedded_checkpoint(descriptor.identifier)
        if payload is None:
            if descriptor.checkpoint is None:
                raise ValueError(
                    f"Frozen opponent {descriptor.identifier!r} has no checkpoint path."
                )
            source = Path(descriptor.checkpoint)
            if not source.is_file():
                raise FileNotFoundError(
                    f"Frozen opponent checkpoint is missing: {source}"
                )
            loaded = torch.load(source, map_location="cpu", weights_only=False)
            if not isinstance(loaded, Mapping):
                raise ValueError(
                    f"Frozen opponent checkpoint is malformed: {source}"
                )
            payload = loaded
        if payload.get("algo") != "battle_dqn" or payload.get(
            "checkpoint_type"
        ) != "frozen_policy":
            raise ValueError(
                f"Frozen opponent {descriptor.identifier!r} is not a compact Battle-DQN policy."
            )
        embedded[descriptor.identifier] = payload
    state["embedded_checkpoints"] = embedded
    return state


def save_battle_training_checkpoint(
    path: str | Path,
    *,
    agent: BattleDQNAgent,
    replay: PackedBattleReplayBuffer,
    opponent_pool: OpponentPool,
    global_step: int,
    episode_index: int,
    epsilon_schedule: LinearSchedule,
    learning_rate_schedule: LinearSchedule,
    training_config: Mapping[str, object],
    at_episode_boundary: bool,
    extra: Optional[Mapping[str, object]] = None,
) -> BattleCheckpointPaths:
    """Atomically publish a trainer checkpoint that references packed replay.

    Mid-episode snapshots are deliberately refused: the battle environment is
    not part of this Python persistence layer, so only an episode boundary can
    be resumed without silently changing a match trajectory.
    """

    if not at_episode_boundary:
        raise ValueError("Battle training checkpoints are only valid at episode boundaries.")
    if int(global_step) < 0 or int(episode_index) < 0:
        raise ValueError("Training counters cannot be negative.")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_replay = destination.parent / f".{destination.name}.replay.tmp.npz"
    replay.save(temporary_replay)
    replay_digest = _sha256(temporary_replay)
    replay_sidecar = destination.parent / (
        f"{destination.name}.replay.{replay_digest}.npz"
    )
    if replay_sidecar.exists():
        if _sha256(replay_sidecar) != replay_digest:
            raise RuntimeError("Replay sidecar hash collision.")
        temporary_replay.unlink()
    else:
        temporary_replay.replace(replay_sidecar)

    pool_state = _pool_state_with_embedded_checkpoints(opponent_pool)
    payload = {
        "format_version": BATTLE_TRAINING_CHECKPOINT_VERSION,
        "algo": "battle_dqn_training",
        "observation_schema": BATTLE_OBSERVATION_SCHEMA,
        "action_order": BATTLE_ACTION_ORDER,
        "episode_boundary": True,
        "agent": agent.state_dict(),
        "replay": {
            "filename": replay_sidecar.name,
            "sha256": replay_digest,
        },
        "opponent_pool": pool_state,
        "counters": {
            "global_step": int(global_step),
            "episode_index": int(episode_index),
        },
        "schedules": {
            "epsilon": epsilon_schedule.state_dict(),
            "learning_rate": learning_rate_schedule.state_dict(),
        },
        "training_config": copy.deepcopy(dict(training_config)),
        "rng": _rng_state(),
        "extra": copy.deepcopy(dict(extra or {})),
    }
    temporary_checkpoint = destination.parent / f".{destination.name}.tmp"
    torch.save(payload, temporary_checkpoint)
    temporary_checkpoint.replace(destination)
    return BattleCheckpointPaths(destination, replay_sidecar)


def load_battle_training_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> BattleTrainingBundle:
    source = Path(path)
    payload = torch.load(source, map_location=device, weights_only=False)
    if (
        payload.get("format_version") != BATTLE_TRAINING_CHECKPOINT_VERSION
        or payload.get("algo") != "battle_dqn_training"
    ):
        raise ValueError("Unsupported battle training checkpoint.")
    if payload.get("observation_schema") != BATTLE_OBSERVATION_SCHEMA:
        raise ValueError("Battle training checkpoint observation schema is incompatible.")
    if payload.get("action_order") != BATTLE_ACTION_ORDER:
        raise ValueError("Battle training checkpoint action order is incompatible.")
    if payload.get("episode_boundary") is not True:
        raise ValueError("Battle training checkpoint was not captured at an episode boundary.")

    replay_state = payload.get("replay")
    if not isinstance(replay_state, Mapping):
        raise ValueError("Battle training checkpoint replay metadata is malformed.")
    replay_sidecar = source.parent / str(replay_state["filename"])
    if not replay_sidecar.is_file():
        raise FileNotFoundError(f"Battle replay sidecar is missing: {replay_sidecar}")
    expected_digest = str(replay_state["sha256"])
    if _sha256(replay_sidecar) != expected_digest:
        raise ValueError("Battle replay sidecar checksum does not match the checkpoint.")

    agent_state = payload.get("agent")
    pool_state = payload.get("opponent_pool")
    schedules = payload.get("schedules")
    counters = payload.get("counters")
    rng = payload.get("rng")
    if not all(
        isinstance(value, Mapping)
        for value in (agent_state, pool_state, schedules, counters, rng)
    ):
        raise ValueError("Battle training checkpoint state is malformed.")

    agent = BattleDQNAgent.from_state_dict(agent_state, device=device)  # type: ignore[arg-type]
    replay = PackedBattleReplayBuffer.load(replay_sidecar)
    opponent_pool = OpponentPool.from_state_dict(pool_state)  # type: ignore[arg-type]
    epsilon = schedules.get("epsilon")  # type: ignore[union-attr]
    learning_rate = schedules.get("learning_rate")  # type: ignore[union-attr]
    if not isinstance(epsilon, Mapping) or not isinstance(learning_rate, Mapping):
        raise ValueError("Battle training checkpoint schedules are malformed.")

    training_config = payload.get("training_config", {})
    extra = payload.get("extra", {})
    if not isinstance(training_config, Mapping) or not isinstance(extra, Mapping):
        raise ValueError("Battle training checkpoint metadata is malformed.")

    # Constructors above consume Torch RNG while allocating model parameters.
    # Restore process-wide streams last so the next training operation is exact.
    _restore_rng_state(rng)  # type: ignore[arg-type]
    return BattleTrainingBundle(
        agent=agent,
        replay=replay,
        opponent_pool=opponent_pool,
        global_step=int(counters["global_step"]),  # type: ignore[index]
        episode_index=int(counters["episode_index"]),  # type: ignore[index]
        epsilon_schedule=LinearSchedule.from_state_dict(epsilon),
        learning_rate_schedule=LinearSchedule.from_state_dict(learning_rate),
        training_config=copy.deepcopy(dict(training_config)),
        extra=copy.deepcopy(dict(extra)),
        checkpoint_path=source,
        replay_sidecar=replay_sidecar,
    )


__all__ = [
    "BATTLE_TRAINING_CHECKPOINT_VERSION",
    "BattleCheckpointPaths",
    "BattleTrainingBundle",
    "load_battle_training_checkpoint",
    "save_battle_training_checkpoint",
]
