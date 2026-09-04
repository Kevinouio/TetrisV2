"""Bit-packed replay storage for battle DQN transitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from tetris_v2.rl.actions import PLACEMENT_ACTION_DIM
from tetris_v2.rl.battle.dqn import (
    BATTLE_OBSERVATION_DIM,
    BATTLE_OBSERVATION_SCHEMA,
)


PACKED_ACTION_MASK_BYTES = (PLACEMENT_ACTION_DIM + 7) // 8
REPLAY_FORMAT_VERSION = 1


class PackedBattleReplayBuffer:
    """Fixed-capacity replay with packed current and next legal-action masks."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int = BATTLE_OBSERVATION_DIM,
        action_dim: int = PLACEMENT_ACTION_DIM,
        seed: int = 0,
    ):
        if int(capacity) <= 0:
            raise ValueError("Replay capacity must be positive.")
        if int(obs_dim) != BATTLE_OBSERVATION_DIM:
            raise ValueError(
                f"Battle replay requires obs_dim={BATTLE_OBSERVATION_DIM}."
            )
        if int(action_dim) != PLACEMENT_ACTION_DIM:
            raise ValueError(
                f"Battle replay requires action_dim={PLACEMENT_ACTION_DIM}."
            )
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=np.uint8)
        self.truncated = np.zeros(self.capacity, dtype=np.uint8)
        self.action_masks = np.zeros(
            (self.capacity, PACKED_ACTION_MASK_BYTES),
            dtype=np.uint8,
        )
        self.next_action_masks = np.zeros_like(self.action_masks)
        self.size = 0
        self.pos = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    @staticmethod
    def _pack(mask: np.ndarray) -> np.ndarray:
        return np.packbits(np.asarray(mask) > 0.5, bitorder="little")

    @staticmethod
    def _unpack(packed: np.ndarray) -> np.ndarray:
        return np.unpackbits(
            packed,
            axis=-1,
            count=PLACEMENT_ACTION_DIM,
            bitorder="little",
        ).astype(np.uint8, copy=False)

    def _mask(self, value: np.ndarray, *, allow_empty: bool) -> np.ndarray:
        mask = np.asarray(value)
        if mask.shape != (self.action_dim,):
            raise ValueError(
                f"Expected action mask shape ({self.action_dim},), got {mask.shape}."
            )
        if not allow_empty and not np.any(mask > 0.5):
            raise ValueError("A nonterminal transition needs at least one legal action.")
        return mask

    def add(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        next_action_mask: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        current_obs = np.asarray(obs, dtype=np.float32)
        following_obs = np.asarray(next_obs, dtype=np.float32)
        if current_obs.shape != (self.obs_dim,) or following_obs.shape != (self.obs_dim,):
            raise ValueError(f"Battle replay observations must have shape ({self.obs_dim},).")
        current_mask = self._mask(action_mask, allow_empty=False)
        terminal = bool(terminated)
        following_mask = self._mask(next_action_mask, allow_empty=terminal)
        selected = int(action)
        if not 0 <= selected < self.action_dim or current_mask[selected] <= 0.5:
            raise ValueError("Replay transition contains an illegal executed action.")

        index = self.pos
        self.obs[index] = current_obs
        self.action_masks[index] = self._pack(current_mask)
        self.actions[index] = selected
        self.rewards[index] = float(reward)
        self.next_obs[index] = following_obs
        self.next_action_masks[index] = self._pack(following_mask)
        self.terminated[index] = terminal
        self.truncated[index] = bool(truncated)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        generator = rng or self.rng
        indices = generator.integers(0, self.size, size=int(batch_size))
        return {
            "obs": self.obs[indices].copy(),
            "action_masks": self._unpack(self.action_masks[indices]),
            "actions": self.actions[indices].copy(),
            "rewards": self.rewards[indices].copy(),
            "next_obs": self.next_obs[indices].copy(),
            "next_action_masks": self._unpack(self.next_action_masks[indices]),
            "terminated": self.terminated[indices].astype(np.float32),
            "truncated": self.truncated[indices].astype(np.float32),
        }

    def transition(self, index: int) -> Dict[str, np.ndarray | int | float | bool]:
        """Return one stored transition in logical ring-buffer index order."""

        if not 0 <= int(index) < self.size:
            raise IndexError(index)
        physical = (self.pos - self.size + int(index)) % self.capacity
        return {
            "obs": self.obs[physical].copy(),
            "action_mask": self._unpack(self.action_masks[physical]).copy(),
            "action": int(self.actions[physical]),
            "reward": float(self.rewards[physical]),
            "next_obs": self.next_obs[physical].copy(),
            "next_action_mask": self._unpack(self.next_action_masks[physical]).copy(),
            "terminated": bool(self.terminated[physical]),
            "truncated": bool(self.truncated[physical]),
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": REPLAY_FORMAT_VERSION,
            "observation_schema": BATTLE_OBSERVATION_SCHEMA,
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "size": self.size,
            "pos": self.pos,
            "rng_state": self.rng.bit_generator.state,
        }
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                metadata=np.asarray(json.dumps(metadata)),
                obs=self.obs,
                next_obs=self.next_obs,
                actions=self.actions,
                rewards=self.rewards,
                terminated=self.terminated,
                truncated=self.truncated,
                action_masks=self.action_masks,
                next_action_masks=self.next_action_masks,
            )
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "PackedBattleReplayBuffer":
        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if int(metadata.get("format_version", -1)) != REPLAY_FORMAT_VERSION:
                raise ValueError("Unsupported battle replay format version.")
            if metadata.get("observation_schema") != BATTLE_OBSERVATION_SCHEMA:
                raise ValueError("Battle replay observation schema is incompatible.")
            replay = cls(
                int(metadata["capacity"]),
                obs_dim=int(metadata["obs_dim"]),
                action_dim=int(metadata["action_dim"]),
                seed=0,
            )
            arrays = {
                "obs": replay.obs,
                "next_obs": replay.next_obs,
                "actions": replay.actions,
                "rewards": replay.rewards,
                "terminated": replay.terminated,
                "truncated": replay.truncated,
                "action_masks": replay.action_masks,
                "next_action_masks": replay.next_action_masks,
            }
            for name, expected in arrays.items():
                value = archive[name]
                if value.shape != expected.shape or value.dtype != expected.dtype:
                    raise ValueError(f"Battle replay array {name!r} is incompatible.")
                expected[...] = value

        replay.size = int(metadata["size"])
        replay.pos = int(metadata["pos"])
        if not 0 <= replay.size <= replay.capacity or not 0 <= replay.pos < replay.capacity:
            raise ValueError("Battle replay ring metadata is invalid.")
        replay.rng.bit_generator.state = metadata["rng_state"]
        return replay


__all__ = [
    "PACKED_ACTION_MASK_BYTES",
    "PackedBattleReplayBuffer",
    "REPLAY_FORMAT_VERSION",
]
