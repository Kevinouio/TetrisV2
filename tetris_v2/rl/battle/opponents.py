"""Deterministic, bounded opponent-pool metadata for battle self-play."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Mapping, Optional, Sequence

import numpy as np


OpponentKind = Literal["random", "cold_clear", "checkpoint", "current"]
_SPECIAL_KINDS: tuple[OpponentKind, ...] = ("random", "cold_clear", "current")
_MIX_ALIASES = {"heuristic": "cold_clear", "checkpoint": "frozen"}


@dataclass(frozen=True)
class OpponentDescriptor:
    """Serializable identity for one opponent policy implementation."""

    identifier: str
    kind: OpponentKind
    checkpoint: Optional[str] = None
    generation: int = 0
    created_step: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("Opponent identifier cannot be empty.")
        if self.kind not in {"random", "cold_clear", "checkpoint", "current"}:
            raise ValueError(f"Unsupported opponent kind: {self.kind!r}")
        if self.kind == "checkpoint" and not self.checkpoint:
            raise ValueError("Frozen checkpoint opponents require a checkpoint path.")
        if self.generation < 0 or self.created_step < 0:
            raise ValueError("Opponent generation and creation step cannot be negative.")

    def to_dict(self) -> Dict[str, object]:
        return {
            "identifier": self.identifier,
            "kind": self.kind,
            "checkpoint": self.checkpoint,
            "generation": self.generation,
            "created_step": self.created_step,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, object]) -> "OpponentDescriptor":
        metadata = state.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("Opponent metadata must be a mapping.")
        return cls(
            identifier=str(state["identifier"]),
            kind=str(state["kind"]),  # type: ignore[arg-type]
            checkpoint=(
                None if state.get("checkpoint") is None else str(state["checkpoint"])
            ),
            generation=int(state.get("generation", 0)),
            created_step=int(state.get("created_step", 0)),
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class OpponentSelection:
    descriptor: OpponentDescriptor
    category: str
    frozen_bucket: Optional[str]
    frozen_pool_size: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "descriptor": self.descriptor.to_dict(),
            "category": self.category,
            "frozen_bucket": self.frozen_bucket,
            "frozen_pool_size": self.frozen_pool_size,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, object]) -> "OpponentSelection":
        descriptor = state.get("descriptor")
        if not isinstance(descriptor, Mapping):
            raise ValueError("Opponent selection descriptor is malformed.")
        return cls(
            descriptor=OpponentDescriptor.from_dict(descriptor),
            category=str(state["category"]),
            frozen_bucket=(
                None
                if state.get("frozen_bucket") is None
                else str(state["frozen_bucket"])
            ),
            frozen_pool_size=int(state.get("frozen_pool_size", 0)),
        )


class OpponentPool:
    """Bounded frozen snapshots plus seeded recent/older sampling."""

    def __init__(
        self,
        max_frozen: int,
        seed: int = 0,
        *,
        recent_window: int = 4,
        recent_probability: float = 0.5,
    ):
        if int(max_frozen) < 2:
            raise ValueError("max_frozen must be at least 2 to retain initial and newest snapshots.")
        if int(recent_window) <= 0:
            raise ValueError("recent_window must be positive.")
        if not 0.0 <= float(recent_probability) <= 1.0:
            raise ValueError("recent_probability must be in [0, 1].")
        self.max_frozen = int(max_frozen)
        self.recent_window = int(recent_window)
        self.recent_probability = float(recent_probability)
        self.rng = np.random.default_rng(seed)
        self._special: Dict[str, OpponentDescriptor] = {
            "random": OpponentDescriptor("random", "random"),
            "cold_clear": OpponentDescriptor("cold_clear", "cold_clear"),
            "current": OpponentDescriptor("current", "current"),
        }
        self._frozen: list[OpponentDescriptor] = []
        self._embedded_checkpoints: Dict[str, Mapping[str, object]] = {}
        self.last_selection: Optional[OpponentSelection] = None

    @staticmethod
    def _sort_key(descriptor: OpponentDescriptor) -> tuple[int, int, str]:
        return (descriptor.generation, descriptor.created_step, descriptor.identifier)

    @property
    def frozen(self) -> tuple[OpponentDescriptor, ...]:
        return tuple(sorted(self._frozen, key=self._sort_key))

    def descriptors(self) -> tuple[OpponentDescriptor, ...]:
        special = tuple(self._special[kind] for kind in _SPECIAL_KINDS)
        return special + self.frozen

    def add(self, descriptor: OpponentDescriptor) -> tuple[OpponentDescriptor, ...]:
        """Add/replace an opponent and return frozen descriptors evicted from the pool."""

        if descriptor.kind != "checkpoint":
            self._special[descriptor.kind] = descriptor
            return ()
        self._frozen = [
            existing
            for existing in self._frozen
            if existing.identifier != descriptor.identifier
        ]
        self._embedded_checkpoints.pop(descriptor.identifier, None)
        self._frozen.append(descriptor)
        evicted: list[OpponentDescriptor] = []
        while len(self._frozen) > self.max_frozen:
            removed = self._select_eviction()
            self._frozen.remove(removed)
            self._embedded_checkpoints.pop(removed.identifier, None)
            evicted.append(removed)
        return tuple(evicted)

    def set_embedded_checkpoint(
        self,
        identifier: str,
        payload: Mapping[str, object],
    ) -> None:
        if identifier not in {item.identifier for item in self._frozen}:
            raise ValueError("Embedded checkpoint must belong to a retained opponent.")
        self._embedded_checkpoints[str(identifier)] = payload

    def embedded_checkpoint(
        self,
        identifier: str,
    ) -> Mapping[str, object] | None:
        return self._embedded_checkpoints.get(str(identifier))

    def _select_eviction(self) -> OpponentDescriptor:
        ordered = sorted(self._frozen, key=self._sort_key)
        if len(ordered) <= 2:
            raise RuntimeError("Cannot evict while preserving initial and newest snapshots.")

        generations = np.asarray([item.generation for item in ordered], dtype=np.float64)
        if np.unique(generations).size != generations.size:
            generations = np.asarray([item.created_step for item in ordered], dtype=np.float64)
        if np.unique(generations).size != generations.size:
            generations = np.arange(len(ordered), dtype=np.float64)

        center = (len(ordered) - 1) / 2.0
        candidates: list[tuple[tuple[float, float, float, float, int], OpponentDescriptor]] = []
        for index in range(1, len(ordered) - 1):
            remaining = np.delete(generations, index)
            gaps = np.diff(remaining)
            score = (
                float(np.min(gaps)),
                -float(np.max(gaps)),
                -float(np.var(gaps)),
                -abs(index - center),
                -index,
            )
            candidates.append((score, ordered[index]))
        return max(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _normalized_mix(mix: Mapping[str, float]) -> Dict[str, float]:
        normalized: Dict[str, float] = {}
        for raw_name, raw_weight in mix.items():
            name = _MIX_ALIASES.get(str(raw_name), str(raw_name))
            if name not in {"random", "cold_clear", "frozen", "current"}:
                raise ValueError(f"Unknown opponent category: {raw_name!r}")
            weight = float(raw_weight)
            if weight < 0:
                raise ValueError("Opponent mix weights cannot be negative.")
            normalized[name] = normalized.get(name, 0.0) + weight
        return normalized

    def sample_selection(self, mix: Mapping[str, float]) -> OpponentSelection:
        weights = self._normalized_mix(mix)
        available = {
            name: weight
            for name, weight in weights.items()
            if weight > 0 and (name != "frozen" or bool(self._frozen))
        }
        total = sum(available.values())
        if total <= 0:
            raise ValueError("Opponent mix has no available positive-weight category.")
        names = tuple(available)
        probabilities = np.asarray([available[name] / total for name in names])
        category = str(self.rng.choice(names, p=probabilities))

        bucket: Optional[str] = None
        if category != "frozen":
            descriptor = self._special[category]
        else:
            ordered = list(self.frozen)
            split = max(0, len(ordered) - self.recent_window)
            older = ordered[:split]
            recent = ordered[split:]
            if older and recent:
                use_recent = self.rng.random() < self.recent_probability
                candidates = recent if use_recent else older
                bucket = "recent" if use_recent else "older"
            elif recent:
                candidates = recent
                bucket = "recent"
            else:
                candidates = older
                bucket = "older"
            descriptor = candidates[int(self.rng.integers(0, len(candidates)))]

        selection = OpponentSelection(
            descriptor=descriptor,
            category=category,
            frozen_bucket=bucket,
            frozen_pool_size=len(self._frozen),
        )
        self.last_selection = selection
        return selection

    def sample(self, mix: Mapping[str, float]) -> OpponentDescriptor:
        return self.sample_selection(mix).descriptor

    def state_dict(self) -> Dict[str, object]:
        return {
            "format_version": 1,
            "max_frozen": self.max_frozen,
            "recent_window": self.recent_window,
            "recent_probability": self.recent_probability,
            "special": {
                kind: descriptor.to_dict()
                for kind, descriptor in self._special.items()
            },
            "frozen": [descriptor.to_dict() for descriptor in self.frozen],
            "embedded_checkpoints": dict(self._embedded_checkpoints),
            "rng_state": self.rng.bit_generator.state,
            "last_selection": (
                None if self.last_selection is None else self.last_selection.to_dict()
            ),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "OpponentPool":
        if int(state.get("format_version", -1)) != 1:
            raise ValueError("Unsupported opponent-pool format version.")
        pool = cls(
            max_frozen=int(state["max_frozen"]),
            seed=0,
            recent_window=int(state["recent_window"]),
            recent_probability=float(state["recent_probability"]),
        )
        special = state.get("special")
        frozen = state.get("frozen")
        if not isinstance(special, Mapping) or not isinstance(frozen, Sequence):
            raise ValueError("Opponent-pool descriptors are malformed.")
        pool._special = {
            str(kind): OpponentDescriptor.from_dict(value)
            for kind, value in special.items()
            if isinstance(value, Mapping)
        }
        if set(pool._special) != set(_SPECIAL_KINDS):
            raise ValueError("Opponent pool is missing a special policy descriptor.")
        pool._frozen = [
            OpponentDescriptor.from_dict(value)
            for value in frozen
            if isinstance(value, Mapping)
        ]
        if len(pool._frozen) != len(frozen) or len(pool._frozen) > pool.max_frozen:
            raise ValueError("Opponent pool contains invalid frozen descriptors.")
        embedded = state.get("embedded_checkpoints", {})
        if not isinstance(embedded, Mapping):
            raise ValueError("Opponent-pool embedded checkpoints are malformed.")
        retained_ids = {item.identifier for item in pool._frozen}
        for identifier, payload in embedded.items():
            if str(identifier) not in retained_ids or not isinstance(payload, Mapping):
                raise ValueError("Opponent-pool embedded checkpoint is invalid.")
            pool._embedded_checkpoints[str(identifier)] = payload
        pool.rng.bit_generator.state = state["rng_state"]  # type: ignore[assignment]
        last = state.get("last_selection")
        pool.last_selection = (
            OpponentSelection.from_dict(last) if isinstance(last, Mapping) else None
        )
        return pool


__all__ = [
    "OpponentDescriptor",
    "OpponentKind",
    "OpponentPool",
    "OpponentSelection",
]
