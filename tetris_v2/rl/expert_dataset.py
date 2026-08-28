"""Expert-label and transition dataset shard I/O for TetrisV2 RL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from tetris_v2.rl.actions import PLACEMENT_ACTION_DIM


REQUIRED_FIELDS = (
    "obs",
    "action_mask",
    "teacher_best_action",
    "seed",
    "episode",
    "step",
    "legal_action_count",
    "placement_count_raw",
    "placement_overflow",
    "nodes",
    "think_ms",
    "budget_miss",
)
DATASET_FIELDS = REQUIRED_FIELDS + ("unexpanded_count",)
TRANSITION_FIELDS = (
    "executed_action",
    "reward",
    "raw_reward",
    "next_obs",
    "next_action_mask",
    "terminated",
    "truncated",
)

TRANSITION_REGENERATION_MESSAGE = (
    "Flow-DQN requires transition dataset schema v3, but this dataset contains only "
    "schema-v2 expert labels. Regenerate it with tetris-generate-expert-data so each "
    "sample includes the executed action, reward, next state, and episode boundaries."
)


@dataclass
class ExpertDataset:
    obs: np.ndarray
    action_mask: np.ndarray
    teacher_best_action: np.ndarray
    seed: np.ndarray
    episode: np.ndarray
    step: np.ndarray
    legal_action_count: np.ndarray
    placement_count_raw: np.ndarray
    placement_overflow: np.ndarray
    nodes: np.ndarray
    think_ms: np.ndarray
    budget_miss: np.ndarray
    unexpanded_count: np.ndarray

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def _sample_indices(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        if len(self) <= 0:
            raise ValueError("Cannot sample from empty expert dataset.")
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")
        return rng.integers(0, len(self), size=int(batch_size))

    def _sample_at(self, idx: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "obs": self.obs[idx],
            "action_mask": self.action_mask[idx],
            "teacher_best_action": self.teacher_best_action[idx],
            "seed": self.seed[idx],
            "episode": self.episode[idx],
            "step": self.step[idx],
            "legal_action_count": self.legal_action_count[idx],
            "placement_count_raw": self.placement_count_raw[idx],
            "placement_overflow": self.placement_overflow[idx],
            "nodes": self.nodes[idx],
            "think_ms": self.think_ms[idx],
            "budget_miss": self.budget_miss[idx],
            "unexpanded_count": self.unexpanded_count[idx],
        }

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        return self._sample_at(self._sample_indices(batch_size, rng))


@dataclass
class ExpertTransitionDataset(ExpertDataset):
    """Schema-v3 expert data containing complete Bellman transitions."""

    executed_action: np.ndarray
    reward: np.ndarray
    raw_reward: np.ndarray
    next_obs: np.ndarray
    next_action_mask: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        idx = self._sample_indices(batch_size, rng)
        batch = self._sample_at(idx)
        batch.update(
            {
                "executed_action": self.executed_action[idx],
                "reward": self.reward[idx],
                "raw_reward": self.raw_reward[idx],
                "next_obs": self.next_obs[idx],
                "next_action_mask": self.next_action_mask[idx],
                "terminated": self.terminated[idx],
                "truncated": self.truncated[idx],
            }
        )
        return batch


def _as_array(records: Sequence[dict], key: str, dtype: np.dtype) -> np.ndarray:
    return np.asarray([r[key] for r in records], dtype=dtype)


def write_shard(path: Path, records: Sequence[dict]) -> None:
    if not records:
        raise ValueError("Cannot write empty expert shard.")

    has_transition = any(key in record for record in records for key in TRANSITION_FIELDS)
    if has_transition:
        for index, record in enumerate(records):
            missing = [key for key in TRANSITION_FIELDS if key not in record]
            if missing:
                raise ValueError(
                    f"Transition record {index} is missing required fields: {', '.join(missing)}"
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "obs": _as_array(records, "obs", np.float32),
        "action_mask": _as_array(records, "action_mask", np.uint8),
        "teacher_best_action": _as_array(records, "teacher_best_action", np.int64),
        "seed": _as_array(records, "seed", np.int64),
        "episode": _as_array(records, "episode", np.int64),
        "step": _as_array(records, "step", np.int64),
        "legal_action_count": _as_array(records, "legal_action_count", np.int64),
        "placement_count_raw": _as_array(records, "placement_count_raw", np.int64),
        "placement_overflow": _as_array(records, "placement_overflow", np.uint8),
        "nodes": _as_array(records, "nodes", np.int64),
        "think_ms": _as_array(records, "think_ms", np.float32),
        "budget_miss": _as_array(records, "budget_miss", np.int64),
        "unexpanded_count": _as_array(records, "unexpanded_count", np.int64),
    }
    if has_transition:
        arrays.update(
            {
                "executed_action": _as_array(records, "executed_action", np.int64),
                "reward": _as_array(records, "reward", np.float32),
                "raw_reward": _as_array(records, "raw_reward", np.float32),
                "next_obs": _as_array(records, "next_obs", np.float32),
                "next_action_mask": _as_array(records, "next_action_mask", np.uint8),
                "terminated": _as_array(records, "terminated", np.uint8),
                "truncated": _as_array(records, "truncated", np.uint8),
            }
        )
    np.savez_compressed(path, **arrays)


def _validate_sample_counts(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    sample_count = arrays["obs"].shape[0]
    if any(values.ndim == 0 or values.shape[0] != sample_count for values in arrays.values()):
        raise ValueError(f"Inconsistent sample counts in {path}")


def load_shard(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k] for k in REQUIRED_FIELDS}
        out["unexpanded_count"] = (
            data["unexpanded_count"]
            if "unexpanded_count" in data
            else np.zeros(out["step"].shape, dtype=np.int64)
        )

    _validate_sample_counts(path, out)
    if out["action_mask"].ndim != 2 or out["action_mask"].shape[1] != PLACEMENT_ACTION_DIM:
        raise ValueError(f"Invalid action_mask dim in {path}: {out['action_mask'].shape}")
    return out


def load_transition_shard(path: Path) -> Dict[str, np.ndarray]:
    """Load one schema-v3 shard, rejecting label-only schema-v2 shards."""

    out = load_shard(path)
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in TRANSITION_FIELDS if key not in data]
        if missing:
            raise ValueError(f"{TRANSITION_REGENERATION_MESSAGE} Shard: {path}")
        out.update({key: data[key] for key in TRANSITION_FIELDS})

    _validate_sample_counts(path, out)
    if out["next_obs"].shape != out["obs"].shape:
        raise ValueError(
            f"next_obs shape does not match obs in {path}: "
            f"{out['next_obs'].shape} != {out['obs'].shape}"
        )
    if out["next_action_mask"].shape != out["action_mask"].shape:
        raise ValueError(
            f"next_action_mask shape does not match action_mask in {path}: "
            f"{out['next_action_mask'].shape} != {out['action_mask'].shape}"
        )
    return out


def load_dataset(paths: Iterable[Path]) -> ExpertDataset:
    arrays: Dict[str, List[np.ndarray]] = {k: [] for k in DATASET_FIELDS}
    for path in paths:
        shard = load_shard(path)
        for key in DATASET_FIELDS:
            arrays[key].append(np.asarray(shard[key]))
    if not arrays["obs"]:
        raise ValueError("No expert dataset shards were loaded.")

    stacked = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    return ExpertDataset(
        obs=stacked["obs"].astype(np.float32, copy=False),
        action_mask=stacked["action_mask"].astype(np.uint8, copy=False),
        teacher_best_action=stacked["teacher_best_action"].astype(np.int64, copy=False),
        seed=stacked["seed"].astype(np.int64, copy=False),
        episode=stacked["episode"].astype(np.int64, copy=False),
        step=stacked["step"].astype(np.int64, copy=False),
        legal_action_count=stacked["legal_action_count"].astype(np.int64, copy=False),
        placement_count_raw=stacked["placement_count_raw"].astype(np.int64, copy=False),
        placement_overflow=stacked["placement_overflow"].astype(np.int64, copy=False),
        nodes=stacked["nodes"].astype(np.int64, copy=False),
        think_ms=stacked["think_ms"].astype(np.float32, copy=False),
        budget_miss=stacked["budget_miss"].astype(np.int64, copy=False),
        unexpanded_count=stacked["unexpanded_count"].astype(np.int64, copy=False),
    )


def load_transition_dataset(paths: Iterable[Path]) -> ExpertTransitionDataset:
    fields = DATASET_FIELDS + TRANSITION_FIELDS
    arrays: Dict[str, List[np.ndarray]] = {key: [] for key in fields}
    for path in paths:
        shard = load_transition_shard(path)
        for key in fields:
            arrays[key].append(np.asarray(shard[key]))
    if not arrays["obs"]:
        raise ValueError("No expert transition dataset shards were loaded.")

    stacked = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    return ExpertTransitionDataset(
        obs=stacked["obs"].astype(np.float32, copy=False),
        action_mask=stacked["action_mask"].astype(np.uint8, copy=False),
        teacher_best_action=stacked["teacher_best_action"].astype(np.int64, copy=False),
        seed=stacked["seed"].astype(np.int64, copy=False),
        episode=stacked["episode"].astype(np.int64, copy=False),
        step=stacked["step"].astype(np.int64, copy=False),
        legal_action_count=stacked["legal_action_count"].astype(np.int64, copy=False),
        placement_count_raw=stacked["placement_count_raw"].astype(np.int64, copy=False),
        placement_overflow=stacked["placement_overflow"].astype(np.int64, copy=False),
        nodes=stacked["nodes"].astype(np.int64, copy=False),
        think_ms=stacked["think_ms"].astype(np.float32, copy=False),
        budget_miss=stacked["budget_miss"].astype(np.int64, copy=False),
        unexpanded_count=stacked["unexpanded_count"].astype(np.int64, copy=False),
        executed_action=stacked["executed_action"].astype(np.int64, copy=False),
        reward=stacked["reward"].astype(np.float32, copy=False),
        raw_reward=stacked["raw_reward"].astype(np.float32, copy=False),
        next_obs=stacked["next_obs"].astype(np.float32, copy=False),
        next_action_mask=stacked["next_action_mask"].astype(np.uint8, copy=False),
        terminated=stacked["terminated"].astype(np.uint8, copy=False),
        truncated=stacked["truncated"].astype(np.uint8, copy=False),
    )


def write_manifest(
    path: Path,
    *,
    shards: Sequence[Path],
    total_samples: int,
    version: int = 2,
) -> None:
    manifest_dir = path.parent.resolve()
    payload = {
        "version": int(version),
        "total_samples": int(total_samples),
        "shards": [str(p.resolve().relative_to(manifest_dir)) for p in shards],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def discover_shards(dataset_dir: Path) -> List[Path]:
    manifest = dataset_dir / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        listed = [Path(p) for p in payload.get("shards", [])]
        if not listed:
            raise ValueError(f"Manifest lists no shards: {manifest}")
        resolved = [p if p.is_absolute() else (dataset_dir / p) for p in listed]
        missing = [p for p in resolved if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Manifest lists missing shard: {missing[0]}")
        return resolved
    return sorted(dataset_dir.glob("expert_shard_*.npz"))


def load_dataset_directory(dataset_dir: Path) -> ExpertDataset:
    dataset = load_dataset(discover_shards(dataset_dir))
    manifest = dataset_dir / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        expected = int(payload["total_samples"])
        if len(dataset) != expected:
            raise ValueError(
                f"Manifest expects {expected} samples, but its shards contain {len(dataset)}"
            )
    return dataset


def load_transition_dataset_directory(dataset_dir: Path) -> ExpertTransitionDataset:
    manifest = dataset_dir / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if int(payload.get("version", 2)) < 3:
            raise ValueError(f"{TRANSITION_REGENERATION_MESSAGE} Manifest: {manifest}")

    dataset = load_transition_dataset(discover_shards(dataset_dir))
    if manifest.exists():
        expected = int(payload["total_samples"])
        if len(dataset) != expected:
            raise ValueError(
                f"Manifest expects {expected} samples, but its shards contain {len(dataset)}"
            )
    return dataset


__all__ = [
    "ExpertDataset",
    "ExpertTransitionDataset",
    "TRANSITION_FIELDS",
    "TRANSITION_REGENERATION_MESSAGE",
    "discover_shards",
    "load_dataset",
    "load_dataset_directory",
    "load_shard",
    "load_transition_dataset",
    "load_transition_dataset_directory",
    "load_transition_shard",
    "write_manifest",
    "write_shard",
]
