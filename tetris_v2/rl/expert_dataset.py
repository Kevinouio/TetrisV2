"""Expert dataset shard I/O for TetrisV2 RL."""

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

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        if len(self) <= 0:
            raise ValueError("Cannot sample from empty expert dataset.")
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")
        idx = rng.integers(0, len(self), size=int(batch_size))
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


def _as_array(records: Sequence[dict], key: str, dtype: np.dtype) -> np.ndarray:
    return np.asarray([r[key] for r in records], dtype=dtype)


def write_shard(path: Path, records: Sequence[dict]) -> None:
    if not records:
        raise ValueError("Cannot write empty expert shard.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=_as_array(records, "obs", np.float32),
        action_mask=_as_array(records, "action_mask", np.uint8),
        teacher_best_action=_as_array(records, "teacher_best_action", np.int64),
        seed=_as_array(records, "seed", np.int64),
        episode=_as_array(records, "episode", np.int64),
        step=_as_array(records, "step", np.int64),
        legal_action_count=_as_array(records, "legal_action_count", np.int64),
        placement_count_raw=_as_array(records, "placement_count_raw", np.int64),
        placement_overflow=_as_array(records, "placement_overflow", np.uint8),
        nodes=_as_array(records, "nodes", np.int64),
        think_ms=_as_array(records, "think_ms", np.float32),
        budget_miss=_as_array(records, "budget_miss", np.int64),
        unexpanded_count=_as_array(records, "unexpanded_count", np.int64),
    )


def load_shard(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k] for k in REQUIRED_FIELDS}
        out["unexpanded_count"] = (
            data["unexpanded_count"]
            if "unexpanded_count" in data
            else np.zeros(out["step"].shape, dtype=np.int64)
        )

    sample_count = out["obs"].shape[0]
    if any(values.ndim == 0 or values.shape[0] != sample_count for values in out.values()):
        raise ValueError(f"Inconsistent sample counts in {path}")
    if out["action_mask"].ndim != 2 or out["action_mask"].shape[1] != PLACEMENT_ACTION_DIM:
        raise ValueError(f"Invalid action_mask dim in {path}: {out['action_mask'].shape}")
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


def write_manifest(path: Path, *, shards: Sequence[Path], total_samples: int) -> None:
    manifest_dir = path.parent.resolve()
    payload = {
        "version": 2,
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


__all__ = [
    "ExpertDataset",
    "discover_shards",
    "load_dataset",
    "load_dataset_directory",
    "load_shard",
    "write_manifest",
    "write_shard",
]
