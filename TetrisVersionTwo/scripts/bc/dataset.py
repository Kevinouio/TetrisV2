from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from .utils import load_json


def load_metadata(data_dir: Path) -> Dict[str, object]:
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    return load_json(meta_path)


class BCDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str):
        self.data_dir = Path(data_dir)
        self.split = split
        self.metadata = load_metadata(self.data_dir)

        splits = self.metadata.get("splits")
        if not isinstance(splits, dict) or split not in splits:
            raise KeyError(f"Split '{split}' not found in metadata.")
        split_meta = splits[split]
        if not isinstance(split_meta, dict):
            raise ValueError(f"Invalid metadata for split '{split}'.")
        shards = split_meta.get("shards", [])
        if not isinstance(shards, list):
            raise ValueError(f"Invalid shard list for split '{split}'.")
        if not shards:
            raise ValueError(f"No shards found for split '{split}'.")

        payloads: List[Dict[str, torch.Tensor]] = []
        for rel_path in shards:
            shard_path = self.data_dir / str(rel_path)
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing shard: {shard_path}")
            shard = torch.load(shard_path, map_location="cpu")
            if not isinstance(shard, dict):
                raise ValueError(f"Invalid shard payload: {shard_path}")
            payloads.append(shard)

        def cat(field: str) -> torch.Tensor:
            tensors = [p[field] for p in payloads]
            return torch.cat(tensors, dim=0)

        self.board = cat("board").float()
        self.piece = cat("piece").float()
        self.hold = cat("hold").float()
        self.queue = cat("queue").float()
        self.scalars = cat("scalars").float()
        self.action_id = cat("action_id").long()
        self.action_tuple = cat("action_tuple").long()
        self.episode_id = cat("episode_id").long()
        self.step_idx = cat("step_idx").long()

        queue_flat = self.queue.flatten(start_dim=1)
        if self.scalars.shape[1] > 0:
            self.aux = torch.cat([self.piece, self.hold, queue_flat, self.scalars], dim=1)
        else:
            self.aux = torch.cat([self.piece, self.hold, queue_flat], dim=1)
        self.action_vocab_size = int(self.metadata.get("action_vocab_size", 0))
        self.aux_dim = int(self.aux.shape[1])

    def __len__(self) -> int:
        return int(self.action_id.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "board": self.board[idx],
            "aux": self.aux[idx],
            "action_id": self.action_id[idx],
            "action_tuple": self.action_tuple[idx],
            "episode_id": self.episode_id[idx],
            "step_idx": self.step_idx[idx],
        }


def class_histogram(action_ids: torch.Tensor, top_k: int = 20) -> List[tuple[int, int]]:
    counts = Counter(int(v) for v in action_ids.tolist())
    return counts.most_common(top_k)

