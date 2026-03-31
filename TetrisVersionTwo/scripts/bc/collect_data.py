from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .config import CollectionConfig, EncoderConfig, SplitConfig, dataclass_to_dict
from .encoders import encode_state
from .utils import (
    ActionCodec,
    BCEnvAdapter,
    NativeAction,
    chunk_list,
    ensure_dir,
    find_library,
    save_json,
    set_global_seeds,
    split_episode_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect top-1 behavioral cloning data from Cold Clear.")
    parser.add_argument("--lib", type=Path, default=None, help="Path to tetris_v2_c_api shared library.")
    parser.add_argument("--out_dir", type=Path, default=Path("data/bc_top1"))
    parser.add_argument("--num_episodes", type=int, default=CollectionConfig.num_episodes)
    parser.add_argument("--max_steps_per_episode", type=int, default=CollectionConfig.max_steps_per_episode)
    parser.add_argument("--think_ms", type=int, default=CollectionConfig.think_ms)
    parser.add_argument("--seed", type=int, default=CollectionConfig.seed)
    parser.add_argument("--episodes_per_shard", type=int, default=CollectionConfig.episodes_per_shard)

    parser.add_argument("--queue_length", type=int, default=EncoderConfig.queue_length)
    parser.add_argument("--include_scalars", action="store_true", default=False)

    parser.add_argument("--train_fraction", type=float, default=SplitConfig.train_fraction)
    parser.add_argument("--val_fraction", type=float, default=SplitConfig.val_fraction)
    parser.add_argument("--test_fraction", type=float, default=SplitConfig.test_fraction)
    parser.add_argument("--split_seed", type=int, default=SplitConfig.seed)
    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def _build_payload(records: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    board = np.stack([r["board"] for r in records], axis=0).astype(np.float32)
    piece = np.stack([r["piece"] for r in records], axis=0).astype(np.float32)
    hold = np.stack([r["hold"] for r in records], axis=0).astype(np.float32)
    queue = np.stack([r["queue"] for r in records], axis=0).astype(np.float32)
    scalars = np.stack([r["scalars"] for r in records], axis=0).astype(np.float32)
    action_id = np.asarray([r["action_id"] for r in records], dtype=np.int64)
    action_tuple = np.stack([r["action_tuple"] for r in records], axis=0).astype(np.int64)
    episode_id = np.asarray([r["episode_id"] for r in records], dtype=np.int64)
    step_idx = np.asarray([r["step_idx"] for r in records], dtype=np.int64)

    return {
        "board": torch.from_numpy(board),
        "piece": torch.from_numpy(piece),
        "hold": torch.from_numpy(hold),
        "queue": torch.from_numpy(queue),
        "scalars": torch.from_numpy(scalars),
        "action_id": torch.from_numpy(action_id),
        "action_tuple": torch.from_numpy(action_tuple),
        "episode_id": torch.from_numpy(episode_id),
        "step_idx": torch.from_numpy(step_idx),
    }


def main() -> int:
    args = parse_args()
    set_global_seeds(args.seed)

    split_cfg = SplitConfig(
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    split_cfg.validate()

    encoder_cfg = EncoderConfig(
        board_height=20,
        board_width=10,
        queue_length=int(args.queue_length),
        include_scalars=bool(args.include_scalars),
    )

    lib_path = find_library(args.lib)
    out_dir = Path(args.out_dir)
    shard_dir = out_dir / "shards"
    ensure_dir(shard_dir)

    codec = ActionCodec()
    episode_records: Dict[int, List[Dict[str, object]]] = {}
    skipped_no_legal = 0
    skipped_invalid_expert = 0
    skipped_missing_tuple = 0

    with BCEnvAdapter(lib_path=lib_path, seed=args.seed) as env:
        for episode_id in range(int(args.num_episodes)):
            env.reset(args.seed + episode_id)
            episode_records[episode_id] = []

            for step_idx in range(int(args.max_steps_per_episode)):
                state = env.get_state()
                if bool(state["game_over"]):
                    break

                legal_actions = env.enumerate_legal_actions()
                if not legal_actions:
                    skipped_no_legal += 1
                    break

                legal_by_native = {native.key(): tup for native, tup in legal_actions}
                legal_tuples = [tup for _, tup in legal_actions]

                expert = env.expert_choose_and_apply(think_ms=args.think_ms)
                if not bool(expert["success"]):
                    skipped_invalid_expert += 1
                    break

                chosen_native = NativeAction(
                    use_hold=bool(expert["used_hold"]),
                    placement_index=int(expert["placement_index"]),
                )
                chosen_tuple = legal_by_native.get(chosen_native.key())
                if chosen_tuple is None:
                    skipped_missing_tuple += 1
                    if bool(expert["game_over"]):
                        break
                    continue

                if chosen_tuple not in legal_tuples:
                    raise AssertionError("Stored label is not legal for this state.")

                action_id = codec.encode_tuple(chosen_tuple, add_if_missing=True)
                encoded = encode_state(state, encoder_cfg)
                record = {
                    "board": encoded["board"],
                    "piece": encoded["piece"],
                    "hold": encoded["hold"],
                    "queue": encoded["queue"],
                    "scalars": encoded["scalars"],
                    "action_id": action_id,
                    "action_tuple": np.asarray(chosen_tuple, dtype=np.int64),
                    "episode_id": int(episode_id),
                    "step_idx": int(step_idx),
                }
                episode_records[episode_id].append(record)
                if bool(expert["game_over"]):
                    break

            if (episode_id + 1) % max(1, int(args.log_every)) == 0:
                current_steps = sum(len(v) for v in episode_records.values())
                print(
                    f"[collect_data] episode={episode_id + 1}/{args.num_episodes} "
                    f"transitions={current_steps} vocab={len(codec)}"
                )

    episodes_with_data = [ep for ep, rows in episode_records.items() if rows]
    if not episodes_with_data:
        raise RuntimeError("No transitions were collected. Check build/library configuration.")

    splits = split_episode_ids(
        episodes_with_data,
        train_fraction=split_cfg.train_fraction,
        val_fraction=split_cfg.val_fraction,
        test_fraction=split_cfg.test_fraction,
        seed=split_cfg.seed,
    )

    split_meta: Dict[str, Dict[str, object]] = {}
    for split_name, split_episode_ids_list in splits.items():
        shard_paths: List[str] = []
        split_transition_count = 0
        for shard_idx, chunk in enumerate(
            chunk_list(split_episode_ids_list, max(1, int(args.episodes_per_shard)))
        ):
            records: List[Dict[str, object]] = []
            for ep_id in chunk:
                records.extend(episode_records[int(ep_id)])
            if not records:
                continue
            split_transition_count += len(records)
            payload = _build_payload(records)
            shard_rel = Path("shards") / f"{split_name}_{shard_idx:04d}.pt"
            shard_abs = out_dir / shard_rel
            torch.save(payload, shard_abs)
            shard_paths.append(str(shard_rel).replace("\\", "/"))

        split_meta[split_name] = {
            "episodes": [int(ep) for ep in split_episode_ids_list],
            "num_episodes": int(len(split_episode_ids_list)),
            "num_transitions": int(split_transition_count),
            "shards": shard_paths,
        }

    all_records = [row for rows in episode_records.values() for row in rows]
    label_counts = Counter(int(r["action_id"]) for r in all_records)
    top_classes = label_counts.most_common(20)

    metadata = {
        "format_version": 1,
        "num_episodes_requested": int(args.num_episodes),
        "num_episodes_with_data": int(len(episodes_with_data)),
        "num_transitions": int(len(all_records)),
        "board_shape": [encoder_cfg.board_height, encoder_cfg.board_width],
        "queue_length": int(encoder_cfg.queue_length),
        "include_scalars": bool(encoder_cfg.include_scalars),
        "encoder_config": dataclass_to_dict(encoder_cfg),
        "collection_config": {
            "seed": int(args.seed),
            "think_ms": int(args.think_ms),
            "max_steps_per_episode": int(args.max_steps_per_episode),
            "episodes_per_shard": int(args.episodes_per_shard),
        },
        "split_config": dataclass_to_dict(split_cfg),
        "splits": split_meta,
        "action_vocab_size": int(len(codec)),
        "id_to_action": [list(tup) for tup in codec.id_to_action],
        "sanity": {
            "skipped_no_legal": int(skipped_no_legal),
            "skipped_invalid_expert": int(skipped_invalid_expert),
            "skipped_missing_tuple": int(skipped_missing_tuple),
            "top_action_classes": [[int(k), int(v)] for k, v in top_classes],
        },
    }
    save_json(out_dir / "metadata.json", metadata)

    print(
        "[collect_data] done "
        f"episodes_with_data={metadata['num_episodes_with_data']} "
        f"transitions={metadata['num_transitions']} "
        f"vocab={metadata['action_vocab_size']}"
    )
    print("[collect_data] top action classes:")
    for class_id, count in top_classes[:10]:
        print(f"  class={class_id:4d} count={count:8d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

