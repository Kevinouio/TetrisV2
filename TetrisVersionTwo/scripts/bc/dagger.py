from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .config import EncoderConfig, TrainConfig
from .encoders import encode_state
from .evaluate import offline_evaluate, online_expert_evaluate, online_policy_evaluate
from .inference_agent import BCAgent
from .utils import (
    ActionCodec,
    ActionTuple,
    BCEnvAdapter,
    NativeAction,
    ensure_dir,
    find_library,
    load_json,
    save_json,
    set_global_seeds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run round-based DAgger on top of BC dataset/model.")
    parser.add_argument("--base_data_dir", type=Path, required=True)
    parser.add_argument("--lib", type=Path, default=None, help="Path to tetris_v2_c_api shared library.")
    parser.add_argument("--run_dir", type=Path, default=Path("runs/bc_dagger"))
    parser.add_argument("--init_checkpoint", type=Path, default=None)

    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--episodes_per_round", type=int, default=200)
    parser.add_argument("--episodes_per_shard", type=int, default=100)
    parser.add_argument("--max_steps_per_episode", type=int, default=2_000)
    parser.add_argument("--think_ms", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument(
        "--beta_schedule",
        type=str,
        default="linear",
        choices=("fixed", "linear", "exponential"),
    )
    parser.add_argument("--beta_start", type=float, default=0.75)
    parser.add_argument("--beta_end", type=float, default=0.0)
    parser.add_argument(
        "--beta_decay",
        type=float,
        default=0.5,
        help="Exponential decay multiplier used when --beta_schedule exponential.",
    )
    parser.add_argument(
        "--beta_decay_rounds",
        type=int,
        default=4,
        help="Linear decay rounds including endpoints.",
    )

    parser.set_defaults(fine_tune=True)
    tune_group = parser.add_mutually_exclusive_group()
    tune_group.add_argument(
        "--fine_tune",
        dest="fine_tune",
        action="store_true",
        help="Fine-tune from previous round checkpoint (default).",
    )
    tune_group.add_argument(
        "--retrain_from_scratch",
        dest="fine_tune",
        action="store_false",
        help="Train from scratch each round.",
    )

    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning_rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--train_seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--conv_channels", type=str, default="32,64,64")
    parser.add_argument("--mlp_hidden", type=str, default="256,256")

    parser.add_argument("--eval_games", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--expert_eval_think_ms", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.num_rounds) < 0:
        raise ValueError("--num_rounds must be >= 0.")
    if int(args.episodes_per_round) <= 0:
        raise ValueError("--episodes_per_round must be > 0.")
    if int(args.episodes_per_shard) <= 0:
        raise ValueError("--episodes_per_shard must be > 0.")
    if int(args.max_steps_per_episode) <= 0:
        raise ValueError("--max_steps_per_episode must be > 0.")
    if int(args.beta_decay_rounds) <= 0:
        raise ValueError("--beta_decay_rounds must be > 0.")
    if not (0.0 <= float(args.beta_start) <= 1.0):
        raise ValueError("--beta_start must be in [0,1].")
    if not (0.0 <= float(args.beta_end) <= 1.0):
        raise ValueError("--beta_end must be in [0,1].")
    if float(args.beta_decay) < 0.0:
        raise ValueError("--beta_decay must be >= 0.")


def _resolve_shard_paths(data_dir: Path, shard_values: Iterable[object]) -> List[str]:
    out: List[str] = []
    for value in shard_values:
        p = Path(str(value))
        if not p.is_absolute():
            p = data_dir / p
        out.append(str(p.resolve()))
    return out


def _load_required_base_metadata(base_data_dir: Path) -> Dict[str, object]:
    metadata_path = base_data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = load_json(metadata_path)
    if not isinstance(metadata.get("splits"), dict):
        raise ValueError("Base metadata must contain 'splits'.")
    for split_name in ("train", "val", "test"):
        split_meta = metadata["splits"].get(split_name)  # type: ignore[index]
        if not isinstance(split_meta, dict):
            raise ValueError(f"Missing split metadata for '{split_name}'.")
        shards = split_meta.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"Split '{split_name}' must have at least one shard.")
    if not isinstance(metadata.get("id_to_action"), list):
        raise ValueError("Base metadata must contain list 'id_to_action'.")
    if not isinstance(metadata.get("encoder_config"), dict):
        raise ValueError("Base metadata must contain dict 'encoder_config'.")
    return metadata


def _build_encoder_config(raw: Dict[str, object]) -> EncoderConfig:
    return EncoderConfig(
        board_height=int(raw["board_height"]),
        board_width=int(raw["board_width"]),
        queue_length=int(raw["queue_length"]),
        include_scalars=bool(raw["include_scalars"]),
    )


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

    round_id = np.asarray([r["round_id"] for r in records], dtype=np.int64)
    learner_action_id = np.asarray([r["learner_action_id"] for r in records], dtype=np.int64)
    learner_action_tuple = np.stack([r["learner_action_tuple"] for r in records], axis=0).astype(np.int64)
    acted_by_expert = np.asarray([r["acted_by_expert"] for r in records], dtype=np.uint8)
    learner_raw_invalid = np.asarray([r["learner_raw_invalid"] for r in records], dtype=np.uint8)
    learner_used_fallback = np.asarray([r["learner_used_fallback"] for r in records], dtype=np.uint8)

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
        "round_id": torch.from_numpy(round_id),
        "learner_action_id": torch.from_numpy(learner_action_id),
        "learner_action_tuple": torch.from_numpy(learner_action_tuple),
        "acted_by_expert": torch.from_numpy(acted_by_expert),
        "learner_raw_invalid": torch.from_numpy(learner_raw_invalid),
        "learner_used_fallback": torch.from_numpy(learner_used_fallback),
    }


def _normalize_tuple(raw: Sequence[int]) -> ActionTuple:
    values = tuple(int(v) for v in raw)
    if len(values) != 5:
        raise ValueError(f"Expected action tuple length 5, got {values}.")
    return (values[0], values[1], values[2], values[3], values[4])


def _beta_for_round(args: argparse.Namespace, round_id: int) -> float:
    start = float(args.beta_start)
    end = float(args.beta_end)
    schedule = str(args.beta_schedule)

    if schedule == "fixed":
        beta = start
    elif schedule == "linear":
        if int(args.beta_decay_rounds) <= 1:
            beta = end
        else:
            span = int(args.beta_decay_rounds) - 1
            clamped = max(0, min(round_id - 1, span))
            t = float(clamped) / float(span)
            beta = start + t * (end - start)
    elif schedule == "exponential":
        beta = start * (float(args.beta_decay) ** max(0, round_id - 1))
        if start >= end:
            beta = max(end, beta)
        else:
            beta = min(end, beta)
    else:
        raise ValueError(f"Unsupported beta schedule: {schedule}")

    return float(max(0.0, min(1.0, beta)))


def _run_train(
    data_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    init_checkpoint: Path | None,
) -> Path:
    ensure_dir(out_dir)
    cmd: List[str] = [
        sys.executable,
        "-m",
        "bc.train",
        "--data_dir",
        str(data_dir),
        "--out_dir",
        str(out_dir),
        "--batch_size",
        str(int(args.batch_size)),
        "--learning_rate",
        str(float(args.learning_rate)),
        "--weight_decay",
        str(float(args.weight_decay)),
        "--epochs",
        str(int(args.epochs)),
        "--patience",
        str(int(args.patience)),
        "--seed",
        str(int(args.train_seed)),
        "--num_workers",
        str(int(args.num_workers)),
        "--conv_channels",
        str(args.conv_channels),
        "--mlp_hidden",
        str(args.mlp_hidden),
    ]
    if args.device is not None:
        cmd.extend(["--device", str(args.device)])
    if init_checkpoint is not None:
        cmd.extend(["--init_checkpoint", str(init_checkpoint)])

    print("[dagger] train:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    best_ckpt = out_dir / "best.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Training completed but missing checkpoint: {best_ckpt}")
    return best_ckpt


def _evaluate_round(
    checkpoint: Path,
    data_dir: Path,
    args: argparse.Namespace,
    lib_path: Path,
    round_seed: int,
) -> Dict[str, object]:
    agent = BCAgent(checkpoint_path=checkpoint, device=args.device)
    offline_val = offline_evaluate(
        agent=agent,
        data_dir=data_dir,
        split="val",
        batch_size=int(args.eval_batch_size),
    )
    offline_test = offline_evaluate(
        agent=agent,
        data_dir=data_dir,
        split="test",
        batch_size=int(args.eval_batch_size),
    )
    payload: Dict[str, object] = {
        "offline_val": offline_val,
        "offline_test": offline_test,
    }
    if int(args.eval_games) > 0:
        policy = online_policy_evaluate(
            checkpoint=checkpoint,
            lib_path=lib_path,
            play_games=int(args.eval_games),
            seed=int(round_seed),
            max_steps_per_episode=int(args.max_steps_per_episode),
            device=args.device,
        )
        expert = online_expert_evaluate(
            lib_path=lib_path,
            play_games=int(args.eval_games),
            seed=int(round_seed),
            max_steps_per_episode=int(args.max_steps_per_episode),
            think_ms=int(args.expert_eval_think_ms),
        )
        payload["online_policy"] = policy
        payload["online_expert"] = expert
    return payload


def _collect_dagger_round(
    *,
    round_id: int,
    beta: float,
    codec: ActionCodec,
    learner_checkpoint: Path,
    encoder_config: EncoderConfig,
    episodes_per_round: int,
    max_steps_per_episode: int,
    think_ms: int,
    lib_path: Path,
    seed: int,
    device: str | None,
    log_every: int,
) -> Tuple[Dict[int, List[Dict[str, object]]], Dict[str, object]]:
    episode_records: Dict[int, List[Dict[str, object]]] = {}
    rng = np.random.default_rng(int(seed) + int(round_id) * 1_000_003)

    episodes_completed = 0
    transitions = 0
    episodes_with_data = 0
    expert_steps = 0
    learner_steps = 0
    skipped_no_legal = 0
    skipped_invalid_expert = 0
    skipped_missing_tuple = 0
    failed_steps = 0
    invalid_learner_raw = 0
    unseen_learner_fallback = 0
    label_illegal_count = 0
    vocab_start = len(codec)
    vocab_added = 0

    started = time.time()
    with BCEnvAdapter(lib_path=lib_path, seed=seed + round_id) as env:
        learner = BCAgent(checkpoint_path=learner_checkpoint, device=device, env_adapter=env)
        for ep_offset in range(int(episodes_per_round)):
            episode_id = int(round_id * 1_000_000 + ep_offset)
            episode_seed = int(seed + episode_id)
            env.reset(episode_seed)
            rows: List[Dict[str, object]] = []

            for step_idx in range(int(max_steps_per_episode)):
                state = env.get_state()
                if bool(state["game_over"]):
                    break

                legal_actions = env.enumerate_legal_actions()
                if not legal_actions:
                    skipped_no_legal += 1
                    break

                legal_by_native = {native.key(): tup for native, tup in legal_actions}
                legal_tuples = set(tup for _, tup in legal_actions)

                expert = env.expert_choose(think_ms=think_ms)
                if not bool(expert["success"]):
                    skipped_invalid_expert += 1
                    break

                expert_native = NativeAction(
                    use_hold=bool(expert["used_hold"]),
                    placement_index=int(expert["placement_index"]),
                )
                expert_tuple = legal_by_native.get(expert_native.key())
                if expert_tuple is None:
                    skipped_missing_tuple += 1
                    break
                if expert_tuple not in legal_tuples:
                    label_illegal_count += 1
                    raise AssertionError("Expert label is not legal in learner-visited state.")

                old_vocab = len(codec)
                expert_action_id = int(codec.encode_tuple(expert_tuple, add_if_missing=True))
                if len(codec) > old_vocab:
                    vocab_added += 1

                learner_native, learner_diag = learner.predict_action_with_diagnostics(
                    state,
                    legal_actions=legal_actions,
                )
                learner_tuple = legal_by_native.get(learner_native.key())
                learner_action_id = (
                    int(codec.action_to_id[learner_tuple])
                    if learner_tuple is not None and learner_tuple in codec.action_to_id
                    else -1
                )
                invalid_learner_raw += int(bool(learner_diag["raw_argmax_invalid"]))
                unseen_learner_fallback += int(bool(learner_diag["used_fallback_unseen_legal"]))

                acted_by_expert = bool(rng.random() < beta)
                executed_native = expert_native if acted_by_expert else learner_native

                step_result = env.step_native_action(executed_native)
                if not bool(step_result["success"]):
                    failed_steps += 1
                    break

                encoded = encode_state(state, encoder_config)
                row = {
                    "board": encoded["board"],
                    "piece": encoded["piece"],
                    "hold": encoded["hold"],
                    "queue": encoded["queue"],
                    "scalars": encoded["scalars"],
                    "action_id": int(expert_action_id),
                    "action_tuple": np.asarray(_normalize_tuple(expert_tuple), dtype=np.int64),
                    "episode_id": int(episode_id),
                    "step_idx": int(step_idx),
                    "round_id": int(round_id),
                    "learner_action_id": int(learner_action_id),
                    "learner_action_tuple": np.asarray(
                        _normalize_tuple(learner_tuple) if learner_tuple is not None else (-1, -1, -1, -1, -1),
                        dtype=np.int64,
                    ),
                    "acted_by_expert": int(acted_by_expert),
                    "learner_raw_invalid": int(bool(learner_diag["raw_argmax_invalid"])),
                    "learner_used_fallback": int(bool(learner_diag["used_fallback_unseen_legal"])),
                }
                if row["action_id"] != expert_action_id:
                    raise AssertionError("DAgger supervision must use expert action_id.")
                rows.append(row)

                transitions += 1
                if acted_by_expert:
                    expert_steps += 1
                else:
                    learner_steps += 1
                if bool(step_result["game_over"]):
                    break

            episode_records[episode_id] = rows
            episodes_completed += 1
            if rows:
                episodes_with_data += 1
            if episodes_completed % max(1, int(log_every)) == 0:
                elapsed = max(1e-9, time.time() - started)
                print(
                    f"[dagger][collect] round={round_id} "
                    f"episodes={episodes_completed}/{episodes_per_round} "
                    f"transitions={transitions} vocab={len(codec)} "
                    f"eps_per_sec={episodes_completed/elapsed:.2f}"
                )

    elapsed = max(0.0, time.time() - started)
    total_actions = expert_steps + learner_steps
    empirical = float(expert_steps / total_actions) if total_actions > 0 else 0.0
    stats = {
        "round_id": int(round_id),
        "beta": float(beta),
        "episodes_requested": int(episodes_per_round),
        "episodes_completed": int(episodes_completed),
        "episodes_with_data": int(episodes_with_data),
        "transitions": int(transitions),
        "elapsed_sec": float(elapsed),
        "episodes_per_sec": float(episodes_completed / elapsed) if elapsed > 0 else 0.0,
        "expert_steps": int(expert_steps),
        "learner_steps": int(learner_steps),
        "empirical_expert_action_rate": float(empirical),
        "skipped_no_legal": int(skipped_no_legal),
        "skipped_invalid_expert": int(skipped_invalid_expert),
        "skipped_missing_tuple": int(skipped_missing_tuple),
        "failed_steps": int(failed_steps),
        "invalid_learner_raw_argmax": int(invalid_learner_raw),
        "unseen_learner_fallback": int(unseen_learner_fallback),
        "label_illegal_count": int(label_illegal_count),
        "vocab_start": int(vocab_start),
        "vocab_end": int(len(codec)),
        "vocab_delta": int(len(codec) - vocab_start),
        "vocab_new_labels_seen": int(vocab_added),
    }
    return episode_records, stats


def _write_round_train_shards(
    out_dir: Path,
    episode_records: Dict[int, List[Dict[str, object]]],
    episodes_per_shard: int,
) -> Tuple[List[str], int, List[int]]:
    shard_dir = out_dir / "shards"
    ensure_dir(shard_dir)

    shard_paths: List[str] = []
    transitions = 0
    episode_ids_with_data = [ep for ep, rows in episode_records.items() if rows]
    episode_ids_with_data.sort()

    for shard_idx, begin in enumerate(range(0, len(episode_ids_with_data), int(episodes_per_shard))):
        chunk_ids = episode_ids_with_data[begin : begin + int(episodes_per_shard)]
        rows: List[Dict[str, object]] = []
        for ep_id in chunk_ids:
            rows.extend(episode_records[ep_id])
        if not rows:
            continue
        transitions += len(rows)
        payload = _build_payload(rows)
        shard_path = (shard_dir / f"dagger_train_{shard_idx:04d}.pt").resolve()
        torch.save(payload, shard_path)
        shard_paths.append(str(shard_path))

    return shard_paths, int(transitions), episode_ids_with_data


def _build_aggregated_metadata(
    *,
    base_data_dir: Path,
    base_metadata: Dict[str, object],
    codec: ActionCodec,
    cumulative_train_shards: List[str],
    cumulative_train_episode_ids: List[int],
    cumulative_train_transitions: int,
    round_id: int,
    round_dir: Path,
) -> Tuple[Path, Dict[str, object]]:
    base_splits = base_metadata["splits"]  # type: ignore[index]
    if not isinstance(base_splits, dict):
        raise ValueError("Invalid base split metadata.")

    train_base_meta = base_splits["train"]
    val_base_meta = base_splits["val"]
    test_base_meta = base_splits["test"]
    if not isinstance(train_base_meta, dict) or not isinstance(val_base_meta, dict) or not isinstance(test_base_meta, dict):
        raise ValueError("Invalid base split metadata entries.")

    train_base_shards = _resolve_shard_paths(base_data_dir, train_base_meta.get("shards", []))
    val_base_shards = _resolve_shard_paths(base_data_dir, val_base_meta.get("shards", []))
    test_base_shards = _resolve_shard_paths(base_data_dir, test_base_meta.get("shards", []))

    base_train_episodes = [int(v) for v in train_base_meta.get("episodes", [])]
    val_episodes = [int(v) for v in val_base_meta.get("episodes", [])]
    test_episodes = [int(v) for v in test_base_meta.get("episodes", [])]

    train_shards = list(train_base_shards) + list(cumulative_train_shards)
    train_episodes = list(base_train_episodes) + list(cumulative_train_episode_ids)
    train_transitions = int(train_base_meta.get("num_transitions", 0)) + int(cumulative_train_transitions)

    val_num_episodes = int(val_base_meta.get("num_episodes", len(val_episodes)))
    test_num_episodes = int(test_base_meta.get("num_episodes", len(test_episodes)))

    metadata = {
        "format_version": int(base_metadata.get("format_version", 1)),
        "generated_by": "bc.dagger",
        "dagger_round": int(round_id),
        "base_data_dir": str(base_data_dir.resolve()),
        "board_shape": list(base_metadata.get("board_shape", [20, 10])),
        "queue_length": int(base_metadata.get("queue_length", 5)),
        "include_scalars": bool(base_metadata.get("include_scalars", False)),
        "encoder_config": dict(base_metadata.get("encoder_config", {})),
        "split_config": dict(base_metadata.get("split_config", {})),
        "action_vocab_size": int(len(codec)),
        "id_to_action": [list(tup) for tup in codec.id_to_action],
        "splits": {
            "train": {
                "episodes": train_episodes,
                "num_episodes": int(len(train_episodes)),
                "num_transitions": int(train_transitions),
                "shards": train_shards,
            },
            "val": {
                "episodes": val_episodes,
                "num_episodes": int(val_num_episodes),
                "num_transitions": int(val_base_meta.get("num_transitions", 0)),
                "shards": val_base_shards,
            },
            "test": {
                "episodes": test_episodes,
                "num_episodes": int(test_num_episodes),
                "num_transitions": int(test_base_meta.get("num_transitions", 0)),
                "shards": test_base_shards,
            },
        },
        "dagger": {
            "train_base_shards": train_base_shards,
            "dagger_train_shards": list(cumulative_train_shards),
            "dagger_train_num_transitions": int(cumulative_train_transitions),
        },
    }

    aggregated_dir = round_dir / "aggregated_data"
    ensure_dir(aggregated_dir)
    meta_path = aggregated_dir / "metadata.json"
    save_json(meta_path, metadata)
    return aggregated_dir, metadata


def _write_round_artifacts(
    *,
    round_dir: Path,
    collection_stats: Dict[str, object],
    train_summary: Dict[str, object],
    eval_metrics: Dict[str, object],
    aggregated_metadata_path: Path,
    checkpoint_path: Path,
) -> None:
    payload = {
        "collection": collection_stats,
        "training_summary": train_summary,
        "evaluation": eval_metrics,
        "aggregated_metadata": str(aggregated_metadata_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
    }
    save_json(round_dir / "metrics.json", payload)


def main() -> int:
    args = parse_args()
    _validate_args(args)
    set_global_seeds(int(args.seed))

    base_data_dir = Path(args.base_data_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    ensure_dir(run_dir)
    lib_path = find_library(args.lib)

    base_metadata = _load_required_base_metadata(base_data_dir)
    encoder_config = _build_encoder_config(base_metadata["encoder_config"])  # type: ignore[arg-type]
    codec = ActionCodec(id_to_action=base_metadata["id_to_action"])  # type: ignore[arg-type]

    component_assessment = {
        "reused_existing_components": [
            "BCEnvAdapter for env state extraction, legal action enumeration, expert integration, and stepping",
            "ActionCodec action tuple/id mapping",
            "BCAgent inference wrapper with legality masking",
            "BC dataset shard format (.pt shards + metadata.json)",
            "bc.train supervised CE training loop",
            "bc.evaluate offline/online evaluation functions",
        ],
        "gaps_filled_by_dagger": [
            "round-based DAgger orchestration",
            "beta-mixture learner/expert rollout collector",
            "expert labeling on learner-visited states",
            "cumulative aggregated metadata across rounds",
            "vocab growth + warm-start compatible fine-tuning loop",
        ],
        "required_adapter_methods_detected": [
            "get_state",
            "enumerate_legal_actions",
            "expert_choose",
            "step_native_action",
        ],
    }
    save_json(run_dir / "component_assessment.json", component_assessment)

    print(f"[dagger] base_data_dir={base_data_dir}")
    print(f"[dagger] run_dir={run_dir}")
    print(f"[dagger] initial_vocab={len(codec)}")

    init_checkpoint: Path
    if args.init_checkpoint is not None:
        init_checkpoint = Path(args.init_checkpoint).resolve()
        if not init_checkpoint.exists():
            raise FileNotFoundError(f"--init_checkpoint not found: {init_checkpoint}")
        current_checkpoint = init_checkpoint
    else:
        bootstrap_dir = run_dir / "round_0_bootstrap"
        train_out = bootstrap_dir / "train"
        current_checkpoint = _run_train(
            data_dir=base_data_dir,
            out_dir=train_out,
            args=args,
            init_checkpoint=None,
        )
        init_checkpoint = current_checkpoint

    summary: Dict[str, object] = {
        "config": {
            "base_data_dir": str(base_data_dir),
            "lib": str(lib_path),
            "run_dir": str(run_dir),
            "init_checkpoint": str(init_checkpoint),
            "num_rounds": int(args.num_rounds),
            "episodes_per_round": int(args.episodes_per_round),
            "max_steps_per_episode": int(args.max_steps_per_episode),
            "think_ms": int(args.think_ms),
            "seed": int(args.seed),
            "beta_schedule": str(args.beta_schedule),
            "beta_start": float(args.beta_start),
            "beta_end": float(args.beta_end),
            "beta_decay": float(args.beta_decay),
            "beta_decay_rounds": int(args.beta_decay_rounds),
            "fine_tune": bool(args.fine_tune),
            "eval_games": int(args.eval_games),
        },
        "rounds": [],
        "initial_checkpoint": str(current_checkpoint),
        "initial_vocab_size": int(len(codec)),
    }

    initial_eval = _evaluate_round(
        checkpoint=current_checkpoint,
        data_dir=base_data_dir,
        args=args,
        lib_path=lib_path,
        round_seed=int(args.seed),
    )
    summary["initial_evaluation"] = initial_eval
    save_json(run_dir / "dagger_summary.json", summary)

    cumulative_round_train_shards: List[str] = []
    cumulative_round_train_episode_ids: List[int] = []
    cumulative_round_train_transitions = 0
    prev_total_train_samples = int(base_metadata["splits"]["train"]["num_transitions"])  # type: ignore[index]

    base_val_shards = _resolve_shard_paths(base_data_dir, base_metadata["splits"]["val"]["shards"])  # type: ignore[index]
    base_test_shards = _resolve_shard_paths(base_data_dir, base_metadata["splits"]["test"]["shards"])  # type: ignore[index]

    for round_id in range(1, int(args.num_rounds) + 1):
        beta = _beta_for_round(args, round_id)
        print(f"[dagger] ===== round={round_id} beta={beta:.4f} =====")
        round_dir = run_dir / f"round_{round_id:02d}"
        ensure_dir(round_dir)

        round_records, collect_stats = _collect_dagger_round(
            round_id=round_id,
            beta=beta,
            codec=codec,
            learner_checkpoint=current_checkpoint,
            encoder_config=encoder_config,
            episodes_per_round=int(args.episodes_per_round),
            max_steps_per_episode=int(args.max_steps_per_episode),
            think_ms=int(args.think_ms),
            lib_path=lib_path,
            seed=int(args.seed),
            device=args.device,
            log_every=int(args.log_every),
        )

        train_shard_root = round_dir / "dagger_train"
        round_train_shards, round_train_transitions, round_episode_ids = _write_round_train_shards(
            out_dir=train_shard_root,
            episode_records=round_records,
            episodes_per_shard=int(args.episodes_per_shard),
        )
        if round_train_transitions <= 0:
            raise RuntimeError(
                f"Round {round_id} collected no transitions; aborting DAgger run."
            )

        cumulative_round_train_shards.extend(round_train_shards)
        cumulative_round_train_episode_ids.extend(round_episode_ids)
        cumulative_round_train_transitions += int(round_train_transitions)

        aggregated_data_dir, aggregated_metadata = _build_aggregated_metadata(
            base_data_dir=base_data_dir,
            base_metadata=base_metadata,
            codec=codec,
            cumulative_train_shards=cumulative_round_train_shards,
            cumulative_train_episode_ids=cumulative_round_train_episode_ids,
            cumulative_train_transitions=cumulative_round_train_transitions,
            round_id=round_id,
            round_dir=round_dir,
        )

        round_total_train_samples = int(aggregated_metadata["splits"]["train"]["num_transitions"])  # type: ignore[index]
        if round_total_train_samples < prev_total_train_samples:
            raise AssertionError("Aggregated train sample count decreased across rounds.")
        prev_total_train_samples = round_total_train_samples

        agg_val_shards = [str(v) for v in aggregated_metadata["splits"]["val"]["shards"]]  # type: ignore[index]
        agg_test_shards = [str(v) for v in aggregated_metadata["splits"]["test"]["shards"]]  # type: ignore[index]
        if agg_val_shards != base_val_shards or agg_test_shards != base_test_shards:
            raise AssertionError("Validation/test split shards changed; must remain fixed.")

        train_out = round_dir / "train"
        train_init_ckpt = current_checkpoint if bool(args.fine_tune) else None
        next_checkpoint = _run_train(
            data_dir=aggregated_data_dir,
            out_dir=train_out,
            args=args,
            init_checkpoint=train_init_ckpt,
        )

        train_summary_path = train_out / "summary.json"
        if not train_summary_path.exists():
            raise FileNotFoundError(f"Missing training summary: {train_summary_path}")
        train_summary = load_json(train_summary_path)

        eval_metrics = _evaluate_round(
            checkpoint=next_checkpoint,
            data_dir=aggregated_data_dir,
            args=args,
            lib_path=lib_path,
            round_seed=int(args.seed + round_id * 10_000),
        )

        round_metrics = {
            "round_id": int(round_id),
            "beta": float(beta),
            "collection": {
                **collect_stats,
                "new_dagger_samples": int(round_train_transitions),
                "total_train_samples": int(round_total_train_samples),
            },
            "training": train_summary,
            "evaluation": eval_metrics,
            "vocab_size": int(len(codec)),
            "train_checkpoint": str(next_checkpoint),
            "aggregated_metadata": str((aggregated_data_dir / "metadata.json").resolve()),
        }
        _write_round_artifacts(
            round_dir=round_dir,
            collection_stats=round_metrics["collection"],
            train_summary=train_summary,
            eval_metrics=eval_metrics,
            aggregated_metadata_path=aggregated_data_dir / "metadata.json",
            checkpoint_path=next_checkpoint,
        )

        summary_rounds = summary["rounds"]
        if not isinstance(summary_rounds, list):
            raise AssertionError("Invalid summary rounds payload.")
        summary_rounds.append(round_metrics)
        summary["latest_checkpoint"] = str(next_checkpoint)
        summary["latest_vocab_size"] = int(len(codec))
        save_json(run_dir / "dagger_summary.json", summary)

        current_checkpoint = next_checkpoint

    print(f"[dagger] complete rounds={args.num_rounds} final_checkpoint={current_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
