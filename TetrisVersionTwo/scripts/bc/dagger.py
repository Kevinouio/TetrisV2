from __future__ import annotations

import atexit
import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
    configure_cpu_runtime,
    ensure_dir,
    find_library,
    load_json,
    save_json,
    set_global_seeds,
)
from .viewer_telemetry import board_for_event, piece_ids_for_event, queue_put_best_effort
from .viewer_runtime import LiveViewerRuntime

_WORKER_ENV: Optional[BCEnvAdapter] = None
_WORKER_AGENT: Optional[BCAgent] = None
_WORKER_ENCODER_CFG: Optional[EncoderConfig] = None
_WORKER_ROUND_ID: int = 0
_WORKER_BETA: float = 0.0
_WORKER_MAX_STEPS: int = 0
_WORKER_THINK_MS: int = 20
_WORKER_BASE_SEED: int = 0
_WORKER_STATE_SOURCE: str = "rollout"
_WORKER_RANDOM_FILL_Y_MAX_EXCLUSIVE: int = 17
_WORKER_RANDOM_FILL_PROB: float = 0.5
_WORKER_RANDOM_MAX_RESAMPLES_PER_SAMPLE: int = 100
_WORKER_RANDOM_POST_CLEAR_STEPS: int = 50
_WORKER_VIEWER_QUEUE: Any = None
_WORKER_VIEWER_PUBLISH_EVERY_STEPS: int = 10
_WORKER_VIEWER_COMPACT_TELEMETRY: bool = True
_WORKER_VIEWER_BOARD_EVERY_STEPS: int = 50
_WORKER_SLOT: int = 1
_WORKER_LABEL: str = "PID"


def _argv_has_flag(argv: Sequence[str], flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in argv)


def _iterator_next_with_timeout(iterator: Any, *, timeout: float) -> Any:
    """Return next pool result while handling both IMapIterator and generator paths."""
    next_with_timeout = getattr(iterator, "next", None)
    if callable(next_with_timeout):
        return next_with_timeout(timeout=float(timeout))
    return next(iterator)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    argv_list = list(sys.argv[1:] if argv is None else argv)
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
        "--state_source",
        type=str,
        choices=("rollout", "random_board"),
        default="rollout",
    )
    parser.add_argument("--random_fill_y_max_exclusive", type=int, default=17)
    parser.add_argument("--random_fill_prob", type=float, default=0.5)
    parser.add_argument("--random_max_resamples_per_sample", type=int, default=100)
    parser.add_argument("--random_post_clear_steps", type=int, default=50)

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
    parser.add_argument(
        "--max_train_transitions",
        type=int,
        default=0,
        help="If > 0, cap train transitions passed to bc.train (--max_train_samples).",
    )
    parser.add_argument("--conv_channels", type=str, default="32,64,64")
    parser.add_argument("--mlp_hidden", type=str, default="256,256")

    parser.add_argument("--eval_games", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--expert_eval_think_ms", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--collect_workers", type=int, default=1)
    parser.add_argument("--worker_chunksize", type=int, default=1)
    parser.add_argument(
        "--worker_maxtasksperchild",
        type=int,
        default=64,
        help="Recycle worker processes after this many tasks (0 disables recycling).",
    )
    parser.add_argument(
        "--progress_mode",
        type=str,
        choices=("console", "json", "both"),
        default="both",
    )
    parser.add_argument("--progress_every_sec", type=float, default=2.0)
    parser.add_argument("--progress_path", type=Path, default=None)
    parser.add_argument(
        "--stop_file",
        type=Path,
        default=None,
        help="If this file appears during collection, stop current round early and keep partial data.",
    )
    parser.add_argument(
        "--rss_warn_mb",
        type=float,
        default=0.0,
        help="Warn once when main-process RSS exceeds this threshold in MiB (0 disables).",
    )
    parser.add_argument(
        "--worker_rss_warn_mb",
        type=float,
        default=0.0,
        help="Warn once when any worker RSS exceeds this threshold in MiB (0 disables).",
    )
    parser.add_argument("--viewer", action="store_true", help="Enable live pygame collection viewer.")
    parser.add_argument(
        "--viewer_fullscreen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start viewer in fullscreen mode (default: false).",
    )
    parser.add_argument("--viewer_fps", type=int, default=20, help="Viewer redraw FPS.")
    parser.add_argument(
        "--viewer_publish_every_steps",
        type=int,
        default=10,
        help="Publish viewer step snapshots every N steps per episode (auto 1 when --viewer unless set).",
    )
    parser.add_argument(
        "--viewer_compact_telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reduce viewer overhead by publishing board snapshots less frequently.",
    )
    parser.add_argument(
        "--viewer_board_every_steps",
        type=int,
        default=50,
        help="When compact telemetry is enabled, include boards every N steps (auto 1 when --viewer unless set).",
    )
    parser.add_argument(
        "--viewer_max_queue",
        type=int,
        default=4096,
        help="Maximum queued viewer events before dropping telemetry.",
    )
    parser.add_argument("--viewer_grid_padding", type=int, default=8, help="Viewer card padding.")
    parser.add_argument("--viewer_min_tile_px", type=int, default=6, help="Minimum viewer tile size.")
    parser.add_argument("--viewer_agent", type=int, default=1, help="Initial selected worker card id.")
    parser.add_argument(
        "--viewer_reopen_file",
        type=Path,
        default=None,
        help="Optional trigger file path used to reopen a closed viewer window.",
    )
    args = parser.parse_args(argv_list)
    if bool(args.viewer):
        if not _argv_has_flag(argv_list, "--viewer_publish_every_steps"):
            args.viewer_publish_every_steps = 1
        if not _argv_has_flag(argv_list, "--viewer_board_every_steps"):
            args.viewer_board_every_steps = 1
    return args


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
    if int(args.collect_workers) <= 0:
        raise ValueError("--collect_workers must be >= 1.")
    if int(args.worker_chunksize) <= 0:
        raise ValueError("--worker_chunksize must be >= 1.")
    if int(args.worker_maxtasksperchild) < 0:
        raise ValueError("--worker_maxtasksperchild must be >= 0.")
    if float(args.progress_every_sec) <= 0.0:
        raise ValueError("--progress_every_sec must be > 0.")
    if int(args.random_fill_y_max_exclusive) < 0 or int(args.random_fill_y_max_exclusive) > 20:
        raise ValueError("--random_fill_y_max_exclusive must be in [0,20].")
    if not (0.0 <= float(args.random_fill_prob) <= 1.0):
        raise ValueError("--random_fill_prob must be in [0,1].")
    if int(args.random_max_resamples_per_sample) <= 0:
        raise ValueError("--random_max_resamples_per_sample must be > 0.")
    if int(args.random_post_clear_steps) < 0:
        raise ValueError("--random_post_clear_steps must be >= 0.")
    if int(args.max_train_transitions) < 0:
        raise ValueError("--max_train_transitions must be >= 0.")
    if int(args.viewer_fps) <= 0:
        raise ValueError("--viewer_fps must be > 0.")
    if int(args.viewer_publish_every_steps) <= 0:
        raise ValueError("--viewer_publish_every_steps must be >= 1.")
    if int(args.viewer_board_every_steps) <= 0:
        raise ValueError("--viewer_board_every_steps must be >= 1.")
    if int(args.viewer_max_queue) <= 0:
        raise ValueError("--viewer_max_queue must be >= 1.")
    if float(args.rss_warn_mb) < 0.0:
        raise ValueError("--rss_warn_mb must be >= 0.")
    if float(args.worker_rss_warn_mb) < 0.0:
        raise ValueError("--worker_rss_warn_mb must be >= 0.")


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


def _pack_dagger_rows_for_ipc(rows: List[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {"n": 0}
    return {
        "n": int(len(rows)),
        "board": np.stack([r["board"] for r in rows], axis=0).astype(np.float32, copy=False),
        "piece": np.stack([r["piece"] for r in rows], axis=0).astype(np.float32, copy=False),
        "hold": np.stack([r["hold"] for r in rows], axis=0).astype(np.float32, copy=False),
        "queue": np.stack([r["queue"] for r in rows], axis=0).astype(np.float32, copy=False),
        "scalars": np.stack([r["scalars"] for r in rows], axis=0).astype(np.float32, copy=False),
        "expert_action_tuple": np.asarray([r["expert_action_tuple"] for r in rows], dtype=np.int64),
        "learner_action_tuple": np.asarray([r["learner_action_tuple"] for r in rows], dtype=np.int64),
        "episode_id": np.asarray([r["episode_id"] for r in rows], dtype=np.int64),
        "step_idx": np.asarray([r["step_idx"] for r in rows], dtype=np.int64),
        "round_id": np.asarray([r["round_id"] for r in rows], dtype=np.int64),
        "acted_by_expert": np.asarray([r["acted_by_expert"] for r in rows], dtype=np.uint8),
        "learner_raw_invalid": np.asarray([r["learner_raw_invalid"] for r in rows], dtype=np.uint8),
        "learner_used_fallback": np.asarray([r["learner_used_fallback"] for r in rows], dtype=np.uint8),
    }


def _unpack_dagger_rows_from_ipc(payload: Dict[str, object]) -> List[Dict[str, object]]:
    n = int(payload.get("n", 0))
    if n <= 0:
        return []
    board = np.asarray(payload["board"], dtype=np.float32)
    piece = np.asarray(payload["piece"], dtype=np.float32)
    hold = np.asarray(payload["hold"], dtype=np.float32)
    queue = np.asarray(payload["queue"], dtype=np.float32)
    scalars = np.asarray(payload["scalars"], dtype=np.float32)
    expert_action_tuple = np.asarray(payload["expert_action_tuple"], dtype=np.int64)
    learner_action_tuple = np.asarray(payload["learner_action_tuple"], dtype=np.int64)
    episode_id = np.asarray(payload["episode_id"], dtype=np.int64)
    step_idx = np.asarray(payload["step_idx"], dtype=np.int64)
    round_id = np.asarray(payload["round_id"], dtype=np.int64)
    acted_by_expert = np.asarray(payload["acted_by_expert"], dtype=np.uint8)
    learner_raw_invalid = np.asarray(payload["learner_raw_invalid"], dtype=np.uint8)
    learner_used_fallback = np.asarray(payload["learner_used_fallback"], dtype=np.uint8)
    out: List[Dict[str, object]] = []
    for i in range(n):
        out.append(
            {
                "board": board[i],
                "piece": piece[i],
                "hold": hold[i],
                "queue": queue[i],
                "scalars": scalars[i],
                "expert_action_tuple": expert_action_tuple[i],
                "learner_action_tuple": learner_action_tuple[i],
                "episode_id": int(episode_id[i]),
                "step_idx": int(step_idx[i]),
                "round_id": int(round_id[i]),
                "acted_by_expert": int(acted_by_expert[i]),
                "learner_raw_invalid": int(learner_raw_invalid[i]),
                "learner_used_fallback": int(learner_used_fallback[i]),
            }
        )
    return out


def _current_rss_mb() -> float:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            parts = handle.read().strip().split()
        if len(parts) >= 2:
            return float(int(parts[1]) * int(page_size) / (1024.0 * 1024.0))
    except Exception:
        pass
    return 0.0


def _torch_load_cpu(path: Path) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location="cpu")


def _spool_path_for_episode(spool_dir: Path, episode_id: int) -> Path:
    return spool_dir / f"episode_{int(episode_id):010d}.pt"


def _write_episode_rows_to_spool(spool_dir: Path, episode_id: int, rows: List[Dict[str, object]]) -> Path:
    path = _spool_path_for_episode(spool_dir, int(episode_id))
    payload = _pack_dagger_rows_for_ipc(rows)
    torch.save(payload, path)
    return path


def _load_episode_rows_from_spool(path: Path) -> List[Dict[str, object]]:
    payload = _torch_load_cpu(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected spool payload type at {path}: {type(payload)}")
    return _unpack_dagger_rows_from_ipc(payload)


def _write_progress_json(path: Path, payload: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp") if path.suffix else Path(str(path) + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _mix_seed(base_seed: int, round_id: int, episode_id: int) -> int:
    mixed = (
        int(base_seed) * 1_000_000_007
        + int(round_id) * 1_000_003
        + int(episode_id) * 97
        + 17
    )
    return int(mixed & ((1 << 63) - 1))


def _mix_seed_with_attempt(base_seed: int, round_id: int, episode_id: int, attempt_idx: int) -> int:
    mixed = (
        int(base_seed) * 1_000_000_007
        + int(round_id) * 1_000_003
        + int(episode_id) * 97
        + int(attempt_idx) * 271
        + 43
    )
    return int(mixed & ((1 << 63) - 1))


def _close_worker_state() -> None:
    global _WORKER_AGENT
    global _WORKER_ENV
    _WORKER_AGENT = None
    if _WORKER_ENV is not None:
        _WORKER_ENV.close()
        _WORKER_ENV = None


def _worker_emit_viewer_event(event: Dict[str, object]) -> None:
    if _WORKER_VIEWER_QUEUE is None:
        return
    payload = dict(event)
    payload.setdefault("type", "step_snapshot")
    payload.setdefault("mode", "dagger")
    payload.setdefault("timestamp", float(time.time()))
    payload.setdefault("worker_slot", int(_WORKER_SLOT))
    payload.setdefault("worker_label", str(_WORKER_LABEL))
    payload.setdefault("worker_key", f"pid:{os.getpid()}")
    queue_put_best_effort(_WORKER_VIEWER_QUEUE, payload)


def _worker_init(
    lib_path_str: str,
    checkpoint_str: str,
    device: Optional[str],
    encoder_cfg_dict: Dict[str, object],
    round_id: int,
    beta: float,
    max_steps_per_episode: int,
    think_ms: int,
    base_seed: int,
    state_source: str,
    random_fill_y_max_exclusive: int,
    random_fill_prob: float,
    random_max_resamples_per_sample: int,
    random_post_clear_steps: int,
    viewer_queue: Any,
    viewer_publish_every_steps: int,
    viewer_compact_telemetry: bool,
    viewer_board_every_steps: int,
) -> None:
    global _WORKER_ENV
    global _WORKER_AGENT
    global _WORKER_ENCODER_CFG
    global _WORKER_ROUND_ID
    global _WORKER_BETA
    global _WORKER_MAX_STEPS
    global _WORKER_THINK_MS
    global _WORKER_BASE_SEED
    global _WORKER_STATE_SOURCE
    global _WORKER_RANDOM_FILL_Y_MAX_EXCLUSIVE
    global _WORKER_RANDOM_FILL_PROB
    global _WORKER_RANDOM_MAX_RESAMPLES_PER_SAMPLE
    global _WORKER_RANDOM_POST_CLEAR_STEPS
    global _WORKER_VIEWER_QUEUE
    global _WORKER_VIEWER_PUBLISH_EVERY_STEPS
    global _WORKER_VIEWER_COMPACT_TELEMETRY
    global _WORKER_VIEWER_BOARD_EVERY_STEPS
    global _WORKER_SLOT
    global _WORKER_LABEL

    configure_cpu_runtime(
        torch_num_threads=1,
        torch_num_interop_threads=1,
        omp_num_threads=1,
        mkl_num_threads=1,
        openblas_num_threads=1,
    )

    _WORKER_ROUND_ID = int(round_id)
    _WORKER_BETA = float(beta)
    _WORKER_MAX_STEPS = int(max_steps_per_episode)
    _WORKER_THINK_MS = int(think_ms)
    _WORKER_BASE_SEED = int(base_seed)
    _WORKER_STATE_SOURCE = str(state_source)
    _WORKER_RANDOM_FILL_Y_MAX_EXCLUSIVE = int(random_fill_y_max_exclusive)
    _WORKER_RANDOM_FILL_PROB = float(random_fill_prob)
    _WORKER_RANDOM_MAX_RESAMPLES_PER_SAMPLE = int(random_max_resamples_per_sample)
    _WORKER_RANDOM_POST_CLEAR_STEPS = int(random_post_clear_steps)
    _WORKER_VIEWER_QUEUE = viewer_queue
    _WORKER_VIEWER_PUBLISH_EVERY_STEPS = max(1, int(viewer_publish_every_steps))
    _WORKER_VIEWER_COMPACT_TELEMETRY = bool(viewer_compact_telemetry)
    _WORKER_VIEWER_BOARD_EVERY_STEPS = max(1, int(viewer_board_every_steps))
    process_identity = getattr(mp.current_process(), "_identity", ())
    if process_identity and len(process_identity) > 0:
        _WORKER_SLOT = int(process_identity[0])
    else:
        _WORKER_SLOT = 1
    _WORKER_LABEL = f"PID {os.getpid()}"
    _WORKER_ENCODER_CFG = EncoderConfig(
        board_height=int(encoder_cfg_dict["board_height"]),
        board_width=int(encoder_cfg_dict["board_width"]),
        queue_length=int(encoder_cfg_dict["queue_length"]),
        include_scalars=bool(encoder_cfg_dict["include_scalars"]),
    )
    _WORKER_ENV = BCEnvAdapter(lib_path=Path(lib_path_str), seed=_WORKER_BASE_SEED + _WORKER_ROUND_ID)
    _WORKER_AGENT = BCAgent(
        checkpoint_path=Path(checkpoint_str),
        device=device,
        env_adapter=_WORKER_ENV,
    )
    atexit.register(_close_worker_state)
    _worker_emit_viewer_event(
        {
            "type": "worker_started",
            "mode": "dagger",
            "status": "active",
            "worker_slot": int(_WORKER_SLOT),
            "worker_label": str(_WORKER_LABEL),
            "worker_key": f"pid:{os.getpid()}",
        }
    )


def _collect_episode_with_env(
    *,
    env: BCEnvAdapter,
    learner: BCAgent,
    encoder_config: EncoderConfig,
    round_id: int,
    beta: float,
    max_steps_per_episode: int,
    think_ms: int,
    base_seed: int,
    episode_id: int,
    state_source: str,
    random_fill_y_max_exclusive: int,
    random_fill_prob: float,
    random_max_resamples_per_sample: int,
    random_post_clear_steps: int,
    publish_event: Optional[Callable[[Dict[str, object]], None]] = None,
    publish_every_steps: int = 10,
    compact_telemetry: bool = True,
    board_every_steps: int = 50,
    worker_slot: int = 1,
    worker_label: str = "PID",
) -> Dict[str, object]:
    worker_pid = int(os.getpid())
    worker_key = f"pid:{worker_pid}"
    if not str(worker_label).strip():
        worker_label = f"PID {worker_pid}"

    env.reset(int(base_seed + episode_id))

    rows: List[Dict[str, object]] = []
    skipped_no_legal = 0
    skipped_invalid_expert = 0
    skipped_missing_tuple = 0
    failed_steps = 0
    invalid_learner_raw = 0
    unseen_learner_fallback = 0
    expert_steps = 0
    learner_steps = 0
    label_illegal_count = 0
    generation_attempts = 0
    resampled_samples = 0
    episodes_cleared_garbage = 0
    episodes_topout_before_clear = 0
    episodes_max_steps_before_clear = 0
    episodes_no_data_after_resamples = 0
    episode_steps = 0
    last_lines_total = 0
    publish_step_interval = max(1, int(publish_every_steps))
    board_step_interval = max(1, int(board_every_steps))

    def count_injected_garbage_cells() -> int:
        return int(env.visible_garbage_count())

    def try_collect_one(state_step_idx: int, mix_rng: np.random.Generator) -> Optional[bool]:
        nonlocal skipped_no_legal
        nonlocal skipped_invalid_expert
        nonlocal skipped_missing_tuple
        nonlocal failed_steps
        nonlocal invalid_learner_raw
        nonlocal unseen_learner_fallback
        nonlocal expert_steps
        nonlocal learner_steps
        nonlocal label_illegal_count

        state = env.get_state()
        if bool(state["game_over"]):
            return None

        legal_actions = env.enumerate_legal_actions()
        if not legal_actions:
            skipped_no_legal += 1
            return None

        legal_by_native = {native.key(): tup for native, tup in legal_actions}
        legal_tuples = set(tup for _, tup in legal_actions)

        expert = env.expert_choose(think_ms=think_ms)
        if not bool(expert["success"]):
            skipped_invalid_expert += 1
            return None

        expert_native = NativeAction(
            use_hold=bool(expert["used_hold"]),
            placement_index=int(expert["placement_index"]),
        )
        expert_tuple = legal_by_native.get(expert_native.key())
        if expert_tuple is None:
            skipped_missing_tuple += 1
            return None
        if expert_tuple not in legal_tuples:
            label_illegal_count += 1
            raise AssertionError("Expert label is not legal in learner-visited state.")

        acted_by_expert = bool(mix_rng.random() < float(beta))
        learner_diag = {
            "raw_argmax_invalid": False,
            "used_fallback_unseen_legal": False,
        }
        learner_tuple: Optional[ActionTuple] = None

        if acted_by_expert:
            executed_native = expert_native
            expert_steps += 1
        else:
            learner_native, learner_diag = learner.predict_action_with_diagnostics(
                state,
                legal_actions=legal_actions,
            )
            learner_tuple = legal_by_native.get(learner_native.key())
            invalid_learner_raw += int(bool(learner_diag["raw_argmax_invalid"]))
            unseen_learner_fallback += int(bool(learner_diag["used_fallback_unseen_legal"]))
            executed_native = learner_native
            learner_steps += 1

        step_result = env.step_native_action(executed_native)
        if not bool(step_result["success"]):
            failed_steps += 1
            return None

        encoded = encode_state(state, encoder_config)
        rows.append(
            {
                "board": encoded["board"],
                "piece": encoded["piece"],
                "hold": encoded["hold"],
                "queue": encoded["queue"],
                "scalars": encoded["scalars"],
                "expert_action_tuple": list(_normalize_tuple(expert_tuple)),
                "learner_action_tuple": list(
                    _normalize_tuple(learner_tuple) if learner_tuple is not None else (-1, -1, -1, -1, -1)
                ),
                "episode_id": int(episode_id),
                "step_idx": int(state_step_idx),
                "round_id": int(round_id),
                "acted_by_expert": int(acted_by_expert),
                "learner_raw_invalid": int(bool(learner_diag["raw_argmax_invalid"])),
                "learner_used_fallback": int(bool(learner_diag["used_fallback_unseen_legal"])),
            }
        )
        done = bool(step_result["game_over"])
        if publish_event is not None and ((state_step_idx + 1) % publish_step_interval == 0 or done):
            include_board = (
                (not bool(compact_telemetry))
                or done
                or ((state_step_idx + 1) % board_step_interval == 0)
            )
            state_now = env.get_state()
            event_payload: Dict[str, object] = {
                "type": "step_snapshot",
                "mode": "dagger",
                "status": "done" if done else "active",
                "worker_slot": int(worker_slot),
                "worker_label": str(worker_label),
                "worker_key": worker_key,
                "round_id": int(round_id),
                "beta": float(beta),
                "episode_id": int(episode_id),
                "step_in_episode": int(state_step_idx + 1),
                "lines_total": int(state_now.get("lines", 0)),
                "transitions_collected": int(len(rows)),
                "expert_steps": int(expert_steps),
                "learner_steps": int(learner_steps),
                "invalid_learner_raw_argmax": int(invalid_learner_raw),
                "unseen_learner_fallback": int(unseen_learner_fallback),
                "skipped_no_legal": int(skipped_no_legal),
                "skipped_invalid_expert": int(skipped_invalid_expert),
                "skipped_missing_tuple": int(skipped_missing_tuple),
                "failed_steps": int(failed_steps),
                "timestamp": float(time.time()),
            }
            if include_board:
                event_payload["board"] = board_for_event(state_now.get("board"))
                try:
                    event_payload["board_piece_ids"] = piece_ids_for_event(
                        env.board_piece_ids(include_active=True)
                    )
                except Exception:
                    pass
            publish_event(event_payload)
        return bool(step_result["game_over"])

    mode = str(state_source).strip().lower()
    if mode == "random_board":
        accepted = False
        attempts_limit = max(1, int(random_max_resamples_per_sample))
        for attempt_idx in range(attempts_limit):
            generation_attempts += 1
            env.reset(int(base_seed + episode_id))

            mask_rng = np.random.default_rng(
                _mix_seed_with_attempt(base_seed, round_id, episode_id, attempt_idx)
            )
            mix_rng = np.random.default_rng(
                _mix_seed_with_attempt(base_seed + 31, round_id, episode_id, attempt_idx)
            )
            mask = np.zeros((20, 10), dtype=np.uint8)
            y_lim = max(0, min(20, int(random_fill_y_max_exclusive)))
            if y_lim > 0:
                lower = mask_rng.random((y_lim, 10)) < float(random_fill_prob)
                for y in range(y_lim):
                    row = 19 - y
                    mask[row, :] = lower[y, :].astype(np.uint8)

            if not env.set_visible_board_mask(mask, reset_meta=True):
                raise RuntimeError("Failed to set randomized visible board mask via C API.")

            if bool(env.meta()["game_over"]):
                resampled_samples += 1
                continue

            remaining_garbage = count_injected_garbage_cells()
            if remaining_garbage <= 0:
                resampled_samples += 1
                continue

            attempt_transitions = 0
            attempt_terminated = False
            cleared_during_attempt = False
            post_clear_done = 0
            for step_idx in range(int(max_steps_per_episode)):
                step_game_over = try_collect_one(state_step_idx=step_idx, mix_rng=mix_rng)
                if step_game_over is None:
                    attempt_terminated = True
                    if attempt_transitions > 0:
                        if not cleared_during_attempt:
                            episodes_topout_before_clear += 1
                        accepted = True
                    else:
                        resampled_samples += 1
                    break

                episode_steps = max(int(episode_steps), int(step_idx + 1))
                try:
                    last_lines_total = int(env.meta().get("lines", last_lines_total))
                except Exception:
                    pass
                attempt_transitions += 1
                remaining_garbage = count_injected_garbage_cells()
                just_cleared_this_step = False
                if (not cleared_during_attempt) and remaining_garbage <= 0:
                    episodes_cleared_garbage += 1
                    cleared_during_attempt = True
                    just_cleared_this_step = True
                    if int(random_post_clear_steps) <= 0:
                        accepted = True
                        attempt_terminated = True
                        break

                if bool(step_game_over):
                    if not cleared_during_attempt:
                        episodes_topout_before_clear += 1
                    accepted = True
                    attempt_terminated = True
                    break

                if cleared_during_attempt and (not just_cleared_this_step):
                    post_clear_done += 1
                    if post_clear_done >= int(random_post_clear_steps):
                        accepted = True
                        attempt_terminated = True
                        break

            if accepted:
                break
            if not attempt_terminated:
                if attempt_transitions > 0:
                    if not cleared_during_attempt:
                        episodes_max_steps_before_clear += 1
                    accepted = True
                    break
                resampled_samples += 1

        if not accepted:
            episodes_no_data_after_resamples += 1
            rows = []
    else:
        mix_rng = np.random.default_rng(_mix_seed(base_seed, round_id, episode_id))
        for step_idx in range(int(max_steps_per_episode)):
            step_game_over = try_collect_one(state_step_idx=step_idx, mix_rng=mix_rng)
            if step_game_over is None:
                break
            episode_steps = max(int(episode_steps), int(step_idx + 1))
            try:
                last_lines_total = int(env.meta().get("lines", last_lines_total))
            except Exception:
                pass
            if bool(step_game_over):
                break

    if publish_event is not None:
        final_state = env.get_state()
        final_payload: Dict[str, object] = {
            "type": "episode_done",
            "mode": "dagger",
            "status": "done" if bool(final_state.get("game_over", False)) else "active",
            "worker_slot": int(worker_slot),
            "worker_label": str(worker_label),
            "worker_key": worker_key,
            "round_id": int(round_id),
            "beta": float(beta),
            "episode_id": int(episode_id),
            "survival_length": int(episode_steps),
            "lines_total": int(final_state.get("lines", last_lines_total)),
            "episode_transitions": int(len(rows)),
            "transitions_collected": int(len(rows)),
            "episodes_completed": 0,
            "expert_steps": int(expert_steps),
            "learner_steps": int(learner_steps),
            "invalid_learner_raw_argmax": int(invalid_learner_raw),
            "unseen_learner_fallback": int(unseen_learner_fallback),
            "timestamp": float(time.time()),
        }
        final_payload["board"] = board_for_event(final_state.get("board"))
        try:
            final_payload["board_piece_ids"] = piece_ids_for_event(env.board_piece_ids(include_active=True))
        except Exception:
            pass
        publish_event(final_payload)

    return {
        "episode_id": int(episode_id),
        "records": rows,
        "num_transitions": int(len(rows)),
        "skipped_no_legal": int(skipped_no_legal),
        "skipped_invalid_expert": int(skipped_invalid_expert),
        "skipped_missing_tuple": int(skipped_missing_tuple),
        "failed_steps": int(failed_steps),
        "invalid_learner_raw_argmax": int(invalid_learner_raw),
        "unseen_learner_fallback": int(unseen_learner_fallback),
        "expert_steps": int(expert_steps),
        "learner_steps": int(learner_steps),
        "label_illegal_count": int(label_illegal_count),
        "generation_attempts": int(generation_attempts),
        "resampled_samples": int(resampled_samples),
        "episodes_cleared_garbage": int(episodes_cleared_garbage),
        "episodes_topout_before_clear": int(episodes_topout_before_clear),
        "episodes_max_steps_before_clear": int(episodes_max_steps_before_clear),
        "episodes_no_data_after_resamples": int(episodes_no_data_after_resamples),
    }


def _collect_episode_worker(episode_id: int) -> Dict[str, object]:
    if _WORKER_ENV is None or _WORKER_AGENT is None or _WORKER_ENCODER_CFG is None:
        raise RuntimeError("DAgger worker state was not initialized.")
    result = _collect_episode_with_env(
        env=_WORKER_ENV,
        learner=_WORKER_AGENT,
        encoder_config=_WORKER_ENCODER_CFG,
        round_id=int(_WORKER_ROUND_ID),
        beta=float(_WORKER_BETA),
        max_steps_per_episode=int(_WORKER_MAX_STEPS),
        think_ms=int(_WORKER_THINK_MS),
        base_seed=int(_WORKER_BASE_SEED),
        episode_id=int(episode_id),
        state_source=str(_WORKER_STATE_SOURCE),
        random_fill_y_max_exclusive=int(_WORKER_RANDOM_FILL_Y_MAX_EXCLUSIVE),
        random_fill_prob=float(_WORKER_RANDOM_FILL_PROB),
        random_max_resamples_per_sample=int(_WORKER_RANDOM_MAX_RESAMPLES_PER_SAMPLE),
        random_post_clear_steps=int(_WORKER_RANDOM_POST_CLEAR_STEPS),
        publish_event=_worker_emit_viewer_event if _WORKER_VIEWER_QUEUE is not None else None,
        publish_every_steps=int(_WORKER_VIEWER_PUBLISH_EVERY_STEPS),
        compact_telemetry=bool(_WORKER_VIEWER_COMPACT_TELEMETRY),
        board_every_steps=int(_WORKER_VIEWER_BOARD_EVERY_STEPS),
        worker_slot=int(_WORKER_SLOT),
        worker_label=str(_WORKER_LABEL),
    )
    result["records_packed"] = _pack_dagger_rows_for_ipc(result.pop("records"))
    result["worker_pid"] = int(os.getpid())
    result["worker_rss_mb"] = float(_current_rss_mb())
    return result


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
        "--max_train_samples",
        str(int(args.max_train_transitions)),
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
    collect_workers: int,
    worker_chunksize: int,
    worker_maxtasksperchild: int,
    progress_mode: str,
    progress_every_sec: float,
    rss_warn_mb: float,
    worker_rss_warn_mb: float,
    progress_path: Path,
    stop_file: Path,
    state_source: str,
    random_fill_y_max_exclusive: int,
    random_fill_prob: float,
    random_max_resamples_per_sample: int,
    random_post_clear_steps: int,
    viewer_enabled: bool,
    viewer_fullscreen: bool,
    viewer_fps: int,
    viewer_publish_every_steps: int,
    viewer_compact_telemetry: bool,
    viewer_board_every_steps: int,
    viewer_max_queue: int,
    viewer_grid_padding: int,
    viewer_min_tile_px: int,
    viewer_agent: int,
    viewer_reopen_file: Optional[Path],
) -> Tuple[Dict[int, Path], Dict[str, object], Path]:
    if stop_file.exists():
        raise RuntimeError(
            f"Stop file already exists at {stop_file}. Remove it before starting round collection."
        )

    progress_mode = str(progress_mode).strip().lower()
    progress_console = progress_mode in ("console", "both")
    progress_json = progress_mode in ("json", "both")
    # Collection-time learner inference is intentionally CPU-first.
    # Multi-process CUDA inference can be unstable and rarely helps throughput here.
    collect_device = "cpu" if device is None else str(device)
    if int(collect_workers) > 1 and collect_device.lower() != "cpu":
        print(
            "[dagger] warning: parallel collection with non-CPU device can be unstable; "
            "forcing CPU for collection workers."
        )
        collect_device = "cpu"

    viewer_reopen_path = (
        Path(viewer_reopen_file)
        if viewer_reopen_file is not None
        else (progress_path.parent / "VIEWER_OPEN")
    )
    viewer_runtime = LiveViewerRuntime(
        log_prefix="dagger",
        enabled=bool(viewer_enabled),
        mode="dagger",
        total_workers=int(collect_workers),
        total_episodes=int(episodes_per_round),
        fullscreen=bool(viewer_fullscreen),
        fps=int(viewer_fps),
        grid_padding=int(viewer_grid_padding),
        min_tile_px=int(viewer_min_tile_px),
        initial_selected_worker=int(viewer_agent),
        run_dir=str(progress_path.parent),
        viewer_max_queue=int(viewer_max_queue),
        reopen_file=viewer_reopen_path,
        round_id=int(round_id),
        beta=float(beta),
    )
    viewer_enabled_local = bool(viewer_runtime.enabled)
    effective_worker_chunksize = int(worker_chunksize)
    if viewer_enabled_local and effective_worker_chunksize > 1:
        print(
            "[dagger] info: viewer is enabled; forcing worker_chunksize=1 "
            "to keep live pumping responsive."
        )
        effective_worker_chunksize = 1

    def emit_viewer_event(event: Dict[str, object]) -> None:
        viewer_runtime.emit(event)

    def pump_viewer(force_draw: bool = False) -> None:
        viewer_runtime.pump(force_draw=bool(force_draw))

    episode_ids = [int(round_id * 1_000_000 + ep) for ep in range(int(episodes_per_round))]
    spool_dir = (
        progress_path.parent
        / f".tmp_dagger_collect_spool_round_{int(round_id):02d}_{int(time.time())}_{os.getpid()}"
    )
    ensure_dir(spool_dir)
    atexit.register(lambda: shutil.rmtree(spool_dir, ignore_errors=True))

    episode_spool_paths: Dict[int, Path] = {}
    worker_rss_by_pid: Dict[int, float] = {}

    episodes_completed = 0
    episodes_with_data = 0
    transitions_collected = 0
    expert_steps = 0
    learner_steps = 0
    skipped_no_legal = 0
    skipped_invalid_expert = 0
    skipped_missing_tuple = 0
    failed_steps = 0
    invalid_learner_raw = 0
    unseen_learner_fallback = 0
    label_illegal_count = 0
    potential_new_vocab: set[ActionTuple] = set()
    generation_attempts = 0
    resampled_samples = 0
    episodes_cleared_garbage = 0
    episodes_topout_before_clear = 0
    episodes_max_steps_before_clear = 0
    episodes_no_data_after_resamples = 0

    stopped_early = False
    stop_reason = ""
    started = time.time()
    last_progress_emit = 0.0
    rss_warned_main = False
    rss_warned_worker = False

    def stop_requested() -> bool:
        return stop_file.exists()

    def make_progress_payload(status: str, final_vocab: Optional[int] = None) -> Dict[str, object]:
        elapsed_sec = max(0.0, time.time() - started)
        eps_per_sec = float(episodes_completed / elapsed_sec) if elapsed_sec > 0 else 0.0
        remaining = max(int(episodes_per_round) - episodes_completed, 0)
        eta_seconds: Optional[float] = None
        if eps_per_sec > 1e-12:
            eta_seconds = float(remaining / eps_per_sec)
        total_actions = expert_steps + learner_steps
        empirical = float(expert_steps / total_actions) if total_actions > 0 else 0.0
        estimated_vocab = (
            int(final_vocab)
            if final_vocab is not None
            else int(len(codec) + len(potential_new_vocab))
        )
        rss_main_mb = float(_current_rss_mb())
        rss_worker_max_mb = (
            float(max(worker_rss_by_pid.values()))
            if worker_rss_by_pid
            else (rss_main_mb if int(collect_workers) == 1 else 0.0)
        )
        payload: Dict[str, object] = {
            "status": status,
            "round_id": int(round_id),
            "beta": float(beta),
            "episodes_total": int(episodes_per_round),
            "episodes_completed": int(episodes_completed),
            "episodes_with_data": int(episodes_with_data),
            "transitions_collected": int(transitions_collected),
            "estimated_vocab_size": int(estimated_vocab),
            "elapsed_seconds": float(elapsed_sec),
            "episodes_per_sec": float(eps_per_sec),
            "eta_seconds": eta_seconds,
            "expert_steps": int(expert_steps),
            "learner_steps": int(learner_steps),
            "empirical_expert_action_rate": float(empirical),
            "invalid_learner_raw_argmax": int(invalid_learner_raw),
            "unseen_learner_fallback": int(unseen_learner_fallback),
            "skipped_no_legal": int(skipped_no_legal),
            "skipped_invalid_expert": int(skipped_invalid_expert),
            "skipped_missing_tuple": int(skipped_missing_tuple),
            "failed_steps": int(failed_steps),
            "label_illegal_count": int(label_illegal_count),
            "generation_attempts": int(generation_attempts),
            "resampled_samples": int(resampled_samples),
            "episodes_cleared_garbage": int(episodes_cleared_garbage),
            "episodes_topout_before_clear": int(episodes_topout_before_clear),
            "episodes_max_steps_before_clear": int(episodes_max_steps_before_clear),
            "episodes_no_data_after_resamples": int(episodes_no_data_after_resamples),
            "collect_workers": int(collect_workers),
            "worker_chunksize": int(effective_worker_chunksize),
            "worker_maxtasksperchild": int(worker_maxtasksperchild),
            "collect_device": str(collect_device),
            "progress_mode": progress_mode,
            "progress_path": str(progress_path).replace("\\", "/"),
            "stop_file": str(stop_file).replace("\\", "/"),
            "rss_main_mb": rss_main_mb,
            "rss_worker_max_mb": rss_worker_max_mb,
            "stopped_early": bool(stopped_early),
            "stop_reason": str(stop_reason),
        }
        if viewer_enabled_local:
            payload.update(viewer_runtime.health_snapshot())
        return payload

    def emit_progress(status: str, force: bool = False, final_vocab: Optional[int] = None) -> None:
        nonlocal last_progress_emit
        nonlocal rss_warned_main
        nonlocal rss_warned_worker

        now = time.time()
        if not force and (now - last_progress_emit) < float(progress_every_sec):
            pump_viewer()
            return

        payload = make_progress_payload(status=status, final_vocab=final_vocab)
        if progress_console:
            eta = payload["eta_seconds"]
            eta_text = "unknown" if eta is None else f"{float(eta):.1f}s"
            print(
                f"[dagger][progress] round={round_id} status={payload['status']} "
                f"episodes={payload['episodes_completed']}/{payload['episodes_total']} "
                f"with_data={payload['episodes_with_data']} transitions={payload['transitions_collected']} "
                f"vocab~={payload['estimated_vocab_size']} eps_per_sec={payload['episodes_per_sec']:.2f} "
                f"eta={eta_text} expert_rate={payload['empirical_expert_action_rate']:.3f} "
                f"rss(main={float(payload['rss_main_mb']):.1f}MiB, "
                f"worker_max={float(payload['rss_worker_max_mb']):.1f}MiB) "
                f"gen_attempts={payload['generation_attempts']} resampled={payload['resampled_samples']} "
                f"clear={payload['episodes_cleared_garbage']} topout={payload['episodes_topout_before_clear']} "
                f"max={payload['episodes_max_steps_before_clear']} no_data={payload['episodes_no_data_after_resamples']} "
                f"skips(no_legal={payload['skipped_no_legal']},invalid_expert={payload['skipped_invalid_expert']},missing={payload['skipped_missing_tuple']})"
            )
            if float(rss_warn_mb) > 0.0 and not rss_warned_main:
                if float(payload["rss_main_mb"]) >= float(rss_warn_mb):
                    print(
                        f"[dagger][warning] main RSS reached {float(payload['rss_main_mb']):.1f} MiB "
                        f"(threshold {float(rss_warn_mb):.1f} MiB)."
                    )
                    rss_warned_main = True
            if float(worker_rss_warn_mb) > 0.0 and not rss_warned_worker:
                if float(payload["rss_worker_max_mb"]) >= float(worker_rss_warn_mb):
                    print(
                        f"[dagger][warning] worker RSS reached {float(payload['rss_worker_max_mb']):.1f} MiB "
                        f"(threshold {float(worker_rss_warn_mb):.1f} MiB)."
                    )
                    rss_warned_worker = True
        if progress_json:
            _write_progress_json(progress_path, payload)
        emit_viewer_event(
            {
                "type": "run_progress",
                "mode": "dagger",
                **payload,
                "run_dir": str(progress_path.parent),
            }
        )
        last_progress_emit = now
        pump_viewer()

    def ingest_episode_result(result: Dict[str, object]) -> None:
        nonlocal episodes_completed
        nonlocal episodes_with_data
        nonlocal transitions_collected
        nonlocal expert_steps
        nonlocal learner_steps
        nonlocal skipped_no_legal
        nonlocal skipped_invalid_expert
        nonlocal skipped_missing_tuple
        nonlocal failed_steps
        nonlocal invalid_learner_raw
        nonlocal unseen_learner_fallback
        nonlocal label_illegal_count
        nonlocal generation_attempts
        nonlocal resampled_samples
        nonlocal episodes_cleared_garbage
        nonlocal episodes_topout_before_clear
        nonlocal episodes_max_steps_before_clear
        nonlocal episodes_no_data_after_resamples

        episode_id = int(result["episode_id"])
        records_obj = result.get("records")
        if records_obj is None and "records_packed" in result:
            records = _unpack_dagger_rows_from_ipc(result["records_packed"])  # type: ignore[arg-type]
        elif isinstance(records_obj, list):
            records = records_obj
        else:
            raise ValueError(f"Expected list records or packed records for episode {episode_id}.")
        if not isinstance(records, list):
            raise ValueError(f"Expected list of records for episode {episode_id}.")

        episodes_completed += 1
        episode_transitions = int(result.get("num_transitions", len(records)))
        transitions_collected += episode_transitions
        if records:
            episodes_with_data += 1
            episode_spool_paths[episode_id] = _write_episode_rows_to_spool(
                spool_dir, episode_id, records
            )

        expert_steps += int(result.get("expert_steps", 0))
        learner_steps += int(result.get("learner_steps", 0))
        skipped_no_legal += int(result.get("skipped_no_legal", 0))
        skipped_invalid_expert += int(result.get("skipped_invalid_expert", 0))
        skipped_missing_tuple += int(result.get("skipped_missing_tuple", 0))
        failed_steps += int(result.get("failed_steps", 0))
        invalid_learner_raw += int(result.get("invalid_learner_raw_argmax", 0))
        unseen_learner_fallback += int(result.get("unseen_learner_fallback", 0))
        label_illegal_count += int(result.get("label_illegal_count", 0))
        generation_attempts += int(result.get("generation_attempts", 0))
        resampled_samples += int(result.get("resampled_samples", 0))
        episodes_cleared_garbage += int(result.get("episodes_cleared_garbage", 0))
        episodes_topout_before_clear += int(result.get("episodes_topout_before_clear", 0))
        episodes_max_steps_before_clear += int(result.get("episodes_max_steps_before_clear", 0))
        episodes_no_data_after_resamples += int(result.get("episodes_no_data_after_resamples", 0))

        worker_pid = int(result.get("worker_pid", 0))
        worker_rss = float(result.get("worker_rss_mb", 0.0))
        if worker_pid > 0 and worker_rss > 0.0:
            worker_rss_by_pid[worker_pid] = worker_rss

        for row in records:
            expert_tup = _normalize_tuple(row["expert_action_tuple"])  # type: ignore[arg-type]
            if expert_tup not in codec.action_to_id:
                potential_new_vocab.add(expert_tup)

        by_episode = episodes_completed % max(1, int(log_every)) == 0
        emit_progress(status="running", force=by_episode)

    keep_spool = False
    try:
        emit_viewer_event(
            {
                "type": "run_started",
                "mode": "dagger",
                "status": "running",
                "round_id": int(round_id),
                "beta": float(beta),
                "episodes_total": int(episodes_per_round),
                "episodes_completed": 0,
                "episodes_with_data": 0,
                "transitions_collected": 0,
                "collect_workers": int(collect_workers),
                "run_dir": str(progress_path.parent),
                "progress_path": str(progress_path).replace("\\", "/"),
            }
        )
        if int(collect_workers) > 1:
            viewer_runtime.emit_starting_workers(
                extra_fields={"round_id": int(round_id), "beta": float(beta)}
            )
        if int(collect_workers) == 1:
            main_pid_label = f"PID {os.getpid()}"
            emit_viewer_event(
                {
                    "type": "worker_started",
                    "mode": "dagger",
                    "status": "active",
                    "worker_slot": 1,
                    "worker_label": main_pid_label,
                    "worker_key": f"pid:{os.getpid()}",
                    "round_id": int(round_id),
                    "beta": float(beta),
                }
            )
        emit_progress(status="running", force=True)
        pump_viewer(force_draw=True)
        try:
            if int(collect_workers) == 1:
                configure_cpu_runtime(
                    torch_num_threads=1,
                    torch_num_interop_threads=1,
                    omp_num_threads=1,
                    mkl_num_threads=1,
                    openblas_num_threads=1,
                )
                with BCEnvAdapter(lib_path=lib_path, seed=seed + round_id) as env:
                    learner = BCAgent(
                        checkpoint_path=learner_checkpoint,
                        device=collect_device,
                        env_adapter=env,
                    )

                    def publish_and_pump(event: Dict[str, object]) -> None:
                        emit_viewer_event(event)
                        pump_viewer()

                    for episode_id in episode_ids:
                        if stop_requested():
                            stopped_early = True
                            stop_reason = f"manual stop file detected: {stop_file}"
                            emit_progress(status="stopping", force=True)
                            break
                        result = _collect_episode_with_env(
                            env=env,
                            learner=learner,
                            encoder_config=encoder_config,
                            round_id=int(round_id),
                            beta=float(beta),
                            max_steps_per_episode=int(max_steps_per_episode),
                            think_ms=int(think_ms),
                            base_seed=int(seed),
                            episode_id=int(episode_id),
                            state_source=str(state_source),
                            random_fill_y_max_exclusive=int(random_fill_y_max_exclusive),
                            random_fill_prob=float(random_fill_prob),
                            random_max_resamples_per_sample=int(random_max_resamples_per_sample),
                            random_post_clear_steps=int(random_post_clear_steps),
                            publish_event=publish_and_pump if viewer_enabled_local else None,
                            publish_every_steps=int(viewer_publish_every_steps),
                            compact_telemetry=bool(viewer_compact_telemetry),
                            board_every_steps=int(viewer_board_every_steps),
                            worker_slot=1,
                            worker_label=main_pid_label,
                        )
                        result["worker_pid"] = int(os.getpid())
                        result["worker_rss_mb"] = float(_current_rss_mb())
                        ingest_episode_result(result)
                        pump_viewer()
            else:
                ctx = mp.get_context("spawn")
                pump_viewer(force_draw=True)
                pool = ctx.Pool(
                    processes=int(collect_workers),
                    initializer=_worker_init,
                    initargs=(
                        str(lib_path),
                        str(learner_checkpoint),
                        collect_device,
                        {
                            "board_height": int(encoder_config.board_height),
                            "board_width": int(encoder_config.board_width),
                            "queue_length": int(encoder_config.queue_length),
                            "include_scalars": bool(encoder_config.include_scalars),
                        },
                        int(round_id),
                        float(beta),
                        int(max_steps_per_episode),
                        int(think_ms),
                        int(seed),
                        str(state_source),
                        int(random_fill_y_max_exclusive),
                        float(random_fill_prob),
                        int(random_max_resamples_per_sample),
                        int(random_post_clear_steps),
                        viewer_runtime.worker_queue if viewer_enabled_local else None,
                        int(viewer_publish_every_steps),
                        bool(viewer_compact_telemetry),
                        int(viewer_board_every_steps),
                    ),
                    maxtasksperchild=(
                        int(worker_maxtasksperchild)
                        if int(worker_maxtasksperchild) > 0
                        else None
                    ),
                )
                pump_viewer(force_draw=True)
                try:
                    iterator = pool.imap_unordered(
                        _collect_episode_worker,
                        episode_ids,
                        chunksize=int(effective_worker_chunksize),
                    )
                    expected_results = len(episode_ids)
                    received_results = 0
                    while received_results < expected_results:
                        if stop_requested():
                            stopped_early = True
                            stop_reason = f"manual stop file detected: {stop_file}"
                            emit_progress(status="stopping", force=True)
                            break
                        try:
                            result = _iterator_next_with_timeout(
                                iterator,
                                timeout=max(0.01, 1.0 / float(max(5, int(viewer_fps)))),
                            )
                        except mp.TimeoutError:
                            pump_viewer()
                            continue
                        except StopIteration:
                            break
                        ingest_episode_result(result)
                        received_results += 1
                        if stop_requested():
                            stopped_early = True
                            stop_reason = f"manual stop file detected: {stop_file}"
                            emit_progress(status="stopping", force=True)
                            break
                        pump_viewer()
                finally:
                    if stopped_early:
                        pool.terminate()
                    else:
                        pool.close()
                    pool.join()
        except Exception as exc:
            emit_viewer_event(
                {
                    "type": "run_done",
                    "mode": "dagger",
                    "status": "failed",
                    "round_id": int(round_id),
                    "beta": float(beta),
                    "episodes_total": int(episodes_per_round),
                    "episodes_completed": int(episodes_completed),
                    "episodes_with_data": int(episodes_with_data),
                    "transitions_collected": int(transitions_collected),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            pump_viewer()
            viewer_runtime.close()
            if progress_json:
                payload = make_progress_payload(status="failed")
                payload["error"] = f"{type(exc).__name__}: {exc}"
                _write_progress_json(progress_path, payload)
            raise

        if transitions_collected <= 0:
            emit_viewer_event(
                {
                    "type": "run_done",
                    "mode": "dagger",
                    "status": "failed",
                    "round_id": int(round_id),
                    "beta": float(beta),
                    "episodes_total": int(episodes_per_round),
                    "episodes_completed": int(episodes_completed),
                    "episodes_with_data": int(episodes_with_data),
                    "transitions_collected": int(transitions_collected),
                    "error": "No DAgger transitions were collected in this round.",
                }
            )
            pump_viewer()
            viewer_runtime.close()
            if progress_json:
                payload = make_progress_payload(status="failed")
                payload["error"] = "No DAgger transitions were collected in this round."
                _write_progress_json(progress_path, payload)
            raise RuntimeError("No DAgger transitions were collected in this round.")

        vocab_start = len(codec)
        for episode_id in sorted(episode_spool_paths):
            rows = _load_episode_rows_from_spool(episode_spool_paths[episode_id])
            rows.sort(key=lambda row: int(row["step_idx"]))
            for row in rows:
                expert_tuple = _normalize_tuple(row["expert_action_tuple"])  # type: ignore[arg-type]
                codec.encode_tuple(expert_tuple, add_if_missing=True)

        elapsed = max(0.0, time.time() - started)
        total_actions = expert_steps + learner_steps
        empirical = float(expert_steps / total_actions) if total_actions > 0 else 0.0
        vocab_end = len(codec)
        stats = {
            "round_id": int(round_id),
            "beta": float(beta),
            "episodes_requested": int(episodes_per_round),
            "episodes_completed": int(episodes_completed),
            "episodes_with_data": int(episodes_with_data),
            "transitions": int(transitions_collected),
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
            "generation_attempts": int(generation_attempts),
            "resampled_samples": int(resampled_samples),
            "episodes_cleared_garbage": int(episodes_cleared_garbage),
            "episodes_topout_before_clear": int(episodes_topout_before_clear),
            "episodes_max_steps_before_clear": int(episodes_max_steps_before_clear),
            "episodes_no_data_after_resamples": int(episodes_no_data_after_resamples),
            "vocab_start": int(vocab_start),
            "vocab_end": int(vocab_end),
            "vocab_delta": int(vocab_end - vocab_start),
            "vocab_new_labels_seen": int(vocab_end - vocab_start),
            "collect_workers": int(collect_workers),
            "worker_chunksize": int(effective_worker_chunksize),
            "worker_maxtasksperchild": int(worker_maxtasksperchild),
            "collect_device": str(collect_device),
            "progress_mode": progress_mode,
            "progress_path": str(progress_path).replace("\\", "/"),
            "stop_file": str(stop_file).replace("\\", "/"),
            "rss_warn_mb": float(rss_warn_mb),
            "worker_rss_warn_mb": float(worker_rss_warn_mb),
            "rss_main_final_mb": float(_current_rss_mb()),
            "rss_worker_max_mb": (
                float(max(worker_rss_by_pid.values()))
                if worker_rss_by_pid
                else float(_current_rss_mb())
            ),
            "stopped_early": bool(stopped_early),
            "stop_reason": str(stop_reason),
            "viewer": bool(viewer_enabled_local),
            "viewer_fullscreen": bool(viewer_fullscreen),
            "viewer_fps": int(viewer_fps),
            "viewer_publish_every_steps": int(viewer_publish_every_steps),
            "viewer_compact_telemetry": bool(viewer_compact_telemetry),
            "viewer_board_every_steps": int(viewer_board_every_steps),
            "viewer_max_queue": int(viewer_max_queue),
            "viewer_grid_padding": int(viewer_grid_padding),
            "viewer_min_tile_px": int(viewer_min_tile_px),
            "viewer_agent": int(viewer_agent),
            "viewer_reopen_file": str(viewer_reopen_path).replace("\\", "/"),
            "state_source": str(state_source),
            "random_fill_y_max_exclusive": int(random_fill_y_max_exclusive),
            "random_fill_prob": float(random_fill_prob),
            "random_max_resamples_per_sample": int(random_max_resamples_per_sample),
            "random_post_clear_steps": int(random_post_clear_steps),
        }

        final_status = "stopped" if stopped_early else "done"
        emit_progress(status=final_status, force=True, final_vocab=vocab_end)
        emit_viewer_event(
            {
                "type": "run_done",
                "mode": "dagger",
                "status": final_status,
                "round_id": int(round_id),
                "beta": float(beta),
                "episodes_total": int(episodes_per_round),
                "episodes_completed": int(episodes_completed),
                "episodes_with_data": int(episodes_with_data),
                "transitions_collected": int(transitions_collected),
                "collect_workers": int(collect_workers),
                "run_dir": str(progress_path.parent),
            }
        )
        pump_viewer()
        viewer_runtime.close()

        keep_spool = True
        return episode_spool_paths, stats, spool_dir
    finally:
        viewer_runtime.close()
        if not keep_spool:
            shutil.rmtree(spool_dir, ignore_errors=True)


def _write_round_train_shards(
    out_dir: Path,
    episode_spool_paths: Dict[int, Path],
    codec: ActionCodec,
    episodes_per_shard: int,
) -> Tuple[List[str], int, List[int]]:
    shard_dir = out_dir / "shards"
    ensure_dir(shard_dir)

    shard_paths: List[str] = []
    transitions = 0
    episode_ids_with_data = sorted(int(ep) for ep in episode_spool_paths.keys())
    episode_ids_with_data.sort()

    for shard_idx, begin in enumerate(range(0, len(episode_ids_with_data), int(episodes_per_shard))):
        chunk_ids = episode_ids_with_data[begin : begin + int(episodes_per_shard)]
        rows: List[Dict[str, object]] = []
        for ep_id in chunk_ids:
            spool_path = episode_spool_paths.get(int(ep_id))
            if spool_path is None:
                continue
            raw_rows = _load_episode_rows_from_spool(spool_path)
            raw_rows.sort(key=lambda row: int(row["step_idx"]))
            for row in raw_rows:
                expert_tuple = _normalize_tuple(row["expert_action_tuple"])  # type: ignore[arg-type]
                learner_tuple = _normalize_tuple(row["learner_action_tuple"])  # type: ignore[arg-type]
                expert_action_id = int(codec.action_to_id[expert_tuple])
                learner_action_id = (
                    int(codec.action_to_id[learner_tuple])
                    if learner_tuple in codec.action_to_id
                    else -1
                )
                rows.append(
                    {
                        "board": row["board"],
                        "piece": row["piece"],
                        "hold": row["hold"],
                        "queue": row["queue"],
                        "scalars": row["scalars"],
                        "action_id": expert_action_id,
                        "action_tuple": np.asarray(expert_tuple, dtype=np.int64),
                        "episode_id": int(row["episode_id"]),
                        "step_idx": int(row["step_idx"]),
                        "round_id": int(row["round_id"]),
                        "learner_action_id": learner_action_id,
                        "learner_action_tuple": np.asarray(learner_tuple, dtype=np.int64),
                        "acted_by_expert": int(row["acted_by_expert"]),
                        "learner_raw_invalid": int(row["learner_raw_invalid"]),
                        "learner_used_fallback": int(row["learner_used_fallback"]),
                    }
                )
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
            "state_source": str(args.state_source),
            "random_fill_y_max_exclusive": int(args.random_fill_y_max_exclusive),
            "random_fill_prob": float(args.random_fill_prob),
            "random_max_resamples_per_sample": int(args.random_max_resamples_per_sample),
            "random_post_clear_steps": int(args.random_post_clear_steps),
            "beta_schedule": str(args.beta_schedule),
            "beta_start": float(args.beta_start),
            "beta_end": float(args.beta_end),
            "beta_decay": float(args.beta_decay),
            "beta_decay_rounds": int(args.beta_decay_rounds),
            "fine_tune": bool(args.fine_tune),
            "eval_games": int(args.eval_games),
            "max_train_transitions": int(args.max_train_transitions),
            "collect_workers": int(args.collect_workers),
            "worker_chunksize": int(args.worker_chunksize),
            "worker_maxtasksperchild": int(args.worker_maxtasksperchild),
            "progress_mode": str(args.progress_mode),
            "progress_every_sec": float(args.progress_every_sec),
            "progress_path": str(args.progress_path) if args.progress_path is not None else None,
            "stop_file": str(args.stop_file) if args.stop_file is not None else None,
            "rss_warn_mb": float(args.rss_warn_mb),
            "worker_rss_warn_mb": float(args.worker_rss_warn_mb),
            "viewer": bool(args.viewer),
            "viewer_fullscreen": bool(args.viewer_fullscreen),
            "viewer_fps": int(args.viewer_fps),
            "viewer_publish_every_steps": int(args.viewer_publish_every_steps),
            "viewer_compact_telemetry": bool(args.viewer_compact_telemetry),
            "viewer_board_every_steps": int(args.viewer_board_every_steps),
            "viewer_max_queue": int(args.viewer_max_queue),
            "viewer_grid_padding": int(args.viewer_grid_padding),
            "viewer_min_tile_px": int(args.viewer_min_tile_px),
            "viewer_agent": int(args.viewer_agent),
            "viewer_reopen_file": str(args.viewer_reopen_file) if args.viewer_reopen_file is not None else None,
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
        round_progress_path = (
            Path(args.progress_path).resolve()
            if args.progress_path is not None
            else (round_dir / "progress.json").resolve()
        )
        round_stop_file = (
            Path(args.stop_file).resolve()
            if args.stop_file is not None
            else (round_dir / "STOP").resolve()
        )

        round_spool_paths, collect_stats, round_spool_dir = _collect_dagger_round(
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
            collect_workers=int(args.collect_workers),
            worker_chunksize=int(args.worker_chunksize),
            worker_maxtasksperchild=int(args.worker_maxtasksperchild),
            progress_mode=str(args.progress_mode),
            progress_every_sec=float(args.progress_every_sec),
            rss_warn_mb=float(args.rss_warn_mb),
            worker_rss_warn_mb=float(args.worker_rss_warn_mb),
            progress_path=round_progress_path,
            stop_file=round_stop_file,
            state_source=str(args.state_source),
            random_fill_y_max_exclusive=int(args.random_fill_y_max_exclusive),
            random_fill_prob=float(args.random_fill_prob),
            random_max_resamples_per_sample=int(args.random_max_resamples_per_sample),
            random_post_clear_steps=int(args.random_post_clear_steps),
            viewer_enabled=bool(args.viewer),
            viewer_fullscreen=bool(args.viewer_fullscreen),
            viewer_fps=int(args.viewer_fps),
            viewer_publish_every_steps=int(args.viewer_publish_every_steps),
            viewer_compact_telemetry=bool(args.viewer_compact_telemetry),
            viewer_board_every_steps=int(args.viewer_board_every_steps),
            viewer_max_queue=int(args.viewer_max_queue),
            viewer_grid_padding=int(args.viewer_grid_padding),
            viewer_min_tile_px=int(args.viewer_min_tile_px),
            viewer_agent=int(args.viewer_agent),
            viewer_reopen_file=(
                Path(args.viewer_reopen_file).resolve()
                if args.viewer_reopen_file is not None
                else None
            ),
        )
        try:
            train_shard_root = round_dir / "dagger_train"
            round_train_shards, round_train_transitions, round_episode_ids = _write_round_train_shards(
                out_dir=train_shard_root,
                episode_spool_paths=round_spool_paths,
                codec=codec,
                episodes_per_shard=int(args.episodes_per_shard),
            )
        finally:
            shutil.rmtree(round_spool_dir, ignore_errors=True)
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
        train_samples_used = int(train_summary.get("num_train_samples_used", train_summary.get("num_train_samples", 0)))

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
                "total_train_samples_available": int(round_total_train_samples),
                "total_train_samples_used": int(train_samples_used),
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
