from __future__ import annotations

import atexit
import argparse
import json
import multiprocessing as mp
import os
import queue as queue_mod
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

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
    configure_cpu_runtime,
    save_json,
    set_global_seeds,
    split_episode_ids,
)
from .viewer_live import board_for_event, piece_ids_for_event, queue_put_best_effort


_WORKER_ENV: Optional[BCEnvAdapter] = None
_WORKER_ENCODER_CFG: Optional[EncoderConfig] = None
_WORKER_BASE_SEED: int = 0
_WORKER_MAX_STEPS: int = 0
_WORKER_THINK_MS: int = 20
_WORKER_VIEWER_QUEUE: Any = None
_WORKER_VIEWER_PUBLISH_EVERY_STEPS: int = 10
_WORKER_VIEWER_COMPACT_TELEMETRY: bool = True
_WORKER_VIEWER_BOARD_EVERY_STEPS: int = 50
_WORKER_SLOT: int = 1
_WORKER_LABEL: str = "PID"


def _argv_has_flag(argv: Sequence[str], flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in argv)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    argv_list = list(sys.argv[1:] if argv is None else argv)
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
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--collect_workers", type=int, default=1)
    parser.add_argument(
        "--progress_mode",
        type=str,
        choices=("console", "json", "both"),
        default="both",
    )
    parser.add_argument("--progress_every_sec", type=float, default=2.0)
    parser.add_argument("--progress_path", type=Path, default=None)
    parser.add_argument("--worker_chunksize", type=int, default=1)
    parser.add_argument(
        "--worker_maxtasksperchild",
        type=int,
        default=64,
        help="Recycle worker processes after this many tasks (0 disables recycling).",
    )
    parser.add_argument(
        "--torch_num_threads",
        type=int,
        default=1,
        help="Torch intra-op threads used by collection workers (0 keeps defaults).",
    )
    parser.add_argument(
        "--torch_num_interop_threads",
        type=int,
        default=1,
        help="Torch inter-op threads used by collection workers (0 keeps defaults).",
    )
    parser.add_argument("--omp_num_threads", type=int, default=1)
    parser.add_argument("--mkl_num_threads", type=int, default=1)
    parser.add_argument("--openblas_num_threads", type=int, default=1)
    parser.add_argument(
        "--stop_file",
        type=Path,
        default=None,
        help="If this file appears while collecting, stop early and save partial dataset.",
    )
    parser.add_argument("--viewer", action="store_true", help="Enable live pygame collection viewer.")
    parser.add_argument(
        "--viewer_fullscreen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start viewer in fullscreen mode (default: true).",
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


def _normalize_action_tuple(raw: Sequence[int] | np.ndarray) -> Tuple[int, int, int, int, int]:
    values = tuple(int(v) for v in raw)
    if len(values) != 5:
        raise ValueError(f"Action tuple must have length 5, got {values}.")
    return (
        int(values[0]),
        int(values[1]),
        int(values[2]),
        int(values[3]),
        int(values[4]),
    )


def _pack_records_for_ipc(records: List[Dict[str, object]]) -> Dict[str, object]:
    if not records:
        return {"n": 0}
    return {
        "n": int(len(records)),
        "board": np.stack([r["board"] for r in records], axis=0).astype(np.float32, copy=False),
        "piece": np.stack([r["piece"] for r in records], axis=0).astype(np.float32, copy=False),
        "hold": np.stack([r["hold"] for r in records], axis=0).astype(np.float32, copy=False),
        "queue": np.stack([r["queue"] for r in records], axis=0).astype(np.float32, copy=False),
        "scalars": np.stack([r["scalars"] for r in records], axis=0).astype(np.float32, copy=False),
        "action_tuple": np.asarray([r["action_tuple"] for r in records], dtype=np.int64),
        "episode_id": np.asarray([r["episode_id"] for r in records], dtype=np.int64),
        "step_idx": np.asarray([r["step_idx"] for r in records], dtype=np.int64),
    }


def _unpack_records_from_ipc(payload: Dict[str, object]) -> List[Dict[str, object]]:
    n = int(payload.get("n", 0))
    if n <= 0:
        return []
    board = np.asarray(payload["board"], dtype=np.float32)
    piece = np.asarray(payload["piece"], dtype=np.float32)
    hold = np.asarray(payload["hold"], dtype=np.float32)
    queue = np.asarray(payload["queue"], dtype=np.float32)
    scalars = np.asarray(payload["scalars"], dtype=np.float32)
    action_tuple = np.asarray(payload["action_tuple"], dtype=np.int64)
    episode_id = np.asarray(payload["episode_id"], dtype=np.int64)
    step_idx = np.asarray(payload["step_idx"], dtype=np.int64)
    out: List[Dict[str, object]] = []
    for i in range(n):
        out.append(
            {
                "board": board[i],
                "piece": piece[i],
                "hold": hold[i],
                "queue": queue[i],
                "scalars": scalars[i],
                "action_tuple": action_tuple[i],
                "episode_id": int(episode_id[i]),
                "step_idx": int(step_idx[i]),
            }
        )
    return out


def _worker_emit_viewer_event(event: Dict[str, object]) -> None:
    if _WORKER_VIEWER_QUEUE is None:
        return
    payload = dict(event)
    payload.setdefault("type", "step_snapshot")
    payload.setdefault("mode", "collect_data")
    payload.setdefault("timestamp", float(time.time()))
    payload.setdefault("worker_slot", int(_WORKER_SLOT))
    payload.setdefault("worker_label", str(_WORKER_LABEL))
    payload.setdefault("worker_key", f"pid:{os.getpid()}")
    queue_put_best_effort(_WORKER_VIEWER_QUEUE, payload)


def _collect_episode_with_env(
    env: BCEnvAdapter,
    episode_id: int,
    base_seed: int,
    max_steps_per_episode: int,
    think_ms: int,
    encoder_cfg: EncoderConfig,
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

    env.reset(base_seed + int(episode_id))
    records: List[Dict[str, object]] = []
    skipped_no_legal = 0
    skipped_invalid_expert = 0
    skipped_missing_tuple = 0
    episode_steps = 0
    last_lines_total = 0
    publish_step_interval = max(1, int(publish_every_steps))
    board_step_interval = max(1, int(board_every_steps))

    for step_idx in range(int(max_steps_per_episode)):
        state = env.get_state()
        if bool(state["game_over"]):
            break

        legal_actions = env.enumerate_legal_actions()
        if not legal_actions:
            skipped_no_legal += 1
            break

        legal_by_native = {native.key(): tup for native, tup in legal_actions}
        legal_tuples = [tup for _, tup in legal_actions]

        expert = env.expert_choose_and_apply(think_ms=think_ms)
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

        encoded = encode_state(state, encoder_cfg)
        records.append(
            {
                "board": encoded["board"],
                "piece": encoded["piece"],
                "hold": encoded["hold"],
                "queue": encoded["queue"],
                "scalars": encoded["scalars"],
                "action_tuple": list(_normalize_action_tuple(chosen_tuple)),
                "episode_id": int(episode_id),
                "step_idx": int(step_idx),
            }
        )
        episode_steps = int(step_idx + 1)
        last_lines_total = int(state.get("lines", 0))

        done = bool(expert["game_over"])
        if publish_event is not None and ((step_idx + 1) % publish_step_interval == 0 or done):
            include_board = (not bool(compact_telemetry)) or done or ((step_idx + 1) % board_step_interval == 0)
            state_now = env.get_state()
            event_payload: Dict[str, object] = {
                "type": "step_snapshot",
                "mode": "collect_data",
                "status": "done" if done else "active",
                "worker_slot": int(worker_slot),
                "worker_label": str(worker_label),
                "worker_key": worker_key,
                "episode_id": int(episode_id),
                "step_in_episode": int(step_idx + 1),
                "lines_total": int(state_now.get("lines", 0)),
                "transitions_collected": int(len(records)),
                "skipped_no_legal": int(skipped_no_legal),
                "skipped_invalid_expert": int(skipped_invalid_expert),
                "skipped_missing_tuple": int(skipped_missing_tuple),
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

        if bool(expert["game_over"]):
            break

    if publish_event is not None:
        final_state = env.get_state()
        final_payload: Dict[str, object] = {
            "type": "episode_done",
            "mode": "collect_data",
            "status": "done" if bool(final_state.get("game_over", False)) else "active",
            "worker_slot": int(worker_slot),
            "worker_label": str(worker_label),
            "worker_key": worker_key,
            "episode_id": int(episode_id),
            "survival_length": int(episode_steps),
            "lines_total": int(final_state.get("lines", last_lines_total)),
            "episode_transitions": int(len(records)),
            "transitions_collected": int(len(records)),
            "episodes_completed": 0,
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
        "records": records,
        "num_transitions": int(len(records)),
        "skipped_no_legal": int(skipped_no_legal),
        "skipped_invalid_expert": int(skipped_invalid_expert),
        "skipped_missing_tuple": int(skipped_missing_tuple),
    }


def _close_worker_env() -> None:
    global _WORKER_ENV
    if _WORKER_ENV is not None:
        _WORKER_ENV.close()
        _WORKER_ENV = None


def _worker_init(
    lib_path_str: str,
    base_seed: int,
    max_steps_per_episode: int,
    think_ms: int,
    encoder_cfg_dict: Dict[str, object],
    torch_num_threads: int,
    torch_num_interop_threads: int,
    omp_num_threads: int,
    mkl_num_threads: int,
    openblas_num_threads: int,
    viewer_queue: Any,
    viewer_publish_every_steps: int,
    viewer_compact_telemetry: bool,
    viewer_board_every_steps: int,
) -> None:
    global _WORKER_ENV
    global _WORKER_ENCODER_CFG
    global _WORKER_BASE_SEED
    global _WORKER_MAX_STEPS
    global _WORKER_THINK_MS
    global _WORKER_VIEWER_QUEUE
    global _WORKER_VIEWER_PUBLISH_EVERY_STEPS
    global _WORKER_VIEWER_COMPACT_TELEMETRY
    global _WORKER_VIEWER_BOARD_EVERY_STEPS
    global _WORKER_SLOT
    global _WORKER_LABEL

    configure_cpu_runtime(
        torch_num_threads=max(0, int(torch_num_threads)),
        torch_num_interop_threads=max(0, int(torch_num_interop_threads)),
        omp_num_threads=max(0, int(omp_num_threads)),
        mkl_num_threads=max(0, int(mkl_num_threads)),
        openblas_num_threads=max(0, int(openblas_num_threads)),
    )

    _WORKER_BASE_SEED = int(base_seed)
    _WORKER_MAX_STEPS = int(max_steps_per_episode)
    _WORKER_THINK_MS = int(think_ms)
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
    _WORKER_ENV = BCEnvAdapter(lib_path=Path(lib_path_str), seed=_WORKER_BASE_SEED)
    atexit.register(_close_worker_env)
    _worker_emit_viewer_event(
        {
            "type": "worker_started",
            "mode": "collect_data",
            "status": "active",
            "worker_slot": int(_WORKER_SLOT),
            "worker_label": str(_WORKER_LABEL),
            "worker_key": f"pid:{os.getpid()}",
        }
    )


def _collect_episode_worker(episode_id: int) -> Dict[str, object]:
    if _WORKER_ENV is None or _WORKER_ENCODER_CFG is None:
        raise RuntimeError("Worker not initialized.")
    result = _collect_episode_with_env(
        env=_WORKER_ENV,
        episode_id=int(episode_id),
        base_seed=_WORKER_BASE_SEED,
        max_steps_per_episode=_WORKER_MAX_STEPS,
        think_ms=_WORKER_THINK_MS,
        encoder_cfg=_WORKER_ENCODER_CFG,
        publish_event=_worker_emit_viewer_event if _WORKER_VIEWER_QUEUE is not None else None,
        publish_every_steps=int(_WORKER_VIEWER_PUBLISH_EVERY_STEPS),
        compact_telemetry=bool(_WORKER_VIEWER_COMPACT_TELEMETRY),
        board_every_steps=int(_WORKER_VIEWER_BOARD_EVERY_STEPS),
        worker_slot=int(_WORKER_SLOT),
        worker_label=str(_WORKER_LABEL),
    )
    result["records_packed"] = _pack_records_for_ipc(result.pop("records"))
    return result


def _write_progress_json(path: Path, payload: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp") if path.suffix else Path(str(path) + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _assign_action_ids(episode_records: Dict[int, List[Dict[str, object]]]) -> ActionCodec:
    codec = ActionCodec()
    for episode_id in sorted(episode_records):
        records = episode_records[int(episode_id)]
        records.sort(key=lambda row: int(row["step_idx"]))
        for row in records:
            action_tuple = _normalize_action_tuple(row["action_tuple"])  # type: ignore[arg-type]
            row["action_tuple"] = np.asarray(action_tuple, dtype=np.int64)
            row["action_id"] = int(codec.encode_tuple(action_tuple, add_if_missing=True))
    return codec


def main() -> int:
    args = parse_args()

    if int(args.collect_workers) <= 0:
        raise ValueError(f"--collect_workers must be >= 1, got {args.collect_workers}.")
    if float(args.progress_every_sec) <= 0.0:
        raise ValueError(f"--progress_every_sec must be > 0, got {args.progress_every_sec}.")
    if int(args.worker_chunksize) <= 0:
        raise ValueError(f"--worker_chunksize must be >= 1, got {args.worker_chunksize}.")
    if int(args.worker_maxtasksperchild) < 0:
        raise ValueError(
            f"--worker_maxtasksperchild must be >= 0, got {args.worker_maxtasksperchild}."
        )
    for name in (
        "torch_num_threads",
        "torch_num_interop_threads",
        "omp_num_threads",
        "mkl_num_threads",
        "openblas_num_threads",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name} must be >= 0, got {getattr(args, name)}.")
    if int(args.viewer_fps) <= 0:
        raise ValueError(f"--viewer_fps must be > 0, got {args.viewer_fps}.")
    if int(args.viewer_publish_every_steps) <= 0:
        raise ValueError(
            f"--viewer_publish_every_steps must be >= 1, got {args.viewer_publish_every_steps}."
        )
    if int(args.viewer_board_every_steps) <= 0:
        raise ValueError(
            f"--viewer_board_every_steps must be >= 1, got {args.viewer_board_every_steps}."
        )
    if int(args.viewer_max_queue) <= 0:
        raise ValueError(f"--viewer_max_queue must be >= 1, got {args.viewer_max_queue}.")

    configure_cpu_runtime(
        torch_num_threads=max(0, int(args.torch_num_threads)),
        torch_num_interop_threads=max(0, int(args.torch_num_interop_threads)),
        omp_num_threads=max(0, int(args.omp_num_threads)),
        mkl_num_threads=max(0, int(args.mkl_num_threads)),
        openblas_num_threads=max(0, int(args.openblas_num_threads)),
    )
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
    progress_path = Path(args.progress_path) if args.progress_path is not None else out_dir / "progress.json"
    progress_mode = str(args.progress_mode).strip().lower()
    progress_console = progress_mode in ("console", "both")
    progress_json = progress_mode in ("json", "both")
    stop_file = Path(args.stop_file) if args.stop_file is not None else out_dir / "STOP"
    if stop_file.exists():
        raise RuntimeError(
            f"Stop file already exists at {stop_file}. Remove it before starting collection."
        )

    viewer_enabled = bool(args.viewer)
    viewer: Any = None
    viewer_queue: Any = None
    viewer_manager: Any = None
    live_viewer_cls: Any = None
    viewer_max_events = max(1, int(args.viewer_max_queue))
    viewer_event_buffer: deque[Dict[str, object]] = deque(maxlen=viewer_max_events)
    viewer_reopen_file = (
        Path(args.viewer_reopen_file)
        if args.viewer_reopen_file is not None
        else (progress_path.parent / "VIEWER_OPEN")
    )

    def create_viewer_instance() -> Any:
        if live_viewer_cls is None:
            return None
        try:
            next_viewer = live_viewer_cls(
                mode="collect_data",
                total_workers=int(args.collect_workers),
                total_episodes=int(args.num_episodes),
                fullscreen=bool(args.viewer_fullscreen),
                fps=int(args.viewer_fps),
                grid_padding=int(args.viewer_grid_padding),
                min_tile_px=int(args.viewer_min_tile_px),
                initial_selected_worker=int(args.viewer_agent),
                run_dir=str(out_dir),
            )
            if not next_viewer.ready:
                return None
            return next_viewer
        except Exception:
            return None

    if viewer_enabled:
        try:
            from .viewer_live import LiveCollectionViewer

            live_viewer_cls = LiveCollectionViewer
            viewer = create_viewer_instance()
            if viewer is None:
                print(
                    "[collect_data] warning: viewer init failed; starting headless. "
                    f"Create '{viewer_reopen_file}' to retry open."
                )
            if int(args.collect_workers) > 1:
                viewer_manager = mp.Manager()
                viewer_queue = viewer_manager.Queue(max(1, int(args.viewer_max_queue)))
        except Exception as exc:
            print(f"[collect_data] warning: viewer unavailable ({exc}); continuing headless.")
            viewer_enabled = False
            viewer = None

    def emit_viewer_event(event: Dict[str, object]) -> None:
        if not viewer_enabled:
            return
        event_payload = dict(event)
        event_payload.setdefault("mode", "collect_data")
        event_payload.setdefault("timestamp", float(time.time()))
        if viewer_queue is not None:
            if not queue_put_best_effort(viewer_queue, event_payload):
                viewer_event_buffer.append(event_payload)
            return
        viewer_event_buffer.append(event_payload)
        if viewer is not None and not viewer.closed:
            viewer.process_event(event_payload)

    def pump_viewer() -> None:
        nonlocal viewer
        if not viewer_enabled:
            return
        if viewer_queue is not None:
            for _ in range(2048):
                try:
                    ev = viewer_queue.get_nowait()
                except queue_mod.Empty:
                    break
                except Exception:
                    break
                viewer_event_buffer.append(ev)
                if viewer is not None and not viewer.closed:
                    viewer.process_event(ev)

        if viewer is not None and viewer.closed:
            viewer = None

        if viewer is None and live_viewer_cls is not None and viewer_reopen_file.exists():
            try:
                viewer_reopen_file.unlink(missing_ok=True)
            except Exception:
                pass
            viewer = create_viewer_instance()
            if viewer is not None:
                for ev in viewer_event_buffer:
                    viewer.process_event(ev)
                print(f"[collect_data] viewer reopened via trigger: {viewer_reopen_file}")
            else:
                print(
                    "[collect_data] warning: viewer reopen failed. "
                    f"Create '{viewer_reopen_file}' again to retry."
                )

        if viewer is not None and not viewer.closed:
            viewer.tick()

    episode_records: Dict[int, List[Dict[str, object]]] = {}
    skipped_no_legal = 0
    skipped_invalid_expert = 0
    skipped_missing_tuple = 0
    episodes_completed = 0
    episodes_with_data = 0
    transitions_collected = 0
    observed_action_tuples: Set[Tuple[int, int, int, int, int]] = set()
    started_at = time.time()
    last_progress_emit = 0.0
    stopped_early = False
    stop_reason = ""

    def stop_requested() -> bool:
        return stop_file.exists()

    def make_progress_payload(status: str, final_vocab: Optional[int] = None) -> Dict[str, object]:
        elapsed_sec = max(0.0, time.time() - started_at)
        eps_per_sec = float(episodes_completed / elapsed_sec) if elapsed_sec > 0 else 0.0
        remaining = max(int(args.num_episodes) - episodes_completed, 0)
        eta_seconds: Optional[float] = None
        if eps_per_sec > 1e-12:
            eta_seconds = float(remaining / eps_per_sec)
        return {
            "status": status,
            "episodes_total": int(args.num_episodes),
            "episodes_completed": int(episodes_completed),
            "episodes_with_data": int(episodes_with_data),
            "transitions_collected": int(transitions_collected),
            "observed_action_tuples": int(
                final_vocab if final_vocab is not None else len(observed_action_tuples)
            ),
            "skipped_no_legal": int(skipped_no_legal),
            "skipped_invalid_expert": int(skipped_invalid_expert),
            "skipped_missing_tuple": int(skipped_missing_tuple),
            "elapsed_seconds": float(elapsed_sec),
            "episodes_per_sec": float(eps_per_sec),
            "eta_seconds": eta_seconds,
            "collect_workers": int(args.collect_workers),
            "progress_mode": progress_mode,
            "progress_path": str(progress_path).replace("\\", "/"),
            "stop_file": str(stop_file).replace("\\", "/"),
            "worker_maxtasksperchild": int(args.worker_maxtasksperchild),
            "torch_num_threads": int(args.torch_num_threads),
            "torch_num_interop_threads": int(args.torch_num_interop_threads),
            "omp_num_threads": int(args.omp_num_threads),
            "mkl_num_threads": int(args.mkl_num_threads),
            "openblas_num_threads": int(args.openblas_num_threads),
            "stopped_early": bool(stopped_early),
            "stop_reason": str(stop_reason),
        }

    def emit_progress(status: str, force: bool = False, final_vocab: Optional[int] = None) -> None:
        nonlocal last_progress_emit
        now = time.time()
        if not force and (now - last_progress_emit) < float(args.progress_every_sec):
            pump_viewer()
            return
        payload = make_progress_payload(status=status, final_vocab=final_vocab)
        if progress_console:
            eta = payload["eta_seconds"]
            eta_text = "unknown" if eta is None else f"{float(eta):.1f}s"
            print(
                f"[collect_data][progress] status={payload['status']} "
                f"episodes={payload['episodes_completed']}/{payload['episodes_total']} "
                f"with_data={payload['episodes_with_data']} "
                f"transitions={payload['transitions_collected']} "
                f"vocab={payload['observed_action_tuples']} "
                f"eps_per_sec={payload['episodes_per_sec']:.2f} "
                f"eta={eta_text} "
                f"skipped(no_legal={payload['skipped_no_legal']}, "
                f"invalid_expert={payload['skipped_invalid_expert']}, "
                f"missing_tuple={payload['skipped_missing_tuple']})"
            )
        if progress_json:
            _write_progress_json(progress_path, payload)
        emit_viewer_event(
            {
                "type": "run_progress",
                "mode": "collect_data",
                **payload,
                "run_dir": str(out_dir),
            }
        )
        last_progress_emit = now
        pump_viewer()

    def ingest_episode_result(result: Dict[str, object]) -> None:
        nonlocal episodes_completed
        nonlocal episodes_with_data
        nonlocal transitions_collected
        nonlocal skipped_no_legal
        nonlocal skipped_invalid_expert
        nonlocal skipped_missing_tuple

        episode_id = int(result["episode_id"])
        records_obj = result.get("records")
        if records_obj is None and "records_packed" in result:
            records = _unpack_records_from_ipc(result["records_packed"])  # type: ignore[arg-type]
        elif isinstance(records_obj, list):
            records = records_obj
        else:
            raise ValueError(f"Expected list records or packed records for episode {episode_id}.")
        if not isinstance(records, list):
            raise ValueError(f"Expected list of records for episode {episode_id}.")
        episode_records[episode_id] = records

        episodes_completed += 1
        episode_transitions = int(result.get("num_transitions", len(records)))
        transitions_collected += episode_transitions
        if records:
            episodes_with_data += 1
        skipped_no_legal += int(result.get("skipped_no_legal", 0))
        skipped_invalid_expert += int(result.get("skipped_invalid_expert", 0))
        skipped_missing_tuple += int(result.get("skipped_missing_tuple", 0))

        for row in records:
            observed_action_tuples.add(_normalize_action_tuple(row["action_tuple"]))  # type: ignore[arg-type]

        by_episode = episodes_completed % max(1, int(args.log_every)) == 0
        emit_progress(status="running", force=by_episode)

    emit_viewer_event(
        {
            "type": "run_started",
            "mode": "collect_data",
            "status": "running",
            "episodes_total": int(args.num_episodes),
            "episodes_completed": 0,
            "episodes_with_data": 0,
            "transitions_collected": 0,
            "collect_workers": int(args.collect_workers),
            "progress_path": str(progress_path).replace("\\", "/"),
            "run_dir": str(out_dir),
        }
    )
    if int(args.collect_workers) == 1:
        main_pid_label = f"PID {os.getpid()}"
        emit_viewer_event(
            {
                "type": "worker_started",
                "mode": "collect_data",
                "status": "active",
                "worker_slot": 1,
                "worker_label": main_pid_label,
                "worker_key": f"pid:{os.getpid()}",
            }
        )
    emit_progress(status="running", force=True)

    try:
        if int(args.collect_workers) == 1:
            with BCEnvAdapter(lib_path=lib_path, seed=args.seed) as env:
                def publish_and_pump(event: Dict[str, object]) -> None:
                    emit_viewer_event(event)
                    pump_viewer()

                for episode_id in range(int(args.num_episodes)):
                    if stop_requested():
                        stopped_early = True
                        stop_reason = f"manual stop file detected: {stop_file}"
                        emit_progress(status="stopping", force=True)
                        break
                    result = _collect_episode_with_env(
                        env=env,
                        episode_id=int(episode_id),
                        base_seed=int(args.seed),
                        max_steps_per_episode=int(args.max_steps_per_episode),
                        think_ms=int(args.think_ms),
                        encoder_cfg=encoder_cfg,
                        publish_event=publish_and_pump if viewer_enabled else None,
                        publish_every_steps=int(args.viewer_publish_every_steps),
                        compact_telemetry=bool(args.viewer_compact_telemetry),
                        board_every_steps=int(args.viewer_board_every_steps),
                        worker_slot=1,
                        worker_label=main_pid_label,
                    )
                    ingest_episode_result(result)
                    pump_viewer()
        else:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(
                processes=int(args.collect_workers),
                initializer=_worker_init,
                initargs=(
                    str(lib_path),
                    int(args.seed),
                    int(args.max_steps_per_episode),
                    int(args.think_ms),
                    dataclass_to_dict(encoder_cfg),
                    int(args.torch_num_threads),
                    int(args.torch_num_interop_threads),
                    int(args.omp_num_threads),
                    int(args.mkl_num_threads),
                    int(args.openblas_num_threads),
                    viewer_queue if viewer_enabled else None,
                    int(args.viewer_publish_every_steps),
                    bool(args.viewer_compact_telemetry),
                    int(args.viewer_board_every_steps),
                ),
                maxtasksperchild=(
                    int(args.worker_maxtasksperchild)
                    if int(args.worker_maxtasksperchild) > 0
                    else None
                ),
            )
            try:
                iterator = pool.imap_unordered(
                    _collect_episode_worker,
                    range(int(args.num_episodes)),
                    chunksize=int(args.worker_chunksize),
                )
                expected_results = int(args.num_episodes)
                received_results = 0
                while received_results < expected_results:
                    if stop_requested():
                        stopped_early = True
                        stop_reason = f"manual stop file detected: {stop_file}"
                        emit_progress(status="stopping", force=True)
                        break
                    try:
                        result = iterator.next(
                            timeout=max(0.01, 1.0 / float(max(5, int(args.viewer_fps))))
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
                "mode": "collect_data",
                "status": "failed",
                "episodes_total": int(args.num_episodes),
                "episodes_completed": int(episodes_completed),
                "episodes_with_data": int(episodes_with_data),
                "transitions_collected": int(transitions_collected),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        pump_viewer()
        if viewer is not None:
            viewer.close()
        if viewer_manager is not None:
            try:
                viewer_manager.shutdown()
            except Exception:
                pass
        if progress_json:
            payload = make_progress_payload(status="failed")
            payload["error"] = f"{type(exc).__name__}: {exc}"
            _write_progress_json(progress_path, payload)
        raise

    episodes_with_data_ids = [ep for ep, rows in episode_records.items() if rows]
    if not episodes_with_data_ids:
        emit_viewer_event(
            {
                "type": "run_done",
                "mode": "collect_data",
                "status": "failed",
                "episodes_total": int(args.num_episodes),
                "episodes_completed": int(episodes_completed),
                "episodes_with_data": int(episodes_with_data),
                "transitions_collected": int(transitions_collected),
                "error": "No transitions were collected. Check build/library configuration.",
            }
        )
        pump_viewer()
        if viewer is not None:
            viewer.close()
        if viewer_manager is not None:
            try:
                viewer_manager.shutdown()
            except Exception:
                pass
        if progress_json:
            payload = make_progress_payload(status="failed")
            payload["error"] = "No transitions were collected. Check build/library configuration."
            _write_progress_json(progress_path, payload)
        raise RuntimeError("No transitions were collected. Check build/library configuration.")

    codec = _assign_action_ids(episode_records)

    splits = split_episode_ids(
        episodes_with_data_ids,
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
    elapsed_total = float(max(0.0, time.time() - started_at))

    metadata = {
        "format_version": 1,
        "num_episodes_requested": int(args.num_episodes),
        "stopped_early": bool(stopped_early),
        "stop_reason": str(stop_reason),
        "num_episodes_with_data": int(len(episodes_with_data_ids)),
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
            "collect_workers": int(args.collect_workers),
            "worker_chunksize": int(args.worker_chunksize),
            "worker_maxtasksperchild": int(args.worker_maxtasksperchild),
            "torch_num_threads": int(args.torch_num_threads),
            "torch_num_interop_threads": int(args.torch_num_interop_threads),
            "omp_num_threads": int(args.omp_num_threads),
            "mkl_num_threads": int(args.mkl_num_threads),
            "openblas_num_threads": int(args.openblas_num_threads),
            "progress_mode": progress_mode,
            "progress_every_sec": float(args.progress_every_sec),
            "progress_path": str(progress_path).replace("\\", "/"),
            "stop_file": str(stop_file).replace("\\", "/"),
            "viewer": bool(viewer_enabled),
            "viewer_fullscreen": bool(args.viewer_fullscreen),
            "viewer_fps": int(args.viewer_fps),
            "viewer_publish_every_steps": int(args.viewer_publish_every_steps),
            "viewer_compact_telemetry": bool(args.viewer_compact_telemetry),
            "viewer_board_every_steps": int(args.viewer_board_every_steps),
            "viewer_max_queue": int(args.viewer_max_queue),
            "viewer_grid_padding": int(args.viewer_grid_padding),
            "viewer_min_tile_px": int(args.viewer_min_tile_px),
            "viewer_agent": int(args.viewer_agent),
            "viewer_reopen_file": str(viewer_reopen_file).replace("\\", "/"),
        },
        "split_config": dataclass_to_dict(split_cfg),
        "splits": split_meta,
        "action_vocab_size": int(len(codec)),
        "id_to_action": [list(tup) for tup in codec.id_to_action],
        "collection_runtime": {
            "elapsed_sec": elapsed_total,
            "episodes_per_sec": float(int(args.num_episodes) / max(1e-9, elapsed_total)),
        },
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
        f"vocab={metadata['action_vocab_size']} "
        f"stopped_early={metadata['stopped_early']}"
    )
    print("[collect_data] top action classes:")
    for class_id, count in top_classes[:10]:
        print(f"  class={class_id:4d} count={count:8d}")

    final_status = "stopped" if stopped_early else "done"
    emit_progress(status=final_status, force=True, final_vocab=int(len(codec)))
    emit_viewer_event(
        {
            "type": "run_done",
            "mode": "collect_data",
            "status": final_status,
            "episodes_total": int(args.num_episodes),
            "episodes_completed": int(episodes_completed),
            "episodes_with_data": int(episodes_with_data),
            "transitions_collected": int(transitions_collected),
            "collect_workers": int(args.collect_workers),
            "run_dir": str(out_dir),
        }
    )
    pump_viewer()
    if viewer is not None:
        viewer.close()
    if viewer_manager is not None:
        try:
            viewer_manager.shutdown()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
