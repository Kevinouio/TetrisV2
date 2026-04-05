import argparse
import ctypes
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame


BOARD_ROWS = 20
BOARD_COLS = 10
EMPTY_CELL_ID = 255

ACTION_NONE = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_SOFT_DROP = 3
ACTION_HARD_DROP = 4
ACTION_CW = 5
ACTION_CCW = 6
ACTION_180 = 7
ACTION_HOLD = 8

MODE_LEGACY = 0
MODE_ZEN = 1
MODE_SCORING = 2
MODE_VERSUS = 3

MODE_NAME_TO_ID = {
    "legacy": MODE_LEGACY,
    "zen": MODE_ZEN,
    "scoring": MODE_SCORING,
    "blitz": MODE_SCORING,
    "versus": MODE_VERSUS,
}

PIECE_NAMES = {
    0: "I",
    1: "O",
    2: "T",
    3: "L",
    4: "J",
    5: "S",
    6: "Z",
    7: "None",
}

ROTATION_NAMES = {0: "N", 1: "E", 2: "S", 3: "W"}

PIECE_COLORS = {
    0: (0, 230, 230),
    1: (240, 220, 0),
    2: (170, 70, 230),
    3: (255, 150, 60),
    4: (80, 110, 255),
    5: (80, 220, 100),
    6: (240, 90, 90),
}
GARBAGE_COLOR = (130, 130, 130)

BG_COLOR = (14, 16, 22)
PANEL_COLOR = (22, 26, 34)
GRID_LINE = (50, 56, 70)
LOCK_TEXT = (230, 235, 245)
PASS_COLOR = (110, 230, 150)
FAIL_COLOR = (240, 120, 120)
SELECT_COLOR = (80, 130, 240)
BOARD_FILL = (70, 80, 100)
ActionTuple = Tuple[int, int, int, int, int]  # (use_hold, piece, rotation, x, y)


@dataclass(frozen=True)
class NativeAction:
    use_hold: bool
    placement_index: int


@dataclass(frozen=True)
class DQNCandidate:
    native_action: NativeAction
    action_tuple: ActionTuple
    feature_vector: np.ndarray
    lines_removed: int
    y_pos: int


def parse_args():
    parser = argparse.ArgumentParser(description="Pygame Placement + Kick Explorer via ctypes.")
    parser.add_argument("--lib", type=Path, default=None, help="Path to tetris_v2_c_api shared library.")
    parser.add_argument("--cell", type=int, default=28, help="Main board cell size.")
    parser.add_argument("--fps", type=int, default=60, help="Render FPS.")
    parser.add_argument("--seed", type=int, default=1234, help="Initial reset seed.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("legacy", "zen", "scoring", "blitz", "versus"),
        default="legacy",
        help="Scoring/attack mode to run.",
    )
    parser.add_argument(
        "--ai-vs-ai",
        action="store_true",
        help="In --mode versus, control both boards with native AI.",
    )
    parser.add_argument("--queue-visible", type=int, default=8, help="How many queued pieces to display.")
    parser.add_argument("--ai", action="store_true", help="Start with AI autoplay enabled.")
    parser.add_argument("--think-ms", type=int, default=20, help="AI think budget per move in milliseconds.")
    parser.add_argument(
        "--ai-pps",
        type=float,
        default=0.0,
        help="Cap AI move rate in pieces/second. Use 0 for unbounded.",
    )
    parser.add_argument(
        "--native-backend",
        type=str,
        choices=("cold_clear", "depth", "beam"),
        default="cold_clear",
        help="Native bot backend to use when no learned checkpoint is provided.",
    )
    parser.add_argument(
        "--depth-search-depth",
        type=int,
        default=1,
        help="Depth backend placement horizon (depth=1 means 1-ply).",
    )
    parser.add_argument(
        "--depth-gamma",
        type=float,
        default=1.0,
        help="Depth backend discount factor.",
    )
    parser.add_argument(
        "--beam-search-depth",
        type=int,
        default=2,
        help="Beam backend placement horizon (depth=1 means 1-ply).",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=8,
        help="Beam backend width (survivors per layer).",
    )
    parser.add_argument(
        "--beam-gamma",
        type=float,
        default=1.0,
        help="Beam backend discount factor.",
    )
    parser.add_argument(
        "--state-source",
        type=str,
        choices=("rollout", "random_board"),
        default="rollout",
        help="Episode start source: normal rollout or DAgger-like random board injection.",
    )
    parser.add_argument(
        "--random-fill-y-max-exclusive",
        type=int,
        default=17,
        help="Bottom rows eligible for random fill in random_board mode.",
    )
    parser.add_argument(
        "--random-fill-prob",
        type=float,
        default=0.5,
        help="Bernoulli fill probability used in random_board mode.",
    )
    parser.add_argument(
        "--random-max-resamples-per-sample",
        type=int,
        default=100,
        help="Max random-board resamples to find a valid non-empty non-terminal start.",
    )
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        default=None,
        help="Path to BC checkpoint. If set, autoplay uses BC instead of Cold Clear.",
    )
    parser.add_argument(
        "--bc-device",
        type=str,
        default=None,
        help="Torch device for BC inference (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--dqn-checkpoint",
        type=Path,
        default=None,
        help="Path to DQN checkpoint. If set, autoplay uses DQN instead of BC/Cold Clear.",
    )
    parser.add_argument(
        "--dqn-device",
        type=str,
        default=None,
        help="Torch device for DQN inference (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--auto-reset",
        action="store_true",
        default=True,
        help="Auto reset to next seed on topout in AI mode (default: on).",
    )
    parser.add_argument(
        "--no-auto-reset",
        action="store_false",
        dest="auto_reset",
        help="Disable AI auto reset on topout.",
    )
    args = parser.parse_args()
    if int(args.random_fill_y_max_exclusive) < 0 or int(args.random_fill_y_max_exclusive) > 20:
        raise SystemExit("--random-fill-y-max-exclusive must be in [0,20].")
    if not (0.0 <= float(args.random_fill_prob) <= 1.0):
        raise SystemExit("--random-fill-prob must be in [0,1].")
    if int(args.random_max_resamples_per_sample) <= 0:
        raise SystemExit("--random-max-resamples-per-sample must be > 0.")
    if int(args.beam_width) <= 0:
        raise SystemExit("--beam-width must be > 0.")
    if float(args.ai_pps) < 0.0:
        raise SystemExit("--ai-pps must be >= 0.")
    return args


_DEPTH_BACKEND_C_API_SYMBOLS = (
    "tetris_cc_bot_set_backend",
    "tetris_cc_bot_get_backend",
    "tetris_cc_bot_set_depth_config",
)
_BEAM_BACKEND_C_API_SYMBOLS = (
    "tetris_cc_bot_set_backend",
    "tetris_cc_bot_get_backend",
    "tetris_cc_bot_set_beam_config",
)


def _symbols_for_native_backend(backend_name: Optional[str]):
    if backend_name == "depth":
        return _DEPTH_BACKEND_C_API_SYMBOLS
    if backend_name == "beam":
        return _BEAM_BACKEND_C_API_SYMBOLS
    return ()


def _has_c_api_symbols(path: Path, symbols) -> bool:
    try:
        lib = ctypes.CDLL(str(path))
    except OSError:
        return False
    return all(hasattr(lib, name) for name in symbols)


def find_library(
    explicit_path: Optional[Path],
    *,
    require_native_backend: Optional[str] = None,
    prefer_native_backend: Optional[str] = None,
) -> Path:
    shared_find_library = None
    try:
        from TetrisVersionTwo.scripts.bc.utils import find_library as shared_find_library
    except Exception:
        # Keep this viewer script usable as a standalone fallback if package imports fail.
        shared_find_library = None
    candidates = [
        Path("build/TetrisVersionTwo/tetris_v2_c_api.dll"),
        Path("build/TetrisVersionTwo/Debug/tetris_v2_c_api.dll"),
        Path("build/TetrisVersionTwo/Release/tetris_v2_c_api.dll"),
        Path("build/TetrisVersionTwo/libtetris_v2_c_api.so"),
        Path("build/TetrisVersionTwo/libtetris_v2_c_api.dylib"),
        Path("TetrisVersionTwo/build/tetris_v2_c_api.dll"),
        Path("TetrisVersionTwo/build/Debug/tetris_v2_c_api.dll"),
        Path("TetrisVersionTwo/build/Release/tetris_v2_c_api.dll"),
        Path("TetrisVersionTwo/build/libtetris_v2_c_api.so"),
        Path("TetrisVersionTwo/build/libtetris_v2_c_api.dylib"),
    ]

    require_symbols = _symbols_for_native_backend(require_native_backend)
    prefer_symbols = _symbols_for_native_backend(prefer_native_backend)
    required_backend_name = str(require_native_backend) if require_native_backend else None

    def has_required_symbols(candidate: Path, symbols) -> bool:
        if not symbols:
            return True
        return _has_c_api_symbols(candidate, symbols)

    def pick_candidate_from_search_paths(*, require_symbols, prefer_symbols) -> Optional[Path]:
        existing = [candidate for candidate in candidates if candidate.exists()]
        if not existing:
            return None
        if require_symbols or prefer_symbols:
            for candidate in existing:
                if has_required_symbols(candidate, require_symbols if require_symbols else prefer_symbols):
                    return candidate
        if require_symbols:
            return None
        return existing[0]

    if shared_find_library is not None:
        candidate = shared_find_library(explicit_path)
        has_required = has_required_symbols(candidate, require_symbols)
        if has_required:
            return candidate
        if require_symbols:
            if explicit_path is not None:
                raise RuntimeError(
                    f"Library '{candidate}' is missing required {required_backend_name} backend symbols: "
                    f"{', '.join(require_symbols)}"
                )
            fallback = pick_candidate_from_search_paths(
                require_symbols=require_symbols, prefer_symbols=require_symbols
            )
            if fallback is not None:
                return fallback
            raise RuntimeError(
                "Found tetris_v2_c_api shared libraries, but none export required "
                f"{required_backend_name} backend symbols."
            )
        if prefer_symbols and explicit_path is None:
            fallback = pick_candidate_from_search_paths(
                require_symbols=(), prefer_symbols=prefer_symbols
            )
            if fallback is not None:
                return fallback
        return candidate

    if explicit_path is not None:
        if explicit_path.exists():
            if require_symbols and not has_required_symbols(explicit_path, require_symbols):
                raise RuntimeError(
                    f"Library '{explicit_path}' is missing required {required_backend_name} backend symbols: "
                    f"{', '.join(require_symbols)}"
                )
            return explicit_path
        raise FileNotFoundError(f"Library not found: {explicit_path}")

    symbol_failures = []
    for candidate in candidates:
        if candidate.exists():
            if require_symbols and not has_required_symbols(candidate, require_symbols):
                symbol_failures.append(str(candidate))
                continue
            if prefer_symbols and not has_required_symbols(candidate, prefer_symbols):
                symbol_failures.append(str(candidate))
                continue
            return candidate
    if prefer_symbols and not require_symbols:
        for candidate in candidates:
            if candidate.exists():
                return candidate
    if require_symbols and symbol_failures:
        raise RuntimeError(
            "Found tetris_v2_c_api shared libraries, but none export required "
            f"{required_backend_name} backend symbols. Tried: {', '.join(symbol_failures)}"
        )
    raise FileNotFoundError("Could not locate tetris_v2_c_api shared library. Build it first.")


class EnvCtypes:
    def __init__(self, lib_path: Path, seed: int):
        self.lib = ctypes.CDLL(str(lib_path))
        self._bind()
        self.handle = self.lib.tetris_cc_env_create(ctypes.c_uint32(seed))
        if not self.handle:
            raise RuntimeError("Failed to create env handle")
        self.bot_handle = None
        if self._has_bot_api:
            self.bot_handle = self.lib.tetris_cc_bot_create_default()
            if not self.bot_handle:
                raise RuntimeError("Failed to create bot handle")
            self.bot_sync()
        self.seed = int(seed)

    def _bind(self):
        void_p = ctypes.c_void_p
        c_int_p = ctypes.POINTER(ctypes.c_int)
        self._has_piece_id_api = all(
            hasattr(self.lib, name)
            for name in (
                "tetris_cc_env_board_piece_ids_write",
                "tetris_cc_env_placement_board_piece_ids_write",
            )
        )
        self._has_bot_api = all(
            hasattr(self.lib, name)
            for name in (
                "tetris_cc_bot_create_default",
                "tetris_cc_bot_destroy",
                "tetris_cc_bot_sync_from_env",
                "tetris_cc_bot_choose",
                "tetris_cc_bot_apply_choice",
                "tetris_cc_bot_choose_and_apply",
            )
        )
        self._has_bot_api_ex = self._has_bot_api and all(
            hasattr(self.lib, name)
            for name in (
                "tetris_cc_bot_choose_ex",
                "tetris_cc_bot_choose_and_apply_ex",
            )
        )
        self._has_bot_backend_api = self._has_bot_api and all(
            hasattr(self.lib, name)
            for name in (
                "tetris_cc_bot_set_backend",
                "tetris_cc_bot_get_backend",
            )
        )
        self._has_depth_backend_config_api = self._has_bot_api and hasattr(
            self.lib, "tetris_cc_bot_set_depth_config"
        )
        self._has_beam_backend_config_api = self._has_bot_api and hasattr(
            self.lib, "tetris_cc_bot_set_beam_config"
        )
        self._has_candidate_batch_api = all(
            hasattr(self.lib, name)
            for name in (
                "tetris_cc_env_candidate_count",
                "tetris_cc_env_candidate_get",
                "tetris_cc_env_candidate_features_write",
            )
        )
        self._has_visible_garbage_count = hasattr(self.lib, "tetris_cc_env_visible_garbage_count")
        self._has_set_visible_board_mask = hasattr(self.lib, "tetris_cc_env_set_visible_board_mask")
        self._has_mode_api = all(
            hasattr(self.lib, name)
            for name in (
                "tetris_cc_env_set_mode",
                "tetris_cc_env_get_mode",
            )
        )
        self._has_last_attack_meta_api = hasattr(self.lib, "tetris_cc_env_last_attack_meta")
        self._has_blitz_meta_api = hasattr(self.lib, "tetris_cc_env_blitz_meta")
        self._has_apply_incoming_garbage_api = hasattr(self.lib, "tetris_cc_env_apply_incoming_garbage")

        self.lib.tetris_cc_env_create.argtypes = [ctypes.c_uint32]
        self.lib.tetris_cc_env_create.restype = void_p

        self.lib.tetris_cc_env_destroy.argtypes = [void_p]
        self.lib.tetris_cc_env_destroy.restype = None

        self.lib.tetris_cc_env_reset.argtypes = [void_p, ctypes.c_uint32]
        self.lib.tetris_cc_env_reset.restype = None
        if self._has_mode_api:
            self.lib.tetris_cc_env_set_mode.argtypes = [void_p, ctypes.c_int]
            self.lib.tetris_cc_env_set_mode.restype = ctypes.c_int
            self.lib.tetris_cc_env_get_mode.argtypes = [void_p, c_int_p]
            self.lib.tetris_cc_env_get_mode.restype = ctypes.c_int

        self.lib.tetris_cc_env_snapshot_create.argtypes = [void_p]
        self.lib.tetris_cc_env_snapshot_create.restype = void_p
        self.lib.tetris_cc_snapshot_destroy.argtypes = [void_p]
        self.lib.tetris_cc_snapshot_destroy.restype = None
        self.lib.tetris_cc_env_restore_snapshot.argtypes = [void_p, void_p]
        self.lib.tetris_cc_env_restore_snapshot.restype = ctypes.c_int

        self.lib.tetris_cc_env_step.argtypes = [void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        self.lib.tetris_cc_env_step.restype = ctypes.c_int

        self.lib.tetris_cc_env_hold.argtypes = [void_p, ctypes.POINTER(ctypes.c_float)]
        self.lib.tetris_cc_env_hold.restype = ctypes.c_int
        if self._has_apply_incoming_garbage_api:
            self.lib.tetris_cc_env_apply_incoming_garbage.argtypes = [
                void_p,
                ctypes.c_int,
                c_int_p,
                c_int_p,
            ]
            self.lib.tetris_cc_env_apply_incoming_garbage.restype = ctypes.c_int

        self.lib.tetris_cc_env_observation_size.argtypes = [void_p, ctypes.c_int]
        self.lib.tetris_cc_env_observation_size.restype = ctypes.c_size_t

        self.lib.tetris_cc_env_board_write.argtypes = [
            void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        self.lib.tetris_cc_env_board_write.restype = ctypes.c_size_t

        if self._has_visible_garbage_count:
            self.lib.tetris_cc_env_visible_garbage_count.argtypes = [void_p]
            self.lib.tetris_cc_env_visible_garbage_count.restype = ctypes.c_size_t
        if self._has_set_visible_board_mask:
            self.lib.tetris_cc_env_set_visible_board_mask.argtypes = [
                void_p,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            self.lib.tetris_cc_env_set_visible_board_mask.restype = ctypes.c_int
        if self._has_last_attack_meta_api:
            self.lib.tetris_cc_env_last_attack_meta.argtypes = [
                void_p,
                c_int_p,
                ctypes.POINTER(ctypes.c_float),
                c_int_p,
                c_int_p,
                c_int_p,
                c_int_p,
                c_int_p,
                c_int_p,
                c_int_p,
                c_int_p,
            ]
            self.lib.tetris_cc_env_last_attack_meta.restype = ctypes.c_int
        if self._has_blitz_meta_api:
            self.lib.tetris_cc_env_blitz_meta.argtypes = [
                void_p,
                c_int_p,
                c_int_p,
                c_int_p,
                c_int_p,
                c_int_p,
            ]
            self.lib.tetris_cc_env_blitz_meta.restype = ctypes.c_int

        if self._has_piece_id_api:
            self.lib.tetris_cc_env_board_piece_ids_write.argtypes = [
                void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t,
            ]
            self.lib.tetris_cc_env_board_piece_ids_write.restype = ctypes.c_size_t

        self.lib.tetris_cc_env_active_piece.argtypes = [void_p, c_int_p, c_int_p, c_int_p, c_int_p]
        self.lib.tetris_cc_env_active_piece.restype = ctypes.c_int

        self.lib.tetris_cc_env_hold_piece.argtypes = [void_p, c_int_p, c_int_p, c_int_p]
        self.lib.tetris_cc_env_hold_piece.restype = ctypes.c_int

        self.lib.tetris_cc_env_queue_count.argtypes = [void_p]
        self.lib.tetris_cc_env_queue_count.restype = ctypes.c_size_t

        self.lib.tetris_cc_env_queue_get.argtypes = [void_p, ctypes.c_size_t, c_int_p]
        self.lib.tetris_cc_env_queue_get.restype = ctypes.c_int

        self.lib.tetris_cc_env_meta.argtypes = [void_p, c_int_p, c_int_p, c_int_p, c_int_p, c_int_p, c_int_p, c_int_p]
        self.lib.tetris_cc_env_meta.restype = ctypes.c_int

        self.lib.tetris_cc_env_placement_count.argtypes = [void_p]
        self.lib.tetris_cc_env_placement_count.restype = ctypes.c_size_t

        self.lib.tetris_cc_env_placement_get.argtypes = [void_p, ctypes.c_size_t, c_int_p, c_int_p, c_int_p, c_int_p]
        self.lib.tetris_cc_env_placement_get.restype = ctypes.c_int

        self.lib.tetris_cc_env_placement_board_write.argtypes = [
            void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        self.lib.tetris_cc_env_placement_board_write.restype = ctypes.c_size_t

        if self._has_piece_id_api:
            self.lib.tetris_cc_env_placement_board_piece_ids_write.argtypes = [
                void_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t,
            ]
            self.lib.tetris_cc_env_placement_board_piece_ids_write.restype = ctypes.c_size_t

        if self._has_candidate_batch_api:
            size_p = ctypes.POINTER(ctypes.c_size_t)
            self.lib.tetris_cc_env_candidate_count.argtypes = [void_p]
            self.lib.tetris_cc_env_candidate_count.restype = ctypes.c_size_t
            self.lib.tetris_cc_env_candidate_get.argtypes = [
                void_p,
                ctypes.c_size_t,
                c_int_p,
                size_p,
                c_int_p,
                c_int_p,
                c_int_p,
                c_int_p,
                c_int_p,
            ]
            self.lib.tetris_cc_env_candidate_get.restype = ctypes.c_int
            self.lib.tetris_cc_env_candidate_features_write.argtypes = [
                void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
            ]
            self.lib.tetris_cc_env_candidate_features_write.restype = ctypes.c_size_t

        self.lib.tetris_cc_env_apply_placement_index.argtypes = [
            void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            c_int_p,
            c_int_p,
        ]
        self.lib.tetris_cc_env_apply_placement_index.restype = ctypes.c_int

        self.lib.tetris_cc_env_rotation_trace_count.argtypes = [void_p, ctypes.c_int]
        self.lib.tetris_cc_env_rotation_trace_count.restype = ctypes.c_size_t

        self.lib.tetris_cc_env_rotation_trace_get.argtypes = [
            void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            c_int_p,
            c_int_p,
            c_int_p,
            c_int_p,
            c_int_p,
            c_int_p,
            c_int_p,
            c_int_p,
            c_int_p,
            c_int_p,
        ]
        self.lib.tetris_cc_env_rotation_trace_get.restype = ctypes.c_int

        self.lib.tetris_cc_env_rotation_trace_meta.argtypes = [void_p, ctypes.c_int, c_int_p, c_int_p, c_int_p, c_int_p]
        self.lib.tetris_cc_env_rotation_trace_meta.restype = ctypes.c_int

        if self._has_bot_api:
            size_p = ctypes.POINTER(ctypes.c_size_t)
            u64_p = ctypes.POINTER(ctypes.c_uint64)
            d_p = ctypes.POINTER(ctypes.c_double)

            self.lib.tetris_cc_bot_create_default.argtypes = []
            self.lib.tetris_cc_bot_create_default.restype = void_p
            self.lib.tetris_cc_bot_destroy.argtypes = [void_p]
            self.lib.tetris_cc_bot_destroy.restype = None
            self.lib.tetris_cc_bot_sync_from_env.argtypes = [void_p, void_p]
            self.lib.tetris_cc_bot_sync_from_env.restype = ctypes.c_int
            self.lib.tetris_cc_bot_choose.argtypes = [
                void_p,
                ctypes.c_int,
                c_int_p,
                size_p,
                ctypes.POINTER(ctypes.c_float),
                u64_p,
                d_p,
                d_p,
                c_int_p,
            ]
            self.lib.tetris_cc_bot_choose.restype = ctypes.c_int
            self.lib.tetris_cc_bot_apply_choice.argtypes = [
                void_p,
                void_p,
                ctypes.POINTER(ctypes.c_float),
                c_int_p,
                c_int_p,
                c_int_p,
                size_p,
            ]
            self.lib.tetris_cc_bot_apply_choice.restype = ctypes.c_int
            self.lib.tetris_cc_bot_choose_and_apply.argtypes = [
                void_p,
                void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                c_int_p,
                c_int_p,
                c_int_p,
                size_p,
                ctypes.POINTER(ctypes.c_float),
                u64_p,
                d_p,
                d_p,
                c_int_p,
            ]
            self.lib.tetris_cc_bot_choose_and_apply.restype = ctypes.c_int
            if self._has_bot_api_ex:
                self.lib.tetris_cc_bot_choose_ex.argtypes = [
                    void_p,
                    ctypes.c_int,
                    c_int_p,
                    size_p,
                    ctypes.POINTER(ctypes.c_float),
                    u64_p,
                    d_p,
                    d_p,
                    c_int_p,
                ]
                self.lib.tetris_cc_bot_choose_ex.restype = ctypes.c_int
                self.lib.tetris_cc_bot_choose_and_apply_ex.argtypes = [
                    void_p,
                    void_p,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_float),
                    c_int_p,
                    c_int_p,
                    c_int_p,
                    size_p,
                    ctypes.POINTER(ctypes.c_float),
                    u64_p,
                    d_p,
                    d_p,
                    c_int_p,
                ]
                self.lib.tetris_cc_bot_choose_and_apply_ex.restype = ctypes.c_int
            if self._has_bot_backend_api:
                self.lib.tetris_cc_bot_set_backend.argtypes = [void_p, ctypes.c_int]
                self.lib.tetris_cc_bot_set_backend.restype = ctypes.c_int
                self.lib.tetris_cc_bot_get_backend.argtypes = [void_p, c_int_p]
                self.lib.tetris_cc_bot_get_backend.restype = ctypes.c_int
            if self._has_depth_backend_config_api:
                self.lib.tetris_cc_bot_set_depth_config.argtypes = [
                    void_p,
                    ctypes.c_int,
                    ctypes.c_double,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_uint64,
                ]
                self.lib.tetris_cc_bot_set_depth_config.restype = ctypes.c_int
            if self._has_beam_backend_config_api:
                self.lib.tetris_cc_bot_set_beam_config.argtypes = [
                    void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_double,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_uint64,
                ]
                self.lib.tetris_cc_bot_set_beam_config.restype = ctypes.c_int

    def close(self):
        if self.bot_handle and self._has_bot_api:
            self.lib.tetris_cc_bot_destroy(self.bot_handle)
            self.bot_handle = None
        if self.handle:
            self.lib.tetris_cc_env_destroy(self.handle)
            self.handle = None

    def reset(self, seed: int):
        self.seed = int(seed)
        self.lib.tetris_cc_env_reset(self.handle, ctypes.c_uint32(seed))
        self.bot_sync()

    def step(self, action: int):
        reward = ctypes.c_float(0.0)
        game_over = self.lib.tetris_cc_env_step(self.handle, int(action), ctypes.byref(reward))
        self.bot_sync()
        return {"reward": float(reward.value), "game_over": bool(game_over)}

    def set_mode(self, mode_name: str) -> bool:
        if not self._has_mode_api:
            return False
        mode_id = MODE_NAME_TO_ID.get(str(mode_name).strip().lower())
        if mode_id is None:
            return False
        ok = self.lib.tetris_cc_env_set_mode(self.handle, ctypes.c_int(int(mode_id)))
        if ok:
            self.bot_sync()
        return bool(ok)

    def get_mode(self) -> Optional[str]:
        if not self._has_mode_api:
            return None
        mode = ctypes.c_int(0)
        ok = self.lib.tetris_cc_env_get_mode(self.handle, ctypes.byref(mode))
        if not ok:
            return None
        for name, mid in MODE_NAME_TO_ID.items():
            if int(mid) == int(mode.value):
                return name
        return None

    def hold(self):
        reward = ctypes.c_float(0.0)
        success = self.lib.tetris_cc_env_hold(self.handle, ctypes.byref(reward))
        if success:
            self.bot_sync()
        return {"success": bool(success), "reward": float(reward.value)}

    def apply_incoming_garbage(self, lines: int):
        if not self._has_apply_incoming_garbage_api:
            return {"success": False, "lines_applied": 0, "top_out": bool(self.meta()["top_out"])}
        lines_applied = ctypes.c_int(0)
        top_out = ctypes.c_int(0)
        ok = self.lib.tetris_cc_env_apply_incoming_garbage(
            self.handle,
            ctypes.c_int(max(0, int(lines))),
            ctypes.byref(lines_applied),
            ctypes.byref(top_out),
        )
        if ok:
            self.bot_sync()
        return {
            "success": bool(ok),
            "lines_applied": int(lines_applied.value),
            "top_out": bool(top_out.value),
        }

    def last_attack_meta(self):
        if not self._has_last_attack_meta_api:
            return {
                "attack_base": 0,
                "attack_combo_scaled": 0.0,
                "attack_rounded": 0,
                "attack_b2b_bonus": 0,
                "attack_all_clear_bonus": 0,
                "attack_total": 0,
                "all_clear": False,
                "b2b_streak": 0,
                "surge_charge": 0,
                "surge_release": 0,
            }
        attack_base = ctypes.c_int(0)
        attack_combo_scaled = ctypes.c_float(0.0)
        attack_rounded = ctypes.c_int(0)
        attack_b2b_bonus = ctypes.c_int(0)
        attack_all_clear_bonus = ctypes.c_int(0)
        attack_total = ctypes.c_int(0)
        all_clear = ctypes.c_int(0)
        b2b_streak = ctypes.c_int(0)
        surge_charge = ctypes.c_int(0)
        surge_release = ctypes.c_int(0)
        ok = self.lib.tetris_cc_env_last_attack_meta(
            self.handle,
            ctypes.byref(attack_base),
            ctypes.byref(attack_combo_scaled),
            ctypes.byref(attack_rounded),
            ctypes.byref(attack_b2b_bonus),
            ctypes.byref(attack_all_clear_bonus),
            ctypes.byref(attack_total),
            ctypes.byref(all_clear),
            ctypes.byref(b2b_streak),
            ctypes.byref(surge_charge),
            ctypes.byref(surge_release),
        )
        if not ok:
            return {
                "attack_base": 0,
                "attack_combo_scaled": 0.0,
                "attack_rounded": 0,
                "attack_b2b_bonus": 0,
                "attack_all_clear_bonus": 0,
                "attack_total": 0,
                "all_clear": False,
                "b2b_streak": 0,
                "surge_charge": 0,
                "surge_release": 0,
            }
        return {
            "attack_base": int(attack_base.value),
            "attack_combo_scaled": float(attack_combo_scaled.value),
            "attack_rounded": int(attack_rounded.value),
            "attack_b2b_bonus": int(attack_b2b_bonus.value),
            "attack_all_clear_bonus": int(attack_all_clear_bonus.value),
            "attack_total": int(attack_total.value),
            "all_clear": bool(all_clear.value),
            "b2b_streak": int(b2b_streak.value),
            "surge_charge": int(surge_charge.value),
            "surge_release": int(surge_release.value),
        }

    def blitz_meta(self):
        if not self._has_blitz_meta_api:
            return {
                "score_total": 0,
                "level": 1,
                "lines_to_next": 0,
                "time_remaining_ms": 0,
                "timed_out": False,
            }
        score_total = ctypes.c_int(0)
        level = ctypes.c_int(1)
        lines_to_next = ctypes.c_int(0)
        time_remaining_ms = ctypes.c_int(0)
        timed_out = ctypes.c_int(0)
        ok = self.lib.tetris_cc_env_blitz_meta(
            self.handle,
            ctypes.byref(score_total),
            ctypes.byref(level),
            ctypes.byref(lines_to_next),
            ctypes.byref(time_remaining_ms),
            ctypes.byref(timed_out),
        )
        if not ok:
            return {
                "score_total": 0,
                "level": 1,
                "lines_to_next": 0,
                "time_remaining_ms": 0,
                "timed_out": False,
            }
        return {
            "score_total": int(score_total.value),
            "level": int(level.value),
            "lines_to_next": int(lines_to_next.value),
            "time_remaining_ms": int(time_remaining_ms.value),
            "timed_out": bool(timed_out.value),
        }

    def has_random_board_api(self):
        return bool(self._has_set_visible_board_mask)

    def visible_garbage_count(self) -> int:
        if self._has_visible_garbage_count:
            return int(self.lib.tetris_cc_env_visible_garbage_count(self.handle))

        board = self.board()
        if self._has_piece_id_api:
            piece_ids = self.board_piece_ids(include_active=False)
            count = 0
            for r in range(BOARD_ROWS):
                for c in range(BOARD_COLS):
                    if int(board[r][c]) != 0 and int(piece_ids[r][c]) == EMPTY_CELL_ID:
                        count += 1
            return int(count)
        return int(sum(int(cell != 0) for row in board for cell in row))

    def set_visible_board_mask(self, mask: np.ndarray, reset_meta: bool = True) -> bool:
        if not self._has_set_visible_board_mask:
            raise RuntimeError("C API does not provide tetris_cc_env_set_visible_board_mask.")
        arr = np.asarray(mask, dtype=np.uint8)
        if arr.shape != (BOARD_ROWS, BOARD_COLS):
            raise ValueError(f"Expected mask shape {(BOARD_ROWS, BOARD_COLS)}, got {arr.shape}")
        flat = np.ascontiguousarray(arr.reshape(-1), dtype=np.uint8)
        ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        ok = self.lib.tetris_cc_env_set_visible_board_mask(
            self.handle,
            ptr,
            ctypes.c_size_t(flat.size),
            ctypes.c_int(1 if bool(reset_meta) else 0),
        )
        if ok:
            self.bot_sync()
        return bool(ok)

    def has_bot(self):
        return bool(self._has_bot_api and self.bot_handle)

    def bot_sync(self):
        if not self.has_bot():
            return False
        ok = self.lib.tetris_cc_bot_sync_from_env(self.bot_handle, self.handle)
        return bool(ok)

    def bot_set_backend(self, backend_name: str) -> bool:
        if not self.has_bot() or not self._has_bot_backend_api:
            return False
        if backend_name == "cold_clear":
            backend_id = 0
        elif backend_name == "depth":
            backend_id = 1
        elif backend_name == "beam":
            backend_id = 2
        else:
            return False
        ok = self.lib.tetris_cc_bot_set_backend(self.bot_handle, ctypes.c_int(backend_id))
        return bool(ok)

    def bot_set_depth_config(self, depth: int, gamma: float) -> bool:
        if not self.has_bot() or not self._has_depth_backend_config_api:
            return False
        ok = self.lib.tetris_cc_bot_set_depth_config(
            self.bot_handle,
            ctypes.c_int(max(1, int(depth))),
            ctypes.c_double(float(gamma)),
            ctypes.c_int(1),  # deduplicate_successors
            ctypes.c_int(0),  # use_transposition_table
            ctypes.c_int(1),  # collect_debug_info
            ctypes.c_uint64(0),  # max_nodes (unlimited)
        )
        return bool(ok)

    def bot_set_beam_config(self, depth: int, beam_width: int, gamma: float) -> bool:
        if not self.has_bot() or not self._has_beam_backend_config_api:
            return False
        ok = self.lib.tetris_cc_bot_set_beam_config(
            self.bot_handle,
            ctypes.c_int(max(1, int(depth))),
            ctypes.c_int(max(1, int(beam_width))),
            ctypes.c_double(float(gamma)),
            ctypes.c_int(1),  # deduplicate_successors
            ctypes.c_int(0),  # use_transposition_table
            ctypes.c_int(1),  # collect_debug_info
            ctypes.c_uint64(0),  # max_nodes (unlimited)
        )
        return bool(ok)

    def bot_choose_and_apply(self, think_ms: int = 20):
        if not self.has_bot():
            return {
                "success": False,
                "reward": 0.0,
                "lines": 0,
                "game_over": False,
                "used_hold": False,
                "placement_index": -1,
                "score": 0.0,
                "nodes": 0,
                "think_ms": 0.0,
                "nps": 0.0,
                "budget_miss": 0,
            }
        reward = ctypes.c_float(0.0)
        lines = ctypes.c_int(0)
        game_over = ctypes.c_int(0)
        used_hold = ctypes.c_int(0)
        placement_index = ctypes.c_size_t(0)
        score = ctypes.c_float(0.0)
        nodes = ctypes.c_uint64(0)
        think = ctypes.c_double(0.0)
        nps = ctypes.c_double(0.0)
        budget_miss = ctypes.c_int(0)
        if self._has_bot_api_ex:
            ok = self.lib.tetris_cc_bot_choose_and_apply_ex(
                self.bot_handle,
                self.handle,
                int(max(1, think_ms)),
                ctypes.byref(reward),
                ctypes.byref(lines),
                ctypes.byref(game_over),
                ctypes.byref(used_hold),
                ctypes.byref(placement_index),
                ctypes.byref(score),
                ctypes.byref(nodes),
                ctypes.byref(think),
                ctypes.byref(nps),
                ctypes.byref(budget_miss),
            )
        else:
            ok = self.lib.tetris_cc_bot_choose_and_apply(
                self.bot_handle,
                self.handle,
                int(max(1, think_ms)),
                ctypes.byref(reward),
                ctypes.byref(lines),
                ctypes.byref(game_over),
                ctypes.byref(used_hold),
                ctypes.byref(placement_index),
                ctypes.byref(score),
                ctypes.byref(nodes),
                ctypes.byref(think),
                ctypes.byref(nps),
                ctypes.byref(budget_miss),
            )
        return {
            "success": bool(ok),
            "reward": float(reward.value),
            "lines": int(lines.value),
            "game_over": bool(game_over.value),
            "used_hold": bool(used_hold.value),
            "placement_index": int(placement_index.value),
            "score": float(score.value),
            "nodes": int(nodes.value),
            "think_ms": float(think.value),
            "nps": float(nps.value),
            "budget_miss": int(budget_miss.value),
        }

    def observation_size(self):
        return int(self.lib.tetris_cc_env_observation_size(self.handle, 1))

    def board(self):
        buf = (ctypes.c_uint8 * (BOARD_ROWS * BOARD_COLS))()
        written = self.lib.tetris_cc_env_board_write(self.handle, 0, buf, len(buf))
        if written != len(buf):
            return [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        flat = [int(v) for v in buf]
        return [flat[r * BOARD_COLS : (r + 1) * BOARD_COLS] for r in range(BOARD_ROWS)]

    def board_piece_ids(self, include_active: bool = True):
        if self._has_piece_id_api:
            buf = (ctypes.c_uint8 * (BOARD_ROWS * BOARD_COLS))()
            written = self.lib.tetris_cc_env_board_piece_ids_write(
                self.handle,
                1 if include_active else 0,
                buf,
                len(buf),
            )
            if written == len(buf):
                flat = [int(v) for v in buf]
                return [flat[r * BOARD_COLS : (r + 1) * BOARD_COLS] for r in range(BOARD_ROWS)]

        occ = self.board()
        return [
            [7 if occ[r][c] else EMPTY_CELL_ID for c in range(BOARD_COLS)]
            for r in range(BOARD_ROWS)
        ]

    def active(self):
        piece = ctypes.c_int(-1)
        rotation = ctypes.c_int(-1)
        x = ctypes.c_int(0)
        y = ctypes.c_int(0)
        ok = self.lib.tetris_cc_env_active_piece(
            self.handle,
            ctypes.byref(piece),
            ctypes.byref(rotation),
            ctypes.byref(x),
            ctypes.byref(y),
        )
        if not ok:
            return None
        return {"piece": int(piece.value), "rotation": int(rotation.value), "x": int(x.value), "y": int(y.value)}

    def hold_info(self):
        has_hold = ctypes.c_int(0)
        hold_piece = ctypes.c_int(7)
        hold_available = ctypes.c_int(0)
        ok = self.lib.tetris_cc_env_hold_piece(
            self.handle,
            ctypes.byref(has_hold),
            ctypes.byref(hold_piece),
            ctypes.byref(hold_available),
        )
        if not ok:
            return {"has_hold": False, "hold_piece": 7, "hold_available": False}
        return {
            "has_hold": bool(has_hold.value),
            "hold_piece": int(hold_piece.value),
            "hold_available": bool(hold_available.value),
        }

    def queue(self):
        count = int(self.lib.tetris_cc_env_queue_count(self.handle))
        out = []
        for i in range(count):
            piece = ctypes.c_int(-1)
            if self.lib.tetris_cc_env_queue_get(self.handle, i, ctypes.byref(piece)):
                out.append(int(piece.value))
        return out

    def meta(self):
        vals = [ctypes.c_int(0) for _ in range(7)]
        ok = self.lib.tetris_cc_env_meta(
            self.handle,
            ctypes.byref(vals[0]),
            ctypes.byref(vals[1]),
            ctypes.byref(vals[2]),
            ctypes.byref(vals[3]),
            ctypes.byref(vals[4]),
            ctypes.byref(vals[5]),
            ctypes.byref(vals[6]),
        )
        if not ok:
            return {
                "game_over": True,
                "top_out": False,
                "combo": -1,
                "b2b": False,
                "lines": 0,
                "lock_timer": 0,
                "lock_resets": 0,
            }
        return {
            "game_over": bool(vals[0].value),
            "top_out": bool(vals[1].value),
            "combo": int(vals[2].value),
            "b2b": bool(vals[3].value),
            "lines": int(vals[4].value),
            "lock_timer": int(vals[5].value),
            "lock_resets": int(vals[6].value),
        }

    def get_state(self):
        active = self.active()
        hold = self.hold_info()
        meta = self.meta()
        return {
            "board": self.board(),
            "current_piece": int(active["piece"]) if active is not None else -1,
            "current_rotation": int(active["rotation"]) if active is not None else -1,
            "current_x": int(active["x"]) if active is not None else -1,
            "current_y": int(active["y"]) if active is not None else -1,
            "hold_piece": int(hold["hold_piece"]) if hold["has_hold"] else 7,
            "hold_available": bool(hold["hold_available"]),
            "queue": self.queue(),
            "lines": int(meta["lines"]),
            "combo": int(meta["combo"]),
            "b2b": bool(meta["b2b"]),
            "game_over": bool(meta["game_over"]),
            "top_out": bool(meta["top_out"]),
        }

    def _enumerate_current_branch_actions(self, use_hold: bool, piece_id: int):
        count = int(self.lib.tetris_cc_env_placement_count(self.handle))
        out: List[Tuple[NativeAction, ActionTuple]] = []
        for idx in range(count):
            x = ctypes.c_int(0)
            y = ctypes.c_int(0)
            rot = ctypes.c_int(0)
            lines = ctypes.c_int(0)
            ok = self.lib.tetris_cc_env_placement_get(
                self.handle,
                idx,
                ctypes.byref(x),
                ctypes.byref(y),
                ctypes.byref(rot),
                ctypes.byref(lines),
            )
            if not ok:
                continue
            action_tuple: ActionTuple = (
                int(bool(use_hold)),
                int(piece_id),
                int(rot.value),
                int(x.value),
                int(y.value),
            )
            out.append((NativeAction(bool(use_hold), int(idx)), action_tuple))
        return out

    def _enumerate_current_branch_dqn_candidates(self, use_hold: bool, piece_id: int):
        from dqn_ref.features import compute_features_from_board

        count = int(self.lib.tetris_cc_env_placement_count(self.handle))
        out: List[DQNCandidate] = []
        for idx in range(count):
            x = ctypes.c_int(0)
            y = ctypes.c_int(0)
            rot = ctypes.c_int(0)
            lines = ctypes.c_int(0)
            ok = self.lib.tetris_cc_env_placement_get(
                self.handle,
                idx,
                ctypes.byref(x),
                ctypes.byref(y),
                ctypes.byref(rot),
                ctypes.byref(lines),
            )
            if not ok:
                continue
            board_after = np.asarray(self.placement_board(int(idx)), dtype=np.uint8)
            features = compute_features_from_board(
                board_after=board_after,
                y_pos=int(y.value),
                lines_removed=int(lines.value),
            ).as_array()
            action_tuple: ActionTuple = (
                int(bool(use_hold)),
                int(piece_id),
                int(rot.value),
                int(x.value),
                int(y.value),
            )
            out.append(
                DQNCandidate(
                    native_action=NativeAction(bool(use_hold), int(idx)),
                    action_tuple=action_tuple,
                    feature_vector=np.asarray(features, dtype=np.float32),
                    lines_removed=int(lines.value),
                    y_pos=int(y.value),
                )
            )
        return out

    def _enumerate_dqn_candidates_batch(self):
        count = int(self.lib.tetris_cc_env_candidate_count(self.handle))
        if count <= 0:
            return []

        needed = int(count * 6)
        feature_buf = (ctypes.c_float * needed)()
        written = self.lib.tetris_cc_env_candidate_features_write(
            self.handle,
            feature_buf,
            needed,
        )
        if int(written) != needed:
            return []

        feature_flat = np.ctypeslib.as_array(feature_buf)[:needed]
        feature_mat = feature_flat.reshape(count, 6)

        out: List[DQNCandidate] = []
        for idx in range(count):
            use_hold = ctypes.c_int(0)
            placement_index = ctypes.c_size_t(0)
            piece = ctypes.c_int(-1)
            rotation = ctypes.c_int(0)
            x = ctypes.c_int(0)
            y = ctypes.c_int(0)
            lines = ctypes.c_int(0)
            ok = self.lib.tetris_cc_env_candidate_get(
                self.handle,
                ctypes.c_size_t(idx),
                ctypes.byref(use_hold),
                ctypes.byref(placement_index),
                ctypes.byref(piece),
                ctypes.byref(rotation),
                ctypes.byref(x),
                ctypes.byref(y),
                ctypes.byref(lines),
            )
            if not ok:
                continue

            out.append(
                DQNCandidate(
                    native_action=NativeAction(bool(use_hold.value), int(placement_index.value)),
                    action_tuple=(
                        int(bool(use_hold.value)),
                        int(piece.value),
                        int(rotation.value),
                        int(x.value),
                        int(y.value),
                    ),
                    feature_vector=np.asarray(feature_mat[idx], dtype=np.float32).copy(),
                    lines_removed=int(lines.value),
                    y_pos=int(y.value),
                )
            )
        return out

    def enumerate_legal_actions(self):
        meta = self.meta()
        if bool(meta["game_over"]):
            return []

        snapshot = self.lib.tetris_cc_env_snapshot_create(self.handle)
        if not snapshot:
            raise RuntimeError("Failed to create env snapshot.")

        out: List[Tuple[NativeAction, ActionTuple]] = []
        try:
            active = self.active()
            if active is not None and 0 <= int(active["piece"]) <= 6:
                out.extend(
                    self._enumerate_current_branch_actions(
                        use_hold=False,
                        piece_id=int(active["piece"]),
                    )
                )

            hold = self.hold_info()
            if bool(hold["hold_available"]):
                self.lib.tetris_cc_env_restore_snapshot(self.handle, snapshot)
                hold_reward = ctypes.c_float(0.0)
                hold_ok = self.lib.tetris_cc_env_hold(self.handle, ctypes.byref(hold_reward))
                if hold_ok:
                    hold_active = self.active()
                    if hold_active is not None and 0 <= int(hold_active["piece"]) <= 6:
                        out.extend(
                            self._enumerate_current_branch_actions(
                                use_hold=True,
                                piece_id=int(hold_active["piece"]),
                            )
                        )
                self.lib.tetris_cc_env_restore_snapshot(self.handle, snapshot)
        finally:
            self.lib.tetris_cc_snapshot_destroy(snapshot)
            self.bot_sync()
        return out

    def enumerate_dqn_candidates(self):
        meta = self.meta()
        if bool(meta["game_over"]):
            return []
        if self._has_candidate_batch_api:
            return self._enumerate_dqn_candidates_batch()

        snapshot = self.lib.tetris_cc_env_snapshot_create(self.handle)
        if not snapshot:
            raise RuntimeError("Failed to create env snapshot.")

        out: List[DQNCandidate] = []
        try:
            active = self.active()
            if active is not None and 0 <= int(active["piece"]) <= 6:
                out.extend(
                    self._enumerate_current_branch_dqn_candidates(
                        use_hold=False,
                        piece_id=int(active["piece"]),
                    )
                )

            hold = self.hold_info()
            if bool(hold["hold_available"]):
                self.lib.tetris_cc_env_restore_snapshot(self.handle, snapshot)
                hold_reward = ctypes.c_float(0.0)
                hold_ok = self.lib.tetris_cc_env_hold(self.handle, ctypes.byref(hold_reward))
                if hold_ok:
                    hold_active = self.active()
                    if hold_active is not None and 0 <= int(hold_active["piece"]) <= 6:
                        out.extend(
                            self._enumerate_current_branch_dqn_candidates(
                                use_hold=True,
                                piece_id=int(hold_active["piece"]),
                            )
                        )
                self.lib.tetris_cc_env_restore_snapshot(self.handle, snapshot)
        finally:
            self.lib.tetris_cc_snapshot_destroy(snapshot)
            self.bot_sync()
        return out

    def step_native_action(self, action: NativeAction):
        used_hold = False
        total_reward = 0.0
        if bool(action.use_hold):
            hold_reward = ctypes.c_float(0.0)
            hold_ok = self.lib.tetris_cc_env_hold(self.handle, ctypes.byref(hold_reward))
            if not hold_ok:
                return {
                    "success": False,
                    "reward": 0.0,
                    "lines": 0,
                    "game_over": bool(self.meta()["game_over"]),
                    "used_hold": True,
                }
            used_hold = True
            total_reward += float(hold_reward.value)

        reward = ctypes.c_float(0.0)
        lines = ctypes.c_int(0)
        game_over = ctypes.c_int(0)
        ok = self.lib.tetris_cc_env_apply_placement_index(
            self.handle,
            int(action.placement_index),
            ctypes.byref(reward),
            ctypes.byref(lines),
            ctypes.byref(game_over),
        )
        total_reward += float(reward.value)
        if ok:
            self.bot_sync()
        return {
            "success": bool(ok),
            "reward": float(total_reward),
            "lines": int(lines.value),
            "game_over": bool(game_over.value),
            "used_hold": used_hold,
        }

    def placements(self):
        count = int(self.lib.tetris_cc_env_placement_count(self.handle))
        out = []
        for i in range(count):
            x = ctypes.c_int(0)
            y = ctypes.c_int(0)
            rot = ctypes.c_int(0)
            lines = ctypes.c_int(0)
            if self.lib.tetris_cc_env_placement_get(
                self.handle, i, ctypes.byref(x), ctypes.byref(y), ctypes.byref(rot), ctypes.byref(lines)
            ):
                out.append({"index": i, "x": int(x.value), "y": int(y.value), "rotation": int(rot.value), "lines": int(lines.value)})
        return out

    def placement_board(self, index: int):
        buf = (ctypes.c_uint8 * (BOARD_ROWS * BOARD_COLS))()
        written = self.lib.tetris_cc_env_placement_board_write(self.handle, int(index), buf, len(buf))
        if written != len(buf):
            return [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        flat = [int(v) for v in buf]
        return [flat[r * BOARD_COLS : (r + 1) * BOARD_COLS] for r in range(BOARD_ROWS)]

    def placement_piece_ids(self, index: int):
        if self._has_piece_id_api:
            buf = (ctypes.c_uint8 * (BOARD_ROWS * BOARD_COLS))()
            written = self.lib.tetris_cc_env_placement_board_piece_ids_write(self.handle, int(index), buf, len(buf))
            if written == len(buf):
                flat = [int(v) for v in buf]
                return [flat[r * BOARD_COLS : (r + 1) * BOARD_COLS] for r in range(BOARD_ROWS)]

        occ = self.placement_board(index)
        return [
            [7 if occ[r][c] else EMPTY_CELL_ID for c in range(BOARD_COLS)]
            for r in range(BOARD_ROWS)
        ]

    def apply_placement(self, index: int):
        reward = ctypes.c_float(0.0)
        lines = ctypes.c_int(0)
        game_over = ctypes.c_int(0)
        ok = self.lib.tetris_cc_env_apply_placement_index(
            self.handle,
            int(index),
            ctypes.byref(reward),
            ctypes.byref(lines),
            ctypes.byref(game_over),
        )
        out = {
            "success": bool(ok),
            "reward": float(reward.value),
            "lines": int(lines.value),
            "game_over": bool(game_over.value),
        }
        if out["success"]:
            self.bot_sync()
        return out

    def rotation_trace(self, action: int):
        count = int(self.lib.tetris_cc_env_rotation_trace_count(self.handle, int(action)))
        tests = []
        for i in range(count):
            vals = [ctypes.c_int(0) for _ in range(10)]
            ok = self.lib.tetris_cc_env_rotation_trace_get(
                self.handle,
                int(action),
                i,
                ctypes.byref(vals[0]),
                ctypes.byref(vals[1]),
                ctypes.byref(vals[2]),
                ctypes.byref(vals[3]),
                ctypes.byref(vals[4]),
                ctypes.byref(vals[5]),
                ctypes.byref(vals[6]),
                ctypes.byref(vals[7]),
                ctypes.byref(vals[8]),
                ctypes.byref(vals[9]),
            )
            if not ok:
                continue
            tests.append(
                {
                    "test_index": int(vals[0].value),
                    "phase": int(vals[1].value),
                    "kick_index": int(vals[2].value),
                    "dx": int(vals[3].value),
                    "dy": int(vals[4].value),
                    "passed": bool(vals[5].value),
                    "x": int(vals[6].value),
                    "y": int(vals[7].value),
                    "rotation": int(vals[8].value),
                    "collides": bool(vals[9].value),
                }
            )

        success = ctypes.c_int(0)
        fx = ctypes.c_int(-1)
        fy = ctypes.c_int(-1)
        fr = ctypes.c_int(-1)
        self.lib.tetris_cc_env_rotation_trace_meta(
            self.handle,
            int(action),
            ctypes.byref(success),
            ctypes.byref(fx),
            ctypes.byref(fy),
            ctypes.byref(fr),
        )
        return {
            "success": bool(success.value),
            "final_x": int(fx.value),
            "final_y": int(fy.value),
            "final_rotation": int(fr.value),
            "tests": tests,
        }


def piece_cells(piece: int, rotation: int):
    base = {
        0: [(-1, 0), (0, 0), (1, 0), (2, 0)],
        1: [(0, 0), (1, 0), (0, 1), (1, 1)],
        2: [(-1, 0), (0, 0), (1, 0), (0, 1)],
        3: [(-1, 0), (0, 0), (1, 0), (1, 1)],
        4: [(-1, 0), (0, 0), (1, 0), (-1, 1)],
        5: [(-1, 0), (0, 0), (0, 1), (1, 1)],
        6: [(-1, 1), (0, 1), (0, 0), (1, 0)],
    }.get(piece, [])

    def rot(cell):
        x, y = cell
        if rotation == 0:
            return (x, y)
        if rotation == 1:
            return (y, -x)
        if rotation == 2:
            return (-x, -y)
        return (-y, x)

    return [rot(c) for c in base]


def draw_board(surface, x0, y0, cell, board_piece_ids, board_occupancy):
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            rect = pygame.Rect(x0 + c * cell, y0 + r * cell, cell, cell)
            piece_id = board_piece_ids[r][c]
            occupied = int(board_occupancy[r][c]) != 0
            if 0 <= int(piece_id) <= 6:
                pygame.draw.rect(surface, PIECE_COLORS.get(piece_id, BOARD_FILL), rect)
            elif int(piece_id) == EMPTY_CELL_ID and occupied:
                pygame.draw.rect(surface, GARBAGE_COLOR, rect)
            pygame.draw.rect(surface, GRID_LINE, rect, width=1)

def draw_small_board(surface, x0, y0, cell, board_piece_ids, board_occupancy):
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            rect = pygame.Rect(x0 + c * cell, y0 + r * cell, cell, cell)
            piece_id = board_piece_ids[r][c]
            occupied = int(board_occupancy[r][c]) != 0
            if 0 <= int(piece_id) <= 6:
                pygame.draw.rect(surface, PIECE_COLORS.get(piece_id, BOARD_FILL), rect)
            elif int(piece_id) == EMPTY_CELL_ID and occupied:
                pygame.draw.rect(surface, GARBAGE_COLOR, rect)
            pygame.draw.rect(surface, (55, 60, 72), rect, width=1)


def draw_garbage_bar(
    surface,
    rect: pygame.Rect,
    incoming_lines: int,
    *,
    max_visible: int,
    small_font,
):
    incoming = max(0, int(incoming_lines))
    shown = min(incoming, int(max_visible))
    if shown <= int(max_visible * 0.33):
        fill_color = (110, 170, 235)
    elif shown <= int(max_visible * 0.66):
        fill_color = (235, 180, 80)
    else:
        fill_color = (235, 100, 100)

    pygame.draw.rect(surface, (26, 30, 40), rect, border_radius=6)
    inner = rect.inflate(-6, -6)
    pygame.draw.rect(surface, (44, 50, 66), inner, border_radius=4)
    if shown > 0:
        fill_h = int(round(inner.height * (shown / float(max(1, max_visible)))))
        fill_rect = pygame.Rect(inner.x, inner.bottom - fill_h, inner.width, fill_h)
        pygame.draw.rect(surface, fill_color, fill_rect, border_radius=4)
    pygame.draw.rect(surface, (80, 90, 115), rect, width=1, border_radius=6)

    label = f"IN {shown}+" if incoming > max_visible else f"IN {shown}"
    text = small_font.render(label, True, LOCK_TEXT)
    tx = rect.centerx - text.get_width() // 2
    ty = rect.top - text.get_height() - 4
    surface.blit(text, (tx, ty))


def draw_side_stats(
    surface,
    panel_rect: pygame.Rect,
    *,
    side: str,
    header: str,
    meta: Dict[str, object],
    attack: Dict[str, object],
    pending_incoming: int,
    stats: Dict[str, object],
    font,
    small_font,
):
    pygame.draw.rect(surface, PANEL_COLOR, panel_rect, border_radius=8)
    pygame.draw.rect(surface, (70, 80, 102), panel_rect, width=1, border_radius=8)

    align_right = side == "right"
    x_pad = 12
    header_surface = font.render(header, True, LOCK_TEXT)
    if align_right:
        hx = panel_rect.right - x_pad - header_surface.get_width()
    else:
        hx = panel_rect.left + x_pad
    hy = panel_rect.top + 10
    surface.blit(header_surface, (hx, hy))

    lines = [
        f"GO={bool(meta.get('game_over', False))} TOP={bool(meta.get('top_out', False))}",
        f"Combo: {int(meta.get('combo', -1))}",
        f"B2B: {int(attack.get('b2b_streak', 0))}",
        (
            f"Last Attack: {int(attack.get('attack_total', 0))} "
            f"(Surge {int(attack.get('surge_charge', 0))}/{int(attack.get('surge_release', 0))})"
        ),
        f"Incoming: {int(pending_incoming)}",
        (
            f"Sent/Cancel/Recv: "
            f"{int(stats.get('garbage_sent_total', 0))}/"
            f"{int(stats.get('garbage_canceled_total', 0))}/"
            f"{int(stats.get('garbage_received_total', 0))}"
        ),
        (
            f"Last S/C/R: "
            f"{int(stats.get('last_sent', 0))}/"
            f"{int(stats.get('last_canceled', 0))}/"
            f"{int(stats.get('last_received', 0))}"
        ),
        (
            f"Pieces: {int(stats.get('pieces_placed', 0))}  "
            f"Lines: {int(stats.get('lines_cleared', 0))}  "
            f"PPS: {float(stats.get('pps', 0.0)):.2f}"
        ),
        f"Topouts: {int(stats.get('topouts', 0))}",
    ]

    line_y = hy + header_surface.get_height() + 12
    line_h = 22
    for line in lines:
        surface_line = small_font.render(line, True, LOCK_TEXT)
        if align_right:
            lx = panel_rect.right - x_pad - surface_line.get_width()
        else:
            lx = panel_rect.left + x_pad
        surface.blit(surface_line, (lx, line_y))
        line_y += line_h


def action_name(action: int):
    return {ACTION_CW: "CW", ACTION_CCW: "CCW", ACTION_180: "180"}.get(action, "?")


def _mix_seed_with_attempt(base_seed: int, round_id: int, episode_id: int, attempt_idx: int) -> int:
    mixed = (
        int(base_seed) * 1_000_000_007
        + int(round_id) * 1_000_003
        + int(episode_id) * 97
        + int(attempt_idx) * 271
        + 43
    )
    return int(mixed & ((1 << 63) - 1))


def run_versus_mode(args, lib_path: Path):
    pygame.init()
    pygame.display.set_caption("Tetris Versus (Human vs AI)")
    font = pygame.font.SysFont("Consolas", 18)
    small_font = pygame.font.SysFont("Consolas", 15)

    cell = max(12, args.cell)
    board_w = BOARD_COLS * cell
    board_h = BOARD_ROWS * cell
    outer_pad = 20
    top_strip_h = 56
    side_panel_w = max(260, int(cell * 7.8))
    cluster_gap = max(20, int(cell * 0.75))
    bar_w = max(16, int(cell * 0.55))
    bar_gap = max(8, int(cell * 0.35))
    between_boards = max(28, int(cell * 1.1))

    center_cluster_w = (bar_w + bar_gap + board_w) + between_boards + (board_w + bar_gap + bar_w)
    screen_w = outer_pad * 2 + side_panel_w * 2 + cluster_gap * 2 + center_cluster_w
    screen_h = max(outer_pad * 2 + top_strip_h + 14 + board_h + 66, 760)

    screen = pygame.display.set_mode((screen_w, screen_h))
    clock = pygame.time.Clock()

    top_strip_rect = pygame.Rect(outer_pad, outer_pad, screen_w - outer_pad * 2, top_strip_h)
    board_y = top_strip_rect.bottom + 14
    left_panel_rect = pygame.Rect(outer_pad, board_y, side_panel_w, board_h)
    cluster_x = left_panel_rect.right + cluster_gap
    left_bar_rect = pygame.Rect(cluster_x, board_y + 4, bar_w, board_h - 8)
    left_board_x = left_bar_rect.right + bar_gap
    right_board_x = left_board_x + board_w + between_boards
    right_bar_rect = pygame.Rect(right_board_x + board_w + bar_gap, board_y + 4, bar_w, board_h - 8)
    right_panel_rect = pygame.Rect(right_bar_rect.right + cluster_gap, board_y, side_panel_w, board_h)

    human_env = EnvCtypes(lib_path, args.seed)
    ai_env = EnvCtypes(lib_path, args.seed + 1)
    status = f"Loaded {lib_path.name}"
    seed = int(args.seed)
    ai_seed = int(args.seed + 1)
    pending_human = 0
    pending_ai = 0

    def zero_attack_stats():
        return {"attack_total": 0, "b2b_streak": 0, "surge_charge": 0, "surge_release": 0}

    def make_side_stats():
        return {
            "pieces_placed": 0,
            "lines_cleared": 0,
            "garbage_sent_total": 0,
            "garbage_received_total": 0,
            "garbage_canceled_total": 0,
            "topouts": 0,
            "last_attack": 0,
            "last_sent": 0,
            "last_canceled": 0,
            "last_received": 0,
            "start_ticks": pygame.time.get_ticks(),
            "pps": 0.0,
        }

    human_attack = zero_attack_stats()
    ai_attack = zero_attack_stats()
    human_side_stats = make_side_stats()
    ai_side_stats = make_side_stats()
    last_h_game_over = False
    last_a_game_over = False

    versus_ai_vs_ai = bool(args.ai_vs_ai)

    if not human_env.set_mode("versus") or not ai_env.set_mode("versus"):
        human_env.close()
        ai_env.close()
        pygame.quit()
        raise SystemExit("Shared library is missing versus mode C API support.")

    ai_backend = str(args.native_backend)
    if ai_backend == "cold_clear":
        ai_backend = "beam"
    if not ai_env.has_bot() or (versus_ai_vs_ai and not human_env.has_bot()):
        human_env.close()
        ai_env.close()
        pygame.quit()
        raise SystemExit("Versus mode requires bot API symbols in the shared library.")

    def configure_bot_backend(env_obj: EnvCtypes) -> bool:
        if ai_backend == "depth":
            return env_obj.bot_set_backend("depth") and env_obj.bot_set_depth_config(
                args.depth_search_depth, args.depth_gamma
            )
        if ai_backend == "beam":
            return env_obj.bot_set_backend("beam") and env_obj.bot_set_beam_config(
                args.beam_search_depth, args.beam_width, args.beam_gamma
            )
        return env_obj.bot_set_backend("cold_clear")

    if not configure_bot_backend(ai_env):
        human_env.close()
        ai_env.close()
        pygame.quit()
        raise SystemExit("Failed to configure opponent backend for versus AI.")
    if versus_ai_vs_ai and not configure_bot_backend(human_env):
        human_env.close()
        ai_env.close()
        pygame.quit()
        raise SystemExit("Failed to configure left-board backend for AI-vs-AI.")
    ai_env.bot_sync()
    if versus_ai_vs_ai:
        human_env.bot_sync()

    ai_enabled = True
    ai_interval_ms = (1000.0 / float(args.ai_pps)) if float(args.ai_pps) > 0.0 else 0.0
    next_left_ai_tick = pygame.time.get_ticks()
    next_right_ai_tick = pygame.time.get_ticks()

    def reset_match(h_seed: int, a_seed: int):
        nonlocal pending_human, pending_ai, human_attack, ai_attack
        nonlocal human_side_stats, ai_side_stats, last_h_game_over, last_a_game_over
        nonlocal next_left_ai_tick, next_right_ai_tick
        human_env.reset(h_seed)
        ai_env.reset(a_seed)
        human_env.set_mode("versus")
        ai_env.set_mode("versus")
        pending_human = 0
        pending_ai = 0
        human_attack = zero_attack_stats()
        ai_attack = zero_attack_stats()
        human_side_stats = make_side_stats()
        ai_side_stats = make_side_stats()
        last_h_game_over = False
        last_a_game_over = False
        configure_bot_backend(ai_env)
        if versus_ai_vs_ai:
            configure_bot_backend(human_env)
        ai_env.bot_sync()
        if versus_ai_vs_ai:
            human_env.bot_sync()
        now_tick = pygame.time.get_ticks()
        next_left_ai_tick = now_tick
        next_right_ai_tick = now_tick

    def refresh_pps(side_stats):
        elapsed_s = max(1e-6, (pygame.time.get_ticks() - int(side_stats["start_ticks"])) / 1000.0)
        side_stats["pps"] = float(side_stats["pieces_placed"]) / elapsed_s

    def resolve_lock(is_human: bool, lines_cleared: int = 0):
        nonlocal pending_human, pending_ai, status, human_attack, ai_attack
        nonlocal human_side_stats, ai_side_stats
        attacker = human_env if is_human else ai_env
        attacker_side = human_side_stats if is_human else ai_side_stats
        attack = attacker.last_attack_meta()
        outgoing = max(0, int(attack.get("attack_total", 0)))
        attacker_side["pieces_placed"] += 1
        attacker_side["lines_cleared"] += max(0, int(lines_cleared))
        if is_human:
            human_attack = attack
            cancel = min(outgoing, pending_human)
            pending_human -= cancel
            sent = outgoing - cancel
            pending_ai += sent
            to_apply = pending_human
            pending_human = 0
        else:
            ai_attack = attack
            cancel = min(outgoing, pending_ai)
            pending_ai -= cancel
            sent = outgoing - cancel
            pending_human += sent
            to_apply = pending_ai
            pending_ai = 0

        applied = 0
        top_out = 0
        if to_apply > 0:
            apply_result = attacker.apply_incoming_garbage(to_apply)
            if not apply_result.get("success", False):
                status = "Incoming garbage apply failed."
                return
            applied = int(apply_result.get("lines_applied", 0))
            top_out = 1 if bool(apply_result.get("top_out", False)) else 0

        attacker_side["garbage_sent_total"] += int(sent)
        attacker_side["garbage_canceled_total"] += int(cancel)
        attacker_side["garbage_received_total"] += int(applied)
        attacker_side["last_attack"] = int(outgoing)
        attacker_side["last_sent"] = int(sent)
        attacker_side["last_canceled"] = int(cancel)
        attacker_side["last_received"] = int(applied)
        refresh_pps(attacker_side)
        status = (
            f"{'Human' if is_human else 'AI'} atk={outgoing} "
            f"cancel={cancel} sent={sent} recv={applied} pending(H/A)=({pending_human}/{pending_ai})"
        )
        if top_out:
            status += " | TOP OUT"

    def do_human_step(action: int):
        pre_meta = human_env.meta()
        pre_queue = tuple(human_env.queue())
        human_env.step(action)
        post_meta = human_env.meta()
        post_queue = tuple(human_env.queue())
        locked = pre_queue != post_queue
        if locked:
            lines_delta = max(0, int(post_meta["lines"]) - int(pre_meta["lines"]))
            resolve_lock(is_human=True, lines_cleared=lines_delta)
        return post_meta

    running = True
    try:
        while running:
            human_acted = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_a:
                        ai_enabled = not ai_enabled
                        status = f"AI {'enabled' if ai_enabled else 'disabled'}."
                        if ai_enabled:
                            now_tick = pygame.time.get_ticks()
                            next_left_ai_tick = now_tick
                            next_right_ai_tick = now_tick
                    elif event.key == pygame.K_r:
                        reset_match(seed, ai_seed)
                        status = f"Reset match seeds=({seed},{ai_seed})"
                    elif event.key == pygame.K_n:
                        seed = random.randint(1, 2**31 - 1)
                        ai_seed = random.randint(1, 2**31 - 1)
                        reset_match(seed, ai_seed)
                        status = f"Reset new seeds=({seed},{ai_seed})"
                    elif not versus_ai_vs_ai:
                        if event.key == pygame.K_LEFT:
                            do_human_step(ACTION_LEFT)
                            human_acted = True
                        elif event.key == pygame.K_RIGHT:
                            do_human_step(ACTION_RIGHT)
                            human_acted = True
                        elif event.key == pygame.K_DOWN:
                            do_human_step(ACTION_SOFT_DROP)
                            human_acted = True
                        elif event.key in (pygame.K_UP, pygame.K_x):
                            do_human_step(ACTION_CW)
                            human_acted = True
                        elif event.key == pygame.K_z:
                            do_human_step(ACTION_CCW)
                            human_acted = True
                        elif event.key == pygame.K_SPACE:
                            do_human_step(ACTION_HARD_DROP)
                            human_acted = True
                        elif event.key in (pygame.K_c, pygame.K_LSHIFT, pygame.K_RSHIFT):
                            hold = human_env.hold()
                            status = "Human hold used." if hold["success"] else "Human hold unavailable."
                            human_acted = True

            h_meta = human_env.meta()
            a_meta = ai_env.meta()
            if not versus_ai_vs_ai and not h_meta["game_over"] and not human_acted:
                h_meta = do_human_step(ACTION_NONE)

            now_tick = pygame.time.get_ticks()
            if (
                ai_enabled
                and versus_ai_vs_ai
                and not h_meta["game_over"]
                and (ai_interval_ms <= 0.0 or now_tick >= next_left_ai_tick)
            ):
                human_result = human_env.bot_choose_and_apply(args.think_ms)
                if human_result["success"]:
                    resolve_lock(is_human=True, lines_cleared=int(human_result["lines"]))
                else:
                    ai_enabled = False
                    status = "Left AI choose/apply failed."
                if ai_interval_ms > 0.0:
                    next_left_ai_tick = now_tick + ai_interval_ms

            h_meta = human_env.meta()
            a_meta = ai_env.meta()

            now_tick = pygame.time.get_ticks()
            if (
                ai_enabled
                and not a_meta["game_over"]
                and (ai_interval_ms <= 0.0 or now_tick >= next_right_ai_tick)
            ):
                ai_result = ai_env.bot_choose_and_apply(args.think_ms)
                if ai_result["success"]:
                    resolve_lock(is_human=False, lines_cleared=int(ai_result["lines"]))
                else:
                    ai_enabled = False
                    status = "AI choose/apply failed."
                if ai_interval_ms > 0.0:
                    next_right_ai_tick = now_tick + ai_interval_ms

            h_meta = human_env.meta()
            a_meta = ai_env.meta()

            if bool(h_meta["game_over"]) and not last_h_game_over:
                human_side_stats["topouts"] += 1
            if bool(a_meta["game_over"]) and not last_a_game_over:
                ai_side_stats["topouts"] += 1
            last_h_game_over = bool(h_meta["game_over"])
            last_a_game_over = bool(a_meta["game_over"])

            if h_meta["game_over"] or a_meta["game_over"]:
                ai_enabled = False

            refresh_pps(human_side_stats)
            refresh_pps(ai_side_stats)

            h_occ = human_env.board()
            h_ids = human_env.board_piece_ids(include_active=True)
            a_occ = ai_env.board()
            a_ids = ai_env.board_piece_ids(include_active=True)

            screen.fill(BG_COLOR)
            pygame.draw.rect(screen, PANEL_COLOR, top_strip_rect, border_radius=8)
            pygame.draw.rect(screen, (70, 80, 102), top_strip_rect, width=1, border_radius=8)

            controls_text = (
                "A toggle AI | R reset | N new seeds | Q quit"
                if versus_ai_vs_ai
                else "Arrows move/drop | Up/X CW | Z CCW | Space hard drop | C/Shift hold | A/R/N/Q"
            )
            strip_line_1 = (
                f"Mode: versus   Seeds H:{seed} A:{ai_seed}   "
                f"Backend: {ai_backend}   AI:{'ON' if ai_enabled else 'OFF'}   "
                f"AI-vs-AI:{versus_ai_vs_ai}   target_pps:{float(args.ai_pps):.2f}"
            )
            strip_line_2 = f"{controls_text}   |   {status}"
            t1 = small_font.render(strip_line_1, True, LOCK_TEXT)
            t2 = small_font.render(strip_line_2, True, (180, 210, 255))
            screen.blit(t1, (top_strip_rect.centerx - t1.get_width() // 2, top_strip_rect.y + 8))
            screen.blit(t2, (top_strip_rect.centerx - t2.get_width() // 2, top_strip_rect.y + 30))

            draw_side_stats(
                screen,
                left_panel_rect,
                side="left",
                header="HUMAN",
                meta=h_meta,
                attack=human_attack,
                pending_incoming=pending_human,
                stats=human_side_stats,
                font=font,
                small_font=small_font,
            )
            draw_side_stats(
                screen,
                right_panel_rect,
                side="right",
                header=f"AI ({ai_backend.upper()})",
                meta=a_meta,
                attack=ai_attack,
                pending_incoming=pending_ai,
                stats=ai_side_stats,
                font=font,
                small_font=small_font,
            )

            draw_garbage_bar(
                screen,
                left_bar_rect,
                pending_human,
                max_visible=20,
                small_font=small_font,
            )
            draw_garbage_bar(
                screen,
                right_bar_rect,
                pending_ai,
                max_visible=20,
                small_font=small_font,
            )

            left_frame = pygame.Rect(left_board_x - 4, board_y - 4, board_w + 8, board_h + 8)
            right_frame = pygame.Rect(right_board_x - 4, board_y - 4, board_w + 8, board_h + 8)
            pygame.draw.rect(screen, PANEL_COLOR, left_frame, border_radius=6)
            pygame.draw.rect(screen, PANEL_COLOR, right_frame, border_radius=6)
            draw_board(screen, left_board_x, board_y, cell, h_ids, h_occ)
            draw_board(screen, right_board_x, board_y, cell, a_ids, a_occ)

            h_label = font.render("Human", True, LOCK_TEXT)
            a_label = font.render(f"AI ({ai_backend})", True, LOCK_TEXT)
            screen.blit(h_label, (left_board_x + board_w // 2 - h_label.get_width() // 2, board_y + board_h + 8))
            screen.blit(a_label, (right_board_x + board_w // 2 - a_label.get_width() // 2, board_y + board_h + 8))

            pygame.display.flip()
            clock.tick(max(10, args.fps))
    finally:
        human_env.close()
        ai_env.close()
        pygame.quit()


def main():
    args = parse_args()
    mode_name = str(args.mode).strip().lower()
    if args.bc_checkpoint is not None and args.dqn_checkpoint is not None:
        print(
            "Error: --bc-checkpoint and --dqn-checkpoint are mutually exclusive. "
            "Pass only one learned backend checkpoint.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    native_backend_requested = (
        str(args.native_backend)
        if args.bc_checkpoint is None and args.dqn_checkpoint is None
        else None
    )
    if mode_name == "versus" and native_backend_requested == "cold_clear":
        native_backend_requested = "beam"
    require_native_backend = (
        str(native_backend_requested)
        if native_backend_requested in ("depth", "beam") and (args.ai or mode_name == "versus")
        else None
    )
    prefer_native_backend = (
        str(native_backend_requested)
        if native_backend_requested in ("depth", "beam")
        else None
    )
    try:
        lib_path = find_library(
            args.lib,
            require_native_backend=require_native_backend,
            prefer_native_backend=prefer_native_backend,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if mode_name == "versus":
        run_versus_mode(args, lib_path)
        return

    pygame.init()
    pygame.display.set_caption("Tetris Placement + Kick Explorer")
    font = pygame.font.SysFont("Consolas", 18)
    small_font = pygame.font.SysFont("Consolas", 15)

    cell = max(12, args.cell)
    board_x, board_y = 20, 20
    board_w, board_h = BOARD_COLS * cell, BOARD_ROWS * cell
    right_x = board_x + board_w + 20
    right_w = 520
    screen_w = right_x + right_w + 20
    screen_h = max(board_y + board_h + 20, 980)

    screen = pygame.display.set_mode((screen_w, screen_h))
    clock = pygame.time.Clock()

    env = EnvCtypes(lib_path, args.seed)
    if mode_name != "legacy":
        if not env.set_mode(mode_name):
            env.close()
            raise SystemExit(f"Library '{lib_path}' does not support --mode {mode_name}.")
    selected_index = 0
    list_scroll = 0
    inspector_actions = [ACTION_CW, ACTION_CCW]
    inspector_idx = 0
    status = f"Loaded {lib_path.name}"
    seed = int(args.seed)
    ai_backend = "cold_clear"
    ai_backend_label = "ColdClear"
    bc_agent = None
    dqn_agent = None

    if args.dqn_checkpoint is not None:
        ai_backend = "dqn"
        ai_backend_label = "DQN"
        try:
            from dqn_ref.inference_agent import DQNRefInferenceAgent  # Lazy import for optional torch dependency

            dqn_agent = DQNRefInferenceAgent(args.dqn_checkpoint, device=args.dqn_device)
            status = (
                f"Loaded DQN checkpoint {args.dqn_checkpoint} "
                f"(device={args.dqn_device or 'auto'})"
            )
        except Exception as exc:
            dqn_agent = None
            status = f"DQN init failed: {exc}"
    elif args.bc_checkpoint is not None:
        ai_backend = "bc"
        ai_backend_label = "BC"
        try:
            from bc.inference_agent import BCAgent  # Lazy import for optional torch dependency

            bc_agent = BCAgent(args.bc_checkpoint, device=args.bc_device)
            status = (
                f"Loaded BC checkpoint {args.bc_checkpoint} "
                f"(device={args.bc_device or 'auto'})"
            )
        except Exception as exc:
            bc_agent = None
            status = f"BC init failed: {exc}"

    else:
        ai_backend = str(args.native_backend)
        ai_backend_label = {
            "depth": "Depth",
            "beam": "Beam",
        }.get(ai_backend, "ColdClear")

    state_source = str(args.state_source).strip().lower()
    random_board_mode = state_source == "random_board"
    native_config_error = None
    random_board_error = None
    last_reset_summary = f"state_source={state_source} (pending)"

    def configure_native_backend() -> Optional[str]:
        if ai_backend not in ("cold_clear", "depth", "beam"):
            return None
        if not env.has_bot():
            return "bot API symbols not exported by shared library."

        if ai_backend == "depth":
            if not getattr(env, "_has_bot_backend_api", False) or not getattr(
                env, "_has_depth_backend_config_api", False
            ):
                return (
                    "depth backend symbols missing "
                    "(tetris_cc_bot_set_backend/tetris_cc_bot_set_depth_config)."
                )
            if not env.bot_set_backend("depth"):
                return "failed to set bot backend to depth."
            if not env.bot_set_depth_config(args.depth_search_depth, args.depth_gamma):
                return (
                    f"failed to set depth config "
                    f"(depth={max(1, int(args.depth_search_depth))}, gamma={float(args.depth_gamma):.3f})."
                )
        elif ai_backend == "beam":
            if not getattr(env, "_has_bot_backend_api", False) or not getattr(
                env, "_has_beam_backend_config_api", False
            ):
                return (
                    "beam backend symbols missing "
                    "(tetris_cc_bot_set_backend/tetris_cc_bot_set_beam_config)."
                )
            if not env.bot_set_backend("beam"):
                return "failed to set bot backend to beam."
            if not env.bot_set_beam_config(
                args.beam_search_depth, args.beam_width, args.beam_gamma
            ):
                return (
                    "failed to set beam config "
                    f"(depth={max(1, int(args.beam_search_depth))}, "
                    f"width={max(1, int(args.beam_width))}, gamma={float(args.beam_gamma):.3f})."
                )
        else:
            if getattr(env, "_has_bot_backend_api", False):
                if not env.bot_set_backend("cold_clear"):
                    return "failed to set bot backend to cold_clear."

        if not env.bot_sync():
            return f"failed to sync {ai_backend} backend from env."
        return None

    def reset_episode_for_seed(seed_value: int) -> Tuple[bool, str]:
        nonlocal native_config_error
        if not random_board_mode:
            env.reset(seed_value)
            native_config_error = configure_native_backend()
            if native_config_error is not None:
                return False, native_config_error
            return True, "state_source=rollout"

        if not env.has_random_board_api():
            return False, "random_board requires tetris_cc_env_set_visible_board_mask."

        attempts_limit = max(1, int(args.random_max_resamples_per_sample))
        y_lim = max(0, min(BOARD_ROWS, int(args.random_fill_y_max_exclusive)))
        for attempt_idx in range(attempts_limit):
            env.reset(seed_value)
            mask_rng = np.random.default_rng(
                _mix_seed_with_attempt(int(args.seed), 0, int(seed_value), int(attempt_idx))
            )
            mask = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.uint8)
            if y_lim > 0:
                lower = mask_rng.random((y_lim, BOARD_COLS)) < float(args.random_fill_prob)
                for y in range(y_lim):
                    row = BOARD_ROWS - 1 - y
                    mask[row, :] = lower[y, :].astype(np.uint8)
            try:
                ok = env.set_visible_board_mask(mask, reset_meta=True)
            except Exception as exc:
                return False, f"failed to inject random board mask: {exc}"
            if not ok:
                return False, "failed to inject random board mask via C API."
            if bool(env.meta()["game_over"]):
                continue
            garbage_cells = int(env.visible_garbage_count())
            if garbage_cells <= 0:
                continue
            native_config_error = configure_native_backend()
            if native_config_error is not None:
                return False, native_config_error
            return (
                True,
                (
                    f"state_source=random_board attempts={attempt_idx + 1} "
                    f"garbage={garbage_cells}"
                ),
            )
        return False, f"failed to sample valid random board after {attempts_limit} attempts."

    startup_reset_ok, startup_reset_message = reset_episode_for_seed(seed)
    if startup_reset_ok:
        last_reset_summary = startup_reset_message
    else:
        last_reset_summary = startup_reset_message
        if random_board_mode:
            random_board_error = startup_reset_message
            # Keep viewer usable even when random-board startup fails.
            env.reset(seed)
            native_config_error = configure_native_backend()
        else:
            native_config_error = startup_reset_message

    if ai_backend == "bc":
        ai_available = bc_agent is not None
    elif ai_backend == "dqn":
        ai_available = dqn_agent is not None
    else:
        ai_available = env.has_bot() and native_config_error is None
    if random_board_error is not None:
        ai_available = False
    ai_enabled = bool(args.ai and ai_available)
    ai_metrics = {
        "pieces": 0,
        "lines": 0,
        "topouts": 0,
        "last_step_ms": 0.0,
        "avg_step_ms": 0.0,
        "step_sum_ms": 0.0,
        "step_samples": 0,
        "invalid_unmasked_predictions": 0,
        "unseen_legal_fallbacks": 0,
        "last_nodes": 0,
        "last_nps": 0.0,
        "last_score": 0.0,
        "last_budget_miss": 0,
        "budget_misses": 0,
        "last_q": 0.0,
        "last_candidate_count": 0,
        "start_ticks": pygame.time.get_ticks(),
    }
    ai_interval_ms = (1000.0 / float(args.ai_pps)) if float(args.ai_pps) > 0.0 else 0.0
    next_ai_tick = pygame.time.get_ticks()
    if args.ai and random_board_error is not None:
        print(
            f"Error: random_board startup failed: {random_board_error}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.ai and not ai_available:
        if ai_backend == "bc":
            status = "AI[BC] requested, but BC checkpoint failed to initialize."
        elif ai_backend == "dqn":
            status = "AI[DQN] requested, but DQN checkpoint failed to initialize."
        else:
            if native_config_error:
                status = f"AI[{ai_backend_label}] requested, but {native_config_error}"
            else:
                status = "AI requested, but bot API symbols were not found in shared library."
    elif ai_enabled:
        if ai_backend == "bc":
            status = f"AI[BC] enabled at startup (device={args.bc_device or 'auto'})"
        elif ai_backend == "dqn":
            status = f"AI[DQN] enabled at startup (device={args.dqn_device or 'auto'})"
        else:
            if ai_backend == "depth":
                status = (
                    f"AI[Depth] enabled at startup "
                    f"(depth={max(1, int(args.depth_search_depth))} gamma={float(args.depth_gamma):.3f})"
                )
            elif ai_backend == "beam":
                status = (
                    f"AI[Beam] enabled at startup "
                    f"(depth={max(1, int(args.beam_search_depth))} "
                    f"width={max(1, int(args.beam_width))} gamma={float(args.beam_gamma):.3f})"
                )
            else:
                status = f"AI[ColdClear] enabled at startup (think={max(1, int(args.think_ms))}ms)"
    elif random_board_error is not None:
        status = f"AI unavailable: random_board startup failed: {random_board_error}"
    elif native_config_error and ai_backend in ("cold_clear", "depth", "beam"):
        status = f"AI[{ai_backend_label}] unavailable: {native_config_error}"

    info_h = 280
    list_y = board_y + info_h + 10
    list_h = 280
    list_header_h = 30
    list_footer_h = 30
    row_h = 18
    list_content_h = max(1, list_h - list_header_h - list_footer_h)
    max_rows = max(1, list_content_h // row_h)
    scrollbar_w = 10
    list_rect = pygame.Rect(right_x, list_y, right_w, list_h)
    list_rows_rect = pygame.Rect(right_x + 6, list_y + list_header_h, right_w - 24, row_h * max_rows)
    list_scrollbar_rect = pygame.Rect(
        list_rows_rect.right + 6,
        list_rows_rect.top,
        scrollbar_w,
        list_rows_rect.height,
    )
    apply_button_rect = pygame.Rect(right_x + right_w - 126, list_y + list_h - 26, 116, 20)

    placements = env.placements()
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_a:
                        if ai_available:
                            ai_enabled = not ai_enabled
                            if ai_enabled:
                                ai_metrics["start_ticks"] = pygame.time.get_ticks()
                                next_ai_tick = pygame.time.get_ticks()
                                if ai_backend == "cold_clear":
                                    env.bot_sync()
                                    status = (
                                        f"AI[ColdClear] enabled (think={max(1, int(args.think_ms))}ms)"
                                    )
                                elif ai_backend == "depth":
                                    env.bot_sync()
                                    status = (
                                        f"AI[Depth] enabled "
                                        f"(depth={max(1, int(args.depth_search_depth))} "
                                        f"gamma={float(args.depth_gamma):.3f})"
                                    )
                                elif ai_backend == "beam":
                                    env.bot_sync()
                                    status = (
                                        f"AI[Beam] enabled "
                                        f"(depth={max(1, int(args.beam_search_depth))} "
                                        f"width={max(1, int(args.beam_width))} "
                                        f"gamma={float(args.beam_gamma):.3f})"
                                    )
                                elif ai_backend == "dqn":
                                    status = f"AI[DQN] enabled (device={args.dqn_device or 'auto'})"
                                else:
                                    status = f"AI[BC] enabled (device={args.bc_device or 'auto'})"
                            else:
                                status = "AI disabled."
                        else:
                            if ai_backend == "bc":
                                status = "AI unavailable: BC checkpoint is not available."
                            elif ai_backend == "dqn":
                                status = "AI unavailable: DQN checkpoint is not available."
                            else:
                                if random_board_error:
                                    status = f"AI unavailable: random_board startup failed: {random_board_error}"
                                elif native_config_error:
                                    status = f"AI[{ai_backend_label}] unavailable: {native_config_error}"
                                else:
                                    status = "AI unavailable: bot symbols not exported by shared library."
                    elif event.key == pygame.K_UP:
                        selected_index = max(0, selected_index - 1)
                        if selected_index < list_scroll:
                            list_scroll = selected_index
                    elif event.key == pygame.K_DOWN:
                        selected_index += 1
                        if selected_index >= list_scroll + max_rows:
                            list_scroll = selected_index - max_rows + 1
                    elif event.key == pygame.K_RETURN and not ai_enabled:
                        result = env.apply_placement(selected_index)
                        if result["success"]:
                            status = f"Applied placement {selected_index}: reward={result['reward']:.1f} lines={result['lines']}"
                        else:
                            status = "Placement apply failed."
                    elif event.key == pygame.K_h and not ai_enabled:
                        result = env.hold()
                        status = "Hold used." if result["success"] else "Hold unavailable."
                    elif event.key == pygame.K_LEFTBRACKET:
                        inspector_idx = (inspector_idx - 1) % len(inspector_actions)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        inspector_idx = (inspector_idx + 1) % len(inspector_actions)
                    elif event.key == pygame.K_r:
                        reset_ok, reset_message = reset_episode_for_seed(seed)
                        selected_index = 0
                        list_scroll = 0
                        if reset_ok:
                            last_reset_summary = reset_message
                            status = f"Reset seed={seed} ({reset_message})"
                        else:
                            last_reset_summary = f"reset_failed: {reset_message}"
                            random_board_error = reset_message if random_board_mode else random_board_error
                            ai_enabled = False
                            ai_available = False
                            status = f"Reset failed for seed={seed}: {reset_message}"
                    elif event.key == pygame.K_n:
                        seed = random.randint(1, 2**31 - 1)
                        reset_ok, reset_message = reset_episode_for_seed(seed)
                        selected_index = 0
                        list_scroll = 0
                        if reset_ok:
                            last_reset_summary = reset_message
                            status = f"Reset new seed={seed} ({reset_message})"
                        else:
                            last_reset_summary = f"reset_failed: {reset_message}"
                            random_board_error = reset_message if random_board_mode else random_board_error
                            ai_enabled = False
                            ai_available = False
                            status = f"Reset failed for seed={seed}: {reset_message}"
                elif event.type == pygame.MOUSEWHEEL:
                    if list_rect.collidepoint(pygame.mouse.get_pos()):
                        max_start = max(0, len(placements) - max_rows)
                        list_scroll = max(0, min(max_start, list_scroll - int(event.y)))
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if apply_button_rect.collidepoint(event.pos) and placements and not ai_enabled:
                        result = env.apply_placement(selected_index)
                        if result["success"]:
                            status = f"Applied placement {selected_index}: reward={result['reward']:.1f} lines={result['lines']}"
                        else:
                            status = "Placement apply failed."
                    elif list_scrollbar_rect.collidepoint(event.pos) and placements and len(placements) > max_rows:
                        max_start = len(placements) - max_rows
                        rel = event.pos[1] - list_scrollbar_rect.top
                        ratio = max(0.0, min(1.0, rel / float(max(1, list_scrollbar_rect.height - 1))))
                        list_scroll = int(round(ratio * max_start))
                    elif list_rows_rect.collidepoint(event.pos):
                        row = (event.pos[1] - list_rows_rect.top) // row_h
                        clicked_index = list_scroll + int(row)
                        if 0 <= row < max_rows and 0 <= clicked_index < len(placements):
                            if clicked_index == selected_index and not ai_enabled:
                                result = env.apply_placement(selected_index)
                                if result["success"]:
                                    status = (
                                        f"Applied placement {selected_index}: reward={result['reward']:.1f} "
                                        f"lines={result['lines']}"
                                    )
                                else:
                                    status = "Placement apply failed."
                            else:
                                selected_index = clicked_index
                                status = f"Selected placement {selected_index}"

            pre_meta = env.meta()
            if ai_enabled and ai_available:
                if pre_meta["game_over"]:
                    ai_metrics["topouts"] += 1
                    if args.auto_reset:
                        seed += 1
                        reset_ok, reset_message = reset_episode_for_seed(seed)
                        selected_index = 0
                        list_scroll = 0
                        if reset_ok:
                            last_reset_summary = reset_message
                            status = f"AI[{ai_backend_label}] auto-reset to seed={seed} ({reset_message})"
                        else:
                            last_reset_summary = f"reset_failed: {reset_message}"
                            random_board_error = reset_message if random_board_mode else random_board_error
                            ai_enabled = False
                            ai_available = False
                            status = f"AI auto-reset failed at seed={seed}: {reset_message}"
                    else:
                        ai_enabled = False
                        status = "AI paused on topout. Press R/N or toggle AI with A."
                else:
                    now_tick = pygame.time.get_ticks()
                    if ai_interval_ms > 0.0 and now_tick < next_ai_tick:
                        pass
                    elif ai_backend == "bc":
                        if bc_agent is None:
                            ai_enabled = False
                            status = "AI[BC] unavailable: checkpoint did not initialize."
                        else:
                            legal_actions = env.enumerate_legal_actions()
                            if not legal_actions:
                                ai_enabled = False
                                status = "AI[BC] no legal actions; autoplay disabled."
                            else:
                                step_start_ms = pygame.time.get_ticks()
                                try:
                                    state = env.get_state()
                                    chosen_action, diag = bc_agent.predict_action_with_diagnostics(
                                        state,
                                        legal_actions=legal_actions,
                                    )
                                    ai_result = env.step_native_action(chosen_action)
                                except Exception as exc:
                                    ai_enabled = False
                                    status = f"AI[BC] inference failed: {exc}"
                                    ai_result = None

                                if ai_result is not None and ai_result["success"]:
                                    step_elapsed_ms = float(pygame.time.get_ticks() - step_start_ms)
                                    ai_metrics["pieces"] += 1
                                    ai_metrics["lines"] += int(ai_result["lines"])
                                    ai_metrics["last_step_ms"] = step_elapsed_ms
                                    ai_metrics["step_sum_ms"] += step_elapsed_ms
                                    ai_metrics["step_samples"] += 1
                                    ai_metrics["avg_step_ms"] = (
                                        ai_metrics["step_sum_ms"] / max(1, ai_metrics["step_samples"])
                                    )
                                    ai_metrics["invalid_unmasked_predictions"] += int(
                                        bool(diag["raw_argmax_invalid"])
                                    )
                                    ai_metrics["unseen_legal_fallbacks"] += int(
                                        bool(diag["used_fallback_unseen_legal"])
                                    )
                                    status = (
                                        f"AI[BC] move hold={int(chosen_action.use_hold)} "
                                        f"idx={int(chosen_action.placement_index)} "
                                        f"lines+={int(ai_result['lines'])}"
                                    )
                                    if ai_result["game_over"]:
                                        ai_metrics["topouts"] += 1
                                        if args.auto_reset:
                                            seed += 1
                                            reset_ok, reset_message = reset_episode_for_seed(seed)
                                            selected_index = 0
                                            list_scroll = 0
                                            if reset_ok:
                                                last_reset_summary = reset_message
                                                status = f"AI[BC] topout -> auto-reset seed={seed} ({reset_message})"
                                            else:
                                                last_reset_summary = f"reset_failed: {reset_message}"
                                                random_board_error = (
                                                    reset_message if random_board_mode else random_board_error
                                                )
                                                ai_enabled = False
                                                ai_available = False
                                                status = f"AI[BC] auto-reset failed at seed={seed}: {reset_message}"
                                        else:
                                            ai_enabled = False
                                            status = "AI[BC] topout. Autoplay stopped (auto-reset disabled)."
                                elif ai_result is not None:
                                    ai_enabled = False
                                    status = "AI[BC] action apply failed. Autoplay disabled."
                            if ai_interval_ms > 0.0:
                                next_ai_tick = now_tick + ai_interval_ms
                    elif ai_backend == "dqn":
                        if dqn_agent is None:
                            ai_enabled = False
                            status = "AI[DQN] unavailable: checkpoint did not initialize."
                        else:
                            candidates = env.enumerate_dqn_candidates()
                            if not candidates:
                                ai_enabled = False
                                status = "AI[DQN] no legal candidates; autoplay disabled."
                            else:
                                step_start_ms = pygame.time.get_ticks()
                                try:
                                    chosen_action, diag = dqn_agent.predict_action_with_diagnostics(candidates)
                                    ai_result = env.step_native_action(chosen_action)
                                except Exception as exc:
                                    ai_enabled = False
                                    status = f"AI[DQN] inference failed: {exc}"
                                    ai_result = None

                                if ai_result is not None and ai_result["success"]:
                                    step_elapsed_ms = float(pygame.time.get_ticks() - step_start_ms)
                                    ai_metrics["pieces"] += 1
                                    ai_metrics["lines"] += int(ai_result["lines"])
                                    ai_metrics["last_step_ms"] = step_elapsed_ms
                                    ai_metrics["step_sum_ms"] += step_elapsed_ms
                                    ai_metrics["step_samples"] += 1
                                    ai_metrics["avg_step_ms"] = (
                                        ai_metrics["step_sum_ms"] / max(1, ai_metrics["step_samples"])
                                    )
                                    ai_metrics["last_q"] = float(diag.get("best_q", 0.0))
                                    ai_metrics["last_candidate_count"] = int(
                                        diag.get("candidate_count", len(candidates))
                                    )
                                    status = (
                                        f"AI[DQN] move hold={int(chosen_action.use_hold)} "
                                        f"idx={int(chosen_action.placement_index)} "
                                        f"q={float(ai_metrics['last_q']):.3f} "
                                        f"lines+={int(ai_result['lines'])}"
                                    )
                                    if ai_result["game_over"]:
                                        ai_metrics["topouts"] += 1
                                        if args.auto_reset:
                                            seed += 1
                                            reset_ok, reset_message = reset_episode_for_seed(seed)
                                            selected_index = 0
                                            list_scroll = 0
                                            if reset_ok:
                                                last_reset_summary = reset_message
                                                status = f"AI[DQN] topout -> auto-reset seed={seed} ({reset_message})"
                                            else:
                                                last_reset_summary = f"reset_failed: {reset_message}"
                                                random_board_error = (
                                                    reset_message if random_board_mode else random_board_error
                                                )
                                                ai_enabled = False
                                                ai_available = False
                                                status = f"AI[DQN] auto-reset failed at seed={seed}: {reset_message}"
                                        else:
                                            ai_enabled = False
                                            status = "AI[DQN] topout. Autoplay stopped (auto-reset disabled)."
                                elif ai_result is not None:
                                    ai_enabled = False
                                    status = "AI[DQN] action apply failed. Autoplay disabled."
                            if ai_interval_ms > 0.0:
                                next_ai_tick = now_tick + ai_interval_ms
                    else:
                        ai_result = env.bot_choose_and_apply(args.think_ms)
                        if ai_result["success"]:
                            ai_metrics["pieces"] += 1
                            ai_metrics["lines"] += int(ai_result["lines"])
                            ai_metrics["last_step_ms"] = float(ai_result["think_ms"])
                            ai_metrics["step_sum_ms"] += float(ai_result["think_ms"])
                            ai_metrics["step_samples"] += 1
                            ai_metrics["avg_step_ms"] = (
                                ai_metrics["step_sum_ms"] / max(1, ai_metrics["step_samples"])
                            )
                            ai_metrics["last_nodes"] = int(ai_result["nodes"])
                            ai_metrics["last_nps"] = float(ai_result["nps"])
                            ai_metrics["last_score"] = float(ai_result["score"])
                            ai_metrics["last_budget_miss"] = int(ai_result["budget_miss"])
                            ai_metrics["budget_misses"] += int(ai_result["budget_miss"])
                            status = (
                                f"AI[{ai_backend_label}] move idx={ai_result['placement_index']} "
                                f"hold={ai_result['used_hold']} score={ai_result['score']:.2f} "
                                f"lines+={ai_result['lines']}"
                            )
                            if ai_result["game_over"]:
                                ai_metrics["topouts"] += 1
                                if args.auto_reset:
                                    seed += 1
                                    reset_ok, reset_message = reset_episode_for_seed(seed)
                                    selected_index = 0
                                    list_scroll = 0
                                    if reset_ok:
                                        last_reset_summary = reset_message
                                        status = (
                                            f"AI[{ai_backend_label}] topout -> auto-reset seed={seed} "
                                            f"({reset_message})"
                                        )
                                    else:
                                        last_reset_summary = f"reset_failed: {reset_message}"
                                        random_board_error = reset_message if random_board_mode else random_board_error
                                        ai_enabled = False
                                        ai_available = False
                                        status = (
                                            f"AI[{ai_backend_label}] auto-reset failed at seed={seed}: "
                                            f"{reset_message}"
                                        )
                                else:
                                    ai_enabled = False
                                    status = "AI topout. Autoplay stopped (auto-reset disabled)."
                        else:
                            ai_enabled = False
                            status = "AI choose/apply failed. Autoplay disabled."
                        if ai_interval_ms > 0.0:
                            next_ai_tick = now_tick + ai_interval_ms

            board_occ = env.board()
            board_piece_ids = env.board_piece_ids(include_active=True)
            hold = env.hold_info()
            queue = env.queue()
            meta = env.meta()
            attack = env.last_attack_meta()
            blitz = env.blitz_meta()
            placements = env.placements()

            if placements:
                selected_index = max(0, min(selected_index, len(placements) - 1))
                max_start = max(0, len(placements) - max_rows)
                list_scroll = max(0, min(max_start, list_scroll))
                preview_occ = env.placement_board(selected_index)
                preview_piece_ids = env.placement_piece_ids(selected_index)
            else:
                selected_index = 0
                list_scroll = 0
                preview_occ = [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
                preview_piece_ids = [[EMPTY_CELL_ID for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]

            inspect_action = inspector_actions[inspector_idx]
            trace = env.rotation_trace(inspect_action)

            screen.fill(BG_COLOR)

            pygame.draw.rect(screen, PANEL_COLOR, (board_x - 4, board_y - 4, board_w + 8, board_h + 8), border_radius=6)
            draw_board(screen, board_x, board_y, cell, board_piece_ids, board_occ)

            # Top-right info panel
            info_y = board_y
            pygame.draw.rect(screen, PANEL_COLOR, (right_x, info_y, right_w, info_h), border_radius=8)
            ai_elapsed_s = max(1e-6, (pygame.time.get_ticks() - ai_metrics["start_ticks"]) / 1000.0)
            ai_pps = ai_metrics["pieces"] / ai_elapsed_s
            ai_backend_info = ai_backend_label
            if ai_backend == "bc":
                ai_backend_info = f"BC ({args.bc_device or 'auto'})"
            elif ai_backend == "dqn":
                ai_backend_info = f"DQN ({args.dqn_device or 'auto'})"
            lines = [
                f"Seed: {seed}",
                f"Mode: {mode_name}",
                f"Obs size: {env.observation_size()}",
                f"Hold: {PIECE_NAMES.get(hold['hold_piece'], '?')}  avail={hold['hold_available']}",
                f"Queue: {' '.join(PIECE_NAMES.get(p, '?') for p in queue[:max(0, args.queue_visible)])}",
                f"GameOver={meta['game_over']} TopOut={meta['top_out']}",
                f"Combo={meta['combo']} B2B={meta['b2b']} Lines={meta['lines']}",
                f"LockTimer={meta['lock_timer']} Resets={meta['lock_resets']}",
                (
                    "StateSource: random_board "
                    f"(y_max_excl={int(args.random_fill_y_max_exclusive)} "
                    f"p={float(args.random_fill_prob):.2f} "
                    f"resamples={int(args.random_max_resamples_per_sample)})"
                    if random_board_mode
                    else "StateSource: rollout"
                ),
                f"Last reset: {last_reset_summary}",
                f"AI: {'ON' if ai_enabled else 'OFF'} backend={ai_backend_info} avail={ai_available}",
                f"AI target PPS={float(args.ai_pps):.2f} (0=unbounded)",
                f"AI pieces={ai_metrics['pieces']} lines={ai_metrics['lines']} topouts={ai_metrics['topouts']}",
                f"AI PPS={ai_pps:.2f} step_ms(last/avg)={ai_metrics['last_step_ms']:.1f}/{ai_metrics['avg_step_ms']:.1f}",
            ]
            if mode_name in ("scoring", "blitz"):
                rem_ms = max(0, int(blitz.get("time_remaining_ms", 0)))
                rem_sec = rem_ms // 1000
                mm = rem_sec // 60
                ss = rem_sec % 60
                lines.insert(8, f"BLITZ Score={int(blitz.get('score_total', 0))}")
                lines.insert(
                    9,
                    (
                        "BLITZ Level="
                        f"{int(blitz.get('level', 1))} "
                        f"LinesToNext={int(blitz.get('lines_to_next', 0))} "
                        f"Time={mm:02d}:{ss:02d}.{(rem_ms % 1000) // 100} "
                        f"TimedOut={int(bool(blitz.get('timed_out', False)))}"
                    ),
                )
            else:
                lines.insert(
                    8,
                    (
                        "Attack:"
                        f" total={attack['attack_total']} base={attack['attack_base']} "
                        f"scaled={attack['attack_combo_scaled']:.2f} round={attack['attack_rounded']}"
                    ),
                )
                lines.insert(
                    9,
                    (
                        "B2B:"
                        f" streak={attack['b2b_streak']} bonus={attack['attack_b2b_bonus']} "
                        f"surge={attack['surge_charge']} release={attack['surge_release']} "
                        f"all_clear={int(attack['all_clear'])}"
                    ),
                )
            if ai_backend in ("cold_clear", "depth", "beam"):
                lines.append(
                    f"AI[{ai_backend_label}] nodes={ai_metrics['last_nodes']} "
                    f"nps={ai_metrics['last_nps']:.0f} score={ai_metrics['last_score']:.2f}"
                )
                lines.append(
                    f"AI[{ai_backend_label}] budget_miss "
                    f"last/total={ai_metrics['last_budget_miss']}/{ai_metrics['budget_misses']}"
                )
            elif ai_backend == "bc":
                lines.append(
                    f"AI[BC] invalid_raw={ai_metrics['invalid_unmasked_predictions']} "
                    f"fallbacks={ai_metrics['unseen_legal_fallbacks']}"
                )
            else:
                lines.append(
                    f"AI[DQN] q={ai_metrics['last_q']:.3f} "
                    f"candidates={int(ai_metrics['last_candidate_count'])}"
                )
            for i, txt in enumerate(lines):
                surface = small_font.render(txt, True, LOCK_TEXT)
                screen.blit(surface, (right_x + 10, info_y + 10 + i * 19))

            # Placement list panel
            pygame.draw.rect(screen, PANEL_COLOR, (right_x, list_y, right_w, list_h), border_radius=8)
            title = font.render(f"Placements ({len(placements)})", True, LOCK_TEXT)
            screen.blit(title, (right_x + 10, list_y + 8))

            start = list_scroll
            end = min(len(placements), start + max_rows)
            for i in range(start, end):
                p = placements[i]
                y = list_rows_rect.top + (i - start) * row_h
                text = f"[{p['index']:03d}] x={p['x']:>2} y={p['y']:>2} rot={ROTATION_NAMES.get(p['rotation'], '?')} lines={p['lines']}"
                color = LOCK_TEXT
                if i == selected_index:
                    pygame.draw.rect(screen, SELECT_COLOR, (list_rows_rect.left, y - 1, list_rows_rect.width, row_h - 1), border_radius=4)
                    color = (255, 255, 255)
                screen.blit(small_font.render(text, True, color), (list_rows_rect.left + 6, y))
            if placements:
                shown = f"Showing {start + 1}-{end} / {len(placements)}"
                screen.blit(small_font.render(shown, True, LOCK_TEXT), (right_x + right_w - 180, list_y + 9))

            # Scrollbar and apply button
            pygame.draw.rect(screen, (42, 48, 60), list_scrollbar_rect, border_radius=4)
            if placements and len(placements) > max_rows:
                max_start = len(placements) - max_rows
                thumb_h = max(24, int(list_scrollbar_rect.height * (max_rows / float(len(placements)))))
                travel = max(0, list_scrollbar_rect.height - thumb_h)
                thumb_top = list_scrollbar_rect.top + int(travel * (list_scroll / float(max_start)))
                thumb_rect = pygame.Rect(list_scrollbar_rect.left, thumb_top, list_scrollbar_rect.width, thumb_h)
            else:
                thumb_rect = pygame.Rect(
                    list_scrollbar_rect.left,
                    list_scrollbar_rect.top,
                    list_scrollbar_rect.width,
                    list_scrollbar_rect.height,
                )
            pygame.draw.rect(screen, (120, 136, 170), thumb_rect, border_radius=4)

            button_enabled = bool(placements and not ai_enabled)
            button_color = (66, 96, 170) if button_enabled else (58, 64, 76)
            pygame.draw.rect(screen, button_color, apply_button_rect, border_radius=5)
            button_label = "Apply (Enter)" if not ai_enabled else "AI running"
            button_text = small_font.render(button_label, True, (255, 255, 255))
            screen.blit(button_text, (apply_button_rect.x + 12, apply_button_rect.y + 2))

            # Preview panel
            preview_y = list_y + list_h + 10
            preview_cell = max(8, cell // 2)
            preview_h = BOARD_ROWS * preview_cell + 36
            pygame.draw.rect(screen, PANEL_COLOR, (right_x, preview_y, right_w, preview_h), border_radius=8)
            screen.blit(font.render("Selected Placement Board", True, LOCK_TEXT), (right_x + 10, preview_y + 8))
            draw_small_board(
                screen,
                right_x + 10,
                preview_y + 30,
                preview_cell,
                preview_piece_ids,
                preview_occ,
            )

            # Kick inspector panel
            kick_y = preview_y + preview_h + 10
            kick_h = screen_h - kick_y - 20
            pygame.draw.rect(screen, PANEL_COLOR, (right_x, kick_y, right_w, kick_h), border_radius=8)
            header = f"Kick Inspector [{action_name(inspect_action)}] success={trace['success']} final=({trace['final_x']},{trace['final_y']},{ROTATION_NAMES.get(trace['final_rotation'], '?')})"
            screen.blit(small_font.render(header, True, LOCK_TEXT), (right_x + 10, kick_y + 8))

            trace_row_h = 17
            max_trace_rows = max(1, (kick_h - 34) // trace_row_h)
            tests = trace["tests"][:max_trace_rows]
            for i, t in enumerate(tests):
                y = kick_y + 28 + i * trace_row_h
                mark = "PASS" if t["passed"] else "fail"
                color = PASS_COLOR if t["passed"] else FAIL_COLOR
                txt = (
                    f"{t['test_index']:02d} ph{t['phase']} k{t['kick_index']} "
                    f"({t['dx']:+d},{t['dy']:+d}) {mark} -> "
                    f"({t['x']},{t['y']},{ROTATION_NAMES.get(t['rotation'], '?')})"
                )
                screen.blit(small_font.render(txt, True, color), (right_x + 10, y))

            controls = "Controls: A toggle AI | Wheel browse | Click row | Enter apply | H | [ ] | R/N | Q"
            screen.blit(small_font.render(controls, True, LOCK_TEXT), (board_x, screen_h - 22))
            screen.blit(small_font.render(status, True, (180, 210, 255)), (board_x, board_y + board_h + 8))

            pygame.display.flip()
            clock.tick(max(10, args.fps))
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
