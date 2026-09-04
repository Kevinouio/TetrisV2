"""Shared ctypes runtime bindings for the `tetris_cc_*` API."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

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
ACTION_ROTATE_CW = ACTION_CW
ACTION_ROTATE_CCW = ACTION_CCW
ACTION_ROTATE_180 = 7
ACTION_HOLD = 8


class _EnvStepResult(ctypes.Structure):
    _fields_ = [
        ("action_succeeded", ctypes.c_int),
        ("piece_locked", ctypes.c_int),
        ("hold_used", ctypes.c_int),
        ("lines_cleared", ctypes.c_int),
        ("spin_clear", ctypes.c_int),
        ("spin_type", ctypes.c_int),
        ("difficult_clear", ctypes.c_int),
        ("b2b_bonus_applied", ctypes.c_int),
        ("combo", ctypes.c_int),
        ("back_to_back", ctypes.c_int),
        ("reward", ctypes.c_float),
        ("game_over", ctypes.c_int),
        ("top_out", ctypes.c_int),
    ]


def _step_result_dict(result: _EnvStepResult) -> Dict[str, object]:
    return {
        "action_succeeded": bool(result.action_succeeded),
        "piece_locked": bool(result.piece_locked),
        "hold_used": bool(result.hold_used),
        "lines_cleared": int(result.lines_cleared),
        "spin_clear": bool(result.spin_clear),
        "spin_type": int(result.spin_type),
        "difficult_clear": bool(result.difficult_clear),
        "b2b_bonus_applied": bool(result.b2b_bonus_applied),
        "combo": int(result.combo),
        "back_to_back": bool(result.back_to_back),
        "reward": float(result.reward),
        "game_over": bool(result.game_over),
        "top_out": bool(result.top_out),
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


def find_library(explicit_path: Optional[Path]) -> Path:
    if explicit_path is not None:
        if explicit_path.exists():
            return explicit_path
        raise FileNotFoundError(f"Library not found: {explicit_path}")

    configured = os.environ.get("TETRIS_V2_LIBRARY")
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.exists():
            return configured_path
        raise FileNotFoundError(f"Library not found: {configured_path}")

    build_dir = Path(__file__).resolve().parents[2] / "build"
    candidates = [
        build_dir / "tetris_v2_c_api.dll",
        build_dir / "Debug" / "tetris_v2_c_api.dll",
        build_dir / "Release" / "tetris_v2_c_api.dll",
        build_dir / "libtetris_v2_c_api.so",
        build_dir / "Debug" / "libtetris_v2_c_api.so",
        build_dir / "Release" / "libtetris_v2_c_api.so",
        build_dir / "libtetris_v2_c_api.dylib",
        build_dir / "Debug" / "libtetris_v2_c_api.dylib",
        build_dir / "Release" / "libtetris_v2_c_api.dylib",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate tetris_v2_c_api shared library. Build it first.")


def _require_abi(condition: bool, operation: str) -> None:
    if not condition:
        raise RuntimeError(f"C++ runtime failed to {operation}.")


class EnvCtypes:
    """ctypes wrapper around the current `tetris_cc_*` C API."""

    def __init__(self, lib_path: Path, seed: int, *, play_mode: bool = False):
        self.lib = ctypes.CDLL(str(lib_path))
        self._bind()
        create = self.lib.tetris_cc_env_create_play if play_mode else self.lib.tetris_cc_env_create
        self.handle = create(ctypes.c_uint32(seed))
        if not self.handle:
            raise RuntimeError("Failed to create env handle")
        self.bot_handle = None
        self.seed = int(seed)
        self.play_mode = bool(play_mode)

    def _bind(self):
        void_p = ctypes.c_void_p
        c_int_p = ctypes.POINTER(ctypes.c_int)

        self.lib.tetris_cc_env_create.argtypes = [ctypes.c_uint32]
        self.lib.tetris_cc_env_create.restype = void_p

        self.lib.tetris_cc_env_create_play.argtypes = [ctypes.c_uint32]
        self.lib.tetris_cc_env_create_play.restype = void_p

        self.lib.tetris_cc_env_destroy.argtypes = [void_p]
        self.lib.tetris_cc_env_destroy.restype = None

        self.lib.tetris_cc_env_reset.argtypes = [void_p, ctypes.c_uint32]
        self.lib.tetris_cc_env_reset.restype = None

        self.lib.tetris_cc_env_step_ex.argtypes = [
            void_p,
            ctypes.c_int,
            ctypes.POINTER(_EnvStepResult),
        ]
        self.lib.tetris_cc_env_step_ex.restype = ctypes.c_int

        self.lib.tetris_cc_env_input_ex.argtypes = [
            void_p,
            ctypes.c_int,
            ctypes.POINTER(_EnvStepResult),
        ]
        self.lib.tetris_cc_env_input_ex.restype = ctypes.c_int

        self.lib.tetris_cc_env_tick_ex.argtypes = [
            void_p,
            ctypes.POINTER(_EnvStepResult),
        ]
        self.lib.tetris_cc_env_tick_ex.restype = ctypes.c_int

        self.lib.tetris_cc_env_hold.argtypes = [void_p, ctypes.POINTER(ctypes.c_float)]
        self.lib.tetris_cc_env_hold.restype = ctypes.c_int

        self.lib.tetris_cc_env_observation_size.argtypes = [void_p, ctypes.c_int]
        self.lib.tetris_cc_env_observation_size.restype = ctypes.c_size_t

        self.lib.tetris_cc_env_observation_write.argtypes = [
            void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        self.lib.tetris_cc_env_observation_write.restype = ctypes.c_size_t

        self.lib.tetris_cc_env_board_write.argtypes = [
            void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        self.lib.tetris_cc_env_board_write.restype = ctypes.c_size_t

        self.lib.tetris_cc_env_board_piece_ids_write.argtypes = [
            void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        self.lib.tetris_cc_env_board_piece_ids_write.restype = ctypes.c_size_t

        self.lib.tetris_cc_env_active_piece.argtypes = [void_p, c_int_p, c_int_p, c_int_p, c_int_p]
        self.lib.tetris_cc_env_active_piece.restype = ctypes.c_int

        self.lib.tetris_cc_env_ghost_piece.argtypes = [void_p, c_int_p, c_int_p, c_int_p, c_int_p]
        self.lib.tetris_cc_env_ghost_piece.restype = ctypes.c_int

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

        self.lib.tetris_cc_env_placement_board_piece_ids_write.argtypes = [
            void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        self.lib.tetris_cc_env_placement_board_piece_ids_write.restype = ctypes.c_size_t

        self.lib.tetris_cc_env_apply_placement_index.argtypes = [
            void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            c_int_p,
            c_int_p,
        ]
        self.lib.tetris_cc_env_apply_placement_index.restype = ctypes.c_int

        self.lib.tetris_cc_env_decision_action_dim.argtypes = []
        self.lib.tetris_cc_env_decision_action_dim.restype = ctypes.c_size_t
        self.lib.tetris_cc_env_decision_mask_write.argtypes = [
            void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        self.lib.tetris_cc_env_decision_mask_write.restype = ctypes.c_size_t
        self.lib.tetris_cc_env_decision_get.argtypes = [
            void_p,
            ctypes.c_size_t,
            c_int_p,
            ctypes.POINTER(ctypes.c_size_t),
            c_int_p,
            c_int_p,
            c_int_p,
        ]
        self.lib.tetris_cc_env_decision_get.restype = ctypes.c_int
        self.lib.tetris_cc_env_decision_action_for_choice.argtypes = [
            void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.tetris_cc_env_decision_action_for_choice.restype = ctypes.c_int
        self.lib.tetris_cc_env_apply_decision.argtypes = [
            void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            c_int_p,
            c_int_p,
            c_int_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.tetris_cc_env_apply_decision.restype = ctypes.c_int

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
        self.lib.tetris_cc_bot_rank_actions.argtypes = [
            void_p,
            void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            u64_p,
            d_p,
            d_p,
            c_int_p,
            c_int_p,
            c_int_p,
            c_int_p,
        ]
        self.lib.tetris_cc_bot_rank_actions.restype = ctypes.c_int

    def close(self):
        if self.bot_handle:
            self.lib.tetris_cc_bot_destroy(self.bot_handle)
            self.bot_handle = None
        if self.handle:
            self.lib.tetris_cc_env_destroy(self.handle)
            self.handle = None

    def reset(self, seed: int):
        self.seed = int(seed)
        self.lib.tetris_cc_env_reset(self.handle, ctypes.c_uint32(seed))
        if self.bot_handle:
            self.bot_sync()

    def step(self, action: int):
        result = _EnvStepResult()
        ok = self.lib.tetris_cc_env_step_ex(self.handle, int(action), ctypes.byref(result))
        _require_abi(bool(ok), f"step action {action}")
        if self.bot_handle:
            self.bot_sync()
        return _step_result_dict(result)

    def input(self, action: int):
        result = _EnvStepResult()
        ok = self.lib.tetris_cc_env_input_ex(self.handle, int(action), ctypes.byref(result))
        _require_abi(bool(ok), f"apply zero-time input {action}")
        if self.bot_handle:
            self.bot_sync()
        return _step_result_dict(result)

    def tick(self):
        result = _EnvStepResult()
        ok = self.lib.tetris_cc_env_tick_ex(self.handle, ctypes.byref(result))
        _require_abi(bool(ok), "advance one simulation tick")
        if self.bot_handle:
            self.bot_sync()
        return _step_result_dict(result)

    def hold(self):
        reward = ctypes.c_float(0.0)
        success = self.lib.tetris_cc_env_hold(self.handle, ctypes.byref(reward))
        if success and self.bot_handle:
            self.bot_sync()
        return {"success": bool(success), "reward": float(reward.value)}

    def _create_bot(self):
        if self.bot_handle:
            return
        self.bot_handle = self.lib.tetris_cc_bot_create_default()
        _require_abi(bool(self.bot_handle), "create the bot")
        ok = self.lib.tetris_cc_bot_sync_from_env(self.bot_handle, self.handle)
        _require_abi(bool(ok), "synchronize the bot")

    def bot_sync(self):
        if not self.bot_handle:
            self._create_bot()
            return
        ok = self.lib.tetris_cc_bot_sync_from_env(self.bot_handle, self.handle)
        _require_abi(bool(ok), "synchronize the bot")

    def bot_choose(self, think_ms: int = 10):
        """Choose an expert action without changing the environment."""

        if not self.bot_handle:
            self._create_bot()
        use_hold = ctypes.c_int(0)
        placement_index = ctypes.c_size_t(0)
        score = ctypes.c_float(0.0)
        nodes = ctypes.c_uint64(0)
        think = ctypes.c_double(0.0)
        nps = ctypes.c_double(0.0)
        budget_miss = ctypes.c_int(0)
        ok = self.lib.tetris_cc_bot_choose(
            self.bot_handle,
            int(think_ms),
            ctypes.byref(use_hold),
            ctypes.byref(placement_index),
            ctypes.byref(score),
            ctypes.byref(nodes),
            ctypes.byref(think),
            ctypes.byref(nps),
            ctypes.byref(budget_miss),
        )
        return {
            "success": bool(ok),
            "use_hold": bool(use_hold.value),
            "placement_index": int(placement_index.value),
            "score": float(score.value),
            "nodes": int(nodes.value),
            "think_ms": float(think.value),
            "nps": float(nps.value),
            "budget_miss": int(budget_miss.value),
        }

    def bot_choose_and_apply(self, think_ms: int = 20):
        if not self.bot_handle:
            self._create_bot()
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
        ok = self.lib.tetris_cc_bot_choose_and_apply(
            self.bot_handle,
            self.handle,
            int(think_ms),
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

    def bot_rank_actions(self, think_ms: int = 10, action_dim: int = 97):
        if not self.bot_handle:
            self._create_bot()
        dim = int(max(1, action_dim))
        scores = (ctypes.c_float * dim)()
        legal_mask = (ctypes.c_uint8 * dim)()
        nodes = ctypes.c_uint64(0)
        think = ctypes.c_double(0.0)
        nps = ctypes.c_double(0.0)
        budget_miss = ctypes.c_int(0)
        placement_count_raw = ctypes.c_int(0)
        placement_overflow = ctypes.c_int(0)
        unexpanded_count = ctypes.c_int(0)

        ok = self.lib.tetris_cc_bot_rank_actions(
            self.bot_handle,
            self.handle,
            int(think_ms),
            scores,
            dim,
            legal_mask,
            dim,
            ctypes.byref(nodes),
            ctypes.byref(think),
            ctypes.byref(nps),
            ctypes.byref(budget_miss),
            ctypes.byref(placement_count_raw),
            ctypes.byref(placement_overflow),
            ctypes.byref(unexpanded_count),
        )

        return {
            "success": bool(ok),
            "scores": np.frombuffer(scores, dtype=np.float32, count=dim).copy(),
            "legal_mask": np.frombuffer(legal_mask, dtype=np.uint8, count=dim).copy(),
            "nodes": int(nodes.value),
            "think_ms": float(think.value),
            "nps": float(nps.value),
            "budget_miss": int(budget_miss.value),
            "placement_count_raw": int(placement_count_raw.value),
            "placement_overflow": bool(placement_overflow.value),
            "unexpanded_count": int(unexpanded_count.value),
        }

    def observation_size(self, include_hidden_rows: bool = True):
        size = int(self.lib.tetris_cc_env_observation_size(self.handle, 1 if include_hidden_rows else 0))
        _require_abi(size > 0, "report the observation size")
        return size

    def observation(self, include_hidden_rows: bool = False) -> np.ndarray:
        size = self.observation_size(include_hidden_rows=include_hidden_rows)
        buf = (ctypes.c_float * size)()
        written = self.lib.tetris_cc_env_observation_write(
            self.handle,
            1 if include_hidden_rows else 0,
            buf,
            size,
        )
        _require_abi(written == size, "write the observation")
        return np.frombuffer(buf, dtype=np.float32).copy()

    def board(self):
        buf = (ctypes.c_uint8 * (BOARD_ROWS * BOARD_COLS))()
        written = self.lib.tetris_cc_env_board_write(self.handle, 0, buf, len(buf))
        _require_abi(written == len(buf), "write the board")
        flat = [int(v) for v in buf]
        return [flat[r * BOARD_COLS : (r + 1) * BOARD_COLS] for r in range(BOARD_ROWS)]

    def board_piece_ids(self, include_active: bool = True):
        buf = (ctypes.c_uint8 * (BOARD_ROWS * BOARD_COLS))()
        written = self.lib.tetris_cc_env_board_piece_ids_write(
            self.handle,
            1 if include_active else 0,
            buf,
            len(buf),
        )
        _require_abi(written == len(buf), "write the board piece IDs")
        flat = [int(v) for v in buf]
        return [flat[r * BOARD_COLS : (r + 1) * BOARD_COLS] for r in range(BOARD_ROWS)]

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
        _require_abi(bool(ok), "read the active piece")
        return {"piece": int(piece.value), "rotation": int(rotation.value), "x": int(x.value), "y": int(y.value)}

    def ghost(self):
        piece = ctypes.c_int(-1)
        rotation = ctypes.c_int(-1)
        x = ctypes.c_int(0)
        landing_y = ctypes.c_int(0)
        ok = self.lib.tetris_cc_env_ghost_piece(
            self.handle,
            ctypes.byref(piece),
            ctypes.byref(rotation),
            ctypes.byref(x),
            ctypes.byref(landing_y),
        )
        if not ok:
            return None
        return {
            "piece": int(piece.value),
            "rotation": int(rotation.value),
            "x": int(x.value),
            "y": int(landing_y.value),
        }

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
        _require_abi(bool(ok), "read the held piece")
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
            ok = self.lib.tetris_cc_env_queue_get(self.handle, i, ctypes.byref(piece))
            _require_abi(bool(ok), f"read queue item {i}")
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
        _require_abi(bool(ok), "read environment metadata")
        return {
            "game_over": bool(vals[0].value),
            "top_out": bool(vals[1].value),
            "combo": int(vals[2].value),
            "b2b": bool(vals[3].value),
            "lines": int(vals[4].value),
            "lock_timer": int(vals[5].value),
            "lock_resets": int(vals[6].value),
        }

    def placements(self):
        count = int(self.lib.tetris_cc_env_placement_count(self.handle))
        out: List[Dict[str, int]] = []
        for i in range(count):
            x = ctypes.c_int(0)
            y = ctypes.c_int(0)
            rot = ctypes.c_int(0)
            lines = ctypes.c_int(0)
            ok = self.lib.tetris_cc_env_placement_get(
                self.handle, i, ctypes.byref(x), ctypes.byref(y), ctypes.byref(rot), ctypes.byref(lines)
            )
            _require_abi(bool(ok), f"read placement {i}")
            out.append(
                {
                    "index": i,
                    "x": int(x.value),
                    "y": int(y.value),
                    "rotation": int(rot.value),
                    "lines": int(lines.value),
                }
            )
        return out

    def placement_count(self) -> int:
        return int(self.lib.tetris_cc_env_placement_count(self.handle))

    def decision_action_dim(self) -> int:
        return int(self.lib.tetris_cc_env_decision_action_dim())

    def decision_mask(self) -> np.ndarray:
        dim = self.decision_action_dim()
        buf = (ctypes.c_uint8 * dim)()
        written = self.lib.tetris_cc_env_decision_mask_write(self.handle, buf, dim)
        _require_abi(written == dim, "write the placement-decision mask")
        return np.frombuffer(buf, dtype=np.uint8).copy()

    def decision(self, action: int) -> Dict[str, int | bool]:
        use_hold = ctypes.c_int(0)
        placement_index = ctypes.c_size_t(0)
        x = ctypes.c_int(0)
        y = ctypes.c_int(0)
        rotation = ctypes.c_int(0)
        ok = self.lib.tetris_cc_env_decision_get(
            self.handle,
            int(action),
            ctypes.byref(use_hold),
            ctypes.byref(placement_index),
            ctypes.byref(x),
            ctypes.byref(y),
            ctypes.byref(rotation),
        )
        _require_abi(bool(ok), f"describe decision {action}")
        return {
            "action": int(action),
            "use_hold": bool(use_hold.value),
            "placement_index": int(placement_index.value),
            "x": int(x.value),
            "y": int(y.value),
            "rotation": int(rotation.value),
        }

    def decision_for_choice(self, *, use_hold: bool, placement_index: int) -> int:
        action = ctypes.c_size_t(0)
        ok = self.lib.tetris_cc_env_decision_action_for_choice(
            self.handle,
            1 if use_hold else 0,
            int(placement_index),
            ctypes.byref(action),
        )
        _require_abi(bool(ok), "map the expert choice to a stable decision")
        return int(action.value)

    def apply_decision(self, action: int) -> Dict[str, int | float | bool]:
        reward = ctypes.c_float(0.0)
        lines = ctypes.c_int(0)
        game_over = ctypes.c_int(0)
        used_hold = ctypes.c_int(0)
        placement_index = ctypes.c_size_t(0)
        ok = self.lib.tetris_cc_env_apply_decision(
            self.handle,
            int(action),
            ctypes.byref(reward),
            ctypes.byref(lines),
            ctypes.byref(game_over),
            ctypes.byref(used_hold),
            ctypes.byref(placement_index),
        )
        out = {
            "success": bool(ok),
            "reward": float(reward.value),
            "lines": int(lines.value),
            "game_over": bool(game_over.value),
            "used_hold": bool(used_hold.value),
            "placement_index": int(placement_index.value),
        }
        if out["success"] and self.bot_handle:
            self.bot_sync()
        return out

    def placement_board(self, index: int):
        buf = (ctypes.c_uint8 * (BOARD_ROWS * BOARD_COLS))()
        written = self.lib.tetris_cc_env_placement_board_write(self.handle, int(index), buf, len(buf))
        _require_abi(written == len(buf), f"write the board for placement {index}")
        flat = [int(v) for v in buf]
        return [flat[r * BOARD_COLS : (r + 1) * BOARD_COLS] for r in range(BOARD_ROWS)]

    def placement_piece_ids(self, index: int):
        buf = (ctypes.c_uint8 * (BOARD_ROWS * BOARD_COLS))()
        written = self.lib.tetris_cc_env_placement_board_piece_ids_write(self.handle, int(index), buf, len(buf))
        _require_abi(written == len(buf), f"write piece IDs for placement {index}")
        flat = [int(v) for v in buf]
        return [flat[r * BOARD_COLS : (r + 1) * BOARD_COLS] for r in range(BOARD_ROWS)]

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
        if out["success"] and self.bot_handle:
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
            _require_abi(bool(ok), f"read rotation trace item {i}")
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
        ok = self.lib.tetris_cc_env_rotation_trace_meta(
            self.handle,
            int(action),
            ctypes.byref(success),
            ctypes.byref(fx),
            ctypes.byref(fy),
            ctypes.byref(fr),
        )
        _require_abi(bool(ok), "read rotation trace metadata")
        return {
            "success": bool(success.value),
            "final_x": int(fx.value),
            "final_y": int(fy.value),
            "final_rotation": int(fr.value),
            "tests": tests,
        }


__all__ = [
    "EnvCtypes",
    "find_library",
    "BOARD_ROWS",
    "BOARD_COLS",
    "EMPTY_CELL_ID",
    "ACTION_NONE",
    "ACTION_LEFT",
    "ACTION_RIGHT",
    "ACTION_SOFT_DROP",
    "ACTION_HARD_DROP",
    "ACTION_CW",
    "ACTION_CCW",
    "ACTION_ROTATE_CW",
    "ACTION_ROTATE_CCW",
    "ACTION_ROTATE_180",
    "ACTION_HOLD",
    "PIECE_NAMES",
]
