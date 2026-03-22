"""Shared ctypes runtime bindings for TetrisVersionTwo (tetris_cc_* API)."""

from __future__ import annotations

import ctypes
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
ACTION_HOLD = 8

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

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate tetris_v2_c_api shared library. Build it first.")


class EnvCtypes:
    """ctypes wrapper around the current VersionTwo `tetris_cc_*` C API."""

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

        self.lib.tetris_cc_env_create.argtypes = [ctypes.c_uint32]
        self.lib.tetris_cc_env_create.restype = void_p

        self.lib.tetris_cc_env_destroy.argtypes = [void_p]
        self.lib.tetris_cc_env_destroy.restype = None

        self.lib.tetris_cc_env_reset.argtypes = [void_p, ctypes.c_uint32]
        self.lib.tetris_cc_env_reset.restype = None

        self.lib.tetris_cc_env_step.argtypes = [void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        self.lib.tetris_cc_env_step.restype = ctypes.c_int

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

    def hold(self):
        reward = ctypes.c_float(0.0)
        success = self.lib.tetris_cc_env_hold(self.handle, ctypes.byref(reward))
        if success:
            self.bot_sync()
        return {"success": bool(success), "reward": float(reward.value)}

    def step_action(self, action: int) -> Dict[str, object]:
        if int(action) == ACTION_HOLD:
            hold_out = self.hold()
            meta = self.meta()
            return {
                "success": bool(hold_out["success"]),
                "reward": float(hold_out["reward"]),
                "game_over": bool(meta["game_over"]),
                "meta": meta,
            }
        reward = ctypes.c_float(0.0)
        game_over = self.lib.tetris_cc_env_step(self.handle, int(action), ctypes.byref(reward))
        meta = self.meta()
        return {
            "success": True,
            "reward": float(reward.value),
            "game_over": bool(game_over) or bool(meta["game_over"]),
            "meta": meta,
        }

    def has_bot(self):
        return bool(self._has_bot_api and self.bot_handle)

    def bot_sync(self):
        if not self.has_bot():
            return False
        ok = self.lib.tetris_cc_bot_sync_from_env(self.bot_handle, self.handle)
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

    def observation_size(self, include_hidden_rows: bool = True):
        return int(self.lib.tetris_cc_env_observation_size(self.handle, 1 if include_hidden_rows else 0))

    def observation(self, include_hidden_rows: bool = False) -> np.ndarray:
        size = self.observation_size(include_hidden_rows=include_hidden_rows)
        out = np.zeros((size,), dtype=np.float32)
        if size == 0:
            return out
        buf = (ctypes.c_float * size)()
        written = self.lib.tetris_cc_env_observation_write(
            self.handle,
            1 if include_hidden_rows else 0,
            buf,
            size,
        )
        if written != size:
            return out
        return np.frombuffer(buf, dtype=np.float32).copy()

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

    def placements(self):
        count = int(self.lib.tetris_cc_env_placement_count(self.handle))
        out: List[Dict[str, int]] = []
        for i in range(count):
            x = ctypes.c_int(0)
            y = ctypes.c_int(0)
            rot = ctypes.c_int(0)
            lines = ctypes.c_int(0)
            if self.lib.tetris_cc_env_placement_get(
                self.handle, i, ctypes.byref(x), ctypes.byref(y), ctypes.byref(rot), ctypes.byref(lines)
            ):
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
    "ACTION_HOLD",
    "PIECE_NAMES",
]
