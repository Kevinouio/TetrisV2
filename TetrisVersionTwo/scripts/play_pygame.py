import argparse
import ctypes
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pygame


BOARD_ROWS = 20
BOARD_COLS = 10
EMPTY_CELL_ID = 255

ACTION_CW = 5
ACTION_CCW = 6
ACTION_180 = 7
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


def parse_args():
    parser = argparse.ArgumentParser(description="Pygame Placement + Kick Explorer via ctypes.")
    parser.add_argument("--lib", type=Path, default=None, help="Path to tetris_v2_c_api shared library.")
    parser.add_argument("--cell", type=int, default=28, help="Main board cell size.")
    parser.add_argument("--fps", type=int, default=60, help="Render FPS.")
    parser.add_argument("--seed", type=int, default=1234, help="Initial reset seed.")
    parser.add_argument("--queue-visible", type=int, default=8, help="How many queued pieces to display.")
    parser.add_argument("--ai", action="store_true", help="Start with AI autoplay enabled.")
    parser.add_argument("--think-ms", type=int, default=20, help="AI think budget per move in milliseconds.")
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
    return parser.parse_args()


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

        self.lib.tetris_cc_env_observation_size.argtypes = [void_p, ctypes.c_int]
        self.lib.tetris_cc_env_observation_size.restype = ctypes.c_size_t

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


def draw_board(surface, x0, y0, cell, board_piece_ids):
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            rect = pygame.Rect(x0 + c * cell, y0 + r * cell, cell, cell)
            piece_id = board_piece_ids[r][c]
            if piece_id != EMPTY_CELL_ID:
                pygame.draw.rect(surface, PIECE_COLORS.get(piece_id, BOARD_FILL), rect)
            pygame.draw.rect(surface, GRID_LINE, rect, width=1)

def draw_small_board(surface, x0, y0, cell, board_piece_ids):
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            rect = pygame.Rect(x0 + c * cell, y0 + r * cell, cell, cell)
            piece_id = board_piece_ids[r][c]
            if piece_id != EMPTY_CELL_ID:
                pygame.draw.rect(surface, PIECE_COLORS.get(piece_id, BOARD_FILL), rect)
            pygame.draw.rect(surface, (55, 60, 72), rect, width=1)


def action_name(action: int):
    return {ACTION_CW: "CW", ACTION_CCW: "CCW", ACTION_180: "180"}.get(action, "?")


def main():
    args = parse_args()
    try:
        lib_path = find_library(args.lib)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

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
    selected_index = 0
    list_scroll = 0
    inspector_actions = [ACTION_CW, ACTION_CCW]
    inspector_idx = 0
    status = f"Loaded {lib_path.name}"
    seed = int(args.seed)
    ai_backend = "cold_clear"
    ai_backend_label = "ColdClear"
    bc_agent = None

    if args.bc_checkpoint is not None:
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

    ai_available = bc_agent is not None if ai_backend == "bc" else env.has_bot()
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
        "start_ticks": pygame.time.get_ticks(),
    }
    if args.ai and not ai_available:
        if ai_backend == "bc":
            status = "AI[BC] requested, but BC checkpoint failed to initialize."
        else:
            status = "AI requested, but bot API symbols were not found in shared library."
    elif ai_enabled:
        if ai_backend == "bc":
            status = f"AI[BC] enabled at startup (device={args.bc_device or 'auto'})"
        else:
            status = f"AI[ColdClear] enabled at startup (think={max(1, int(args.think_ms))}ms)"

    info_h = 240
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
                                if ai_backend == "cold_clear":
                                    env.bot_sync()
                                    status = (
                                        f"AI[ColdClear] enabled (think={max(1, int(args.think_ms))}ms)"
                                    )
                                else:
                                    status = f"AI[BC] enabled (device={args.bc_device or 'auto'})"
                            else:
                                status = "AI disabled."
                        else:
                            if ai_backend == "bc":
                                status = "AI unavailable: BC checkpoint is not available."
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
                        env.reset(seed)
                        selected_index = 0
                        list_scroll = 0
                        status = f"Reset seed={seed}"
                    elif event.key == pygame.K_n:
                        seed = random.randint(1, 2**31 - 1)
                        env.reset(seed)
                        selected_index = 0
                        list_scroll = 0
                        status = f"Reset new seed={seed}"
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
                        env.reset(seed)
                        status = f"AI[{ai_backend_label}] auto-reset to seed={seed}"
                    else:
                        ai_enabled = False
                        status = "AI paused on topout. Press R/N or toggle AI with A."
                else:
                    if ai_backend == "bc":
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
                                            env.reset(seed)
                                            status = f"AI[BC] topout -> auto-reset seed={seed}"
                                        else:
                                            ai_enabled = False
                                            status = "AI[BC] topout. Autoplay stopped (auto-reset disabled)."
                                elif ai_result is not None:
                                    ai_enabled = False
                                    status = "AI[BC] action apply failed. Autoplay disabled."
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
                                f"AI[ColdClear] move idx={ai_result['placement_index']} "
                                f"hold={ai_result['used_hold']} score={ai_result['score']:.2f} "
                                f"lines+={ai_result['lines']}"
                            )
                            if ai_result["game_over"]:
                                ai_metrics["topouts"] += 1
                                if args.auto_reset:
                                    seed += 1
                                    env.reset(seed)
                                    status = f"AI[ColdClear] topout -> auto-reset seed={seed}"
                                else:
                                    ai_enabled = False
                                    status = "AI topout. Autoplay stopped (auto-reset disabled)."
                        else:
                            ai_enabled = False
                            status = "AI choose/apply failed. Autoplay disabled."

            board_piece_ids = env.board_piece_ids(include_active=True)
            hold = env.hold_info()
            queue = env.queue()
            meta = env.meta()
            placements = env.placements()

            if placements:
                selected_index = max(0, min(selected_index, len(placements) - 1))
                max_start = max(0, len(placements) - max_rows)
                list_scroll = max(0, min(max_start, list_scroll))
                preview_piece_ids = env.placement_piece_ids(selected_index)
            else:
                selected_index = 0
                list_scroll = 0
                preview_piece_ids = [[EMPTY_CELL_ID for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]

            inspect_action = inspector_actions[inspector_idx]
            trace = env.rotation_trace(inspect_action)

            screen.fill(BG_COLOR)

            pygame.draw.rect(screen, PANEL_COLOR, (board_x - 4, board_y - 4, board_w + 8, board_h + 8), border_radius=6)
            draw_board(screen, board_x, board_y, cell, board_piece_ids)

            # Top-right info panel
            info_y = board_y
            pygame.draw.rect(screen, PANEL_COLOR, (right_x, info_y, right_w, info_h), border_radius=8)
            ai_elapsed_s = max(1e-6, (pygame.time.get_ticks() - ai_metrics["start_ticks"]) / 1000.0)
            ai_pps = ai_metrics["pieces"] / ai_elapsed_s
            ai_backend_info = ai_backend_label
            if ai_backend == "bc":
                ai_backend_info = f"BC ({args.bc_device or 'auto'})"
            lines = [
                f"Seed: {seed}",
                f"Obs size: {env.observation_size()}",
                f"Hold: {PIECE_NAMES.get(hold['hold_piece'], '?')}  avail={hold['hold_available']}",
                f"Queue: {' '.join(PIECE_NAMES.get(p, '?') for p in queue[:max(0, args.queue_visible)])}",
                f"GameOver={meta['game_over']} TopOut={meta['top_out']}",
                f"Combo={meta['combo']} B2B={meta['b2b']} Lines={meta['lines']}",
                f"LockTimer={meta['lock_timer']} Resets={meta['lock_resets']}",
                f"AI: {'ON' if ai_enabled else 'OFF'} backend={ai_backend_info} avail={ai_available}",
                f"AI pieces={ai_metrics['pieces']} lines={ai_metrics['lines']} topouts={ai_metrics['topouts']}",
                f"AI PPS={ai_pps:.2f} step_ms(last/avg)={ai_metrics['last_step_ms']:.1f}/{ai_metrics['avg_step_ms']:.1f}",
            ]
            if ai_backend == "cold_clear":
                lines.append(
                    f"AI nodes={ai_metrics['last_nodes']} nps={ai_metrics['last_nps']:.0f} score={ai_metrics['last_score']:.2f}"
                )
                lines.append(
                    f"AI budget_miss last/total={ai_metrics['last_budget_miss']}/{ai_metrics['budget_misses']}"
                )
            else:
                lines.append(
                    f"AI[BC] invalid_raw={ai_metrics['invalid_unmasked_predictions']} "
                    f"fallbacks={ai_metrics['unseen_legal_fallbacks']}"
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
            draw_small_board(screen, right_x + 10, preview_y + 30, preview_cell, preview_piece_ids)

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
