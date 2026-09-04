"""ctypes bindings for the deterministic ``tetris_cc_battle_*`` C ABI."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, Optional

import numpy as np

from tetris_v2.rl.battle.config import BattleRulesConfig
from tetris_v2.rl.runtime import BOARD_COLS, BOARD_ROWS, find_library


_PLAYER_COUNT = 2
_ATTACK_TABLE_SIZE = 5
_NATIVE_BATTLE_OBSERVATION_SIZE = 470


def _validate_native_observation_size(size: int) -> None:
    if int(size) != _NATIVE_BATTLE_OBSERVATION_SIZE:
        raise RuntimeError(
            "native battle observation schema mismatch: "
            f"expected {_NATIVE_BATTLE_OBSERVATION_SIZE}, got {int(size)}"
        )


class _BattleConfig(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_uint32),
        ("attack_table", ctypes.c_int * _ATTACK_TABLE_SIZE),
        ("garbage_delay", ctypes.c_int),
        ("max_joint_steps", ctypes.c_int),
        ("same_piece_sequence", ctypes.c_int),
    ]


class _BattlePlayerStepResult(ctypes.Structure):
    _fields_ = [
        ("action_succeeded", ctypes.c_int),
        ("used_hold", ctypes.c_int),
        ("placement_index", ctypes.c_size_t),
        ("reward", ctypes.c_float),
        ("lines_cleared", ctypes.c_int),
        ("attack_generated", ctypes.c_int),
        ("garbage_cancelled", ctypes.c_int),
        ("garbage_sent", ctypes.c_int),
        ("garbage_received", ctypes.c_int),
        ("garbage_applied", ctypes.c_int),
        ("incoming_garbage", ctypes.c_int),
        ("next_garbage_delay", ctypes.c_int),
        ("top_out", ctypes.c_int),
    ]


class _BattleStepResult(ctypes.Structure):
    _fields_ = [
        ("success", ctypes.c_int),
        ("terminated", ctypes.c_int),
        ("winner", ctypes.c_int),
        ("joint_step", ctypes.c_int),
        ("players", _BattlePlayerStepResult * _PLAYER_COUNT),
    ]


class _BattlePlayerStats(ctypes.Structure):
    _fields_ = [
        ("placements", ctypes.c_int),
        ("score", ctypes.c_float),
        ("lines_cleared", ctypes.c_int),
        ("attack_generated", ctypes.c_int),
        ("garbage_cancelled", ctypes.c_int),
        ("garbage_sent", ctypes.c_int),
        ("garbage_received", ctypes.c_int),
        ("garbage_applied", ctypes.c_int),
        ("top_outs", ctypes.c_int),
    ]


class _BattleMeta(ctypes.Structure):
    _fields_ = [
        ("joint_steps", ctypes.c_int),
        ("terminated", ctypes.c_int),
        ("winner", ctypes.c_int),
        ("pending_garbage", ctypes.c_int * _PLAYER_COUNT),
        ("next_garbage_delay", ctypes.c_int * _PLAYER_COUNT),
        ("players", _BattlePlayerStats * _PLAYER_COUNT),
    ]


def _winner(value: int) -> int | None:
    return int(value) if int(value) in (0, 1) else None


class BattleRuntime:
    """Owned native two-player match with stable placement decisions."""

    def __init__(
        self,
        *,
        lib_path: Optional[Path] = None,
        seed: int = 1,
        rules: BattleRulesConfig | None = None,
    ):
        self.rules = rules or BattleRulesConfig()
        self.seed = int(seed)
        self.lib_path = find_library(lib_path)
        self.lib = ctypes.CDLL(str(self.lib_path))
        self._bind()
        config = _BattleConfig()
        self.lib.tetris_cc_battle_config_default(ctypes.byref(config))
        config.seed = ctypes.c_uint32(self.seed)
        config.attack_table[:] = self.rules.attack_table
        config.garbage_delay = int(self.rules.garbage_delay)
        config.max_joint_steps = int(self.rules.max_steps)
        config.same_piece_sequence = 1 if self.rules.mirrored_piece_seeds else 0
        self.handle = self.lib.tetris_cc_battle_create(ctypes.byref(config))
        if not self.handle:
            raise RuntimeError("failed to create native battle environment")
        size = self.observation_size()
        try:
            _validate_native_observation_size(size)
        except RuntimeError:
            self.close()
            raise

    def _bind(self) -> None:
        void_p = ctypes.c_void_p
        size_t = ctypes.c_size_t
        size_p = ctypes.POINTER(size_t)
        int_p = ctypes.POINTER(ctypes.c_int)

        self.lib.tetris_cc_battle_config_default.argtypes = [ctypes.POINTER(_BattleConfig)]
        self.lib.tetris_cc_battle_config_default.restype = None
        self.lib.tetris_cc_battle_create.argtypes = [ctypes.POINTER(_BattleConfig)]
        self.lib.tetris_cc_battle_create.restype = void_p
        self.lib.tetris_cc_battle_destroy.argtypes = [void_p]
        self.lib.tetris_cc_battle_destroy.restype = None
        self.lib.tetris_cc_battle_reset.argtypes = [void_p, ctypes.c_uint32]
        self.lib.tetris_cc_battle_reset.restype = ctypes.c_int
        self.lib.tetris_cc_battle_action_dim.argtypes = []
        self.lib.tetris_cc_battle_action_dim.restype = size_t
        self.lib.tetris_cc_battle_observation_size.argtypes = [void_p]
        self.lib.tetris_cc_battle_observation_size.restype = size_t
        self.lib.tetris_cc_battle_observation_write.argtypes = [
            void_p,
            size_t,
            ctypes.POINTER(ctypes.c_float),
            size_t,
        ]
        self.lib.tetris_cc_battle_observation_write.restype = size_t
        self.lib.tetris_cc_battle_decision_mask_write.argtypes = [
            void_p,
            size_t,
            ctypes.POINTER(ctypes.c_uint8),
            size_t,
        ]
        self.lib.tetris_cc_battle_decision_mask_write.restype = size_t
        self.lib.tetris_cc_battle_step.argtypes = [
            void_p,
            size_t,
            size_t,
            ctypes.POINTER(_BattleStepResult),
        ]
        self.lib.tetris_cc_battle_step.restype = ctypes.c_int
        self.lib.tetris_cc_battle_meta_get.argtypes = [void_p, ctypes.POINTER(_BattleMeta)]
        self.lib.tetris_cc_battle_meta_get.restype = ctypes.c_int
        self.lib.tetris_cc_battle_board_write.argtypes = [
            void_p,
            size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            size_t,
        ]
        self.lib.tetris_cc_battle_board_write.restype = size_t
        self.lib.tetris_cc_battle_board_piece_ids_write.argtypes = [
            void_p,
            size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            size_t,
        ]
        self.lib.tetris_cc_battle_board_piece_ids_write.restype = size_t
        self.lib.tetris_cc_battle_enqueue_garbage.argtypes = [
            void_p,
            size_t,
            int_p,
            size_t,
            ctypes.c_int,
        ]
        self.lib.tetris_cc_battle_enqueue_garbage.restype = ctypes.c_int
        self.lib.tetris_cc_battle_bot_choose.argtypes = [
            void_p,
            size_t,
            ctypes.c_int,
            size_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            int_p,
        ]
        self.lib.tetris_cc_battle_bot_choose.restype = ctypes.c_int

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.tetris_cc_battle_destroy(self.handle)
            self.handle = None

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        if not self.lib.tetris_cc_battle_reset(self.handle, ctypes.c_uint32(self.seed)):
            raise RuntimeError("native battle reset failed")

    def action_dim(self) -> int:
        return int(self.lib.tetris_cc_battle_action_dim())

    def observation_size(self) -> int:
        return int(self.lib.tetris_cc_battle_observation_size(self.handle))

    def observation(self, player: int) -> np.ndarray:
        size = self.observation_size()
        buffer = (ctypes.c_float * size)()
        written = self.lib.tetris_cc_battle_observation_write(
            self.handle, int(player), buffer, size
        )
        if int(written) != size:
            raise RuntimeError(f"native battle observation write failed for player {player}")
        return np.frombuffer(buffer, dtype=np.float32).copy()

    def decision_mask(self, player: int) -> np.ndarray:
        size = self.action_dim()
        buffer = (ctypes.c_uint8 * size)()
        written = self.lib.tetris_cc_battle_decision_mask_write(
            self.handle, int(player), buffer, size
        )
        if int(written) != size:
            raise RuntimeError(f"native battle mask write failed for player {player}")
        return np.frombuffer(buffer, dtype=np.uint8).copy()

    def board(self, player: int, *, include_active: bool = False) -> np.ndarray:
        size = BOARD_ROWS * BOARD_COLS
        buffer = (ctypes.c_uint8 * size)()
        written = self.lib.tetris_cc_battle_board_write(
            self.handle, int(player), 1 if include_active else 0, buffer, size
        )
        if int(written) != size:
            raise RuntimeError(f"native battle board write failed for player {player}")
        return np.frombuffer(buffer, dtype=np.uint8).copy().reshape(BOARD_ROWS, BOARD_COLS)

    def board_piece_ids(self, player: int, *, include_active: bool = False) -> np.ndarray:
        size = BOARD_ROWS * BOARD_COLS
        buffer = (ctypes.c_uint8 * size)()
        written = self.lib.tetris_cc_battle_board_piece_ids_write(
            self.handle, int(player), 1 if include_active else 0, buffer, size
        )
        if int(written) != size:
            raise RuntimeError(f"native battle piece-ID write failed for player {player}")
        return np.frombuffer(buffer, dtype=np.uint8).copy().reshape(BOARD_ROWS, BOARD_COLS)

    @staticmethod
    def _cumulative_player(value: _BattlePlayerStats, pending: int, delay: int) -> dict[str, Any]:
        return {
            "placements": int(value.placements),
            "score": float(value.score),
            "lines_cleared": int(value.lines_cleared),
            "attack_generated": int(value.attack_generated),
            "garbage_cancelled": int(value.garbage_cancelled),
            "garbage_sent": int(value.garbage_sent),
            "garbage_received": int(value.garbage_received),
            "garbage_applied": int(value.garbage_applied),
            "top_outs": int(value.top_outs),
            "top_out": bool(value.top_outs),
            "incoming_garbage": int(pending),
            "next_garbage_delay": int(delay),
        }

    def meta(self) -> dict[str, Any]:
        value = _BattleMeta()
        if not self.lib.tetris_cc_battle_meta_get(self.handle, ctypes.byref(value)):
            raise RuntimeError("native battle metadata read failed")
        players = [
            self._cumulative_player(
                value.players[player],
                value.pending_garbage[player],
                value.next_garbage_delay[player],
            )
            for player in (0, 1)
        ]
        return {
            "joint_step": int(value.joint_steps),
            "terminated": bool(value.terminated),
            "winner": _winner(value.winner),
            "players": players,
        }

    def step(self, actions: tuple[int, int]) -> dict[str, Any]:
        value = _BattleStepResult()
        ok = self.lib.tetris_cc_battle_step(
            self.handle,
            int(actions[0]),
            int(actions[1]),
            ctypes.byref(value),
        )
        players = []
        for native in value.players:
            players.append(
                {
                    "action_succeeded": bool(native.action_succeeded),
                    "used_hold": bool(native.used_hold),
                    "placement_index": int(native.placement_index),
                    "raw_reward": float(native.reward),
                    "lines_cleared": int(native.lines_cleared),
                    "attack_generated": int(native.attack_generated),
                    "garbage_cancelled": int(native.garbage_cancelled),
                    "garbage_sent": int(native.garbage_sent),
                    "garbage_received": int(native.garbage_received),
                    "garbage_applied": int(native.garbage_applied),
                    "incoming_garbage": int(native.incoming_garbage),
                    "next_garbage_delay": int(native.next_garbage_delay),
                    "top_out": bool(native.top_out),
                }
            )
        return {
            "success": bool(ok and value.success),
            "terminated": bool(value.terminated),
            "winner": _winner(value.winner),
            "joint_step": int(value.joint_step),
            "players": players,
        }

    def enqueue_garbage(self, player: int, holes: list[int], delay: int = 0) -> bool:
        columns = [int(hole) for hole in holes]
        if columns:
            values = (ctypes.c_int * len(columns))(*columns)
            pointer = values
        else:
            pointer = None
        return bool(
            self.lib.tetris_cc_battle_enqueue_garbage(
                self.handle,
                int(player),
                pointer,
                len(columns),
                int(delay),
            )
        )

    def bot_choose(self, player: int, think_ms: int = 0) -> dict[str, Any]:
        """Choose without mutation; zero uses native deterministic fixed effort."""

        action = ctypes.c_size_t(0)
        score = ctypes.c_float(0.0)
        nodes = ctypes.c_uint64(0)
        elapsed = ctypes.c_double(0.0)
        nps = ctypes.c_double(0.0)
        budget_miss = ctypes.c_int(0)
        success = self.lib.tetris_cc_battle_bot_choose(
            self.handle,
            int(player),
            int(think_ms),
            ctypes.byref(action),
            ctypes.byref(score),
            ctypes.byref(nodes),
            ctypes.byref(elapsed),
            ctypes.byref(nps),
            ctypes.byref(budget_miss),
        )
        return {
            "success": bool(success),
            "action": int(action.value),
            "score": float(score.value),
            "nodes": int(nodes.value),
            "think_ms": float(elapsed.value),
            "nps": float(nps.value),
            "budget_miss": int(budget_miss.value),
        }


__all__ = ["BattleRuntime"]
