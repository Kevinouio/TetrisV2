from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..bc.utils import BCEnvAdapter, NativeAction, canonical_action_tuple
from .features import AfterstateFeatures, compute_features_from_board


@dataclass(frozen=True)
class CandidateAfterstate:
    native_action: NativeAction
    action_tuple: tuple[int, int, int, int, int]
    features: AfterstateFeatures
    feature_vector: np.ndarray
    lines_removed: int
    y_pos: int


class DQNRefEnvBridge:
    BOARD_ROWS = 20
    BOARD_COLS = 10

    def __init__(self, lib_path: Path, seed: int):
        self.adapter = BCEnvAdapter(lib_path=lib_path, seed=seed)
        self.lib = self.adapter.lib
        self.handle = self.adapter.handle
        self._bind_extra()
        self._board_buf = (ctypes.c_uint8 * (self.BOARD_ROWS * self.BOARD_COLS))()
        self._candidate_feature_buf: Optional[ctypes.Array[ctypes.c_float]] = None
        self._candidate_feature_capacity = 0

    def _bind_extra(self) -> None:
        void_p = ctypes.c_void_p
        int_p = ctypes.POINTER(ctypes.c_int)
        size_t_p = ctypes.POINTER(ctypes.c_size_t)
        self.lib.tetris_cc_env_placement_board_write.argtypes = [
            void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        self.lib.tetris_cc_env_placement_board_write.restype = ctypes.c_size_t
        self.lib.tetris_cc_env_placement_get.argtypes = [void_p, ctypes.c_size_t, int_p, int_p, int_p, int_p]
        self.lib.tetris_cc_env_placement_get.restype = ctypes.c_int

        self._supports_candidate_batch = all(
            hasattr(self.lib, name)
            for name in (
                "tetris_cc_env_candidate_count",
                "tetris_cc_env_candidate_get",
                "tetris_cc_env_candidate_features_write",
            )
        )
        self._has_candidate_batch = bool(self._supports_candidate_batch)
        if self._supports_candidate_batch:
            self.lib.tetris_cc_env_candidate_count.argtypes = [void_p]
            self.lib.tetris_cc_env_candidate_count.restype = ctypes.c_size_t
            self.lib.tetris_cc_env_candidate_get.argtypes = [
                void_p,
                ctypes.c_size_t,
                int_p,
                size_t_p,
                int_p,
                int_p,
                int_p,
                int_p,
                int_p,
            ]
            self.lib.tetris_cc_env_candidate_get.restype = ctypes.c_int
            self.lib.tetris_cc_env_candidate_features_write.argtypes = [
                void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
            ]
            self.lib.tetris_cc_env_candidate_features_write.restype = ctypes.c_size_t

    def close(self) -> None:
        self.adapter.close()

    def __enter__(self) -> "DQNRefEnvBridge":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def reset(self, seed: int) -> None:
        self.adapter.reset(seed)

    def state(self) -> Dict[str, object]:
        out = self.adapter.get_state()
        try:
            out["board_piece_ids"] = self.adapter.board_piece_ids(include_active=True)
        except Exception:
            # Keep state() resilient on older libs or unexpected C API read failures.
            pass
        return out

    def step(self, action: NativeAction) -> Dict[str, object]:
        return self.adapter.step_native_action(action)

    def _placement_board(self, placement_index: int) -> np.ndarray:
        written = self.lib.tetris_cc_env_placement_board_write(
            self.handle,
            ctypes.c_size_t(int(placement_index)),
            self._board_buf,
            ctypes.c_size_t(len(self._board_buf)),
        )
        if int(written) != len(self._board_buf):
            raise RuntimeError(f"Failed to read placement board for index={placement_index}.")
        flat = np.frombuffer(self._board_buf, dtype=np.uint8, count=len(self._board_buf))
        return flat.reshape(self.BOARD_ROWS, self.BOARD_COLS)

    def _ensure_feature_buffer(self, required_floats: int) -> None:
        if required_floats <= self._candidate_feature_capacity:
            return
        new_capacity = max(required_floats, max(64, self._candidate_feature_capacity * 2))
        self._candidate_feature_buf = (ctypes.c_float * new_capacity)()
        self._candidate_feature_capacity = int(new_capacity)

    def _enumerate_candidates_batch(self) -> List[CandidateAfterstate]:
        count = int(self.lib.tetris_cc_env_candidate_count(self.handle))
        if count <= 0:
            return []

        needed = int(count * 6)
        self._ensure_feature_buffer(needed)
        if self._candidate_feature_buf is None:
            return []

        written = self.lib.tetris_cc_env_candidate_features_write(
            self.handle,
            self._candidate_feature_buf,
            ctypes.c_size_t(needed),
        )
        if int(written) != needed:
            raise RuntimeError(
                f"Failed to read candidate feature matrix: wrote={int(written)} expected={needed}"
            )

        feature_flat = np.ctypeslib.as_array(self._candidate_feature_buf)[:needed]
        feature_mat = feature_flat.reshape(count, 6)

        out: List[CandidateAfterstate] = []
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

            feature_vec = np.asarray(feature_mat[idx], dtype=np.float32).copy()
            features = AfterstateFeatures(
                total_height=float(feature_vec[0]),
                bumpiness=float(feature_vec[1]),
                lines_removed=float(feature_vec[2]),
                holes=float(feature_vec[3]),
                y_pos=float(feature_vec[4]),
                pillar=float(feature_vec[5]),
            )
            use_hold_bool = bool(use_hold.value)
            native_action = NativeAction(
                use_hold=use_hold_bool,
                placement_index=int(placement_index.value),
            )
            action_tuple = canonical_action_tuple(
                use_hold=use_hold_bool,
                piece=int(piece.value),
                rotation=int(rotation.value),
                x=int(x.value),
                y=int(y.value),
            )
            out.append(
                CandidateAfterstate(
                    native_action=native_action,
                    action_tuple=action_tuple,
                    features=features,
                    feature_vector=feature_vec,
                    lines_removed=int(lines.value),
                    y_pos=int(y.value),
                )
            )
        return out

    def _enumerate_current_branch(self, use_hold: bool, piece_id: int) -> List[CandidateAfterstate]:
        count = int(self.lib.tetris_cc_env_placement_count(self.handle))
        out: List[CandidateAfterstate] = []
        for idx in range(count):
            x = ctypes.c_int(0)
            y = ctypes.c_int(0)
            rot = ctypes.c_int(0)
            lines = ctypes.c_int(0)
            ok = self.lib.tetris_cc_env_placement_get(
                self.handle,
                ctypes.c_size_t(idx),
                ctypes.byref(x),
                ctypes.byref(y),
                ctypes.byref(rot),
                ctypes.byref(lines),
            )
            if not ok:
                continue

            board_after = self._placement_board(idx)
            features = compute_features_from_board(
                board_after=board_after,
                y_pos=int(y.value),
                lines_removed=int(lines.value),
            )
            feature_vec = features.as_array()
            native_action = NativeAction(use_hold=bool(use_hold), placement_index=int(idx))
            action_tuple = canonical_action_tuple(
                use_hold=bool(use_hold),
                piece=int(piece_id),
                rotation=int(rot.value),
                x=int(x.value),
                y=int(y.value),
            )
            out.append(
                CandidateAfterstate(
                    native_action=native_action,
                    action_tuple=action_tuple,
                    features=features,
                    feature_vector=feature_vec,
                    lines_removed=int(lines.value),
                    y_pos=int(y.value),
                )
            )
        return out

    def enumerate_candidates(self) -> List[CandidateAfterstate]:
        if getattr(self, "_has_candidate_batch", False):
            return self._enumerate_candidates_batch()

        meta = self.adapter.meta()
        if bool(meta["game_over"]):
            return []

        snapshot = self.lib.tetris_cc_env_snapshot_create(self.handle)
        if not snapshot:
            raise RuntimeError("Failed to create env snapshot.")

        out: List[CandidateAfterstate] = []
        try:
            active = self.adapter.active_piece()
            if active is not None and 0 <= int(active["piece"]) <= 6:
                out.extend(
                    self._enumerate_current_branch(
                        use_hold=False,
                        piece_id=int(active["piece"]),
                    )
                )

            hold = self.adapter.hold_info()
            if bool(hold["hold_available"]):
                self.lib.tetris_cc_env_restore_snapshot(self.handle, snapshot)
                hold_reward = ctypes.c_float(0.0)
                hold_ok = self.lib.tetris_cc_env_hold(self.handle, ctypes.byref(hold_reward))
                if hold_ok:
                    hold_active = self.adapter.active_piece()
                    if hold_active is not None and 0 <= int(hold_active["piece"]) <= 6:
                        out.extend(
                            self._enumerate_current_branch(
                                use_hold=True,
                                piece_id=int(hold_active["piece"]),
                            )
                        )
                self.lib.tetris_cc_env_restore_snapshot(self.handle, snapshot)
        finally:
            self.lib.tetris_cc_snapshot_destroy(snapshot)
            self.adapter.bot_sync()

        return out
