from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..bc.utils import BCEnvAdapter, NativeAction, canonical_action_tuple
from .features import AfterstateFeatures, compute_features_from_board


class _CandidateRow(ctypes.Structure):
    _fields_ = [
        ("use_hold", ctypes.c_int),
        ("placement_index", ctypes.c_size_t),
        ("piece", ctypes.c_int),
        ("rotation", ctypes.c_int),
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("lines_cleared", ctypes.c_int),
        ("features", ctypes.c_float * 6),
    ]


@dataclass(frozen=True)
class CandidateAfterstate:
    native_action: NativeAction
    action_tuple: tuple[int, int, int, int, int]
    features: AfterstateFeatures
    feature_vector: np.ndarray
    lines_removed: int
    y_pos: int


@dataclass(frozen=True)
class CandidateBatch:
    """Compact candidate representation for fast action selection."""

    feature_matrix: np.ndarray
    use_hold: np.ndarray
    placement_index: np.ndarray
    piece: np.ndarray
    rotation: np.ndarray
    x: np.ndarray
    y: np.ndarray
    lines_removed: np.ndarray

    @property
    def count(self) -> int:
        return int(self.feature_matrix.shape[0])

    def native_action_at(self, index: int) -> NativeAction:
        idx = int(index)
        return NativeAction(
            use_hold=bool(self.use_hold[idx]),
            placement_index=int(self.placement_index[idx]),
        )

    def action_tuple_at(self, index: int) -> tuple[int, int, int, int, int]:
        idx = int(index)
        return canonical_action_tuple(
            use_hold=bool(self.use_hold[idx]),
            piece=int(self.piece[idx]),
            rotation=int(self.rotation[idx]),
            x=int(self.x[idx]),
            y=int(self.y[idx]),
        )


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
        self._candidate_row_buf: Optional[ctypes.Array[_CandidateRow]] = None
        self._candidate_row_capacity = 0

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
        self._supports_candidate_rows = bool(
            self._supports_candidate_batch and hasattr(self.lib, "tetris_cc_env_candidate_rows_write")
        )

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

        if self._supports_candidate_rows:
            self.lib.tetris_cc_env_candidate_rows_write.argtypes = [
                void_p,
                ctypes.POINTER(_CandidateRow),
                ctypes.c_size_t,
            ]
            self.lib.tetris_cc_env_candidate_rows_write.restype = ctypes.c_size_t

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

    @staticmethod
    def _empty_candidate_batch() -> CandidateBatch:
        empty_f = np.zeros((0, 6), dtype=np.float32)
        empty_i = np.zeros((0,), dtype=np.int32)
        empty_u = np.zeros((0,), dtype=np.uint8)
        empty_idx = np.zeros((0,), dtype=np.int64)
        return CandidateBatch(
            feature_matrix=empty_f,
            use_hold=empty_u,
            placement_index=empty_idx,
            piece=empty_i,
            rotation=empty_i.copy(),
            x=empty_i.copy(),
            y=empty_i.copy(),
            lines_removed=empty_i.copy(),
        )

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

    def _ensure_row_buffer(self, required_rows: int) -> None:
        if required_rows <= self._candidate_row_capacity:
            return
        new_capacity = max(required_rows, max(32, self._candidate_row_capacity * 2))
        self._candidate_row_buf = (_CandidateRow * new_capacity)()
        self._candidate_row_capacity = int(new_capacity)

    def _candidate_batch_from_rows_api(self) -> CandidateBatch:
        count = int(self.lib.tetris_cc_env_candidate_count(self.handle))
        if count <= 0:
            return self._empty_candidate_batch()

        self._ensure_row_buffer(count)
        if self._candidate_row_buf is None:
            return self._empty_candidate_batch()

        written = int(
            self.lib.tetris_cc_env_candidate_rows_write(
                self.handle,
                self._candidate_row_buf,
                ctypes.c_size_t(count),
            )
        )
        if written != count:
            raise RuntimeError(f"Failed to read candidate rows: wrote={written} expected={count}")

        feature_mat = np.empty((count, 6), dtype=np.float32)
        use_hold = np.empty((count,), dtype=np.uint8)
        placement_index = np.empty((count,), dtype=np.int64)
        piece = np.empty((count,), dtype=np.int32)
        rotation = np.empty((count,), dtype=np.int32)
        x = np.empty((count,), dtype=np.int32)
        y = np.empty((count,), dtype=np.int32)
        lines = np.empty((count,), dtype=np.int32)

        for idx in range(count):
            row = self._candidate_row_buf[idx]
            use_hold[idx] = np.uint8(1 if row.use_hold else 0)
            placement_index[idx] = np.int64(int(row.placement_index))
            piece[idx] = np.int32(int(row.piece))
            rotation[idx] = np.int32(int(row.rotation))
            x[idx] = np.int32(int(row.x))
            y[idx] = np.int32(int(row.y))
            lines[idx] = np.int32(int(row.lines_cleared))
            feature_mat[idx, :] = np.ctypeslib.as_array(row.features, shape=(6,))

        return CandidateBatch(
            feature_matrix=feature_mat,
            use_hold=use_hold,
            placement_index=placement_index,
            piece=piece,
            rotation=rotation,
            x=x,
            y=y,
            lines_removed=lines,
        )

    def _candidate_batch_from_legacy_batch_api(self) -> CandidateBatch:
        count = int(self.lib.tetris_cc_env_candidate_count(self.handle))
        if count <= 0:
            return self._empty_candidate_batch()

        needed = int(count * 6)
        self._ensure_feature_buffer(needed)
        if self._candidate_feature_buf is None:
            return self._empty_candidate_batch()

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
        feature_mat = np.asarray(feature_flat.reshape(count, 6), dtype=np.float32).copy()

        use_hold = np.empty((count,), dtype=np.uint8)
        placement_index = np.empty((count,), dtype=np.int64)
        piece = np.empty((count,), dtype=np.int32)
        rotation = np.empty((count,), dtype=np.int32)
        x = np.empty((count,), dtype=np.int32)
        y = np.empty((count,), dtype=np.int32)
        lines = np.empty((count,), dtype=np.int32)

        for idx in range(count):
            use_hold_v = ctypes.c_int(0)
            placement_index_v = ctypes.c_size_t(0)
            piece_v = ctypes.c_int(-1)
            rotation_v = ctypes.c_int(0)
            x_v = ctypes.c_int(0)
            y_v = ctypes.c_int(0)
            lines_v = ctypes.c_int(0)
            ok = self.lib.tetris_cc_env_candidate_get(
                self.handle,
                ctypes.c_size_t(idx),
                ctypes.byref(use_hold_v),
                ctypes.byref(placement_index_v),
                ctypes.byref(piece_v),
                ctypes.byref(rotation_v),
                ctypes.byref(x_v),
                ctypes.byref(y_v),
                ctypes.byref(lines_v),
            )
            if not ok:
                raise RuntimeError(f"Failed to read candidate metadata for index={idx}")
            use_hold[idx] = np.uint8(1 if use_hold_v.value else 0)
            placement_index[idx] = np.int64(int(placement_index_v.value))
            piece[idx] = np.int32(int(piece_v.value))
            rotation[idx] = np.int32(int(rotation_v.value))
            x[idx] = np.int32(int(x_v.value))
            y[idx] = np.int32(int(y_v.value))
            lines[idx] = np.int32(int(lines_v.value))

        return CandidateBatch(
            feature_matrix=feature_mat,
            use_hold=use_hold,
            placement_index=placement_index,
            piece=piece,
            rotation=rotation,
            x=x,
            y=y,
            lines_removed=lines,
        )

    def enumerate_candidate_batch(self) -> Optional[CandidateBatch]:
        if not getattr(self, "_has_candidate_batch", False):
            return None
        if bool(getattr(self, "_supports_candidate_rows", False)):
            return self._candidate_batch_from_rows_api()
        return self._candidate_batch_from_legacy_batch_api()

    @staticmethod
    def _materialize_batch(batch: CandidateBatch) -> List[CandidateAfterstate]:
        out: List[CandidateAfterstate] = []
        for idx in range(batch.count):
            feature_vec = np.asarray(batch.feature_matrix[idx], dtype=np.float32).copy()
            features = AfterstateFeatures(
                total_height=float(feature_vec[0]),
                bumpiness=float(feature_vec[1]),
                lines_removed=float(feature_vec[2]),
                holes=float(feature_vec[3]),
                y_pos=float(feature_vec[4]),
                pillar=float(feature_vec[5]),
            )
            out.append(
                CandidateAfterstate(
                    native_action=batch.native_action_at(idx),
                    action_tuple=batch.action_tuple_at(idx),
                    features=features,
                    feature_vector=feature_vec,
                    lines_removed=int(batch.lines_removed[idx]),
                    y_pos=int(batch.y[idx]),
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
        batch = self.enumerate_candidate_batch()
        if batch is not None:
            return self._materialize_batch(batch)

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
