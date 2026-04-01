from __future__ import annotations

import ctypes
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ActionTuple = Tuple[int, int, int, int, int]  # (use_hold, piece, rotation, x, y)


@dataclass(frozen=True)
class NativeAction:
    use_hold: bool
    placement_index: int

    def key(self) -> Tuple[int, int]:
        return (int(self.use_hold), int(self.placement_index))


def canonical_action_tuple(
    use_hold: bool,
    piece: int,
    rotation: int,
    x: int,
    y: int,
) -> ActionTuple:
    return (int(bool(use_hold)), int(piece), int(rotation), int(x), int(y))


class ActionCodec:
    def __init__(self, id_to_action: Optional[Iterable[Sequence[int]]] = None):
        self.action_to_id: Dict[ActionTuple, int] = {}
        self.id_to_action: List[ActionTuple] = []
        if id_to_action is not None:
            for action in id_to_action:
                tup = tuple(int(v) for v in action)
                if len(tup) != 5:
                    raise ValueError(f"Action tuple must have length 5, got {tup}.")
                self.add_tuple(tup)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self.id_to_action)

    def add_tuple(self, action_tuple: ActionTuple) -> int:
        existing = self.action_to_id.get(action_tuple)
        if existing is not None:
            return existing
        idx = len(self.id_to_action)
        self.id_to_action.append(action_tuple)
        self.action_to_id[action_tuple] = idx
        return idx

    def encode_tuple(self, action_tuple: ActionTuple, add_if_missing: bool = False) -> int:
        action_id = self.action_to_id.get(action_tuple)
        if action_id is not None:
            return action_id
        if not add_if_missing:
            raise KeyError(f"Action tuple not in codec: {action_tuple}")
        return self.add_tuple(action_tuple)

    def decode_id(self, action_id: int) -> ActionTuple:
        return self.id_to_action[int(action_id)]

    def legal_ids(self, legal_tuples: Iterable[ActionTuple]) -> Tuple[List[int], int]:
        out: List[int] = []
        unseen = 0
        for tup in legal_tuples:
            action_id = self.action_to_id.get(tup)
            if action_id is None:
                unseen += 1
                continue
            out.append(action_id)
        return out, unseen

    def legal_mask(self, legal_tuples: Iterable[ActionTuple]) -> np.ndarray:
        mask = np.zeros((len(self.id_to_action),), dtype=np.bool_)
        for action_id in self.legal_ids(legal_tuples)[0]:
            mask[action_id] = True
        return mask


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def split_episode_ids(
    episode_ids: Sequence[int],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, List[int]]:
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total:.6f}.")
    ids = sorted(set(int(ep) for ep in episode_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    n_test = n - n_train - n_val
    if n > 0 and n_test < 0:
        raise ValueError("Invalid split counts.")
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val : n_train + n_val + n_test]
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def chunk_list(values: Sequence[int], chunk_size: int) -> List[List[int]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    return [list(values[i : i + chunk_size]) for i in range(0, len(values), chunk_size)]


def find_library(explicit_path: Optional[Path] = None) -> Path:
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
        Path("build-wsl/TetrisVersionTwo/libtetris_v2_c_api.so"),
        Path("TetrisVersionTwo/build/tetris_v2_c_api.dll"),
        Path("TetrisVersionTwo/build/Debug/tetris_v2_c_api.dll"),
        Path("TetrisVersionTwo/build/Release/tetris_v2_c_api.dll"),
        Path("TetrisVersionTwo/build/libtetris_v2_c_api.so"),
        Path("TetrisVersionTwo/build/libtetris_v2_c_api.dylib"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate tetris_v2_c_api shared library. "
        "Build CMake targets first or pass --lib explicitly."
    )


class BCEnvAdapter:
    BOARD_ROWS = 20
    BOARD_COLS = 10

    def __init__(self, lib_path: Path, seed: int):
        self.lib = ctypes.CDLL(str(lib_path))
        self._bind()
        self.handle = self.lib.tetris_cc_env_create(ctypes.c_uint32(seed))
        if not self.handle:
            raise RuntimeError("Failed to create env handle.")

        self.bot_handle = None
        if self._has_bot_api:
            self.bot_handle = self.lib.tetris_cc_bot_create_default()
            if not self.bot_handle:
                raise RuntimeError("Failed to create bot handle.")
            self.bot_sync()
        self.seed = int(seed)

    def _bind(self) -> None:
        void_p = ctypes.c_void_p
        int_p = ctypes.POINTER(ctypes.c_int)
        size_t_p = ctypes.POINTER(ctypes.c_size_t)

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

        self.lib.tetris_cc_env_hold.argtypes = [void_p, ctypes.POINTER(ctypes.c_float)]
        self.lib.tetris_cc_env_hold.restype = ctypes.c_int
        self.lib.tetris_cc_env_board_write.argtypes = [
            void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        self.lib.tetris_cc_env_board_write.restype = ctypes.c_size_t
        self.lib.tetris_cc_env_active_piece.argtypes = [void_p, int_p, int_p, int_p, int_p]
        self.lib.tetris_cc_env_active_piece.restype = ctypes.c_int
        self.lib.tetris_cc_env_hold_piece.argtypes = [void_p, int_p, int_p, int_p]
        self.lib.tetris_cc_env_hold_piece.restype = ctypes.c_int
        self.lib.tetris_cc_env_queue_count.argtypes = [void_p]
        self.lib.tetris_cc_env_queue_count.restype = ctypes.c_size_t
        self.lib.tetris_cc_env_queue_get.argtypes = [void_p, ctypes.c_size_t, int_p]
        self.lib.tetris_cc_env_queue_get.restype = ctypes.c_int
        self.lib.tetris_cc_env_meta.argtypes = [void_p, int_p, int_p, int_p, int_p, int_p, int_p, int_p]
        self.lib.tetris_cc_env_meta.restype = ctypes.c_int

        self.lib.tetris_cc_env_placement_count.argtypes = [void_p]
        self.lib.tetris_cc_env_placement_count.restype = ctypes.c_size_t
        self.lib.tetris_cc_env_placement_get.argtypes = [void_p, ctypes.c_size_t, int_p, int_p, int_p, int_p]
        self.lib.tetris_cc_env_placement_get.restype = ctypes.c_int
        self.lib.tetris_cc_env_apply_placement_index.argtypes = [
            void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            int_p,
            int_p,
        ]
        self.lib.tetris_cc_env_apply_placement_index.restype = ctypes.c_int

        self._has_bot_api = all(
            hasattr(self.lib, name)
            for name in (
                "tetris_cc_bot_create_default",
                "tetris_cc_bot_destroy",
                "tetris_cc_bot_sync_from_env",
                "tetris_cc_bot_choose_and_apply",
            )
        )
        self._has_bot_choose_api = self._has_bot_api and hasattr(
            self.lib, "tetris_cc_bot_choose"
        )
        self._has_bot_choose_api_ex = self._has_bot_api and hasattr(
            self.lib, "tetris_cc_bot_choose_ex"
        )
        self._has_bot_api_ex = self._has_bot_api and hasattr(
            self.lib, "tetris_cc_bot_choose_and_apply_ex"
        )
        if self._has_bot_api:
            self.lib.tetris_cc_bot_create_default.argtypes = []
            self.lib.tetris_cc_bot_create_default.restype = void_p
            self.lib.tetris_cc_bot_destroy.argtypes = [void_p]
            self.lib.tetris_cc_bot_destroy.restype = None
            self.lib.tetris_cc_bot_sync_from_env.argtypes = [void_p, void_p]
            self.lib.tetris_cc_bot_sync_from_env.restype = ctypes.c_int

            choose_only_args = [
                void_p,
                ctypes.c_int,
                int_p,
                size_t_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                int_p,
            ]
            if self._has_bot_choose_api:
                self.lib.tetris_cc_bot_choose.argtypes = choose_only_args
                self.lib.tetris_cc_bot_choose.restype = ctypes.c_int
            if self._has_bot_choose_api_ex:
                self.lib.tetris_cc_bot_choose_ex.argtypes = choose_only_args
                self.lib.tetris_cc_bot_choose_ex.restype = ctypes.c_int

            choose_args = [
                void_p,
                void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                int_p,
                int_p,
                int_p,
                size_t_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                int_p,
            ]
            self.lib.tetris_cc_bot_choose_and_apply.argtypes = choose_args
            self.lib.tetris_cc_bot_choose_and_apply.restype = ctypes.c_int
            if self._has_bot_api_ex:
                self.lib.tetris_cc_bot_choose_and_apply_ex.argtypes = choose_args
                self.lib.tetris_cc_bot_choose_and_apply_ex.restype = ctypes.c_int

    def close(self) -> None:
        if self.bot_handle is not None and self._has_bot_api:
            self.lib.tetris_cc_bot_destroy(self.bot_handle)
            self.bot_handle = None
        if getattr(self, "handle", None):
            self.lib.tetris_cc_env_destroy(self.handle)
            self.handle = None

    def __enter__(self) -> "BCEnvAdapter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        self.lib.tetris_cc_env_reset(self.handle, ctypes.c_uint32(seed))
        self.bot_sync()

    def bot_sync(self) -> bool:
        if not self._has_bot_api or self.bot_handle is None:
            return False
        return bool(self.lib.tetris_cc_bot_sync_from_env(self.bot_handle, self.handle))

    def board_occupancy(self) -> np.ndarray:
        buf = (ctypes.c_uint8 * (self.BOARD_ROWS * self.BOARD_COLS))()
        written = self.lib.tetris_cc_env_board_write(self.handle, 0, buf, len(buf))
        if written != len(buf):
            raise RuntimeError("Failed to read board occupancy from C API.")
        flat = np.frombuffer(buf, dtype=np.uint8, count=len(buf))
        return flat.reshape(self.BOARD_ROWS, self.BOARD_COLS)

    def active_piece(self) -> Optional[Dict[str, int]]:
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
        return {
            "piece": int(piece.value),
            "rotation": int(rotation.value),
            "x": int(x.value),
            "y": int(y.value),
        }

    def hold_info(self) -> Dict[str, object]:
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

    def queue(self) -> List[int]:
        count = int(self.lib.tetris_cc_env_queue_count(self.handle))
        out: List[int] = []
        for idx in range(count):
            piece = ctypes.c_int(-1)
            ok = self.lib.tetris_cc_env_queue_get(self.handle, idx, ctypes.byref(piece))
            if ok:
                out.append(int(piece.value))
        return out

    def meta(self) -> Dict[str, object]:
        values = [ctypes.c_int(0) for _ in range(7)]
        ok = self.lib.tetris_cc_env_meta(
            self.handle,
            ctypes.byref(values[0]),
            ctypes.byref(values[1]),
            ctypes.byref(values[2]),
            ctypes.byref(values[3]),
            ctypes.byref(values[4]),
            ctypes.byref(values[5]),
            ctypes.byref(values[6]),
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
            "game_over": bool(values[0].value),
            "top_out": bool(values[1].value),
            "combo": int(values[2].value),
            "b2b": bool(values[3].value),
            "lines": int(values[4].value),
            "lock_timer": int(values[5].value),
            "lock_resets": int(values[6].value),
        }

    def get_state(self) -> Dict[str, object]:
        active = self.active_piece()
        hold = self.hold_info()
        meta = self.meta()
        board = self.board_occupancy()
        return {
            "board": board,
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

    def _enumerate_current_branch_actions(
        self,
        use_hold: bool,
        piece_id: int,
    ) -> List[Tuple[NativeAction, ActionTuple]]:
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
            native_action = NativeAction(use_hold=bool(use_hold), placement_index=int(idx))
            tuple_action = canonical_action_tuple(
                use_hold=use_hold,
                piece=piece_id,
                rotation=int(rot.value),
                x=int(x.value),
                y=int(y.value),
            )
            out.append((native_action, tuple_action))
        return out

    def enumerate_legal_actions(self) -> List[Tuple[NativeAction, ActionTuple]]:
        meta = self.meta()
        if bool(meta["game_over"]):
            return []

        snapshot = self.lib.tetris_cc_env_snapshot_create(self.handle)
        if not snapshot:
            raise RuntimeError("Failed to create env snapshot.")

        out: List[Tuple[NativeAction, ActionTuple]] = []
        try:
            active = self.active_piece()
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
                reward = ctypes.c_float(0.0)
                hold_ok = self.lib.tetris_cc_env_hold(self.handle, ctypes.byref(reward))
                if hold_ok:
                    hold_active = self.active_piece()
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

    def expert_choose(self, think_ms: int = 20) -> Dict[str, object]:
        if not self._has_bot_api or self.bot_handle is None:
            return {
                "success": False,
                "used_hold": False,
                "placement_index": -1,
                "score": 0.0,
                "nodes": 0,
                "think_ms": 0.0,
                "nps": 0.0,
                "budget_miss": 0,
                "via_snapshot_rollback": False,
            }

        if self._has_bot_choose_api or self._has_bot_choose_api_ex:
            used_hold = ctypes.c_int(0)
            placement_index = ctypes.c_size_t(0)
            score = ctypes.c_float(0.0)
            nodes = ctypes.c_uint64(0)
            think = ctypes.c_double(0.0)
            nps = ctypes.c_double(0.0)
            budget_miss = ctypes.c_int(0)
            fn = self.lib.tetris_cc_bot_choose_ex if self._has_bot_choose_api_ex else self.lib.tetris_cc_bot_choose
            ok = fn(
                self.bot_handle,
                int(max(1, think_ms)),
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
                "used_hold": bool(used_hold.value),
                "placement_index": int(placement_index.value),
                "score": float(score.value),
                "nodes": int(nodes.value),
                "think_ms": float(think.value),
                "nps": float(nps.value),
                "budget_miss": int(budget_miss.value),
                "via_snapshot_rollback": False,
            }

        snapshot = self.lib.tetris_cc_env_snapshot_create(self.handle)
        if not snapshot:
            return {
                "success": False,
                "used_hold": False,
                "placement_index": -1,
                "score": 0.0,
                "nodes": 0,
                "think_ms": 0.0,
                "nps": 0.0,
                "budget_miss": 0,
                "via_snapshot_rollback": True,
            }
        try:
            chosen = self.expert_choose_and_apply(think_ms=think_ms)
            restored = bool(self.lib.tetris_cc_env_restore_snapshot(self.handle, snapshot))
            self.bot_sync()
            return {
                "success": bool(chosen["success"]) and restored,
                "used_hold": bool(chosen["used_hold"]),
                "placement_index": int(chosen["placement_index"]),
                "score": float(chosen["score"]),
                "nodes": int(chosen["nodes"]),
                "think_ms": float(chosen["think_ms"]),
                "nps": float(chosen["nps"]),
                "budget_miss": int(chosen["budget_miss"]),
                "via_snapshot_rollback": True,
            }
        finally:
            self.lib.tetris_cc_snapshot_destroy(snapshot)

    def expert_choose_and_apply(self, think_ms: int = 20) -> Dict[str, object]:
        if not self._has_bot_api or self.bot_handle is None:
            return {
                "success": False,
                "used_hold": False,
                "placement_index": -1,
                "reward": 0.0,
                "lines": 0,
                "game_over": True,
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
        fn = (
            self.lib.tetris_cc_bot_choose_and_apply_ex
            if self._has_bot_api_ex
            else self.lib.tetris_cc_bot_choose_and_apply
        )
        ok = fn(
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
            "used_hold": bool(used_hold.value),
            "placement_index": int(placement_index.value),
            "reward": float(reward.value),
            "lines": int(lines.value),
            "game_over": bool(game_over.value),
            "score": float(score.value),
            "nodes": int(nodes.value),
            "think_ms": float(think.value),
            "nps": float(nps.value),
            "budget_miss": int(budget_miss.value),
        }

    def step_native_action(self, action: NativeAction) -> Dict[str, object]:
        used_hold = False
        total_reward = 0.0
        if action.use_hold:
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

