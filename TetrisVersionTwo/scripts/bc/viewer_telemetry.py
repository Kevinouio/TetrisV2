from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

BOARD_ROWS = 20
BOARD_COLS = 10
EMPTY_CELL_ID = 255


def _event_type_name(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("type", "")).strip()


def queue_put_best_effort(event_queue: Any, event: Dict[str, object]) -> bool:
    if event_queue is None:
        return False
    try:
        event_queue.put_nowait(event)
        return True
    except Exception:
        pass

    event_type = _event_type_name(event)
    if event_type == "step_snapshot":
        return False

    # If queue is full, preferentially evict stale step snapshots so terminal
    # and run-level events can still get through.
    drained: List[object] = []
    removed_step = False
    for _ in range(64):
        try:
            existing = event_queue.get_nowait()
        except Exception:
            break
        existing_type = _event_type_name(existing)
        if (not removed_step) and existing_type == "step_snapshot":
            removed_step = True
            continue
        drained.append(existing)

    if not removed_step:
        for existing in drained:
            try:
                event_queue.put_nowait(existing)
            except Exception:
                break
        return False

    pushed = False
    try:
        event_queue.put_nowait(event)
        pushed = True
    except Exception:
        pushed = False

    for existing in drained:
        try:
            event_queue.put_nowait(existing)
        except Exception:
            break
    return pushed


def board_for_event(board: object) -> List[int]:
    arr = np.asarray(board, dtype=np.uint8)
    if arr.size != BOARD_ROWS * BOARD_COLS:
        return [0] * (BOARD_ROWS * BOARD_COLS)
    return arr.reshape(-1).astype(np.uint8, copy=False).tolist()


def piece_ids_for_event(piece_ids: object) -> List[int]:
    arr = np.asarray(piece_ids, dtype=np.uint8)
    if arr.size != BOARD_ROWS * BOARD_COLS:
        return [EMPTY_CELL_ID] * (BOARD_ROWS * BOARD_COLS)
    return arr.reshape(-1).astype(np.uint8, copy=False).tolist()
