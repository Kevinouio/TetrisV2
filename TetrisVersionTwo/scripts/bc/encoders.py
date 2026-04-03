from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from .config import EncoderConfig

NUM_PIECES = 7
HOLD_CLASSES = 8  # 7 pieces + empty hold


def _piece_one_hot(piece: int, classes: int = NUM_PIECES) -> np.ndarray:
    out = np.zeros((classes,), dtype=np.float32)
    if 0 <= piece < classes:
        out[piece] = 1.0
    return out


def _hold_one_hot(hold_piece: int) -> np.ndarray:
    out = np.zeros((HOLD_CLASSES,), dtype=np.float32)
    if 0 <= hold_piece < NUM_PIECES:
        out[hold_piece] = 1.0
    else:
        out[NUM_PIECES] = 1.0
    return out


def _queue_one_hot(queue: Iterable[int], queue_length: int) -> np.ndarray:
    out = np.zeros((queue_length, NUM_PIECES), dtype=np.float32)
    for idx, piece in enumerate(list(queue)[:queue_length]):
        if 0 <= piece < NUM_PIECES:
            out[idx, piece] = 1.0
    return out


def encode_state(raw_state: Dict[str, object], config: EncoderConfig) -> Dict[str, np.ndarray]:
    board = np.asarray(raw_state["board"], dtype=np.uint8)
    if board.shape != (config.board_height, config.board_width):
        raise ValueError(
            "Unexpected board shape. "
            f"Expected {(config.board_height, config.board_width)}, got {tuple(board.shape)}."
        )
    board = (board > 0).astype(np.float32)[None, :, :]

    current_piece = int(raw_state.get("current_piece", -1))
    hold_piece = int(raw_state.get("hold_piece", NUM_PIECES))
    queue = raw_state.get("queue", [])
    if not isinstance(queue, (list, tuple)):
        queue = []

    piece = _piece_one_hot(current_piece, classes=NUM_PIECES)
    hold = _hold_one_hot(hold_piece)
    queue_oh = _queue_one_hot(queue, config.queue_length)

    if config.include_scalars:
        scalars = np.asarray(
            [
                float(raw_state.get("lines", 0)),
                float(raw_state.get("combo", -1)),
                float(1.0 if raw_state.get("b2b", False) else 0.0),
            ],
            dtype=np.float32,
        )
    else:
        scalars = np.zeros((0,), dtype=np.float32)

    return {
        "board": board,
        "piece": piece,
        "hold": hold,
        "queue": queue_oh,
        "scalars": scalars,
    }


def flatten_aux_features(encoded_state: Dict[str, np.ndarray]) -> np.ndarray:
    chunks = [
        encoded_state["piece"].reshape(-1),
        encoded_state["hold"].reshape(-1),
        encoded_state["queue"].reshape(-1),
    ]
    scalars = encoded_state.get("scalars")
    if scalars is not None and scalars.size > 0:
        chunks.append(scalars.reshape(-1))
    return np.concatenate(chunks, axis=0).astype(np.float32)

