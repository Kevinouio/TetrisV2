"""Utilities for defining and applying custom board presets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from . import utils

BoardArray = np.ndarray


@dataclass
class PieceSpec:
    piece: str
    rotation: int = 0
    row: int = utils.NES_SPAWN_ROW
    col: int = utils.NES_SPAWN_COL


@dataclass
class BoardPreset:
    name: str
    env: str
    board: List[str] = field(default_factory=list)
    current: Optional[PieceSpec] = None
    queue: List[str] = field(default_factory=list)
    hold: Optional[str] = None
    hold_available: Optional[bool] = None
    meta: Dict[str, Any] = field(default_factory=dict)


BoardPresetLibrary = Dict[str, BoardPreset]


def load_board_presets(path: Path | str) -> BoardPresetLibrary:
    """Load a JSON file describing preset board states."""
    file_path = Path(path)
    try:
        data = json.loads(file_path.read_text())
    except OSError as exc:
        raise ValueError(f"Unable to read board preset file '{file_path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in board preset file '{file_path}': {exc}") from exc
    raw_presets = data.get("presets", data)
    library: BoardPresetLibrary = {}
    for name, spec in raw_presets.items():
        env = str(spec.get("env") or spec.get("environment") or "").strip().lower()
        if env not in {"nes", "modern"}:
            raise ValueError(f"Preset '{name}' must specify env as 'nes' or 'modern'.")
        board_lines = _normalise_board_lines(spec.get("board"))
        preset = BoardPreset(
            name=name,
            env=env,
            board=board_lines,
            current=_parse_piece_spec(spec.get("current")),
            queue=_normalise_piece_list(spec.get("queue")),
            hold=_normalise_hold(spec.get("hold")),
            hold_available=spec.get("hold_available"),
            meta={
                key: value
                for key, value in spec.items()
                if key
                not in {"env", "environment", "board", "current", "queue", "hold", "hold_available"}
            },
        )
        library[name] = preset
    return library


def apply_board_preset(env, preset: BoardPreset):
    """Apply a preset to an environment and return the new observation."""
    env_type = preset.env.lower()
    if env_type == "nes":
        return _apply_nes_preset(env, preset)
    if env_type == "modern":
        return _apply_modern_preset(env, preset)
    raise ValueError(f"Unsupported preset env '{preset.env}'.")


def _normalise_board_lines(board_spec: Optional[Iterable[str]]) -> List[str]:
    if board_spec is None:
        return []
    if isinstance(board_spec, str):
        lines = [line.rstrip() for line in board_spec.splitlines() if line.strip()]
    else:
        lines = [str(line).rstrip() for line in board_spec if str(line).strip()]
    return lines


def _parse_piece_spec(raw: Optional[Dict[str, Any]]) -> Optional[PieceSpec]:
    if not raw:
        return None
    piece = raw.get("piece")
    if not piece:
        return None
    return PieceSpec(
        piece=str(piece),
        rotation=int(raw.get("rotation", 0)),
        row=int(raw.get("row", utils.NES_SPAWN_ROW)),
        col=int(raw.get("col", utils.NES_SPAWN_COL)),
    )


def _normalise_piece_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        tokens = [token.strip() for token in raw.replace(",", " ").split() if token.strip()]
        return tokens
    return [str(item).strip() for item in raw if str(item).strip()]


def _normalise_hold(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    value = str(raw).strip()
    return value if value else None


def _parse_board(board_lines: List[str]) -> BoardArray:
    if not board_lines:
        return utils.create_board()
    width = utils.BOARD_WIDTH
    arr = utils.create_board()
    if any(len(line) != width for line in board_lines):
        raise ValueError(f"Board rows must be length {width}.")
    start = utils.TOTAL_ROWS - len(board_lines)
    if start < 0:
        raise ValueError("Board specification has too many rows.")
    for idx, line in enumerate(board_lines):
        arr[start + idx] = [_cell_value(ch) for ch in line]
    return arr


def _cell_value(ch: str) -> int:
    ch = ch.strip()
    if not ch:
        return 0
    token = ch[0]
    if token in {".", "_"}:
        return 0
    if token == "#":
        return 8
    if token.isdigit():
        return int(token)
    token = token.upper()
    if token in utils.NAME_TO_ID:
        return utils.NAME_TO_ID[token] + 1
    raise ValueError(f"Unrecognised board cell token '{ch}'.")


def _piece_state_from_spec(spec: Optional[PieceSpec], default_row: int, default_col: int) -> Optional[utils.PieceState]:
    if spec is None:
        return None
    name = spec.piece.upper()
    if name not in utils.NAME_TO_ID:
        raise ValueError(f"Unknown piece '{spec.piece}'.")
    piece_id = utils.NAME_TO_ID[name]
    row = spec.row if spec.row is not None else default_row
    col = spec.col if spec.col is not None else default_col
    rotation = spec.rotation
    return utils.PieceState(piece_id=piece_id, rotation=rotation, row=row, col=col)


def _apply_nes_preset(env, preset: BoardPreset):
    from .nes_tetris_env import NesTetrisEnv

    base = env.unwrapped
    if not isinstance(base, NesTetrisEnv):
        raise ValueError(f"Preset '{preset.name}' requires the NES environment.")
    base._board = _parse_board(preset.board)
    base._current = _piece_state_from_spec(
        preset.current,
        default_row=utils.NES_SPAWN_ROW,
        default_col=utils.NES_SPAWN_COL,
    )
    if preset.queue:
        queue_ids = [_piece_name_to_id(piece) for piece in preset.queue]
        base._preview_queue = queue_ids[: base.preview_pieces]
    base._frames = 0
    base._steps = 0
    base._gravity_timer = 0
    base._top_out = False
    base._lines = int(preset.meta.get("lines", base._lines))
    base._level = int(preset.meta.get("level", base._level))
    base._score = int(preset.meta.get("score", base._score))
    base._lines_until_level_up = utils.lines_to_next_level(base._level)
    base._consecutive_rotations = 0
    return base._observe()


def _apply_modern_preset(env, preset: BoardPreset):
    from .modern_tetris_env import ModernTetrisEnv

    base = env.unwrapped
    if not isinstance(base, ModernTetrisEnv):
        raise ValueError(f"Preset '{preset.name}' requires the Modern environment.")
    rules = base._rules
    rules._board = _parse_board(preset.board)
    rules._current = _piece_state_from_spec(
        preset.current,
        default_row=utils.NES_SPAWN_ROW,
        default_col=utils.NES_SPAWN_COL,
    )
    if preset.queue:
        rules._queue.clear()
        for piece in preset.queue:
            rules._queue.append(_piece_name_to_id(piece))
    if preset.hold is not None:
        rules._hold_piece = _piece_name_to_id(preset.hold) if preset.hold else None
        rules._hold_available = bool(preset.hold_available if preset.hold_available is not None else True)
    pending = preset.meta.get("pending_garbage")
    if pending is not None:
        rules._pending_garbage.clear()
        for entry in pending:
            rules._pending_garbage.append(int(entry))
    rules._combo = int(preset.meta.get("combo", rules._combo))
    rules._back_to_back = bool(preset.meta.get("back_to_back", rules._back_to_back))
    rules._lines = int(preset.meta.get("lines", rules._lines))
    rules._level = int(preset.meta.get("level", rules._level))
    rules._score = int(preset.meta.get("score", rules._score))
    rules._steps = 0
    rules._top_out = False
    rules._line_clear_timer = 0
    rules._ground_frames = 0
    rules._gravity_progress = 0.0
    rules._last_action = "none"
    return rules.snapshot()


def _piece_name_to_id(piece: str) -> int:
    name = str(piece).strip().upper()
    if name not in utils.NAME_TO_ID:
        raise ValueError(f"Unknown piece '{piece}'.")
    return utils.NAME_TO_ID[name]
