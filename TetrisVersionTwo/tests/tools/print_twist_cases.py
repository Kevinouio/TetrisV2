#!/usr/bin/env python3
"""Print twist dataset cases with piece metadata and board views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VISIBLE_ROWS = 20
BOARD_WIDTH = 10
ROT_NAMES = ["N", "E", "S", "W"]
BOARD_KEYS = ("board_top_to_bottom", "board_rows_bottom", "board_rows_by_y", "filled_cells")

# Matches src/piece_defs.cpp base_cells()
BASE_CELLS: dict[str, list[tuple[int, int]]] = {
    "I": [(-1, 0), (0, 0), (1, 0), (2, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "T": [(-1, 0), (0, 0), (1, 0), (0, 1)],
    "L": [(-1, 0), (0, 0), (1, 0), (1, 1)],
    "J": [(-1, 0), (0, 0), (1, 0), (-1, 1)],
    "S": [(-1, 0), (0, 0), (0, 1), (1, 1)],
    "Z": [(-1, 1), (0, 1), (0, 0), (1, 0)],
}


def rotate_cell(rotation: int, x: int, y: int) -> tuple[int, int]:
    if rotation == 0:  # North
        return (x, y)
    if rotation == 1:  # East
        return (y, -x)
    if rotation == 2:  # South
        return (-x, -y)
    if rotation == 3:  # West
        return (-y, x)
    raise ValueError(f"Invalid rotation {rotation}")


def piece_cells(piece: str, rotation: int) -> list[tuple[int, int]]:
    base = BASE_CELLS[piece]
    return [rotate_cell(rotation, x, y) for (x, y) in base]


def validate_board_row(row: Any) -> str:
    if not isinstance(row, str) or len(row) != BOARD_WIDTH or any(ch not in ".#" for ch in row):
        raise ValueError(f"Invalid board row '{row}'")
    return row


def empty_board_top_to_bottom() -> list[str]:
    return ["." * BOARD_WIDTH for _ in range(VISIBLE_ROWS)]


def normalize_board(case: dict[str, Any]) -> list[str]:
    present = [k for k in BOARD_KEYS if k in case]
    if len(present) != 1:
        raise ValueError(
            f"Case {case.get('id', '<unknown>')} must define exactly one board field: {BOARD_KEYS}; got {present}"
        )

    key = present[0]
    if key == "board_top_to_bottom":
        board = case[key]
        if not isinstance(board, list) or len(board) != VISIBLE_ROWS:
            raise ValueError("board_top_to_bottom must be a 20-element list")
        return [validate_board_row(r) for r in board]

    if key == "board_rows_bottom":
        rows_bottom = case[key]
        if not isinstance(rows_bottom, list) or len(rows_bottom) > VISIBLE_ROWS:
            raise ValueError("board_rows_bottom must be a list with at most 20 rows")
        board = empty_board_top_to_bottom()
        for y, row in enumerate(rows_bottom):
            board[(VISIBLE_ROWS - 1) - y] = validate_board_row(row)
        return board

    if key == "board_rows_by_y":
        rows_by_y = case[key]
        if not isinstance(rows_by_y, dict):
            raise ValueError("board_rows_by_y must be an object/dict")
        board = empty_board_top_to_bottom()
        for y_raw, row in rows_by_y.items():
            y = int(y_raw)
            if y < 0 or y >= VISIBLE_ROWS:
                raise ValueError(f"Invalid y '{y}' in board_rows_by_y")
            board[(VISIBLE_ROWS - 1) - y] = validate_board_row(row)
        return board

    filled = case[key]
    if not isinstance(filled, list):
        raise ValueError("filled_cells must be a list")
    rows = [list("." * BOARD_WIDTH) for _ in range(VISIBLE_ROWS)]
    for cell in filled:
        if not isinstance(cell, list) or len(cell) != 2:
            raise ValueError(f"Invalid filled cell '{cell}', expected [x,y]")
        x, y = cell
        if not isinstance(x, int) or not isinstance(y, int):
            raise ValueError(f"Invalid filled cell '{cell}', x/y must be int")
        if x < 0 or x >= BOARD_WIDTH or y < 0 or y >= VISIBLE_ROWS:
            raise ValueError(f"Invalid filled cell '{cell}', out of bounds")
        rows[(VISIBLE_ROWS - 1) - y][x] = "#"
    return ["".join(r) for r in rows]


def overlay_active(board_rows: list[str], case: dict[str, Any]) -> list[str]:
    rows = [list(r) for r in board_rows]
    active = case["active"]
    piece = active["piece"]
    rotation = active["rotation"]
    x0 = active["x"]
    y0 = active["y"]

    for dx, dy in piece_cells(piece, rotation):
        x = x0 + dx
        y = y0 + dy
        if x < 0 or x >= BOARD_WIDTH or y < 0 or y >= VISIBLE_ROWS:
            continue
        row = (VISIBLE_ROWS - 1) - y
        rows[row][x] = "!" if rows[row][x] == "#" else piece

    return ["".join(r) for r in rows]


def print_case(case: dict[str, Any], index: int, total: int, overlay: bool) -> None:
    board = normalize_board(case)
    if overlay:
        board = overlay_active(board, case)

    active = case["active"]
    state = case["state"]
    expect = case["expect"]
    actions = " ".join(case["actions"])
    rotation = active["rotation"]
    rotation_name = ROT_NAMES[rotation] if 0 <= rotation < len(ROT_NAMES) else str(rotation)

    print("=" * 90)
    print(f"[{index}/{total}] {case['id']} family={case['family']}")
    print(f"source: {case['source_url']}")
    print(f"note:   {case['source_note']}")
    print(
        f"active: piece={active['piece']} rot={rotation}({rotation_name}) "
        f"x={active['x']} y={active['y']}"
    )
    print(
        "state:  "
        f"b2b={state['back_to_back']} combo={state['combo']} "
        f"spin_eligible={state['spin_eligible']} "
        f"last_rotate_used_kick={state['last_rotate_used_kick']}"
    )
    print(f"actions: {actions}")
    print(
        "expect: "
        f"lines={expect['lines_cleared']} spin={expect['spin_clear']} "
        f"difficult={expect['difficult_clear']} b2b_out={expect['b2b_out']} "
        f"b2b_bonus={expect['b2b_bonus_applied']} "
        f"must_observe_kick={expect['must_observe_kick']}"
    )
    if overlay:
        print("board_top_to_bottom (overlay: piece letter, '!' means overlap with stack):")
    else:
        print("board_top_to_bottom:")
    print("    0123456789")
    for row_idx, row in enumerate(board):
        print(f"{row_idx:2d}  {row}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_json = script_dir.parent / "data" / "twist_cases_srs.json"

    parser = argparse.ArgumentParser(
        description="Print twist dataset cases with board and piece metadata."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_json,
        help=f"Path to twist cases JSON (default: {default_json})",
    )
    parser.add_argument(
        "--family",
        action="append",
        choices=["I", "O", "T", "L", "J", "S", "Z"],
        help="Filter by family (can pass multiple times).",
    )
    parser.add_argument(
        "--id",
        dest="ids",
        action="append",
        help="Filter by case id (can pass multiple times).",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Do not draw the active piece on top of the board.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"input file not found: {args.input}")
        return 2

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        print("Invalid dataset: top-level JSON must be a list.")
        return 2

    selected: list[dict[str, Any]] = []
    family_filter = set(args.family or [])
    id_filter = set(args.ids or [])

    for case in raw:
        if not isinstance(case, dict):
            continue
        if family_filter and case.get("family") not in family_filter:
            continue
        if id_filter and case.get("id") not in id_filter:
            continue
        selected.append(case)

    if not selected:
        print("No cases matched the provided filters.")
        return 1

    print(f"Printing {len(selected)} case(s) from {args.input}")
    for idx, case in enumerate(selected, start=1):
        print_case(case, idx, len(selected), overlay=not args.no_overlay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
