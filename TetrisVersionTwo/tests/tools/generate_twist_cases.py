#!/usr/bin/env python3
"""Generate a C++ header from twist dataset JSON fixtures.

Supported board encodings per case (exactly one):
  1) board_top_to_bottom: [20 strings]
  2) board_rows_bottom: [rows from y=0 upward]
  3) board_rows_by_y: {"0": "..........", "1": "###..#####", ...}
  4) filled_cells: [[x, y], ...]  # marks those cells as '#'
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


VALID_ACTIONS = {"L", "R", "D", "HD", "CW", "CCW", "R180", "NONE", "HOLD"}
VALID_PIECES = {"I", "O", "T", "L", "J", "S", "Z"}
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BOARD_KEYS = ("board_top_to_bottom", "board_rows_bottom", "board_rows_by_y", "filled_cells")


def cpp_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def expect_bool(obj: dict[str, Any], key: str) -> bool:
    value = obj.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean for '{key}', got {type(value).__name__}")
    return value


def expect_int(obj: dict[str, Any], key: str) -> int:
    value = obj.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Expected integer for '{key}', got {type(value).__name__}")
    return value


def expect_str(obj: dict[str, Any], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Expected string for '{key}', got {type(value).__name__}")
    return value


def validate_board_row(row: Any) -> str:
    if not isinstance(row, str) or len(row) != BOARD_WIDTH or any(ch not in ".#" for ch in row):
        raise ValueError(f"Invalid board row '{row}'")
    return row


def empty_board_top_to_bottom() -> list[str]:
    return ["." * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]


def board_from_rows_bottom(rows_bottom: Any) -> list[str]:
    if not isinstance(rows_bottom, list):
        raise ValueError("board_rows_bottom must be a list")
    if len(rows_bottom) > BOARD_HEIGHT:
        raise ValueError(f"board_rows_bottom can have at most {BOARD_HEIGHT} rows")

    board = empty_board_top_to_bottom()
    for y, row in enumerate(rows_bottom):
        row_text = validate_board_row(row)
        top_index = (BOARD_HEIGHT - 1) - y
        board[top_index] = row_text
    return board


def board_from_rows_by_y(rows_by_y: Any) -> list[str]:
    if not isinstance(rows_by_y, dict):
        raise ValueError("board_rows_by_y must be an object/dict")

    board = empty_board_top_to_bottom()
    for y_raw, row in rows_by_y.items():
        try:
            y = int(y_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"board_rows_by_y key '{y_raw}' is not an integer") from exc
        if y < 0 or y >= BOARD_HEIGHT:
            raise ValueError(f"board_rows_by_y y must be in [0,{BOARD_HEIGHT - 1}], got {y}")
        row_text = validate_board_row(row)
        top_index = (BOARD_HEIGHT - 1) - y
        board[top_index] = row_text
    return board


def board_from_filled_cells(filled_cells: Any) -> list[str]:
    if not isinstance(filled_cells, list):
        raise ValueError("filled_cells must be a list")

    rows = [list("." * BOARD_WIDTH) for _ in range(BOARD_HEIGHT)]
    for cell in filled_cells:
        if not isinstance(cell, list) or len(cell) != 2:
            raise ValueError(f"Invalid filled cell '{cell}', expected [x, y]")
        x, y = cell
        if not isinstance(x, int) or not isinstance(y, int):
            raise ValueError(f"Invalid filled cell '{cell}', coordinates must be integers")
        if x < 0 or x >= BOARD_WIDTH or y < 0 or y >= BOARD_HEIGHT:
            raise ValueError(
                f"Invalid filled cell '{cell}', x in [0,{BOARD_WIDTH - 1}] and y in [0,{BOARD_HEIGHT - 1}]"
            )
        top_index = (BOARD_HEIGHT - 1) - y
        rows[top_index][x] = "#"

    return ["".join(r) for r in rows]


def normalize_board(case: dict[str, Any]) -> list[str]:
    present_keys = [k for k in BOARD_KEYS if k in case]
    if len(present_keys) != 1:
        raise ValueError(
            f"Case {case.get('id', '<unknown>')} must specify exactly one of {BOARD_KEYS}, got {present_keys}"
        )

    key = present_keys[0]
    if key == "board_top_to_bottom":
        board = case["board_top_to_bottom"]
        if not isinstance(board, list) or len(board) != BOARD_HEIGHT:
            raise ValueError(f"board_top_to_bottom must be a {BOARD_HEIGHT}-element list")
        return [validate_board_row(row) for row in board]
    if key == "board_rows_bottom":
        return board_from_rows_bottom(case["board_rows_bottom"])
    if key == "board_rows_by_y":
        return board_from_rows_by_y(case["board_rows_by_y"])
    return board_from_filled_cells(case["filled_cells"])


def normalize_case(case: dict[str, Any]) -> dict[str, Any]:
    out = dict(case)
    out["board_top_to_bottom"] = normalize_board(case)
    for key in BOARD_KEYS:
        if key != "board_top_to_bottom":
            out.pop(key, None)
    return out


def validate_case(case: dict[str, Any]) -> None:
    required = {
        "id",
        "family",
        "source_url",
        "source_note",
        "board_top_to_bottom",
        "active",
        "state",
        "actions",
        "expect",
    }
    missing = required.difference(case.keys())
    if missing:
        raise ValueError(f"Missing keys for case {case.get('id', '<unknown>')}: {sorted(missing)}")

    family = expect_str(case, "family")
    if family not in VALID_PIECES:
        raise ValueError(f"Invalid family '{family}'")

    board = case["board_top_to_bottom"]
    if not isinstance(board, list) or len(board) != BOARD_HEIGHT:
        raise ValueError(f"board_top_to_bottom must be a {BOARD_HEIGHT}-element list")
    for row in board:
        validate_board_row(row)

    active = case["active"]
    if not isinstance(active, dict):
        raise ValueError("active must be an object")
    piece = expect_str(active, "piece")
    if piece != family:
        raise ValueError(f"active.piece '{piece}' must match family '{family}'")
    if piece not in VALID_PIECES:
        raise ValueError(f"Invalid active piece '{piece}'")
    rotation = expect_int(active, "rotation")
    if rotation < 0 or rotation > 3:
        raise ValueError("active.rotation must be in [0,3]")
    expect_int(active, "x")
    expect_int(active, "y")

    state = case["state"]
    if not isinstance(state, dict):
        raise ValueError("state must be an object")
    expect_bool(state, "back_to_back")
    expect_int(state, "combo")
    expect_bool(state, "spin_eligible")
    expect_bool(state, "last_rotate_used_kick")

    actions = case["actions"]
    if not isinstance(actions, list) or not actions:
        raise ValueError("actions must be a non-empty list")
    for action in actions:
        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action token '{action}'")

    expect_obj = case["expect"]
    if not isinstance(expect_obj, dict):
        raise ValueError("expect must be an object")
    lines_cleared = expect_int(expect_obj, "lines_cleared")
    if lines_cleared < 0 or lines_cleared > 4:
        raise ValueError("expect.lines_cleared must be in [0,4]")
    expect_bool(expect_obj, "spin_clear")
    expect_bool(expect_obj, "difficult_clear")
    expect_bool(expect_obj, "b2b_out")
    expect_bool(expect_obj, "b2b_bonus_applied")
    expect_bool(expect_obj, "must_observe_kick")


def emit_header(cases: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <array>")
    lines.append("#include <string_view>")
    lines.append("#include <vector>")
    lines.append("")
    lines.append("namespace tetris_v2::twist_dataset {")
    lines.append("")
    lines.append("struct TwistActive {")
    lines.append("    char piece;")
    lines.append("    int rotation;")
    lines.append("    int x;")
    lines.append("    int y;")
    lines.append("};")
    lines.append("")
    lines.append("struct TwistState {")
    lines.append("    bool back_to_back;")
    lines.append("    int combo;")
    lines.append("    bool spin_eligible;")
    lines.append("    bool last_rotate_used_kick;")
    lines.append("};")
    lines.append("")
    lines.append("struct TwistExpect {")
    lines.append("    int lines_cleared;")
    lines.append("    bool spin_clear;")
    lines.append("    bool difficult_clear;")
    lines.append("    bool b2b_out;")
    lines.append("    bool b2b_bonus_applied;")
    lines.append("    bool must_observe_kick;")
    lines.append("};")
    lines.append("")
    lines.append("struct TwistCase {")
    lines.append("    std::string_view id;")
    lines.append("    std::string_view family;")
    lines.append("    std::string_view source_url;")
    lines.append("    std::string_view source_note;")
    lines.append("    std::array<std::string_view, 20> board_top_to_bottom;")
    lines.append("    TwistActive active;")
    lines.append("    TwistState state;")
    lines.append("    std::vector<std::string_view> actions;")
    lines.append("    TwistExpect expect;")
    lines.append("};")
    lines.append("")
    lines.append("inline const std::vector<TwistCase> kTwistCases{")

    for case in cases:
        active = case["active"]
        state = case["state"]
        expect = case["expect"]
        board = ", ".join(cpp_string(row) for row in case["board_top_to_bottom"])
        actions = ", ".join(cpp_string(token) for token in case["actions"])
        lines.append("    TwistCase{")
        lines.append(f"        {cpp_string(case['id'])},")
        lines.append(f"        {cpp_string(case['family'])},")
        lines.append(f"        {cpp_string(case['source_url'])},")
        lines.append(f"        {cpp_string(case['source_note'])},")
        lines.append(f"        std::array<std::string_view, 20>{{{board}}},")
        lines.append(
            "        TwistActive{"
            f"{cpp_string(active['piece'])}[0], {active['rotation']}, {active['x']}, {active['y']}"
            "},")
        lines.append(
            "        TwistState{"
            f"{str(state['back_to_back']).lower()}, {state['combo']}, "
            f"{str(state['spin_eligible']).lower()}, {str(state['last_rotate_used_kick']).lower()}"
            "},")
        lines.append(f"        std::vector<std::string_view>{{{actions}}},")
        lines.append(
            "        TwistExpect{"
            f"{expect['lines_cleared']}, "
            f"{str(expect['spin_clear']).lower()}, "
            f"{str(expect['difficult_clear']).lower()}, "
            f"{str(expect['b2b_out']).lower()}, "
            f"{str(expect['b2b_bonus_applied']).lower()}, "
            f"{str(expect['must_observe_kick']).lower()}"
            "},")
        lines.append("    },")

    lines.append("};")
    lines.append("")
    lines.append("}  // namespace tetris_v2::twist_dataset")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: generate_twist_cases.py <input_json> <output_header>", file=sys.stderr)
        return 2

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    if not input_path.exists():
        print(f"input file not found: {input_path}", file=sys.stderr)
        return 2

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Top-level JSON must be a list of cases")

    cases: list[dict[str, Any]] = []
    ids: set[str] = set()
    for case in raw:
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object")
        normalized_case = normalize_case(case)
        validate_case(normalized_case)
        case_id = normalized_case["id"]
        if case_id in ids:
            raise ValueError(f"Duplicate case id '{case_id}'")
        ids.add(case_id)
        cases.append(normalized_case)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(emit_header(cases), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
