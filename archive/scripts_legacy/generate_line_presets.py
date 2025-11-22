#!/usr/bin/env python3
"""Generate NES line-clear presets with configurable hole patterns."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional


def _build_row(width: int, holes: List[int]) -> str:
    row = ["#"] * width
    for idx in holes:
        row[idx] = "."
    return "".join(row)


def generate_presets(
    *,
    width: int,
    height: int,
    hole_counts: List[int],
    filled_rows: int,
    piece: Optional[str],
    rotation: int,
    env: str,
    name_prefix: str,
) -> Dict[str, Dict[str, object]]:
    presets: Dict[str, Dict[str, object]] = {}
    empty_line = "." * width
    top_rows = height - filled_rows
    for holes in hole_counts:
        if holes <= 0 or holes > width:
            continue
        for combo in itertools.combinations(range(width), holes):
            bottom = [_build_row(width, list(combo)) for _ in range(filled_rows)]
            board = [empty_line for _ in range(top_rows)] + bottom
            key = f"{name_prefix}_{holes}_{'_'.join(str(idx) for idx in combo)}"
            entry = {
                "env": env,
                "board": board,
            }
            if piece:
                entry["current"] = {
                    "piece": piece,
                    "rotation": rotation,
                }
            presets[key] = entry
    return presets


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate simple line-clear board presets.")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the generated JSON file.")
    parser.add_argument("--width", type=int, default=10, help="Board width (default: 10).")
    parser.add_argument("--height", type=int, default=20, help="Board height (default: 20).")
    parser.add_argument("--hole-counts", type=int, nargs="+", default=[1, 2, 3, 4], help="List of hole counts to generate variations for.")
    parser.add_argument("--filled-rows", type=int, default=1, help="Number of identical filled rows at the bottom (default: 1).")
    parser.add_argument("--piece", default=None, help="Active piece id (default: none / leave to env).")
    parser.add_argument("--rotation", type=int, default=0, help="Active piece rotation (default: 0).")
    parser.add_argument("--env", default="nes", help="Environment id stored in each preset (default: nes).")
    parser.add_argument("--name-prefix", default="nes_line", help="Prefix for generated preset keys.")
    args = parser.parse_args()

    presets = generate_presets(
        width=args.width,
        height=args.height,
        hole_counts=args.hole_counts,
        filled_rows=args.filled_rows,
        piece=args.piece,
        rotation=args.rotation,
        env=args.env,
        name_prefix=args.name_prefix,
    )

    payload = {"presets": presets}
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(presets)} presets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
