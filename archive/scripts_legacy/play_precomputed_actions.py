#!/usr/bin/env python
"""Interactive CLI to play with placement-based macro actions."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

import gymnasium as gym

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tetris_v2.envs.registration import register_envs
from tetris_v2.envs.wrappers import PlacementActionWrapper


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Tetris using precomputed placement actions.")
    parser.add_argument("--env", choices=("nes", "modern"), default="nes", help="Environment preset to load.")
    parser.add_argument("--env-id", help="Override Gymnasium id when using --env=custom.")
    parser.add_argument("--render", choices=("none", "human", "rgb_array"), default="human")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-hold", action="store_true", help="Force-enable hold even on NES envs.")
    parser.add_argument("--no-hold", action="store_true", help="Disable hold macro actions.")
    return parser.parse_args()


def _resolve_env_id(args: argparse.Namespace) -> str:
    if args.env_id:
        return args.env_id
    if args.env == "nes":
        return "KevinNES/Tetris-v0"
    return "KevinModern/Tetris-v0"


def main() -> None:
    args = _parse_args()
    register_envs()
    env_id = _resolve_env_id(args)
    render_kw = {}
    if args.render != "none":
        render_kw["render_mode"] = args.render
    env = gym.make(env_id, **render_kw)
    allow_hold: Optional[bool]
    if args.no_hold:
        allow_hold = False
    elif args.allow_hold:
        allow_hold = True
    else:
        allow_hold = None
    env = PlacementActionWrapper(env, allow_hold=allow_hold)
    obs, info = env.reset(seed=args.seed)
    terminated = truncated = False
    step_idx = 0
    while True:
        if args.render == "human":
            env.render()
        catalog = env.available_action_descriptions()
        if not catalog:
            print("No available placement actions; stepping noop.")
        else:
            print("\nAvailable placements (index: hold rotation column landing sequence_length):")
            for entry in catalog:
                print(
                    f"  {entry['index']:2d}:  {'H' if entry['use_hold'] else '-'}  "
                    f"rot={entry['rotation']} col={entry['column']} land={entry['landing_row']} "
                    f"len={entry['sequence_length']}"
                )
        choice = input("Enter action index (or 'q' to quit, 'r' to render next frame, 'reset' to restart): ").strip()
        if choice.lower() in {"q", "quit"}:
            break
        if choice.lower() in {"reset", "restart"}:
            obs, info = env.reset()
            terminated = truncated = False
            step_idx = 0
            continue
        if choice.lower() == "r":
            env.render()
            continue
        try:
            action = int(choice)
        except ValueError:
            print("Invalid selection; enter an action index or a command.")
            continue
        if catalog and action not in [entry["index"] for entry in catalog]:
            print("Action not currently available.")
            continue
        obs, reward, terminated, truncated, info = env.step(action)
        step_idx += 1
        score = info.get("score")
        print(
            f"Step {step_idx}: reward={reward:.2f} score={score} "
            f"flags={{'term': {terminated}, 'trunc': {truncated}}}"
        )
        if terminated or truncated:
            print("Episode finished. Resetting...\n")
            obs, info = env.reset()
            terminated = truncated = False
            step_idx = 0

    env.close()


if __name__ == "__main__":
    main()
