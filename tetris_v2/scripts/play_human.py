"""Minimal CLI to play the environments manually."""

from __future__ import annotations

import argparse
from typing import Dict, Optional

import gymnasium as gym

from tetris_v2.envs.registration import register_envs


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Tetris with keyboard input.")
    parser.add_argument("--env", choices=("nes", "modern"), default="nes")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def _resolve_env_id(env_key: str) -> str:
    return "KevinNES/Tetris-v0" if env_key == "nes" else "KevinModern/Tetris-v0"


def _action_mapping(env_key: str) -> Dict[str, int]:
    if env_key == "nes":
        return {
            "": 0,
            "a": 1,
            "d": 2,
            "w": 3,
            "s": 4,
            " ": 5,
        }
    return {
        "": 0,
        "a": 1,
        "d": 2,
        "w": 3,
        "x": 3,
        "z": 4,
        "s": 5,
        " ": 6,
        "c": 7,
    }


def _prompt_action(env_key: str) -> Optional[int]:
    mapping = _action_mapping(env_key)
    raw = input("Action [a/d/w/s/space, '?' for help, 'r' reset, 'q' quit]: ").strip().lower()
    if raw == "?":
        if env_key == "nes":
            print("Controls: a=left, d=right, w=rotate, s=soft drop, space=hard drop, Enter=wait")
        else:
            print("Controls: a=left, d=right, w/x=rotate CW, z=rotate CCW, s=soft drop, space=hard drop, c=hold")
        return _prompt_action(env_key)
    if raw == "q":
        return None
    if raw == "r":
        return -1
    if raw in mapping:
        return mapping[raw]
    print("Unknown input, try again.")
    return _prompt_action(env_key)


def _print_hud(obs, reward, info) -> None:
    score = int(obs.get("score", 0))
    lines = int(obs.get("lines", 0))
    level = int(obs.get("level", 0))
    print(f"Score: {score}  Lines: {lines}  Level: {level}  Reward: {reward:.2f}")
    board = obs["board"]
    print("\n".join("".join(" .:#%&@AB"[cell] for cell in row) for row in board[::-1]))


def main(argv=None) -> int:
    args = _parse_args(argv)
    register_envs()
    env_id = _resolve_env_id(args.env)
    env = gym.make(env_id, render_mode="human")
    obs, _ = env.reset(seed=args.seed)
    done = False
    while True:
        env.render()
        action = _prompt_action(args.env)
        if action is None:
            break
        if action == -1:  # reset
            obs, _ = env.reset()
            continue
        obs, reward, terminated, truncated, info = env.step(action)
        _print_hud(obs, reward, info)
        done = terminated or truncated
        if done:
            print("Game over. Press Enter to play again or type 'q' to quit.")
            if input().strip().lower() == "q":
                break
            obs, _ = env.reset()
            done = False
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
