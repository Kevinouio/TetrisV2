#!/usr/bin/env python
"""Visualise a random agent using placement-based actions."""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import gymnasium as gym

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tetris_v2.envs.registration import register_envs
from tetris_v2.envs.wrappers import PlacementActionWrapper


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a random placement-action agent.")
    parser.add_argument("--env", choices=("nes", "modern"), default="modern")
    parser.add_argument("--env-id", help="Custom Gymnasium id to use instead of the presets.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--allow-hold", action="store_true", help="Force-enable hold even if env wouldn't.")
    parser.add_argument("--no-hold", action="store_true", help="Force-disable hold actions.")
    parser.add_argument("--render-mode", choices=("human", "rgb_array"), default="human")
    return parser.parse_args()


def _resolve_env_id(env_name: str, env_id: str | None) -> str:
    if env_id:
        return env_id
    return "KevinModern/Tetris-v0" if env_name == "modern" else "KevinNES/Tetris-v0"


def main() -> None:
    args = _parse_args()
    register_envs()
    env_id = _resolve_env_id(args.env, args.env_id)
    env = gym.make(env_id, render_mode=args.render_mode)
    if args.no_hold:
        allow_hold = False
    elif args.allow_hold:
        allow_hold = True
    else:
        allow_hold = None
    env = PlacementActionWrapper(env, allow_hold=allow_hold)

    episodes = 0
    rng = np.random.default_rng(args.seed)
    while episodes < args.max_episodes:
        obs, info = env.reset(seed=None if args.seed is None else args.seed + episodes)
        done = False
        total_reward = 0.0
        while not done:
            mask = info.get("action_mask")
            if mask is None or not mask.any():
                action = int(env.action_space.sample())
            else:
                valid_indices = np.flatnonzero(mask)
                action = int(rng.choice(valid_indices))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            done = terminated or truncated
        print(f"Episode {episodes + 1}: reward={total_reward:.2f}")
        episodes += 1
    env.close()


if __name__ == "__main__":
    main()
