"""Utilities to evaluate trained checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import gymnasium as gym

from tetris_v2.envs.registration import register_envs
from tetris_v2.envs.wrappers import FloatBoardWrapper


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN policy.")
    parser.add_argument("checkpoint", type=Path, help="Path to the SB3 .zip checkpoint.")
    parser.add_argument("--env", choices=("nes", "modern", "custom"), default="nes")
    parser.add_argument("--env-id", help="Gymnasium id when --env=custom.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true", help="Render episodes to console.")
    return parser.parse_args(argv)


def _resolve_env_id(args: argparse.Namespace) -> str:
    if args.env == "nes":
        return "KevinNES/Tetris-v0"
    if args.env == "modern":
        return "KevinModern/Tetris-v0"
    if not args.env_id:
        raise SystemExit("--env-id must be supplied when --env=custom.")
    return args.env_id


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    register_envs()

    try:
        from stable_baselines3 import DQN
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "stable-baselines3 is required to load checkpoints. "
            "Install with `pip install stable-baselines3[extra]`."
        ) from exc

    env_id = _resolve_env_id(args)
    env = gym.make(env_id, render_mode="human" if args.render else None)
    env = FloatBoardWrapper(env)

    model = DQN.load(str(args.checkpoint))

    returns = []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode if args.seed is not None else None)
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if args.render:
                env.render()
        returns.append(total_reward)
        print(f"Episode {episode+1}: reward={total_reward:.1f}, score={info.get('score')}")

    env.close()
    avg_return = sum(returns) / max(len(returns), 1)
    print(f"Average return over {len(returns)} episodes: {avg_return:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
