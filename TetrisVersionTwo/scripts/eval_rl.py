"""Evaluate a VersionTwo PPO or DQN checkpoint on the C++ env."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VersionTwo RL checkpoint (ppo or dqn).")
    parser.add_argument("checkpoint", type=Path, help="Checkpoint path (.pt)")
    parser.add_argument("--algo", choices=("ppo", "dqn"), required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--temperature", type=float, default=1.0, help="PPO only")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Exploration rate when --stochastic")
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=True,
        help="Use greedy policy actions (default).",
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Sample/explore instead of greedy actions.",
    )
    parser.add_argument("--lib", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        from TetrisVersionTwo.rl.env import CCTetrisEnv
        from TetrisVersionTwo.rl.policy import load_policy
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc}. Install with: pip install -r requirements.txt", file=sys.stderr)
        return 1

    try:
        policy = load_policy(args.algo, args.checkpoint, device=args.device)
        env = CCTetrisEnv(seed=args.seed, max_steps=args.max_steps, lib_path=args.lib)
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc}. Install with: pip install -r requirements.txt", file=sys.stderr)
        return 1

    if policy.action_dim != int(env.action_space.n):
        raise SystemExit(
            f"Checkpoint action_dim={policy.action_dim} incompatible with env action space={env.action_space.n}."
        )

    returns = []
    final_lines = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        terminated = False
        truncated = False
        total_reward = 0.0
        last_lines = 0
        while not (terminated or truncated):
            action = policy.act(
                obs,
                deterministic=args.deterministic,
                temperature=args.temperature,
                epsilon=args.epsilon,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            last_lines = int(info.get("lines", last_lines))
        returns.append(total_reward)
        final_lines.append(last_lines)
        print(f"Episode {ep + 1}: reward={total_reward:.2f} lines={last_lines}")

    env.close()
    avg_return = float(np.mean(returns)) if returns else 0.0
    avg_lines = float(np.mean(final_lines)) if final_lines else 0.0
    print(f"Average reward over {len(returns)} episodes: {avg_return:.2f}")
    print(f"Average lines over {len(final_lines)} episodes: {avg_lines:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
