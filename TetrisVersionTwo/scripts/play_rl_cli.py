"""CLI playback for VersionTwo PPO/DQN checkpoints."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play VersionTwo RL policy in CLI mode.")
    parser.add_argument("checkpoint", type=Path, help="Checkpoint path (.pt)")
    parser.add_argument("--algo", choices=("ppo", "dqn"), required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--episodes", type=int, default=1)
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
    parser.add_argument("--delay-ms", type=int, default=0, help="Optional delay between actions.")
    parser.add_argument("--render-board", action="store_true", help="Print board after each action.")
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

    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0
            while not (terminated or truncated):
                action = policy.act(
                    obs,
                    deterministic=args.deterministic,
                    temperature=args.temperature,
                    epsilon=args.epsilon,
                )
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                if args.render_board:
                    print("\033[2J\033[H", end="")
                    print(env.render())
                    print(
                        f"Episode={episode + 1} Step={steps} Reward={total_reward:.2f} "
                        f"Lines={int(info.get('lines', 0))} GameOver={bool(info.get('game_over', False))}"
                    )
                if args.delay_ms > 0:
                    time.sleep(args.delay_ms / 1000.0)

            print(
                f"Episode {episode + 1}: steps={steps} reward={total_reward:.2f} "
                f"lines={int(info.get('lines', 0))} game_over={bool(info.get('game_over', False))}"
            )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
