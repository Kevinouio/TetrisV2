"""Evaluate a trained scratch DQN checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import gymnasium as gym

from tetris_v2.envs.registration import register_envs
from tetris_v2.envs.wrappers import FloatBoardWrapper, RewardScaleWrapper
from .models import DQNAgent, ObservationProcessor


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a native DQN checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Path to the saved .pt checkpoint.")
    parser.add_argument("--env", choices=("nes", "modern", "custom"), default="nes")
    parser.add_argument("--env-id", help="Gymnasium id when --env=custom.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true", help="Render gameplay to the pygame window.")
    parser.add_argument("--device", help="Torch device override for loading the checkpoint.")
    parser.add_argument("--reward-scale", type=float, default=1.0)
    return parser.parse_args(argv)


def _resolve_env_id(args: argparse.Namespace) -> str:
    if args.env == "nes":
        return "KevinNES/Tetris-v0"
    if args.env == "modern":
        return "KevinModern/Tetris-v0"
    if not args.env_id:
        raise SystemExit("--env-id must be provided when --env=custom.")
    return args.env_id


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    register_envs()
    env_id = _resolve_env_id(args)
    env = gym.make(env_id, render_mode="human" if args.render else None)
    env = FloatBoardWrapper(env)
    if args.reward_scale != 1.0:
        env = RewardScaleWrapper(env, scale=args.reward_scale)

    processor = ObservationProcessor(env.observation_space)
    agent, _ = DQNAgent.load(str(args.checkpoint), device=args.device)
    if agent.obs_dim != processor.flat_dim or agent.action_dim != env.action_space.n:
        raise SystemExit("Checkpoint is incompatible with the selected environment.")

    returns = []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=None if args.seed is None else args.seed + episode)
        flat = processor.flatten(obs)
        terminated = truncated = False
        total_reward = 0.0
        last_score = 0.0
        while not (terminated or truncated):
            action = agent.act(flat, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            flat = processor.flatten(obs)
            total_reward += reward
            last_score = float(info.get("score", last_score))
            if args.render:
                env.render()
        returns.append(total_reward)
        print(f"Episode {episode+1}: reward={total_reward:.1f}, score={last_score:.1f}")

    env.close()
    avg_return = sum(returns) / max(len(returns), 1)
    print(f"Average return over {len(returns)} episodes: {avg_return:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
