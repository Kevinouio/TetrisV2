"""Evaluate a trained PPO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import gymnasium as gym

from tetris_v2.agents.single_agent.common import build_advanced_reward_config
from tetris_v2.agents.single_agent.dqn.models import ObservationProcessor
from tetris_v2.envs.registration import register_envs
from tetris_v2.envs.wrappers import AdvancedRewardWrapper, FloatBoardWrapper, RewardScaleWrapper
from .models import PPOAgent


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PPO agent checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Path to the saved .pt checkpoint.")
    parser.add_argument("--env", choices=("nes", "modern", "custom"), default="nes")
    parser.add_argument("--env-id", help="Gymnasium id when --env=custom.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", help="Torch device override for loading the checkpoint.")
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--rotation-penalty", type=float, default=0.0)
    parser.add_argument(
        "--line-clear-reward",
        type=float,
        nargs=4,
        metavar=("SINGLE", "DOUBLE", "TRIPLE", "TETRIS"),
        help="Additional reward (NES score units) per line clear type.",
    )
    parser.add_argument("--step-penalty", type=float, default=0.0)
    parser.add_argument(
        "--advanced-reward",
        action="store_true",
        help="Enable the shaped reward wrapper during evaluation.",
    )
    parser.add_argument(
        "--advanced-reward-weight",
        dest="advanced_reward_weights",
        metavar="KEY=VALUE",
        action="append",
        help="Override AdvancedRewardConfig fields when --advanced-reward is set.",
    )
    return parser.parse_args(argv)


def _resolve_env_id(args: argparse.Namespace) -> str:
    if args.env == "nes":
        return "KevinNES/Tetris-v0"
    if args.env == "modern":
        return "KevinModern/Tetris-v0"
    if not args.env_id:
        raise SystemExit("--env-id must be provided when --env=custom.")
    return args.env_id


def _wrap_env(
    env: gym.Env,
    *,
    reward_scale: float,
    use_advanced: bool,
    advanced_reward_weights: Optional[list[str]],
):
    try:
        config = build_advanced_reward_config(use_advanced, advanced_reward_weights)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if config is not None:
        env = AdvancedRewardWrapper(env, config=config)
    env = FloatBoardWrapper(env)
    if reward_scale != 1.0:
        env = RewardScaleWrapper(env, scale=reward_scale)
    return env


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    register_envs()
    env_id = _resolve_env_id(args)
    env_kwargs = {}
    if env_id.startswith("KevinNES"):
        if args.rotation_penalty:
            env_kwargs["rotation_penalty"] = args.rotation_penalty
        if args.step_penalty:
            env_kwargs["step_penalty"] = args.step_penalty
        if args.line_clear_reward:
            env_kwargs["line_clear_reward"] = args.line_clear_reward
    env = gym.make(env_id, render_mode="human" if args.render else None, **env_kwargs)
    env = _wrap_env(
        env,
        reward_scale=args.reward_scale,
        use_advanced=args.advanced_reward,
        advanced_reward_weights=args.advanced_reward_weights,
    )

    processor = ObservationProcessor(env.observation_space)
    agent, _ = PPOAgent.load(str(args.checkpoint), device=args.device)
    if agent.config.obs_dim != processor.flat_dim or agent.config.action_dim != env.action_space.n:
        raise SystemExit("Checkpoint is incompatible with the selected environment.")

    returns = []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=None if args.seed is None else args.seed + episode)
        flat = processor.flatten(obs)
        terminated = truncated = False
        total_reward = 0.0
        last_score = 0.0
        while not (terminated or truncated):
            action, _, _ = agent.act(flat, temperature=1e-6, epsilon=0.0, deterministic=True)
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
