"""Evaluate a trained native PPO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np

from TetrisVersionOne.agents.common import build_agent_reward_config, build_environment_reward_config
from TetrisVersionOne.agents.observation import ObservationProcessor
from TetrisVersionOne.env.registration import register_envs
from TetrisVersionOne.env.wrappers import (
    FloatBoardWrapper,
    PlacementActionWrapper,
    RewardScaleWrapper,
    UniversalRewardWrapper,
)
from .models import PPOAgent


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a native PPO checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Path to the saved .pt checkpoint.")
    parser.add_argument("--env", choices=("nes", "modern", "custom"), default="nes")
    parser.add_argument("--env-id", help="Gymnasium id when --env=custom.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true", help="Render gameplay to a pygame window.")
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
        "--agent-reward-weight",
        dest="agent_reward_weights",
        metavar="KEY=VALUE",
        action="append",
        help="Override AgentRewardConfig fields (e.g., hole_penalty=1.2).",
    )
    parser.add_argument(
        "--environment-reward-weight",
        "--env-reward-weight",
        dest="environment_reward_weights",
        metavar="KEY=VALUE",
        action="append",
        help="Override EnvironmentRewardConfig fields (e.g., combo_reward=0.5).",
    )
    parser.add_argument(
        "--advanced-reward-weight",
        dest="agent_reward_weights",
        metavar="KEY=VALUE",
        action="append",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Action sampling temperature.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Epsilon-greedy exploration rate.")
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=True,
        help="Use greedy actions (default).",
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Sample actions instead of running deterministically.",
    )
    parser.add_argument(
        "--placement-actions",
        action="store_true",
        help="Evaluate a placement-action policy (must match training).",
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
    env_kind: str,
    agent_reward_weights: Optional[list[str]],
    environment_reward_weights: Optional[list[str]],
    use_placement_actions: bool,
):
    try:
        agent_cfg = build_agent_reward_config(agent_reward_weights)
        env_cfg = build_environment_reward_config(environment_reward_weights)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    env = UniversalRewardWrapper(
        env,
        env_kind=env_kind,
        agent_config=agent_cfg,
        env_config=env_cfg,
    )
    env = FloatBoardWrapper(env)
    if reward_scale != 1.0:
        env = RewardScaleWrapper(env, scale=reward_scale)
    if use_placement_actions:
        env = PlacementActionWrapper(env, allow_hold=(env_kind != "nes"))
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
    env_kind = args.env.lower()
    if env_kind == "custom":
        lowered = env_id.lower()
        if "modern" in lowered:
            env_kind = "modern"
        elif "nes" in lowered:
            env_kind = "nes"

    env = gym.make(env_id, render_mode="human" if args.render else None, **env_kwargs)
    env = _wrap_env(
        env,
        reward_scale=args.reward_scale,
        env_kind=env_kind,
        agent_reward_weights=args.agent_reward_weights,
        environment_reward_weights=args.environment_reward_weights,
        use_placement_actions=args.placement_actions,
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
            action, _, _ = agent.act(
                flat,
                temperature=args.temperature,
                epsilon=args.epsilon,
                deterministic=args.deterministic,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            flat = processor.flatten(obs)
            total_reward += reward
            last_score = float(info.get("score", last_score))
            if args.render:
                env.render()
        returns.append(total_reward)
        print(f"Episode {episode + 1}: reward={total_reward:.1f}, score={last_score:.1f}")

    env.close()
    avg_return = float(np.mean(returns)) if returns else 0.0
    print(f"Average return over {len(returns)} episodes: {avg_return:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
