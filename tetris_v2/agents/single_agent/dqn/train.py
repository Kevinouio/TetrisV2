"""Command-line trainer for DQN agents on Tetris environments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym

from tetris_v2.envs.registration import register_envs
from tetris_v2.envs.wrappers import FloatBoardWrapper, RewardScaleWrapper
from .models import build_dqn_agent


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent on Tetris.")
    parser.add_argument("--env", choices=("nes", "modern", "custom"), default="nes")
    parser.add_argument("--env-id", help="Gymnasium env id when --env=custom.")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/dqn"))
    parser.add_argument("--checkpoint-frequency", type=int, default=100_000)
    parser.add_argument("--eval-frequency", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", help="Torch device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--resume", type=Path, help="Path to resume checkpoint (.zip).")
    return parser.parse_args(argv)


def _resolve_env_id(args: argparse.Namespace) -> str:
    if args.env == "nes":
        return "KevinNES/Tetris-v0"
    if args.env == "modern":
        return "KevinModern/Tetris-v0"
    if not args.env_id:
        raise SystemExit("--env-id is required when --env=custom.")
    return args.env_id


def _make_env_factory(env_id: str, seed: int, reward_scale: float) -> Callable[[], gym.Env]:
    def wrap(env: gym.Env) -> gym.Env:
        env = FloatBoardWrapper(env)
        if reward_scale != 1.0:
            env = RewardScaleWrapper(env, scale=reward_scale)
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=100)
        return env

    def factory(rank: int = 0) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            env = gym.make(env_id)
            env.reset(seed=seed + rank if seed is not None else None)
            return wrap(env)

        return _init

    return factory


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    register_envs()
    env_id = _resolve_env_id(args)

    try:  # lazy import so --help works without SB3
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "stable-baselines3[extra] is required for training. "
            "Install it with `pip install stable-baselines3[extra] torch`."
        ) from exc

    env_factory = _make_env_factory(env_id, args.seed, args.reward_scale)
    vec_env = DummyVecEnv([env_factory(rank=i) for i in range(args.n_envs)])

    eval_env = DummyVecEnv([env_factory(rank=10_000 + i) for i in range(1)])  # independent seed

    args.log_dir.mkdir(parents=True, exist_ok=True)
    callbacks = []
    if args.checkpoint_frequency:
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_frequency // max(args.n_envs, 1),
                save_path=str(args.log_dir),
                name_prefix="dqn_tetris",
            )
        )
    if args.eval_frequency:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(args.log_dir / "best"),
                n_eval_episodes=args.eval_episodes,
                eval_freq=args.eval_frequency,
                deterministic=True,
            )
        )

    model = build_dqn_agent(
        vec_env,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    if args.resume:
        model.set_parameters(str(args.resume))

    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=not args.no_progress_bar,
        callback=callbacks if callbacks else None,
    )
    model.save(str(args.log_dir / "final_model"))
    vec_env.close()
    eval_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
