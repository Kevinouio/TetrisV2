"""Record evaluation rollouts to MP4/GIF using Gymnasium's RecordVideo wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import gymnasium as gym

from tetris_v2.envs.registration import register_envs
from tetris_v2.envs.wrappers import FloatBoardWrapper


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record gameplay footage.")
    parser.add_argument("--env", choices=("nes", "modern", "custom"), default="nes")
    parser.add_argument("--env-id", help="Gymnasium id when --env=custom.")
    parser.add_argument("--checkpoint", type=Path, help="Optional SB3 checkpoint for deterministic play.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--video-dir", type=Path, default=Path("videos"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render", action="store_true", help="Also render to the console while recording.")
    return parser.parse_args(argv)


def _resolve_env_id(args: argparse.Namespace) -> str:
    if args.env == "nes":
        return "KevinNES/Tetris-v0"
    if args.env == "modern":
        return "KevinModern/Tetris-v0"
    if not args.env_id:
        raise SystemExit("--env-id required when --env=custom.")
    return args.env_id


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    register_envs()
    env_id = _resolve_env_id(args)

    model = None
    if args.checkpoint is not None:
        try:
            from stable_baselines3 import DQN
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit(
                "stable-baselines3 is required to load checkpoints. "
                "Install with `pip install stable-baselines3[extra]`."
            ) from exc
        model = DQN.load(str(args.checkpoint))

    args.video_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(args.video_dir),
        name_prefix="tetris",
        episode_trigger=lambda episode_id: True,
    )
    env = FloatBoardWrapper(env)

    obs, _ = env.reset(seed=args.seed)
    for episode in range(args.episodes):
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if args.render:
                env.render()
        print(f"Recorded episode {episode+1}: score={info.get('score')} reward={total_reward:.1f}")
        obs, _ = env.reset()

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
