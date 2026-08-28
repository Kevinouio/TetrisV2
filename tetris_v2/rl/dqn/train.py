"""Train the custom DQN agent on the TetrisV2 environment."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import numpy as np
import torch

from tetris_v2.rl.dqn.core import DQNAgent, DQNConfig, ReplayBuffer
from tetris_v2.rl.env import CCTetrisEnv


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train custom DQN on the TetrisV2 C++ environment.")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000_000)
    parser.add_argument("--buffer-size", type=int, default=250_000)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-sync-interval", type=int, default=1_000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=200_000)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--log-interval", type=int, default=10_000)
    parser.add_argument("--eval-frequency", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-frequency", type=int, default=100_000)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/dqn"))
    parser.add_argument("--lib", type=Path, default=None)
    return parser.parse_args(argv)


def epsilon_at_step(step: int, *, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return float(end)
    alpha = min(1.0, max(0.0, step / float(decay_steps)))
    return float(start + (end - start) * alpha)


def evaluate(agent: DQNAgent, *, episodes: int, seed: int, max_steps: int, lib_path: Optional[Path]) -> float:
    returns = []
    env = CCTetrisEnv(seed=seed, max_steps=max_steps, lib_path=lib_path)
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        action_mask = np.asarray(info["action_mask"], dtype=np.float32)
        done = False
        trunc = False
        total = 0.0
        while not (done or trunc):
            action = agent.select_action(obs, deterministic=True, action_mask=action_mask)
            obs, reward, done, trunc, info = env.step(action)
            action_mask = np.asarray(info["action_mask"], dtype=np.float32)
            total += reward
        returns.append(total)
    env.close()
    return float(np.mean(returns)) if returns else 0.0


def run(args: argparse.Namespace) -> int:
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be >= 1")
    if args.buffer_size <= 0:
        raise SystemExit("--buffer-size must be >= 1")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = CCTetrisEnv(seed=args.seed, max_steps=args.max_steps, lib_path=args.lib)
    obs, info = env.reset(seed=args.seed)
    action_mask = np.asarray(info["action_mask"], dtype=np.float32)
    obs_dim = int(obs.shape[-1])
    action_dim = int(env.action_space.n)

    cfg = DQNConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=tuple(args.hidden_sizes),
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        target_sync_interval=args.target_sync_interval,
        device=args.device,
    )
    agent = DQNAgent(cfg)
    replay = ReplayBuffer(args.buffer_size, obs_dim=obs_dim, action_dim=action_dim)

    args.log_dir.mkdir(parents=True, exist_ok=True)

    next_log = args.log_interval
    next_eval = args.eval_frequency
    next_ckpt = args.checkpoint_frequency
    episode_idx = 0
    episode_return = 0.0
    recent_returns: Deque[float] = deque(maxlen=200)
    recent_losses: Deque[float] = deque(maxlen=1_000)
    last_loss = 0.0

    try:
        for global_step in range(1, args.total_timesteps + 1):
            epsilon = epsilon_at_step(
                global_step,
                start=args.epsilon_start,
                end=args.epsilon_end,
                decay_steps=args.epsilon_decay_steps,
            )
            action = agent.select_action(obs, epsilon=epsilon, action_mask=action_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_action_mask = np.asarray(info["action_mask"], dtype=np.float32)
            episode_done = bool(terminated or truncated)
            replay.add(
                obs,
                action,
                float(reward),
                next_obs,
                bool(terminated),
                next_action_mask=next_action_mask,
            )

            obs = next_obs
            action_mask = next_action_mask
            episode_return += float(reward)
            if episode_done:
                recent_returns.append(episode_return)
                episode_return = 0.0
                episode_idx += 1
                obs, info = env.reset(seed=args.seed + episode_idx)
                action_mask = np.asarray(info["action_mask"], dtype=np.float32)

            if (
                global_step >= args.warmup_steps
                and (global_step % max(1, args.train_frequency)) == 0
                and len(replay) >= args.batch_size
            ):
                batch = replay.sample(args.batch_size)
                last_loss = agent.update(batch)
                if np.isfinite(last_loss):
                    recent_losses.append(last_loss)

            if global_step >= next_log:
                avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
                avg_loss = float(np.mean(recent_losses)) if recent_losses else 0.0
                print(
                    f"[dqn step {global_step:,}] return={avg_return:.2f} loss={avg_loss:.5f} "
                    f"epsilon={epsilon:.3f} replay={len(replay)}"
                )
                next_log += args.log_interval

            if global_step >= next_eval:
                eval_return = evaluate(
                    agent,
                    episodes=args.eval_episodes,
                    seed=args.seed + 10_000,
                    max_steps=args.max_steps,
                    lib_path=args.lib,
                )
                print(f"[dqn eval step {global_step:,}] avg_return={eval_return:.2f}")
                next_eval += args.eval_frequency

            if global_step >= next_ckpt:
                ckpt = args.log_dir / f"dqn_checkpoint_step_{global_step}.pt"
                agent.save(
                    str(ckpt),
                    metadata={
                        "global_step": float(global_step),
                        "obs_dim": float(obs_dim),
                        "action_dim": float(action_dim),
                        "epsilon": float(epsilon),
                        "avg_loss": float(np.mean(recent_losses)) if recent_losses else float(last_loss),
                    },
                )
                next_ckpt += args.checkpoint_frequency
    finally:
        env.close()

    final_path = args.log_dir / "dqn_final.pt"
    agent.save(
        str(final_path),
        metadata={
            "global_step": float(args.total_timesteps),
            "obs_dim": float(obs_dim),
            "action_dim": float(action_dim),
            "epsilon": float(epsilon_at_step(args.total_timesteps, start=args.epsilon_start, end=args.epsilon_end, decay_steps=args.epsilon_decay_steps)),
            "avg_loss": float(np.mean(recent_losses)) if recent_losses else float(last_loss),
        },
    )
    print(f"Saved DQN checkpoint to {final_path}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
