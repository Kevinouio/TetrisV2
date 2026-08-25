"""Train the custom PPO agent on the TetrisV2 environment."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Deque, Optional

from gymnasium.vector import AutoresetMode, SyncVectorEnv
import numpy as np
import torch

from tetris_v2.rl.env import CCTetrisEnv
from tetris_v2.rl.ppo.core import PPOAgent, PPOConfig, RolloutBuffer


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train custom PPO on the TetrisV2 C++ environment.")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--log-interval", type=int, default=10_000)
    parser.add_argument("--eval-frequency", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-frequency", type=int, default=1_000_000)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--lib", type=Path, default=None)
    return parser.parse_args(argv)


def evaluate(agent: PPOAgent, *, episodes: int, seed: int, max_steps: int, lib_path: Optional[Path]) -> float:
    returns = []
    env = CCTetrisEnv(seed=seed, max_steps=max_steps, lib_path=lib_path)
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        action_mask = np.asarray(info["action_mask"], dtype=np.float32)
        done = False
        trunc = False
        total = 0.0
        while not (done or trunc):
            action, _, _ = agent.act(obs, deterministic=True, action_mask=action_mask)
            obs, reward, done, trunc, info = env.step(action)
            action_mask = np.asarray(info["action_mask"], dtype=np.float32)
            total += reward
        returns.append(total)
    env.close()
    return float(np.mean(returns)) if returns else 0.0


def _make_env(seed: int, max_steps: int, lib_path: Optional[Path]):
    def _init():
        return CCTetrisEnv(seed=seed, max_steps=max_steps, lib_path=lib_path)

    return _init


def _bootstrap_truncated_rewards(
    agent: PPOAgent,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    infos: dict,
    gamma: float,
) -> np.ndarray:
    training_rewards = np.asarray(rewards, dtype=np.float32).copy()
    truncated_only = np.logical_and(truncated, np.logical_not(terminated))
    indices = np.flatnonzero(truncated_only)
    if indices.size == 0:
        return training_rewards

    final_observations = infos.get("final_obs")
    if final_observations is None:
        raise RuntimeError("Vector environment omitted final observations for truncated episodes.")
    final_batch = np.stack(
        [np.asarray(final_observations[i], dtype=np.float32) for i in indices]
    )
    training_rewards[indices] += float(gamma) * agent.value_batch(final_batch)
    return training_rewards


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.num_envs <= 0:
        raise SystemExit("--num-envs must be >= 1")
    if args.n_steps <= 0:
        raise SystemExit("--n-steps must be >= 1")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_fns = [_make_env(args.seed + i, args.max_steps, args.lib) for i in range(args.num_envs)]
    env = SyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)

    obs_batch, infos = env.reset(seed=[args.seed + i for i in range(args.num_envs)])
    obs_dim = int(obs_batch.shape[-1])
    action_dim = int(env.single_action_space.n)
    action_masks = np.asarray(infos["action_mask"], dtype=np.float32)

    config = PPOConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=tuple(args.hidden_sizes),
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
    )
    agent = PPOAgent(config)
    buffer = RolloutBuffer(args.n_steps, args.num_envs, obs_dim)

    args.log_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    next_log = args.log_interval
    next_eval = args.eval_frequency
    next_ckpt = args.checkpoint_frequency

    ep_returns = np.zeros(args.num_envs, dtype=np.float32)
    recent_returns: Deque[float] = deque(maxlen=200)
    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0

    try:
        while global_step < args.total_timesteps:
            buffer.reset()
            last_dones = np.zeros(args.num_envs, dtype=np.float32)

            for _ in range(args.n_steps):
                actions, log_probs, values = agent.act_batch(obs_batch, action_mask=action_masks)
                next_obs, rewards, terminated, truncated, infos = env.step(actions)
                dones = np.logical_or(terminated, truncated)
                next_action_masks = np.asarray(infos["action_mask"], dtype=np.float32)
                environment_rewards = np.asarray(rewards, dtype=np.float32)
                training_rewards = _bootstrap_truncated_rewards(
                    agent,
                    environment_rewards,
                    terminated,
                    truncated,
                    infos,
                    args.gamma,
                )

                buffer.add(
                    obs_batch,
                    actions,
                    training_rewards,
                    np.asarray(dones, dtype=np.float32),
                    values,
                    log_probs,
                    action_masks,
                )

                ep_returns += environment_rewards
                for i in range(args.num_envs):
                    if dones[i]:
                        recent_returns.append(float(ep_returns[i]))
                        ep_returns[i] = 0.0

                obs_batch = next_obs
                action_masks = next_action_masks
                last_dones = np.asarray(dones, dtype=np.float32)
                global_step += args.num_envs
                if global_step >= args.total_timesteps:
                    break

            last_values = agent.value_batch(obs_batch)
            buffer.compute_returns_and_advantages(last_values, last_dones, args.gamma, args.gae_lambda)
            update_metrics = agent.update(buffer, batch_size=args.minibatch_size, epochs=args.update_epochs)
            last_policy_loss = float(update_metrics["policy_loss"])
            last_value_loss = float(update_metrics["value_loss"])
            last_entropy = float(update_metrics["entropy"])

            if global_step >= next_log:
                avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
                print(
                    f"[ppo step {global_step:,}] return={avg_return:.2f} "
                    f"policy={update_metrics['policy_loss']:.4f} "
                    f"value={update_metrics['value_loss']:.4f} "
                    f"entropy={update_metrics['entropy']:.4f}"
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
                print(f"[ppo eval step {global_step:,}] avg_return={eval_return:.2f}")
                next_eval += args.eval_frequency

            if global_step >= next_ckpt:
                ckpt = args.log_dir / f"ppo_checkpoint_step_{global_step}.pt"
                agent.save(
                    str(ckpt),
                    metadata={
                        "global_step": float(global_step),
                        "obs_dim": float(obs_dim),
                        "action_dim": float(action_dim),
                        "policy_loss": last_policy_loss,
                        "value_loss": last_value_loss,
                        "entropy": last_entropy,
                    },
                )
                next_ckpt += args.checkpoint_frequency
    finally:
        env.close()

    final_path = args.log_dir / "ppo_final.pt"
    agent.save(
        str(final_path),
        metadata={
            "global_step": float(global_step),
            "obs_dim": float(obs_dim),
            "action_dim": float(action_dim),
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "entropy": last_entropy,
        },
    )
    print(f"Saved PPO checkpoint to {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
