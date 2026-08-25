"""Hybrid DQN training with expert auxiliary losses (DQN-first pipeline)."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np
import torch
from tetris_v2.rl.dqn.core import DQNAgent, DQNConfig, ReplayBuffer
from tetris_v2.rl.dqn.expert_losses import expert_aux_losses
from tetris_v2.rl.env import CCTetrisEnv
from tetris_v2.rl.expert import ExpertRanker
from tetris_v2.rl.expert_dataset import load_dataset_directory


class OnlineExpertBuffer:
    """Ring buffer of online expert-labeled states."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be >= 1")
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.action_mask = np.zeros((self.capacity, action_dim), dtype=np.uint8)
        self.teacher_best_action = np.zeros((self.capacity,), dtype=np.int64)
        self.size = 0
        self.pos = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        teacher_best_action: int,
    ) -> None:
        idx = self.pos
        self.obs[idx] = obs
        self.action_mask[idx] = action_mask
        self.teacher_best_action[idx] = int(teacher_best_action)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        if self.size <= 0:
            raise ValueError("OnlineExpertBuffer is empty")
        idx = rng.integers(0, self.size, size=int(batch_size))
        return {
            "obs": self.obs[idx],
            "action_mask": self.action_mask[idx],
            "teacher_best_action": self.teacher_best_action[idx],
        }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hybrid DQN with expert auxiliary losses.")
    parser.add_argument("--offline-dataset-dir", type=Path, required=True)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
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
    parser.add_argument("--log-dir", type=Path, default=Path("runs/dqn_hybrid"))
    parser.add_argument("--lib", type=Path, default=None)
    parser.add_argument("--expert-think-ms", type=int, default=10)
    parser.add_argument("--online-expert-interval", type=int, default=4)
    parser.add_argument("--online-expert-capacity", type=int, default=50_000)
    parser.add_argument("--lambda-bc-start", type=float, default=1.0)
    parser.add_argument("--lambda-bc-end", type=float, default=0.1)
    parser.add_argument("--lambda-pair-start", type=float, default=0.0)
    parser.add_argument("--lambda-pair-end", type=float, default=0.0)
    parser.add_argument("--lambda-anneal-steps", type=int, default=500_000)
    parser.add_argument("--pairs-per-sample", type=int, default=0)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    return parser.parse_args(argv)


def epsilon_at_step(step: int, *, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return float(end)
    alpha = min(1.0, max(0.0, step / float(decay_steps)))
    return float(start + (end - start) * alpha)


def linear_anneal(step: int, *, start: float, end: float, total_steps: int) -> float:
    if total_steps <= 0:
        return float(end)
    alpha = min(1.0, max(0.0, step / float(total_steps)))
    return float(start + (end - start) * alpha)


def evaluate(agent, *, episodes: int, seed: int, max_steps: int, lib_path: Optional[Path]) -> float:
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


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be >= 1")
    if args.buffer_size <= 0:
        raise SystemExit("--buffer-size must be >= 1")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    offline_dataset = load_dataset_directory(args.offline_dataset_dir)

    env = CCTetrisEnv(seed=args.seed, max_steps=args.max_steps, lib_path=args.lib)
    obs, info = env.reset(seed=args.seed)
    action_mask = np.asarray(info["action_mask"], dtype=np.float32)
    obs_dim = int(obs.shape[-1])
    action_dim = int(env.action_space.n)
    if int(offline_dataset.obs.shape[-1]) != obs_dim or int(offline_dataset.action_mask.shape[-1]) != action_dim:
        env.close()
        raise SystemExit(
            f"Offline expert dataset shape mismatch: dataset=({offline_dataset.obs.shape[-1]},"
            f"{offline_dataset.action_mask.shape[-1]}) env=({obs_dim},{action_dim})"
        )

    if args.init_checkpoint is not None:
        agent, _ = DQNAgent.load(str(args.init_checkpoint), device=args.device)
        if int(agent.config.obs_dim) != obs_dim or int(agent.config.action_dim) != action_dim:
            env.close()
            raise SystemExit(
                f"Init checkpoint shape mismatch: ckpt=({agent.config.obs_dim},{agent.config.action_dim}) "
                f"env=({obs_dim},{action_dim})"
            )
        agent.config.learning_rate = float(args.learning_rate)
        agent.config.gamma = float(args.gamma)
        agent.config.target_sync_interval = int(args.target_sync_interval)
        for group in agent.optimizer.param_groups:
            group["lr"] = float(args.learning_rate)
    else:
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
    online_expert = OnlineExpertBuffer(
        capacity=args.online_expert_capacity,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    ranker = ExpertRanker(env.runtime, think_ms=args.expert_think_ms)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    next_log = args.log_interval
    next_eval = args.eval_frequency
    next_ckpt = args.checkpoint_frequency

    episode_idx = 0
    episode_return = 0.0
    recent_returns: Deque[float] = deque(maxlen=200)
    recent_lines: Deque[int] = deque(maxlen=200)
    recent_td: Deque[float] = deque(maxlen=1000)
    recent_bc: Deque[float] = deque(maxlen=1000)
    recent_pair: Deque[float] = deque(maxlen=1000)
    recent_agree: Deque[float] = deque(maxlen=1000)
    last_metrics: Dict[str, float] = {"td_loss": 0.0, "bc_loss": 0.0, "pair_loss": 0.0, "total_loss": 0.0}
    last_agreement = 0.0

    try:
        for global_step in range(1, args.total_timesteps + 1):
            if args.online_expert_interval > 0 and (global_step % args.online_expert_interval) == 0:
                rank = ranker.rank_current_state(think_ms=args.expert_think_ms)
                online_expert.add(
                    np.asarray(obs, dtype=np.float32),
                    np.asarray(rank.action_mask, dtype=np.float32),
                    int(rank.teacher_best_action),
                )

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
                recent_lines.append(int(info.get("lines", 0)))
                episode_return = 0.0
                episode_idx += 1
                obs, info = env.reset(seed=args.seed + episode_idx)
                action_mask = np.asarray(info["action_mask"], dtype=np.float32)

            if (
                global_step >= args.warmup_steps
                and (global_step % max(1, args.train_frequency)) == 0
                and len(replay) >= args.batch_size
            ):
                rl_batch = replay.sample(args.batch_size)

                offline_count = max(1, args.batch_size // 2)
                online_count = max(1, args.batch_size - offline_count)
                offline_batch = offline_dataset.sample(offline_count, rng)
                if len(online_expert) >= online_count:
                    online_batch = online_expert.sample(online_count, rng)
                else:
                    online_batch = offline_dataset.sample(online_count, rng)

                expert_obs = np.concatenate([offline_batch["obs"], online_batch["obs"]], axis=0)
                expert_mask = np.concatenate(
                    [offline_batch["action_mask"], online_batch["action_mask"]], axis=0
                )
                expert_best = np.concatenate(
                    [offline_batch["teacher_best_action"], online_batch["teacher_best_action"]], axis=0
                )

                expert_obs_t = torch.from_numpy(expert_obs).to(agent.device)
                expert_mask_t = torch.from_numpy(expert_mask).to(agent.device)
                expert_best_t = torch.from_numpy(expert_best).to(agent.device)
                expert_q = agent.online(expert_obs_t)
                bc_loss, pair_loss, agreement = expert_aux_losses(
                    expert_q,
                    expert_best_t,
                    expert_mask_t,
                    rng=rng,
                    pairs_per_sample=max(0, int(args.pairs_per_sample)),
                )

                lambda_bc = linear_anneal(
                    global_step,
                    start=args.lambda_bc_start,
                    end=args.lambda_bc_end,
                    total_steps=args.lambda_anneal_steps,
                )
                lambda_pair = linear_anneal(
                    global_step,
                    start=args.lambda_pair_start,
                    end=args.lambda_pair_end,
                    total_steps=args.lambda_anneal_steps,
                )
                last_metrics = agent.update_combined(
                    rl_batch,
                    bc_loss=bc_loss,
                    pair_loss=pair_loss,
                    lambda_bc=lambda_bc,
                    lambda_pair=lambda_pair,
                )
                last_agreement = float(agreement.detach().item())
                if np.isfinite(last_metrics["td_loss"]):
                    recent_td.append(float(last_metrics["td_loss"]))
                if np.isfinite(last_metrics["bc_loss"]):
                    recent_bc.append(float(last_metrics["bc_loss"]))
                if np.isfinite(last_metrics["pair_loss"]):
                    recent_pair.append(float(last_metrics["pair_loss"]))
                if np.isfinite(last_agreement):
                    recent_agree.append(last_agreement)

            if global_step >= next_log:
                avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
                avg_lines = float(np.mean(recent_lines)) if recent_lines else 0.0
                avg_td = float(np.mean(recent_td)) if recent_td else 0.0
                avg_bc = float(np.mean(recent_bc)) if recent_bc else 0.0
                avg_pair = float(np.mean(recent_pair)) if recent_pair else 0.0
                avg_agree = float(np.mean(recent_agree)) if recent_agree else 0.0
                print(
                    f"[hybrid dqn step {global_step:,}] return={avg_return:.2f} lines={avg_lines:.2f} "
                    f"td={avg_td:.5f} bc={avg_bc:.5f} pair={avg_pair:.5f} agree={avg_agree:.3f} "
                    f"eps={epsilon:.3f} replay={len(replay)} online_expert={len(online_expert)}"
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
                print(f"[hybrid dqn eval step {global_step:,}] avg_return={eval_return:.2f}")
                next_eval += args.eval_frequency

            if global_step >= next_ckpt:
                ckpt = args.log_dir / f"dqn_hybrid_checkpoint_step_{global_step}.pt"
                agent.save(
                    str(ckpt),
                    metadata={
                        "global_step": float(global_step),
                        "obs_dim": float(obs_dim),
                        "action_dim": float(action_dim),
                        "epsilon": float(epsilon),
                        "td_loss": float(last_metrics["td_loss"]),
                        "bc_loss": float(last_metrics["bc_loss"]),
                        "pair_loss": float(last_metrics["pair_loss"]),
                        "teacher_top1_agreement": float(last_agreement),
                    },
                )
                next_ckpt += args.checkpoint_frequency
    finally:
        env.close()

    final_path = args.log_dir / "dqn_hybrid_final.pt"
    agent.save(
        str(final_path),
        metadata={
            "global_step": float(args.total_timesteps),
            "obs_dim": float(obs_dim),
            "action_dim": float(action_dim),
            "epsilon": float(
                epsilon_at_step(
                    args.total_timesteps,
                    start=args.epsilon_start,
                    end=args.epsilon_end,
                    decay_steps=args.epsilon_decay_steps,
                )
            ),
            "td_loss": float(last_metrics["td_loss"]),
            "bc_loss": float(last_metrics["bc_loss"]),
            "pair_loss": float(last_metrics["pair_loss"]),
            "teacher_top1_agreement": float(last_agreement),
        },
    )
    print(f"Saved hybrid DQN checkpoint to {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
