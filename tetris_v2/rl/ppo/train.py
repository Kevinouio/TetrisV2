"""Train the custom PPO agent on the TetrisV2 environment."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any, Deque, Optional

from gymnasium.vector import AutoresetMode, SyncVectorEnv
import numpy as np
import torch

from tetris_v2.rl.env import CCTetrisEnv
from tetris_v2.rl.expert_dataset import (
    discover_shards,
    load_dataset,
    load_dataset_directory,
)
from tetris_v2.rl.ppo.core import PPOAgent, PPOConfig, RolloutBuffer


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train custom PPO on TetrisV2.")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-clip-range", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[512, 256],
        help="Hidden layers for legacy MLP PPO; structured PPO uses 32 channels.",
    )
    parser.add_argument(
        "--network-type",
        choices=("auto", "placement_conv", "mlp"),
        default="auto",
    )
    parser.add_argument(
        "--lr-anneal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Linearly anneal the learning rate to zero.",
    )
    parser.add_argument("--expert-dataset-dir", type=Path, default=None)
    parser.add_argument("--extra-expert-dataset-dir", type=Path, action="append", default=[])
    parser.add_argument("--expert-batch-size", type=int, default=512)
    parser.add_argument("--bc-coef-start", type=float, default=1.0)
    parser.add_argument("--bc-coef-end", type=float, default=0.1)
    parser.add_argument("--bc-anneal-timesteps", type=int, default=1_000_000)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--init-checkpoint", type=Path, default=None)
    checkpoint_group.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--log-interval", type=int, default=10_000)
    parser.add_argument("--eval-frequency", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-frequency", type=int, default=250_000)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--lib", type=Path, default=None)
    return parser.parse_args(argv)


def _evaluate_metrics(
    agent: PPOAgent,
    *,
    episodes: int,
    seed: int,
    max_steps: int,
    lib_path: Optional[Path],
) -> dict[str, float]:
    returns: list[float] = []
    placements: list[int] = []
    lines: list[int] = []
    env = CCTetrisEnv(seed=seed, max_steps=max_steps, lib_path=lib_path)
    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)
            action_mask = np.asarray(info["action_mask"], dtype=np.float32)
            terminated = False
            truncated = False
            total = 0.0
            while not (terminated or truncated):
                action, _, _ = agent.act(obs, deterministic=True, action_mask=action_mask)
                obs, reward, terminated, truncated, info = env.step(action)
                action_mask = np.asarray(info["action_mask"], dtype=np.float32)
                total += float(reward)
            returns.append(total)
            placements.append(int(info["placements"]))
            lines.append(int(info["lines"]))
    finally:
        env.close()
    return {
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "min_placements": float(min(placements)) if placements else 0.0,
        "mean_placements": float(np.mean(placements)) if placements else 0.0,
        "min_lines": float(min(lines)) if lines else 0.0,
        "mean_lines": float(np.mean(lines)) if lines else 0.0,
    }


def evaluate(
    agent: PPOAgent,
    *,
    episodes: int,
    seed: int,
    max_steps: int,
    lib_path: Optional[Path],
) -> float:
    """Compatibility wrapper returning the mean evaluation return."""

    return _evaluate_metrics(
        agent,
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
        lib_path=lib_path,
    )["mean_return"]


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
    indices = np.flatnonzero(np.logical_and(truncated, np.logical_not(terminated)))
    if indices.size == 0:
        return training_rewards

    final_observations = infos.get("final_obs")
    if final_observations is None:
        raise RuntimeError("Vector environment omitted final observations for truncated episodes.")
    final_batch = np.stack(
        [np.asarray(final_observations[index], dtype=np.float32) for index in indices]
    )
    training_rewards[indices] += float(gamma) * agent.value_batch(final_batch)
    return training_rewards


def _linear_schedule(start: float, end: float, step: int, duration: int) -> float:
    if duration <= 0:
        return float(end)
    progress = min(max(float(step) / float(duration), 0.0), 1.0)
    return float(start + progress * (end - start))


def _next_event(step: int, frequency: int) -> float:
    if frequency <= 0:
        return float("inf")
    return float((step // frequency + 1) * frequency)


def _load_expert_data(primary: Path, extras: list[Path]):
    if not extras:
        return load_dataset_directory(primary)
    shards = discover_shards(primary)
    for directory in extras:
        shards.extend(discover_shards(directory))
    return load_dataset(shards)


def _apply_training_config(agent: PPOAgent, args: argparse.Namespace) -> None:
    agent.config.learning_rate = float(args.learning_rate)
    agent.config.gamma = float(args.gamma)
    agent.config.gae_lambda = float(args.gae_lambda)
    agent.config.clip_range = float(args.clip_range)
    agent.config.value_clip_range = (
        None if args.value_clip_range <= 0 else float(args.value_clip_range)
    )
    agent.config.target_kl = None if args.target_kl <= 0 else float(args.target_kl)
    agent.config.entropy_coef = float(args.entropy_coef)
    agent.config.value_coef = float(args.value_coef)
    agent.config.max_grad_norm = float(args.max_grad_norm)


def _checkpoint_metadata(
    *,
    global_step: int,
    obs_dim: int,
    action_dim: int,
    metrics: dict[str, float],
    bc_coef: float,
    learning_rate: float,
    best_score: Optional[tuple[float, ...]],
) -> dict[str, Any]:
    return {
        "global_step": global_step,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "bc_coef": bc_coef,
        "learning_rate": learning_rate,
        "best_eval_score": list(best_score) if best_score is not None else None,
        **metrics,
    }


def run(args: argparse.Namespace) -> int:
    if args.total_timesteps <= 0:
        raise SystemExit("--total-timesteps must be >= 1")
    if args.num_envs <= 0:
        raise SystemExit("--num-envs must be >= 1")
    if args.n_steps <= 0:
        raise SystemExit("--n-steps must be >= 1")
    if args.minibatch_size <= 0:
        raise SystemExit("--minibatch-size must be >= 1")
    if args.expert_batch_size <= 0:
        raise SystemExit("--expert-batch-size must be >= 1")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    expert_rng = np.random.default_rng(args.seed + 1)

    env_fns = [_make_env(args.seed + i, args.max_steps, args.lib) for i in range(args.num_envs)]
    env = SyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    obs_batch, infos = env.reset(seed=[args.seed + i for i in range(args.num_envs)])
    obs_dim = int(obs_batch.shape[-1])
    action_dim = int(env.single_action_space.n)
    action_masks = np.asarray(infos["action_mask"], dtype=np.float32)

    resume_metadata: dict[str, Any] = {}
    if args.resume_checkpoint is not None:
        agent, resume_metadata = PPOAgent.load(
            str(args.resume_checkpoint),
            device=args.device,
            restore_optimizer=True,
        )
        global_step = int(resume_metadata.get("global_step", 0))
    elif args.init_checkpoint is not None:
        agent, _ = PPOAgent.load(
            str(args.init_checkpoint),
            device=args.device,
            restore_optimizer=False,
        )
        if args.network_type != "auto" and agent.network_type != args.network_type:
            env.close()
            raise SystemExit(
                f"Init checkpoint network_type={agent.network_type}, requested={args.network_type}"
            )
        _apply_training_config(agent, args)
        agent.reset_optimizer(args.learning_rate)
        global_step = 0
    else:
        agent = PPOAgent(
            PPOConfig(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=tuple(args.hidden_sizes),
                learning_rate=args.learning_rate,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                value_clip_range=(
                    None if args.value_clip_range <= 0 else args.value_clip_range
                ),
                target_kl=None if args.target_kl <= 0 else args.target_kl,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
                max_grad_norm=args.max_grad_norm,
                network_type=args.network_type,
                device=args.device,
            )
        )
        global_step = 0

    if agent.config.obs_dim != obs_dim or agent.config.action_dim != action_dim:
        env.close()
        raise SystemExit(
            f"Checkpoint shape ({agent.config.obs_dim},{agent.config.action_dim}) is incompatible "
            f"with environment ({obs_dim},{action_dim})"
        )

    expert_dataset = None
    if args.expert_dataset_dir is not None:
        expert_dataset = _load_expert_data(
            args.expert_dataset_dir,
            args.extra_expert_dataset_dir,
        )
        expert_shape = (int(expert_dataset.obs.shape[-1]), int(expert_dataset.action_mask.shape[-1]))
        if expert_shape != (obs_dim, action_dim):
            env.close()
            raise SystemExit(
                f"Expert dataset shape {expert_shape} is incompatible with environment "
                f"({obs_dim}, {action_dim})"
            )

    buffer = RolloutBuffer(args.n_steps, args.num_envs, obs_dim)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    schedule_start_step = global_step if args.resume_checkpoint is not None else 0
    schedule_start_lr = float(agent.optimizer.param_groups[0]["lr"])
    next_log = _next_event(global_step, args.log_interval)
    next_eval = _next_event(global_step, args.eval_frequency)
    next_ckpt = _next_event(global_step, args.checkpoint_frequency)
    ep_returns = np.zeros(args.num_envs, dtype=np.float32)
    recent_returns: Deque[float] = deque(maxlen=200)
    last_metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "explained_variance": 0.0,
        "bc_loss": 0.0,
        "teacher_top1_agreement": 0.0,
        "early_stopped": 0.0,
    }
    stored_best = resume_metadata.get("best_eval_score")
    best_score = tuple(float(value) for value in stored_best) if stored_best else None
    current_bc_coef = (
        _linear_schedule(
            args.bc_coef_start,
            args.bc_coef_end,
            global_step,
            args.bc_anneal_timesteps,
        )
        if expert_dataset is not None
        else 0.0
    )
    current_learning_rate = float(agent.optimizer.param_groups[0]["lr"])

    try:
        while global_step < args.total_timesteps:
            if args.lr_anneal:
                schedule_span = max(1, args.total_timesteps - schedule_start_step)
                schedule_progress = min(
                    max((global_step - schedule_start_step) / float(schedule_span), 0.0),
                    1.0,
                )
                current_learning_rate = schedule_start_lr * (1.0 - schedule_progress)
                agent.set_learning_rate(current_learning_rate)
            else:
                current_learning_rate = schedule_start_lr
                agent.set_learning_rate(current_learning_rate)
            current_bc_coef = (
                _linear_schedule(
                    args.bc_coef_start,
                    args.bc_coef_end,
                    global_step,
                    args.bc_anneal_timesteps,
                )
                if expert_dataset is not None
                else 0.0
            )

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
                    agent.config.gamma,
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
                for index in range(args.num_envs):
                    if dones[index]:
                        recent_returns.append(float(ep_returns[index]))
                        ep_returns[index] = 0.0

                obs_batch = next_obs
                action_masks = next_action_masks
                last_dones = np.asarray(dones, dtype=np.float32)
                global_step += args.num_envs
                if global_step >= args.total_timesteps:
                    break

            last_values = agent.value_batch(obs_batch)
            buffer.compute_returns_and_advantages(
                last_values,
                last_dones,
                agent.config.gamma,
                agent.config.gae_lambda,
            )
            last_metrics = agent.update(
                buffer,
                batch_size=args.minibatch_size,
                epochs=args.update_epochs,
                expert_dataset=expert_dataset,
                expert_batch_size=args.expert_batch_size,
                bc_coef=current_bc_coef,
                expert_rng=expert_rng,
            )

            if global_step >= next_log:
                avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
                print(
                    f"[ppo step {global_step:,}] return={avg_return:.2f} "
                    f"policy={last_metrics['policy_loss']:.4f} "
                    f"value={last_metrics['value_loss']:.4f} "
                    f"entropy={last_metrics['entropy']:.4f} "
                    f"kl={last_metrics['approx_kl']:.5f} "
                    f"clip={last_metrics['clip_fraction']:.3f} "
                    f"ev={last_metrics['explained_variance']:.3f} "
                    f"bc={last_metrics['bc_loss']:.4f}@{current_bc_coef:.3f}"
                )
                next_log = _next_event(global_step, args.log_interval)

            if global_step >= next_eval:
                eval_metrics = _evaluate_metrics(
                    agent,
                    episodes=args.eval_episodes,
                    seed=args.seed + 10_000,
                    max_steps=args.max_steps,
                    lib_path=args.lib,
                )
                print(
                    f"[ppo eval step {global_step:,}] return={eval_metrics['mean_return']:.2f} "
                    f"placements(min/mean)={eval_metrics['min_placements']:.0f}/"
                    f"{eval_metrics['mean_placements']:.1f} "
                    f"lines(min/mean)={eval_metrics['min_lines']:.0f}/"
                    f"{eval_metrics['mean_lines']:.1f}"
                )
                score = (
                    eval_metrics["min_placements"],
                    eval_metrics["min_lines"],
                    eval_metrics["mean_lines"],
                    eval_metrics["mean_return"],
                )
                if best_score is None or score > best_score:
                    best_score = score
                    agent.save(
                        str(args.log_dir / "ppo_best.pt"),
                        metadata={
                            **_checkpoint_metadata(
                                global_step=global_step,
                                obs_dim=obs_dim,
                                action_dim=action_dim,
                                metrics=last_metrics,
                                bc_coef=current_bc_coef,
                                learning_rate=current_learning_rate,
                                best_score=best_score,
                            ),
                            **{f"eval_{key}": value for key, value in eval_metrics.items()},
                        },
                    )
                next_eval = _next_event(global_step, args.eval_frequency)

            if global_step >= next_ckpt:
                agent.save(
                    str(args.log_dir / f"ppo_checkpoint_step_{global_step}.pt"),
                    metadata=_checkpoint_metadata(
                        global_step=global_step,
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        metrics=last_metrics,
                        bc_coef=current_bc_coef,
                        learning_rate=current_learning_rate,
                        best_score=best_score,
                    ),
                )
                next_ckpt = _next_event(global_step, args.checkpoint_frequency)
    finally:
        env.close()

    final_path = args.log_dir / "ppo_final.pt"
    agent.save(
        str(final_path),
        metadata=_checkpoint_metadata(
            global_step=global_step,
            obs_dim=obs_dim,
            action_dim=action_dim,
            metrics=last_metrics,
            bc_coef=current_bc_coef,
            learning_rate=current_learning_rate,
            best_score=best_score,
        ),
    )
    print(f"Saved PPO checkpoint to {final_path}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
