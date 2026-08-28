"""Offline-to-online training for TetrisV2's discrete Flow-DQN agent."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np
import torch

from tetris_v2.rl.actions import PLACEMENT_ACTION_DIM
from tetris_v2.rl.env import CCTetrisEnv
from tetris_v2.rl.expert_dataset import load_transition_dataset_directory
from tetris_v2.rl.flow_dqn.core import (
    SOURCE_NOISE_STD,
    FlowDQNAgent,
    FlowDQNConfig,
    FlowReplayBuffer,
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train discrete Flow-DQN offline, online, or offline-to-online."
    )
    parser.add_argument("--offline-dataset-dir", type=Path, default=None)
    parser.add_argument("--offline-updates", type=int, default=0)
    parser.add_argument("--online-timesteps", type=int, default=1_000_000)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--flow-learning-rate", type=float, default=3e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument(
        "--distill-q-coef",
        "--distill-coef",
        dest="distill_q_coef",
        type=float,
        default=1.0,
        help="One-step distillation coefficient relative to normalized Q guidance.",
    )
    parser.add_argument(
        "--q-guidance-coef",
        type=float,
        default=1.0,
        help="Extra multiplier for Q guidance; the paper-aligned value is 1.0.",
    )
    parser.add_argument(
        "--flow-t0-ce-coef",
        type=float,
        default=1.0,
        help="Masked executed-action CE applied to the flow field at t=0.",
    )
    parser.add_argument(
        "--actor-bc-coef",
        type=float,
        default=1.0,
        help="Masked executed-action CE applied directly to the one-step actor.",
    )
    parser.add_argument(
        "--action-logit-scale",
        type=float,
        default=10.0,
        help="Scale from flow-map values to masked categorical training logits.",
    )
    parser.add_argument(
        "--normalized-q",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--exploration-temperature", type=float, default=1.0)
    parser.add_argument(
        "--online-offline-fraction",
        type=float,
        default=None,
        help="Expert-transition fraction in online update batches (default: 0.5 with a dataset).",
    )
    parser.add_argument("--source-noise-std", type=float, default=SOURCE_NOISE_STD)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=4_000)
    parser.add_argument("--log-interval", type=int, default=1_000)
    parser.add_argument("--eval-frequency", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-frequency", type=int, default=50_000)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/flow_dqn"))
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--lib", type=Path, default=None)
    return parser.parse_args(argv)


def _offline_batch(dataset, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    sample = dataset.sample(batch_size, rng)
    return {
        "obs": sample["obs"],
        "action_masks": sample["action_mask"],
        "actions": sample["executed_action"],
        "rewards": sample["reward"],
        "next_obs": sample["next_obs"],
        "next_action_masks": sample["next_action_mask"],
        "terminated": sample["terminated"],
        "truncated": sample["truncated"],
    }


def _mixed_online_batch(
    replay: FlowReplayBuffer,
    dataset,
    batch_size: int,
    offline_fraction: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    offline_count = int(round(batch_size * offline_fraction)) if dataset is not None else 0
    offline_count = min(batch_size, max(0, offline_count))
    online_count = batch_size - offline_count
    parts = []
    if online_count:
        parts.append(replay.sample(online_count, rng))
    if offline_count:
        parts.append(_offline_batch(dataset, offline_count, rng))
    if len(parts) == 1:
        return parts[0]

    permutation = rng.permutation(batch_size)
    return {
        key: np.concatenate([part[key] for part in parts], axis=0)[permutation]
        for key in parts[0]
    }


def evaluate(
    agent: FlowDQNAgent,
    *,
    episodes: int,
    seed: int,
    max_steps: int,
    lib_path: Optional[Path],
) -> float:
    returns = []
    env = CCTetrisEnv(seed=seed, max_steps=max_steps, lib_path=lib_path)
    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)
            action_mask = np.asarray(info["action_mask"], dtype=np.float32)
            episode_return = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.select_action(
                    obs,
                    deterministic=True,
                    action_mask=action_mask,
                )
                obs, reward, terminated, truncated, info = env.step(action)
                action_mask = np.asarray(info["action_mask"], dtype=np.float32)
                episode_return += float(reward)
            returns.append(episode_return)
    finally:
        env.close()
    return float(np.mean(returns)) if returns else 0.0


def _config_from_args(args: argparse.Namespace, obs_dim: int, action_dim: int) -> FlowDQNConfig:
    return FlowDQNConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        channels=args.channels,
        critic_learning_rate=args.critic_learning_rate,
        flow_learning_rate=args.flow_learning_rate,
        actor_learning_rate=args.actor_learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        flow_steps=args.flow_steps,
        distill_coef=args.distill_q_coef,
        q_guidance_coef=args.q_guidance_coef,
        flow_t0_ce_coef=args.flow_t0_ce_coef,
        actor_bc_coef=args.actor_bc_coef,
        action_logit_scale=args.action_logit_scale,
        normalized_q=args.normalized_q,
        policy_temperature=args.policy_temperature,
        source_noise_std=args.source_noise_std,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
    )


def _load_or_create_agent(
    args: argparse.Namespace,
    obs_dim: int,
    action_dim: int,
) -> FlowDQNAgent:
    if args.init_checkpoint is not None and args.resume_checkpoint is not None:
        raise ValueError("Use only one of --init-checkpoint and --resume-checkpoint.")
    checkpoint = args.resume_checkpoint or args.init_checkpoint
    if checkpoint is None:
        return FlowDQNAgent(_config_from_args(args, obs_dim, action_dim))

    agent, _ = FlowDQNAgent.load(checkpoint, device=args.device)
    if agent.config.obs_dim != obs_dim or agent.config.action_dim != action_dim:
        raise ValueError("Checkpoint observation/action schema does not match the training data.")
    if args.init_checkpoint is not None:
        # Initialization transfers learned parameters but starts a fresh run.
        config = _config_from_args(args, obs_dim, action_dim)
        initialized = FlowDQNAgent(config)
        initialized.actor.load_state_dict(agent.actor.state_dict())
        initialized.flow.load_state_dict(agent.flow.state_dict())
        initialized.critic1.load_state_dict(agent.critic1.state_dict())
        initialized.critic2.load_state_dict(agent.critic2.state_dict())
        initialized.target_critic1.load_state_dict(agent.target_critic1.state_dict())
        initialized.target_critic2.load_state_dict(agent.target_critic2.state_dict())
        return initialized
    return agent


def _checkpoint_metadata(agent: FlowDQNAgent, metrics: Dict[str, float]) -> Dict[str, object]:
    return {
        "update_steps": agent.update_steps,
        "offline_update_steps": agent.offline_update_steps,
        "online_update_steps": agent.online_update_steps,
        "environment_steps": agent.environment_steps,
        **metrics,
    }


def run(args: argparse.Namespace) -> int:
    if args.offline_updates > 0 and args.offline_dataset_dir is None:
        raise ValueError("--offline-updates requires --offline-dataset-dir with schema-v3 transitions.")
    if args.offline_updates < 0 or args.online_timesteps < 0:
        raise ValueError("Training step counts cannot be negative.")
    if args.offline_updates == 0 and args.online_timesteps == 0:
        raise ValueError("At least one offline update or online timestep is required.")

    offline_fraction = (
        (0.5 if args.offline_dataset_dir is not None else 0.0)
        if args.online_offline_fraction is None
        else float(args.online_offline_fraction)
    )
    if not 0.0 <= offline_fraction <= 1.0:
        raise ValueError("--online-offline-fraction must be between 0 and 1.")
    if offline_fraction > 0.0 and args.offline_dataset_dir is None:
        raise ValueError("A positive --online-offline-fraction requires --offline-dataset-dir.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.online_timesteps > 0:
        online_batch_size = args.batch_size - int(round(args.batch_size * offline_fraction))
        required_replay = max(args.warmup_steps, online_batch_size)
        if args.buffer_size < required_replay:
            raise ValueError(
                "--buffer-size must be at least max(--warmup-steps, online batch share); "
                "otherwise online updates can never begin."
            )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dataset = None
    obs_dim = 0
    action_dim = PLACEMENT_ACTION_DIM
    if args.offline_dataset_dir is not None:
        dataset = load_transition_dataset_directory(args.offline_dataset_dir)
        obs_dim = int(dataset.obs.shape[1])
        action_dim = int(dataset.action_mask.shape[1])

    env = None
    obs = None
    action_mask = None
    if args.online_timesteps > 0:
        env = CCTetrisEnv(seed=args.seed, max_steps=args.max_steps, lib_path=args.lib)
        obs, info = env.reset(seed=args.seed)
        action_mask = np.asarray(info["action_mask"], dtype=np.float32)
        env_obs_dim = int(obs.shape[0])
        env_action_dim = int(env.action_space.n)
        if obs_dim and (env_obs_dim != obs_dim or env_action_dim != action_dim):
            env.close()
            raise ValueError("Offline transition schema does not match the online environment.")
        obs_dim, action_dim = env_obs_dim, env_action_dim

    if obs_dim == 0:
        raise ValueError("Could not infer the observation schema.")

    agent = _load_or_create_agent(args, obs_dim, action_dim)
    replay = (
        FlowReplayBuffer(args.buffer_size, obs_dim, action_dim)
        if args.online_timesteps > 0
        else None
    )
    args.log_dir.mkdir(parents=True, exist_ok=True)
    latest_metrics: Dict[str, float] = {}

    try:
        if args.offline_updates > 0:
            assert dataset is not None
            for update in range(1, args.offline_updates + 1):
                latest_metrics = agent.update(
                    _offline_batch(dataset, args.batch_size, rng), source="offline"
                )
                cumulative_update = agent.offline_update_steps
                if args.log_interval > 0 and cumulative_update % args.log_interval == 0:
                    print(
                        f"[flow-dqn offline +{update:,}/{args.offline_updates:,} "
                        f"(total {cumulative_update:,})] "
                        f"critic={latest_metrics['critic_loss']:.5f} "
                        f"flow={latest_metrics['flow_loss']:.5f} "
                        f"actor={latest_metrics['actor_loss']:.5f} "
                        f"t0_acc={latest_metrics['flow_t0_agreement']:.3f} "
                        f"bc_acc={latest_metrics['actor_executed_agreement']:.3f} "
                        f"entropy={latest_metrics['actor_masked_entropy']:.3f}"
                    )
                if (
                    args.checkpoint_frequency > 0
                    and cumulative_update % args.checkpoint_frequency == 0
                ):
                    agent.save(
                        args.log_dir / f"flow_dqn_offline_update_{cumulative_update}.pt",
                        metadata=_checkpoint_metadata(agent, latest_metrics),
                    )

        if offline_fraction == 0.0:
            dataset = None

        if args.online_timesteps > 0:
            assert env is not None and replay is not None and obs is not None and action_mask is not None
            episode_return = 0.0
            episode_index = 0
            recent_returns: Deque[float] = deque(maxlen=200)
            for online_step in range(1, args.online_timesteps + 1):
                legal = np.flatnonzero(action_mask > 0.5)
                if len(replay) < args.warmup_steps:
                    action = int(rng.choice(legal))
                else:
                    action = agent.select_action(
                        obs,
                        deterministic=False,
                        temperature=args.exploration_temperature,
                        action_mask=action_mask,
                    )
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_action_mask = np.asarray(info["action_mask"], dtype=np.float32)
                replay.add(
                    obs,
                    action_mask,
                    action,
                    float(reward),
                    next_obs,
                    next_action_mask,
                    bool(terminated),
                    bool(truncated),
                )
                agent.environment_steps += 1
                cumulative_step = agent.environment_steps
                episode_return += float(reward)
                obs = next_obs
                action_mask = next_action_mask

                if terminated or truncated:
                    recent_returns.append(episode_return)
                    episode_return = 0.0
                    episode_index += 1
                    obs, info = env.reset(seed=args.seed + episode_index)
                    action_mask = np.asarray(info["action_mask"], dtype=np.float32)

                if (
                    len(replay) >= max(online_batch_size, args.warmup_steps)
                    and cumulative_step % max(1, args.train_frequency) == 0
                ):
                    for _ in range(max(1, args.gradient_steps)):
                        latest_metrics = agent.update(
                            _mixed_online_batch(
                                replay,
                                dataset,
                                args.batch_size,
                                offline_fraction,
                                rng,
                            ),
                            source="online",
                        )

                if args.log_interval > 0 and cumulative_step % args.log_interval == 0:
                    mean_return = float(np.mean(recent_returns)) if recent_returns else 0.0
                    print(
                        f"[flow-dqn online +{online_step:,}/{args.online_timesteps:,} "
                        f"(total {cumulative_step:,})] "
                        f"return={mean_return:.2f} replay={len(replay):,} "
                        f"critic={latest_metrics.get('critic_loss', 0.0):.5f} "
                        f"actor={latest_metrics.get('actor_loss', 0.0):.5f} "
                        f"t0_acc={latest_metrics.get('flow_t0_agreement', 0.0):.3f} "
                        f"bc_acc={latest_metrics.get('actor_executed_agreement', 0.0):.3f} "
                        f"entropy={latest_metrics.get('actor_masked_entropy', 0.0):.3f}"
                    )
                if args.eval_frequency > 0 and cumulative_step % args.eval_frequency == 0:
                    eval_return = evaluate(
                        agent,
                        episodes=args.eval_episodes,
                        seed=args.seed + 100_000,
                        max_steps=args.max_steps,
                        lib_path=args.lib,
                    )
                    print(f"[flow-dqn eval {cumulative_step:,}] avg_return={eval_return:.2f}")
                if (
                    args.checkpoint_frequency > 0
                    and cumulative_step % args.checkpoint_frequency == 0
                ):
                    agent.save(
                        args.log_dir / f"flow_dqn_online_step_{cumulative_step}.pt",
                        metadata=_checkpoint_metadata(agent, latest_metrics),
                    )
    finally:
        if env is not None:
            env.close()

    final_path = args.log_dir / "flow_dqn_final.pt"
    agent.save(final_path, metadata=_checkpoint_metadata(agent, latest_metrics))
    print(f"Saved Flow-DQN checkpoint to {final_path}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate", "main", "parse_args", "run"]
