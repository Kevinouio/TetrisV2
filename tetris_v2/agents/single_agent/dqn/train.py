"""Command-line trainer for the native PyTorch DQN agent."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from tetris_v2.agents.single_agent.common import (
    build_advanced_reward_config,
    linear_schedule,
)
from tetris_v2.envs.registration import register_envs
from tetris_v2.envs.wrappers import (
    AdvancedRewardConfig,
    AdvancedRewardWrapper,
    FloatBoardWrapper,
    RewardScaleWrapper,
)
from .models import AgentConfig, DQNAgent, ObservationProcessor, PrioritizedReplayBuffer, ReplayBuffer


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a scratch DQN agent on Tetris.")
    parser.add_argument("--env", choices=("nes", "modern", "custom"), default="nes")
    parser.add_argument("--env-id", help="Gymnasium id to use when --env=custom.")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-sync-interval", type=int, default=1_000)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=int, default=200_000)
    parser.add_argument(
        "--exploration-strategy",
        choices=("epsilon", "boltzmann"),
        default="epsilon",
        help="Choose epsilon-greedy or Boltzmann action sampling.",
    )
    parser.add_argument("--boltzmann-temp-start", type=float, default=1.5, help="Initial Boltzmann temperature.")
    parser.add_argument("--boltzmann-temp-end", type=float, default=0.3, help="Final Boltzmann temperature.")
    parser.add_argument(
        "--boltzmann-temp-decay",
        type=int,
        default=200_000,
        help="Steps over which the Boltzmann temperature is annealed.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="Torch device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--log-interval", type=int, default=5_000)
    parser.add_argument("--eval-frequency", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-frequency", type=int, default=100_000)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/dqn_native"))
    parser.add_argument("--resume-from", type=Path, help="Resume training from a saved .pt checkpoint.")
    parser.add_argument("--rotation-penalty", type=float, default=0.0, help="Per extra rotation penalty .")
    parser.add_argument(
        "--line-clear-reward",
        type=float,
        nargs=4,
        metavar=("SINGLE", "DOUBLE", "TRIPLE", "TETRIS"),
        help="Additional reward (NES score units) per line clear type.",
    )
    parser.add_argument(
        "--step-penalty",
        type=float,
        default=0.0,
        help="Penalty (NES score units) applied every step to discourage stalling.",
    )
    parser.add_argument("--prioritized-replay", action="store_true", help="Enable prioritized experience replay.")
    parser.add_argument("--per-alpha", type=float, default=0.6, help="Prioritized replay alpha.")
    parser.add_argument("--per-beta", type=float, default=0.4, help="Initial prioritized replay beta.")
    parser.add_argument(
        "--per-beta-anneal",
        type=float,
        default=1e-6,
        help="Increment applied to beta after each sample when using prioritized replay.",
    )
    parser.add_argument(
        "--advanced-reward",
        action="store_true",
        help="Enable shaped rewards that penalise holes/height and reward survival.",
    )
    parser.add_argument(
        "--advanced-reward-weight",
        dest="advanced_reward_weights",
        metavar="KEY=VALUE",
        action="append",
        help="Override AdvancedRewardConfig fields (requires --advanced-reward).",
    )
    return parser.parse_args(argv)


def _resolve_env_id(args: argparse.Namespace) -> str:
    if args.env == "nes":
        return "KevinNES/Tetris-v0"
    if args.env == "modern":
        return "KevinModern/Tetris-v0"
    if not args.env_id:
        raise SystemExit("--env-id is required when --env=custom.")
    return args.env_id


def _make_env(
    env_id: str,
    *,
    reward_scale: float,
    env_kwargs: Optional[Dict[str, Any]] = None,
    advanced_reward: Optional[AdvancedRewardConfig] = None,
) -> gym.Env:
    kwargs = dict(env_kwargs or {})
    env = gym.make(env_id, **kwargs)
    if advanced_reward is not None:
        env = AdvancedRewardWrapper(env, config=advanced_reward)
    env = FloatBoardWrapper(env)
    if reward_scale != 1.0:
        env = RewardScaleWrapper(env, scale=reward_scale)
    return env


def _evaluate_policy(
    agent: DQNAgent,
    processor: ObservationProcessor,
    env_id: str,
    *,
    episodes: int,
    seed: Optional[int],
    reward_scale: float,
    env_kwargs: Optional[Dict[str, Any]] = None,
    advanced_reward: Optional[AdvancedRewardConfig] = None,
) -> Tuple[float, float]:
    env = _make_env(
        env_id,
        reward_scale=reward_scale,
        env_kwargs=env_kwargs,
        advanced_reward=advanced_reward,
    )
    returns: list[float] = []
    scores: list[float] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=None if seed is None else seed + episode)
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
        returns.append(total_reward)
        scores.append(last_score)
    env.close()
    return float(np.mean(returns)), float(np.mean(scores))


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    register_envs()
    np.random.seed(args.seed)
    env_id = _resolve_env_id(args)
    env_kwargs: Dict[str, Any] = {}
    if env_id.startswith("KevinNES"):
        if args.rotation_penalty:
            env_kwargs["rotation_penalty"] = args.rotation_penalty
        if args.step_penalty:
            env_kwargs["step_penalty"] = args.step_penalty
        if args.line_clear_reward:
            env_kwargs["line_clear_reward"] = args.line_clear_reward
    try:
        advanced_reward_cfg = build_advanced_reward_config(
            args.advanced_reward,
            args.advanced_reward_weights,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    env = _make_env(
        env_id,
        reward_scale=args.reward_scale,
        env_kwargs=env_kwargs,
        advanced_reward=advanced_reward_cfg,
    )
    processor = ObservationProcessor(env.observation_space)
    if args.prioritized_replay:
        buffer = PrioritizedReplayBuffer(
            capacity=args.buffer_size,
            obs_dim=processor.flat_dim,
            alpha=args.per_alpha,
            beta=args.per_beta,
            beta_increment=args.per_beta_anneal,
        )
    else:
        buffer = ReplayBuffer(capacity=args.buffer_size, obs_dim=processor.flat_dim)
    rng = np.random.default_rng(args.seed)

    if not args.hidden_sizes:
        raise SystemExit("--hidden-sizes must contain at least one layer size.")

    if args.resume_from:
        agent, metadata = DQNAgent.load(str(args.resume_from), device=args.device)
        if agent.obs_dim != processor.flat_dim or agent.action_dim != env.action_space.n:
            raise SystemExit("Checkpoint is incompatible with the selected environment.")
        start_step = int(metadata.get("global_step", 0))
        best_eval = float(metadata.get("best_eval_return", float("-inf")))
    else:
        config = AgentConfig(
            obs_dim=processor.flat_dim,
            action_dim=env.action_space.n,
            hidden_sizes=tuple(args.hidden_sizes),
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            target_sync_interval=args.target_sync_interval,
            max_grad_norm=args.max_grad_norm,
            device=args.device,
            board_dim=processor.board_dim,
            board_shape=processor.board_shape,
        )
        agent = DQNAgent(config)
        start_step = 0
        best_eval = float("-inf")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    obs, _ = env.reset(seed=args.seed)
    flat = processor.flatten(obs)
    episodic_return = 0.0
    episodic_length = 0
    episodic_score = 0.0
    episodes_run = 0
    recent_returns: Deque[float] = deque(maxlen=100)
    recent_lengths: Deque[int] = deque(maxlen=100)
    recent_losses: Deque[float] = deque(maxlen=1_000)

    global_step = start_step
    if global_step >= args.total_timesteps:
        print("Checkpoint already reached requested total timesteps; exiting.")
        env.close()
        return 0

    while global_step < args.total_timesteps:
        epsilon = linear_schedule(args.epsilon_start, args.epsilon_end, args.epsilon_decay, global_step)
        temperature = None
        if args.exploration_strategy == "epsilon":
            action = agent.act(flat, epsilon=epsilon, rng=rng, strategy="epsilon")
        else:
            temp_value = linear_schedule(
                args.boltzmann_temp_start,
                args.boltzmann_temp_end,
                args.boltzmann_temp_decay,
                global_step,
            )
            temperature = max(temp_value, 1e-3)
            action = agent.act(
                flat,
                epsilon=epsilon,
                temperature=temperature,
                strategy="boltzmann",
                rng=rng,
            )
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_flat = processor.flatten(next_obs)
        buffer.add(flat, action, float(reward), next_flat, done)

        episodic_return += reward
        episodic_length += 1
        episodic_score = float(info.get("score", episodic_score))
        flat = next_flat
        global_step += 1

        should_learn = (
            buffer.size >= args.batch_size
            and global_step >= args.learning_starts
            and global_step % args.train_frequency == 0
        )
        if should_learn:
            for _ in range(args.gradient_steps):
                batch = buffer.sample(args.batch_size)
                loss, td_errors = agent.update(batch)
                recent_losses.append(loss)
                if args.prioritized_replay:
                    buffer.update_priorities(batch["indices"], np.abs(td_errors))

        if done:
            recent_returns.append(episodic_return)
            recent_lengths.append(episodic_length)
            episodes_run += 1
            if episodes_run % 500 == 0:
                print(
                    f"[episode {episodes_run:,}] return={episodic_return:.2f} "
                    f"len={episodic_length} score={episodic_score:.1f}"
                )
            obs, _ = env.reset()
            flat = processor.flatten(obs)
            episodic_return = 0.0
            episodic_length = 0
            episodic_score = 0.0

        if args.log_interval and global_step % args.log_interval == 0:
            avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
            avg_len = float(np.mean(recent_lengths)) if recent_lengths else 0.0
            avg_loss = float(np.mean(recent_losses)) if recent_losses else 0.0
            if temperature is None:
                explore_msg = f"eps={epsilon:.3f}"
            else:
                explore_msg = f"temp={temperature:.3f} eps-mix={epsilon:.3f}"
            print(
                f"[step {global_step:,} | episode {episodes_run:,}] {explore_msg} "
                f"return={avg_return:.1f} len={avg_len:.1f} "
                f"loss={avg_loss:.4f} buffer={buffer.size:,}"
            )

        if args.eval_frequency and global_step % args.eval_frequency == 0:
            eval_return, eval_score = _evaluate_policy(
                agent,
                processor,
                env_id,
                episodes=args.eval_episodes,
                seed=args.seed,
                reward_scale=args.reward_scale,
                env_kwargs=env_kwargs,
                advanced_reward=advanced_reward_cfg,
            )
            print(
                f"[eval step {global_step:,}] avg_return={eval_return:.1f} "
                f"avg_score={eval_score:.1f}"
            )
            if eval_return > best_eval:
                best_eval = eval_return
                best_path = args.log_dir / "best_model.pt"
                agent.save(str(best_path), metadata={"global_step": global_step, "best_eval_return": best_eval})

        if args.checkpoint_frequency and global_step % args.checkpoint_frequency == 0:
            ckpt_path = args.log_dir / f"checkpoint_step_{global_step}.pt"
            agent.save(str(ckpt_path), metadata={"global_step": global_step, "best_eval_return": best_eval})

    final_path = args.log_dir / "final_model.pt"
    agent.save(str(final_path), metadata={"global_step": global_step, "best_eval_return": best_eval})
    env.close()
    print(f"Training complete. Final checkpoint saved to {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
