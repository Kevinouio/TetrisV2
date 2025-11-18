"""Command-line trainer for the native PyTorch DQN agent."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np

from tetris_v2.agents.single_agent.common import (
    build_agent_reward_config,
    build_environment_reward_config,
    linear_schedule,
)
from tetris_v2.envs.curriculum import CurriculumEpisodeWrapper, CurriculumManager, CurriculumStage, apply_overrides, build_default_curriculum
from tetris_v2.envs.registration import register_envs
from tetris_v2.envs.state_presets import BoardPresetLibrary, load_board_presets
from tetris_v2.envs.wrappers import (
    AgentRewardConfig,
    EnvironmentRewardConfig,
    FloatBoardWrapper,
    RewardScaleWrapper,
    UniversalRewardWrapper,
)
from .models import AgentConfig, DQNAgent, ObservationProcessor, PrioritizedReplayBuffer, ReplayBuffer


@dataclass
class _StageRuntime:
    stage: Optional[CurriculumStage]
    agent_reward: AgentRewardConfig
    env_reward: EnvironmentRewardConfig
    env_kwargs: Dict[str, Any]


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a scratch DQN agent on Tetris.")
    parser.add_argument("--env", choices=("nes", "modern", "custom"), default="nes")
    parser.add_argument("--env-id", help="Gymnasium id to use when --env=custom.")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=1, help="How many parallel env workers to launch.")
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
    parser.add_argument(
        "--double-dqn",
        dest="double_dqn",
        action="store_true",
        default=True,
        help="Enable Double DQN target selection (on by default).",
    )
    parser.add_argument(
        "--no-double-dqn",
        dest="double_dqn",
        action="store_false",
        help="Disable Double DQN; fall back to vanilla DQN targets.",
    )
    parser.add_argument(
        "--dueling",
        dest="dueling",
        action="store_true",
        default=True,
        help="Use dueling value/advantage heads (default).",
    )
    parser.add_argument(
        "--no-dueling",
        dest="dueling",
        action="store_false",
        help="Use a single shared head (vanilla DQN).",
    )
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
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable staged curriculum learning (line clears -> survival -> full game).",
    )
    parser.add_argument(
        "--board-preset-file",
        type=Path,
        help="JSON file containing named board presets for curriculum stages.",
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
    env_kind: str,
    agent_reward: Optional[AgentRewardConfig] = None,
    env_reward: Optional[EnvironmentRewardConfig] = None,
    curriculum_stage: Optional[CurriculumStage] = None,
    preset_library: Optional[BoardPresetLibrary] = None,
) -> gym.Env:
    kwargs = dict(env_kwargs or {})
    env = gym.make(env_id, **kwargs)
    if agent_reward is not None or env_reward is not None:
        env = UniversalRewardWrapper(
            env,
            env_kind=env_kind,
            agent_config=agent_reward,
            env_config=env_reward,
        )
    env = FloatBoardWrapper(env)
    if reward_scale != 1.0:
        env = RewardScaleWrapper(env, scale=reward_scale)
    if curriculum_stage is not None:
        env = CurriculumEpisodeWrapper(env, stage=curriculum_stage, presets=preset_library)
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
    env_kind: str,
    agent_reward: Optional[AgentRewardConfig] = None,
    env_reward: Optional[EnvironmentRewardConfig] = None,
    curriculum_stage: Optional[CurriculumStage] = None,
    preset_library: Optional[BoardPresetLibrary] = None,
) -> Tuple[float, float]:
    env = _make_env(
        env_id,
        reward_scale=reward_scale,
        env_kwargs=env_kwargs,
        env_kind=env_kind,
        agent_reward=agent_reward,
        env_reward=env_reward,
        curriculum_stage=curriculum_stage,
        preset_library=preset_library,
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


def _make_env_factory(
    env_id: str,
    *,
    env_kwargs: Optional[Dict[str, Any]],
    reward_scale: float,
    env_kind: str,
    agent_reward: Optional[AgentRewardConfig],
    env_reward: Optional[EnvironmentRewardConfig],
    preset_library: Optional[BoardPresetLibrary],
    curriculum_stage: Optional[CurriculumStage],
    seed: Optional[int],
) -> Callable[[], gym.Env]:
    kwargs = dict(env_kwargs or {})

    def _init():
        register_envs()
        env = gym.make(env_id, **kwargs)
        if agent_reward is not None or env_reward is not None:
            env = UniversalRewardWrapper(
                env,
                env_kind=env_kind,
                agent_config=agent_reward,
                env_config=env_reward,
            )
        env = FloatBoardWrapper(env)
        if reward_scale != 1.0:
            env = RewardScaleWrapper(env, scale=reward_scale)
        if curriculum_stage is not None:
            env = CurriculumEpisodeWrapper(env, stage=curriculum_stage, presets=preset_library)
        if seed is not None:
            env.reset(seed=seed)
        return env

    return _init


def _build_stage_runtime(
    stage: Optional[CurriculumStage],
    base_agent: AgentRewardConfig,
    base_env: EnvironmentRewardConfig,
    base_env_kwargs: Dict[str, Any],
) -> _StageRuntime:
    env_kwargs = dict(base_env_kwargs)
    agent_cfg = base_agent
    env_cfg = base_env
    if stage is not None:
        if stage.agent_reward_overrides:
            agent_cfg = apply_overrides(base_agent, stage.agent_reward_overrides)
        if stage.env_reward_overrides:
            env_cfg = apply_overrides(base_env, stage.env_reward_overrides)
        env_kwargs.update(stage.env_kwargs)
    return _StageRuntime(stage=stage, agent_reward=agent_cfg, env_reward=env_cfg, env_kwargs=env_kwargs)


def _split_observations(obs_batch: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    num_envs = len(next(iter(obs_batch.values())))
    result: List[Dict[str, Any]] = []
    for idx in range(num_envs):
        obs = {key: np.array(value[idx]) for key, value in obs_batch.items()}
        result.append(obs)
    return result


def _flatten_batch(processor: ObservationProcessor, obs_batch: Dict[str, np.ndarray]) -> np.ndarray:
    obs_list = _split_observations(obs_batch)
    flats = [processor.flatten(observation) for observation in obs_list]
    return np.stack(flats, axis=0)


def _split_infos(infos: Any, num_envs: int) -> List[Dict[str, Any]]:
    if isinstance(infos, (list, tuple)):
        items = list(infos)
        if len(items) == num_envs:
            return [dict(item) for item in items]
    if isinstance(infos, dict):
        split: List[Dict[str, Any]] = [dict() for _ in range(num_envs)]
        for key, value in infos.items():
            if key in {"final_observation", "final_info"} and isinstance(value, dict):
                for idx in range(num_envs):
                    split[idx][key] = {kk: np.array(vv)[idx] for kk, vv in value.items()}
                continue
            arr = list(value) if isinstance(value, (list, tuple)) else np.array(value)
            for idx in range(num_envs):
                split[idx][key] = arr[idx]
        return split
    return [dict() for _ in range(num_envs)]


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if args.num_envs <= 0:
        raise SystemExit("--num-envs must be at least 1.")
    register_envs()
    np.random.seed(args.seed)
    env_id = _resolve_env_id(args)
    env_kind = args.env.lower()
    if env_kind == "custom":
        lowered = env_id.lower()
        if "modern" in lowered:
            env_kind = "modern"
        elif "nes" in lowered:
            env_kind = "nes"
    base_env_kwargs: Dict[str, Any] = {}
    if env_id.startswith("KevinNES"):
        if args.rotation_penalty:
            base_env_kwargs["rotation_penalty"] = args.rotation_penalty
        if args.step_penalty:
            base_env_kwargs["step_penalty"] = args.step_penalty
        if args.line_clear_reward:
            base_env_kwargs["line_clear_reward"] = args.line_clear_reward
    try:
        base_agent_reward_cfg = build_agent_reward_config(args.agent_reward_weights)
        base_env_reward_cfg = build_environment_reward_config(args.environment_reward_weights)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    stage_manager: Optional[CurriculumManager] = None
    try:
        preset_library = load_board_presets(args.board_preset_file) if args.board_preset_file else None
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    stage_runtime = _StageRuntime(
        stage=None,
        agent_reward=base_agent_reward_cfg,
        env_reward=base_env_reward_cfg,
        env_kwargs=dict(base_env_kwargs),
    )
    if args.curriculum:
        stage_manager = build_default_curriculum(args.total_timesteps)
        active_stage = stage_manager.stage_for_step(0)
        stage_runtime = _build_stage_runtime(
            active_stage,
            base_agent_reward_cfg,
            base_env_reward_cfg,
            base_env_kwargs,
        )

    reference_env = _make_env(
        env_id,
        reward_scale=args.reward_scale,
        env_kwargs=stage_runtime.env_kwargs,
        env_kind=env_kind,
        agent_reward=stage_runtime.agent_reward,
        env_reward=stage_runtime.env_reward,
        curriculum_stage=stage_runtime.stage,
        preset_library=preset_library,
    )
    processor = ObservationProcessor(reference_env.observation_space)
    action_dim = reference_env.action_space.n
    reference_env.close()

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
        if agent.obs_dim != processor.flat_dim or agent.action_dim != action_dim:
            raise SystemExit("Checkpoint is incompatible with the selected environment.")
        start_step = int(metadata.get("global_step", 0))
        best_eval = float(metadata.get("best_eval_return", float("-inf")))
    else:
        config = AgentConfig(
            obs_dim=processor.flat_dim,
            action_dim=reference_env.action_space.n,
            hidden_sizes=tuple(args.hidden_sizes),
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            target_sync_interval=args.target_sync_interval,
            max_grad_norm=args.max_grad_norm,
            device=args.device,
            board_dim=processor.board_dim,
            board_shape=processor.board_shape,
            use_double_q=args.double_dqn,
            use_dueling=args.dueling,
        )
        agent = DQNAgent(config)
        start_step = 0
        best_eval = float("-inf")

    args.log_dir.mkdir(parents=True, exist_ok=True)

    def _build_vector_env(stage_rt: _StageRuntime):
        env_fns = []
        for idx in range(args.num_envs):
            seed_val = None if args.seed is None else args.seed + idx
            env_fns.append(
                _make_env_factory(
                    env_id,
                env_kwargs=stage_rt.env_kwargs,
                reward_scale=args.reward_scale,
                env_kind=env_kind,
                agent_reward=stage_rt.agent_reward,
                env_reward=stage_rt.env_reward,
                preset_library=preset_library,
                curriculum_stage=stage_rt.stage,
                seed=seed_val,
            )
            )
        vector = AsyncVectorEnv(env_fns)
        seed_seq = None if args.seed is None else [args.seed + idx for idx in range(args.num_envs)]
        obs_batch, _ = vector.reset(seed=seed_seq)
        return vector, obs_batch

    train_env, obs_batch = _build_vector_env(stage_runtime)
    flat_batch = _flatten_batch(processor, obs_batch)
    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    episode_scores = np.zeros(args.num_envs, dtype=np.float32)
    episodes_run = 0
    recent_returns: Deque[float] = deque(maxlen=100)
    recent_lengths: Deque[int] = deque(maxlen=100)
    recent_losses: Deque[float] = deque(maxlen=1_000)

    global_step = start_step
    next_log = args.log_interval if args.log_interval else None
    next_eval = args.eval_frequency if args.eval_frequency else None
    next_checkpoint = args.checkpoint_frequency if args.checkpoint_frequency else None
    if global_step >= args.total_timesteps:
        print("Checkpoint already reached requested total timesteps; exiting.")
        train_env.close()
        return 0

    while global_step < args.total_timesteps:
        if stage_manager is not None:
            desired_stage = stage_manager.stage_for_step(global_step)
            current_name = stage_runtime.stage.name if stage_runtime.stage else None
            if desired_stage.name != current_name:
                print(f"[curriculum] Switching to stage '{desired_stage.name}' at step {global_step:,}.")
                train_env.close()
                stage_runtime = _build_stage_runtime(
                    desired_stage,
                    base_agent_reward_cfg,
                    base_env_reward_cfg,
                    base_env_kwargs,
                )
                train_env, obs_batch = _build_vector_env(stage_runtime)
                flat_batch = _flatten_batch(processor, obs_batch)
                episode_returns.fill(0.0)
                episode_lengths.fill(0)
                episode_scores.fill(0.0)
                continue
        epsilon = linear_schedule(args.epsilon_start, args.epsilon_end, args.epsilon_decay, global_step)
        temperature = None
        if args.exploration_strategy == "boltzmann":
            temp_value = linear_schedule(
                args.boltzmann_temp_start,
                args.boltzmann_temp_end,
                args.boltzmann_temp_decay,
                global_step,
            )
            temperature = max(temp_value, 1e-3)
        actions = agent.act_batch(
            flat_batch,
            epsilon=epsilon,
            temperature=temperature or 1.0,
            strategy=args.exploration_strategy,
            rng=rng,
        )
        next_obs_batch, rewards, terminated, truncated, infos = train_env.step(actions)
        done = np.logical_or(terminated, truncated)
        info_list = _split_infos(infos, args.num_envs)
        next_flat_batch = _flatten_batch(processor, next_obs_batch)
        rewards = np.asarray(rewards, dtype=np.float32)
        episode_returns += rewards
        episode_lengths += 1
        for idx, info in enumerate(info_list):
            if "score" in info and info["score"] is not None:
                episode_scores[idx] = float(info["score"])

        for env_idx in range(args.num_envs):
            actual_next = next_flat_batch[env_idx]
            if done[env_idx]:
                final_obs = info_list[env_idx].get("final_observation")
                if final_obs is not None:
                    actual_next = processor.flatten(final_obs)
            buffer.add(
                flat_batch[env_idx],
                int(actions[env_idx]),
                float(rewards[env_idx]),
                actual_next,
                bool(done[env_idx]),
            )
            global_step += 1

        flat_batch = next_flat_batch
        if global_step >= args.total_timesteps:
            break

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

        for env_idx in range(args.num_envs):
            if done[env_idx]:
                final_info = info_list[env_idx].get("final_info") or info_list[env_idx]
                score_val = float(final_info.get("score", episode_scores[env_idx]))
                recent_returns.append(float(episode_returns[env_idx]))
                recent_lengths.append(int(episode_lengths[env_idx]))
                episodes_run += 1
                if episodes_run % 500 == 0:
                    print(
                        f"[episode {episodes_run:,}] return={episode_returns[env_idx]:.2f} "
                        f"len={episode_lengths[env_idx]} score={score_val:.1f}"
                    )
                episode_returns[env_idx] = 0.0
                episode_lengths[env_idx] = 0
                episode_scores[env_idx] = 0.0

        if next_log and global_step >= next_log:
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
            next_log += args.log_interval

        if next_eval and global_step >= next_eval:
            eval_return, eval_score = _evaluate_policy(
                agent,
                processor,
                env_id,
                episodes=args.eval_episodes,
                seed=args.seed,
                reward_scale=args.reward_scale,
                env_kwargs=stage_runtime.env_kwargs,
                env_kind=env_kind,
                agent_reward=stage_runtime.agent_reward,
                env_reward=stage_runtime.env_reward,
                curriculum_stage=stage_runtime.stage,
                preset_library=preset_library,
            )
            print(
                f"[eval step {global_step:,}] avg_return={eval_return:.1f} "
                f"avg_score={eval_score:.1f}"
            )
            if eval_return > best_eval:
                best_eval = eval_return
                best_path = args.log_dir / "best_model.pt"
                agent.save(str(best_path), metadata={"global_step": global_step, "best_eval_return": best_eval})
            next_eval += args.eval_frequency

        if next_checkpoint and global_step >= next_checkpoint:
            ckpt_path = args.log_dir / f"checkpoint_step_{global_step}.pt"
            agent.save(str(ckpt_path), metadata={"global_step": global_step, "best_eval_return": best_eval})
            next_checkpoint += args.checkpoint_frequency

    final_path = args.log_dir / "final_model.pt"
    agent.save(str(final_path), metadata={"global_step": global_step, "best_eval_return": best_eval})
    train_env.close()
    print(f"Training complete. Final checkpoint saved to {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
