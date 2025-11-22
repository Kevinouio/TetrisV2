"""Command-line trainer for the native single-agent PPO implementation."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np
import torch

from TetrisVersionOne.agents.common import (
    build_agent_reward_config,
    build_environment_reward_config,
)
from TetrisVersionOne.agents.observation import ObservationProcessor
from TetrisVersionOne.env.curriculum import (
    CurriculumEpisodeWrapper,
    CurriculumManager,
    CurriculumStage,
    apply_overrides,
    build_default_curriculum,
    build_modern_placement_curriculum,
)
from TetrisVersionOne.env.registration import register_envs
from TetrisVersionOne.env.state_presets import BoardPresetLibrary, load_board_presets
from TetrisVersionOne.env.wrappers import (
    AgentRewardConfig,
    EnvironmentRewardConfig,
    FloatBoardWrapper,
    PlacementActionWrapper,
    RewardScaleWrapper,
    UniversalRewardWrapper,
)
from .models import PPOAgent, PPOConfig, RolloutBuffer


@dataclass
class _StageRuntime:
    stage: Optional[CurriculumStage]
    agent_reward: AgentRewardConfig
    env_reward: EnvironmentRewardConfig
    env_kwargs: Dict[str, Any]


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a scratch PPO agent on Tetris.")
    parser.add_argument("--env", choices=("nes", "modern", "custom"), default="nes")
    parser.add_argument("--env-id", help="Gymnasium id to use when --env=custom.")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=4_096, help="Rollout horizon per environment.")
    parser.add_argument("--minibatch-size", type=int, default=2_048)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--anneal-lr", action="store_true", help="Linearly anneal the learning rate to zero.")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--action-temperature", type=float, default=1.0)
    parser.add_argument("--action-epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", help="Torch device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--log-interval", type=int, default=10_000)
    parser.add_argument("--eval-frequency", type=int, default=200_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-temperature", type=float, default=1.0)
    parser.add_argument("--eval-epsilon", type=float, default=0.0)
    parser.add_argument(
        "--eval-deterministic",
        dest="eval_deterministic",
        action="store_true",
        default=True,
        help="Use greedy action selection during evaluation (default).",
    )
    parser.add_argument(
        "--stochastic-eval",
        dest="eval_deterministic",
        action="store_false",
        help="Sample actions during evaluation instead of running greedily.",
    )
    parser.add_argument("--checkpoint-frequency", type=int, default=1_000_000)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/ppo_native"))
    parser.add_argument("--resume-from", type=Path, help="Resume training from a saved .pt checkpoint.")
    parser.add_argument("--rotation-penalty", type=float, default=0.0, help="Per extra rotation penalty.")
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
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable staged curriculum learning.",
    )
    parser.add_argument(
        "--curriculum-profile",
        choices=("macro", "none"),
        default="none",
        help="Optional curriculum preset; overrides --curriculum when set.",
    )
    parser.add_argument(
        "--board-preset-file",
        type=Path,
        help="JSON file containing named board presets for curriculum stages.",
    )
    parser.add_argument(
        "--placement-actions",
        action="store_true",
        help="Collapse inputs into placement macros (rotate/move/drop per piece).",
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
    env_kwargs: Optional[Dict[str, Any]],
    env_kind: str,
    agent_reward: Optional[AgentRewardConfig],
    env_reward: Optional[EnvironmentRewardConfig],
    curriculum_stage: Optional[CurriculumStage],
    preset_library: Optional[BoardPresetLibrary],
    use_placement_actions: bool,
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
    if use_placement_actions:
        env = PlacementActionWrapper(env, allow_hold=(env_kind != "nes"))
    return env


def _evaluate_policy(
    agent: PPOAgent,
    processor: ObservationProcessor,
    env_id: str,
    *,
    episodes: int,
    seed: Optional[int],
    reward_scale: float,
    env_kwargs: Optional[Dict[str, Any]],
    env_kind: str,
    agent_reward: Optional[AgentRewardConfig],
    env_reward: Optional[EnvironmentRewardConfig],
    curriculum_stage: Optional[CurriculumStage],
    preset_library: Optional[BoardPresetLibrary],
    deterministic: bool,
    temperature: float,
    epsilon: float,
    use_placement_actions: bool,
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
        use_placement_actions=use_placement_actions,
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
            action, _, _ = agent.act(
                flat,
                temperature=temperature,
                epsilon=epsilon,
                deterministic=deterministic,
            )
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
    use_placement_actions: bool,
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
        if use_placement_actions:
            env = PlacementActionWrapper(env, allow_hold=(env_kind != "nes"))
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
    if args.n_steps <= 0:
        raise SystemExit("--n-steps must be positive.")
    register_envs()
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)

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
    use_curriculum = args.curriculum or args.curriculum_profile != "none"
    if use_curriculum:
        if args.curriculum_profile != "none" and env_kind == "modern":
            stage_manager = build_modern_placement_curriculum(args.total_timesteps)
        else:
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
        use_placement_actions=args.placement_actions,
    )
    processor = ObservationProcessor(reference_env.observation_space)
    action_dim = reference_env.action_space.n
    reference_env.close()

    buffer = RolloutBuffer(args.n_steps, args.num_envs, processor.flat_dim)

    if not args.hidden_sizes:
        raise SystemExit("--hidden-sizes must contain at least one layer size.")

    if args.resume_from:
        agent, metadata = PPOAgent.load(str(args.resume_from), device=args.device)
        if agent.config.obs_dim != processor.flat_dim or agent.config.action_dim != action_dim:
            raise SystemExit("Checkpoint is incompatible with the selected environment.")
        start_step = int(metadata.get("global_step", 0))
        best_eval = float(metadata.get("best_eval_return", float("-inf")))
    else:
        config = PPOConfig(
            obs_dim=processor.flat_dim,
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
                    use_placement_actions=args.placement_actions,
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
    recent_returns: Deque[float] = deque(maxlen=200)
    recent_lengths: Deque[int] = deque(maxlen=200)
    last_update_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    rng = np.random.default_rng(args.seed)

    global_step = start_step
    next_log = args.log_interval if args.log_interval else None
    next_eval = args.eval_frequency if args.eval_frequency else None
    next_checkpoint = args.checkpoint_frequency if args.checkpoint_frequency else None
    if global_step >= args.total_timesteps:
        print("Checkpoint already reached requested total timesteps; exiting.")
        train_env.close()
        return 0

    try:
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

            buffer.reset()
            steps_collected = 0
            last_dones = np.zeros(args.num_envs, dtype=np.float32)
            while steps_collected < args.n_steps and global_step < args.total_timesteps:
                actions, log_probs, values = agent.act_batch(
                    flat_batch,
                    temperature=max(args.action_temperature, 1e-6),
                    epsilon=args.action_epsilon,
                    rng=rng,
                )
                next_obs_batch, rewards, terminated, truncated, infos = train_env.step(actions)
                done = np.logical_or(terminated, truncated)
                info_list = _split_infos(infos, args.num_envs)
                rewards = np.asarray(rewards, dtype=np.float32)
                buffer.add(flat_batch, actions, rewards, done, values, log_probs)
                episode_returns += rewards
                episode_lengths += 1
                for idx, info in enumerate(info_list):
                    if "score" in info and info["score"] is not None:
                        episode_scores[idx] = float(info["score"])

                next_flat_batch = _flatten_batch(processor, next_obs_batch)
                flat_batch = next_flat_batch
                last_dones = done.astype(np.float32)
                for env_idx in range(args.num_envs):
                    if done[env_idx]:
                        final_info = info_list[env_idx].get("final_info") or info_list[env_idx]
                        score_val = float(final_info.get("score", episode_scores[env_idx]))
                        episode_return = float(episode_returns[env_idx])
                        episode_length = int(episode_lengths[env_idx])
                        recent_returns.append(episode_return)
                        recent_lengths.append(episode_length)
                        episodes_run += 1
                        if episodes_run % 500 == 0:
                            print(
                                f"[episode {episodes_run:,}] return={episode_return:.2f} "
                                f"len={episode_length} score={score_val:.1f}"
                            )
                        episode_returns[env_idx] = 0.0
                        episode_lengths[env_idx] = 0
                        episode_scores[env_idx] = score_val
                steps_collected += 1
                global_step += args.num_envs

            last_values = agent.value_batch(flat_batch)
            buffer.compute_returns_and_advantages(
                last_values,
                last_dones,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )

            if args.anneal_lr:
                frac = max(0.0, 1.0 - (global_step / float(args.total_timesteps)))
                new_lr = args.learning_rate * frac
                for group in agent.optimizer.param_groups:
                    group["lr"] = new_lr

            last_update_metrics = agent.update(buffer, batch_size=args.minibatch_size, epochs=args.update_epochs)

            if next_log and global_step >= next_log:
                avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
                avg_len = float(np.mean(recent_lengths)) if recent_lengths else 0.0
                lr_value = agent.optimizer.param_groups[0]["lr"]
                stage_label = stage_runtime.stage.name if stage_runtime.stage else "free_play"
                print(
                    f"[step {global_step:,} | episode {episodes_run:,}] "
                    f"return={avg_return:.1f} len={avg_len:.1f} "
                    f"policy={last_update_metrics['policy_loss']:.4f} "
                    f"value={last_update_metrics['value_loss']:.4f} "
                    f"entropy={last_update_metrics['entropy']:.4f} "
                    f"lr={lr_value:.2e} stage={stage_label}"
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
                    deterministic=args.eval_deterministic,
                    temperature=args.eval_temperature,
                    epsilon=args.eval_epsilon,
                    use_placement_actions=args.placement_actions,
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

    finally:
        train_env.close()

    final_path = args.log_dir / "final_model.pt"
    agent.save(str(final_path), metadata={"global_step": global_step, "best_eval_return": best_eval})
    print(f"Training complete. Final checkpoint saved to {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
