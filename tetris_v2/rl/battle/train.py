"""CPU-first shared-policy Battle-DQN curriculum training."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Mapping, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from tetris_v2.rl.battle.checkpoint import (
    load_battle_training_checkpoint,
    save_battle_training_checkpoint,
)
from tetris_v2.rl.battle.config import BattleRewardConfig, BattleRulesConfig
from tetris_v2.rl.battle.curriculum import CurriculumState
from tetris_v2.rl.battle.dqn import (
    BATTLE_OBSERVATION_DIM,
    BattleDQNAgent,
    BattleDQNConfig,
    LinearSchedule,
)
from tetris_v2.rl.battle.env import BattleEnv
from tetris_v2.rl.battle.evaluation import evaluate_paired_battles
from tetris_v2.rl.battle.metrics import append_jsonl
from tetris_v2.rl.battle.opponents import OpponentDescriptor, OpponentPool
from tetris_v2.rl.battle.policies import (
    BattleDQNPolicy,
    ColdClearBattlePolicy,
    RandomBattlePolicy,
    load_battle_dqn_policy,
    load_embedded_battle_dqn_policy,
)
from tetris_v2.rl.battle.replay import PackedBattleReplayBuffer


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train shared masked Double-DQN in battle mode.")
    parser.add_argument("--total-timesteps", type=_positive_int, default=1_000_000)
    parser.add_argument("--buffer-size", type=_positive_int, default=50_000)
    parser.add_argument("--warmup-steps", type=_nonnegative_int, default=2_000)
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--train-frequency", type=_positive_int, default=1)
    parser.add_argument("--gradient-steps", type=_positive_int, default=1)
    parser.add_argument("--learning-rate-start", type=float, default=1e-4)
    parser.add_argument("--learning-rate-end", type=float, default=1e-5)
    parser.add_argument("--learning-rate-decay-steps", type=_nonnegative_int, default=1_000_000)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--target-sync-interval", type=_positive_int, default=2_000)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--epsilon-start", type=float, default=0.20)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-steps", type=_nonnegative_int, default=500_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lib", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)

    parser.add_argument("--attack-table", type=int, nargs=5, default=[0, 0, 1, 2, 4])
    parser.add_argument("--garbage-delay", type=_nonnegative_int, default=1)
    parser.add_argument("--max-match-steps", type=_positive_int, default=500)
    parser.add_argument("--independent-piece-seeds", action="store_true")
    parser.add_argument("--reward-terminal", type=float, default=20.0)
    parser.add_argument("--reward-attack", type=float, default=0.05)
    parser.add_argument("--reward-cancellation", type=float, default=0.03)
    parser.add_argument("--reward-line-clear", type=float, default=0.01)
    parser.add_argument("--reward-board-quality", type=float, default=0.02)
    parser.add_argument("--reward-height", type=float, default=0.02)
    parser.add_argument("--reward-holes", type=float, default=0.03)
    parser.add_argument("--reward-garbage", type=float, default=0.04)

    parser.add_argument("--opponent-pool-size", type=_positive_int, default=20)
    parser.add_argument("--pool-checkpoint-frequency", type=_positive_int, default=100_000)
    parser.add_argument(
        "--cold-clear-think-ms",
        type=_nonnegative_int,
        default=0,
        help="0 uses deterministic fixed-work search; positive values are wall-clock milliseconds",
    )
    parser.add_argument("--random-opponent-weight", type=float, default=0.20)
    parser.add_argument("--heuristic-opponent-weight", type=float, default=0.30)
    parser.add_argument("--frozen-opponent-weight", type=float, default=0.50)
    parser.add_argument("--current-opponent-weight", type=float, default=0.0)
    parser.add_argument("--disable-curriculum", action="store_true")
    parser.add_argument(
        "--curriculum-config",
        type=Path,
        default=None,
        help=(
            "YAML stage mixes and fixed-seed promotion thresholds "
            "(default: packaged tetris_v2/conf/battle_curriculum.yaml)."
        ),
    )

    parser.add_argument("--log-frequency", type=_positive_int, default=10_000)
    parser.add_argument("--eval-frequency", type=_positive_int, default=50_000)
    parser.add_argument("--eval-matches", type=_positive_int, default=100)
    parser.add_argument("--eval-seed", type=int, default=900_000)
    parser.add_argument("--checkpoint-frequency", type=_positive_int, default=100_000)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/battle_dqn"))
    return parser.parse_args(argv)


def _rules(args: argparse.Namespace) -> BattleRulesConfig:
    return BattleRulesConfig(
        attack_table=tuple(int(value) for value in args.attack_table),
        garbage_delay=int(args.garbage_delay),
        max_steps=int(args.max_match_steps),
        mirrored_piece_seeds=not bool(args.independent_piece_seeds),
    )


def _reward_config(args: argparse.Namespace) -> BattleRewardConfig:
    return BattleRewardConfig(
        terminal=float(args.reward_terminal),
        attack=float(args.reward_attack),
        cancellation=float(args.reward_cancellation),
        line_clear=float(args.reward_line_clear),
        board_quality=float(args.reward_board_quality),
        height=float(args.reward_height),
        holes=float(args.reward_holes),
        garbage=float(args.reward_garbage),
    )


def _manual_mix(args: argparse.Namespace) -> dict[str, float]:
    values = {
        "random": float(args.random_opponent_weight),
        "cold_clear": float(args.heuristic_opponent_weight),
        "frozen": float(args.frozen_opponent_weight),
        "current": float(args.current_opponent_weight),
    }
    if any(weight < 0 for weight in values.values()):
        raise SystemExit("Opponent weights cannot be negative")
    total = sum(values.values())
    if total <= 0:
        raise SystemExit("At least one opponent weight must be positive")
    return {name: weight / total for name, weight in values.items()}


def _configured_curriculum(path: Path | None) -> CurriculumState:
    source = path or (
        Path(__file__).resolve().parents[2] / "conf" / "battle_curriculum.yaml"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Battle curriculum config is missing: {source}")
    raw = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("stages"), list):
        raise ValueError("Battle curriculum YAML must contain a non-empty stages list.")
    return CurriculumState.from_state_dict(
        {
            "stages": raw["stages"],
            "stage_index": 0,
            "entered_at_step": 0,
            "promotion_history": [],
        }
    )


def _serializable_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


_RESUME_COMPATIBLE_ARGUMENTS = (
    "batch_size",
    "buffer_size",
    "cold_clear_think_ms",
    "disable_curriculum",
    "device",
    "epsilon_decay_steps",
    "epsilon_end",
    "epsilon_start",
    "eval_frequency",
    "eval_matches",
    "eval_seed",
    "gamma",
    "gradient_steps",
    "heuristic_opponent_weight",
    "learning_rate_decay_steps",
    "learning_rate_end",
    "learning_rate_start",
    "max_grad_norm",
    "opponent_pool_size",
    "pool_checkpoint_frequency",
    "random_opponent_weight",
    "frozen_opponent_weight",
    "current_opponent_weight",
    "target_sync_interval",
    "train_frequency",
    "warmup_steps",
)


def _validate_resume_config(
    stored: Mapping[str, object],
    current: Mapping[str, object],
) -> None:
    """Reject continuation settings that would silently change the run."""

    for section in ("rules", "rewards"):
        if stored.get(section) != current.get(section):
            raise ValueError(
                f"Resume {section} differ from the checkpoint; use the original settings."
            )
    # Older checkpoints already retain the live curriculum in their extra
    # state, but predate this resolved-config field. Keep them resumable while
    # preventing newer checkpoints from silently switching stage definitions.
    stored_stages = stored.get("curriculum_stages")
    if (
        stored_stages is not None
        and stored_stages != current.get("curriculum_stages")
    ):
        raise ValueError(
            "Resume curriculum_stages differ from the checkpoint; "
            "use the original settings."
        )
    if stored.get("observation_dim") != current.get("observation_dim"):
        raise ValueError("Resume observation schema differs from the checkpoint.")
    stored_args = stored.get("arguments")
    current_args = current.get("arguments")
    if not isinstance(stored_args, Mapping) or not isinstance(current_args, Mapping):
        raise ValueError("Resume checkpoint training arguments are malformed.")
    differences = [
        name
        for name in _RESUME_COMPATIBLE_ARGUMENTS
        if stored_args.get(name) != current_args.get(name)
    ]
    if differences:
        names = ", ".join(differences)
        raise ValueError(
            f"Resume would change trajectory-affecting arguments: {names}. "
            "Use the checkpoint values."
        )


def _next_boundary(step: int, frequency: int) -> int:
    return ((int(step) // int(frequency)) + 1) * int(frequency)


def _make_env(
    args: argparse.Namespace,
    rules: BattleRulesConfig,
    rewards: BattleRewardConfig,
    *,
    seed: int,
) -> BattleEnv:
    return BattleEnv(
        seed=seed,
        lib_path=args.lib,
        rules=rules,
        reward_config=rewards,
    )


class _PolicyCache:
    def __init__(
        self,
        agent: BattleDQNAgent,
        *,
        pool: OpponentPool,
        device: str,
        think_ms: int,
    ):
        self.agent = agent
        self.pool = pool
        self.device = device
        self.think_ms = think_ms
        self.frozen: dict[str, BattleDQNPolicy] = {}

    def policy(
        self,
        descriptor: OpponentDescriptor,
        *,
        seed: int,
        epsilon: float,
    ):
        if descriptor.kind == "random":
            return RandomBattlePolicy(seed=seed)
        if descriptor.kind == "cold_clear":
            return ColdClearBattlePolicy(think_ms=self.think_ms)
        if descriptor.kind == "current":
            return BattleDQNPolicy(
                self.agent,
                identifier="current",
                kind="current",
                epsilon=epsilon,
                deterministic=False,
                seed=seed,
            )
        assert descriptor.checkpoint is not None
        cached = self.frozen.get(descriptor.identifier)
        if cached is None:
            checkpoint = Path(descriptor.checkpoint)
            if checkpoint.is_file():
                cached = load_battle_dqn_policy(
                    checkpoint,
                    device=self.device,
                    identifier=descriptor.identifier,
                )
            else:
                payload = self.pool.embedded_checkpoint(descriptor.identifier)
                if payload is None:
                    raise FileNotFoundError(
                        f"Frozen opponent checkpoint is missing: {checkpoint}"
                    )
                cached = load_embedded_battle_dqn_policy(
                    payload,
                    device=self.device,
                    identifier=descriptor.identifier,
                )
            self.frozen[descriptor.identifier] = cached
        cached.reset(seed)
        return cached


def _save_pool_snapshot(
    agent: BattleDQNAgent,
    pool: OpponentPool,
    *,
    pool_dir: Path,
    global_step: int,
    generation: int,
    wall_seconds: float,
    rules: BattleRulesConfig,
    rewards: BattleRewardConfig,
) -> tuple[OpponentDescriptor, tuple[OpponentDescriptor, ...]]:
    pool_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = (pool_dir / f"battle_policy_step_{global_step:012d}.pt").resolve()
    identifier = f"battle_step_{global_step:012d}"
    agent.save_frozen(
        checkpoint,
        metadata={
            "identifier": identifier,
            "global_step": global_step,
            "generation": generation,
            "wall_seconds": wall_seconds,
            "rules": rules.to_dict(),
            "rewards": rewards.to_dict(),
        },
    )
    descriptor = OpponentDescriptor(
        identifier=identifier,
        kind="checkpoint",
        checkpoint=str(checkpoint),
        generation=generation,
        created_step=global_step,
    )
    evicted = pool.add(descriptor)
    for removed in evicted:
        if removed.checkpoint:
            path = Path(removed.checkpoint)
            if path.parent.resolve() == pool_dir.resolve() and path.is_file():
                path.unlink()
    return descriptor, evicted


def _episode_log(
    *,
    episode: int,
    match_seed: int,
    learner_seat: int,
    descriptor: OpponentDescriptor,
    selection: Mapping[str, object],
    info: Mapping[str, Any],
    learner_return: float,
    epsilon: float,
    learning_rate: float,
    loss: float,
    last_loss: float,
    optimizer_updates: int,
    reward_components: Mapping[str, float],
    global_step: int,
    wall_seconds: float,
) -> dict[str, object]:
    player = info["players"][learner_seat]
    winner = info.get("winner")
    result = "draw" if winner is None else ("win" if winner == learner_seat else "loss")
    return {
        "episode": episode,
        "global_step": global_step,
        "learner_checkpoint_identifier": f"learner_step_{global_step:012d}",
        "seed": match_seed,
        "learner_seat": learner_seat,
        "opponent_type": descriptor.kind,
        "opponent_identifier": descriptor.identifier,
        "opponent_checkpoint": descriptor.checkpoint,
        "opponent_selection": dict(selection),
        "result": result,
        "winner": winner,
        "episode_return": learner_return,
        "raw_score": float(player.get("score", 0.0)),
        "match_length": int(info["step"]),
        "pieces_placed": int(player["placements"]),
        "lines_cleared": int(player["lines_cleared"]),
        "attack_generated": int(player["attack_generated"]),
        "attack_sent": int(player["garbage_sent"]),
        "garbage_sent": int(player["garbage_sent"]),
        "garbage_cancelled": int(player["garbage_cancelled"]),
        "garbage_received": int(player["garbage_received"]),
        "garbage_applied": int(player.get("garbage_applied", 0)),
        "board_stats": dict(info["board_stats"][learner_seat]),
        "epsilon": epsilon,
        "learning_rate": learning_rate,
        "episode_mean_loss": loss,
        "loss": loss,
        "last_loss": last_loss,
        "optimizer_updates": optimizer_updates,
        "illegal_actions": 0,
        "reward_components": dict(reward_components),
        "wall_seconds": wall_seconds,
    }


def _sum_components(target: dict[str, float], row: Mapping[str, float]) -> None:
    for name, value in row.items():
        target[name] = target.get(name, 0.0) + float(value)


def _evaluation_opponent(
    name: str,
    *,
    pool: OpponentPool,
    cache: _PolicyCache,
    seed: int,
):
    if name == "random":
        return RandomBattlePolicy(seed=seed)
    if name in {"cold_clear", "heuristic"}:
        return ColdClearBattlePolicy(think_ms=cache.think_ms)
    if name == "frozen":
        if not pool.frozen:
            raise RuntimeError("Frozen-opponent evaluation requires a pool checkpoint")
        descriptor = pool.frozen[-1]
        return cache.policy(descriptor, seed=seed, epsilon=0.0)
    if name == "current":
        return BattleDQNPolicy(
            cache.agent,
            identifier="current",
            kind="current",
            deterministic=True,
            epsilon=0.0,
            seed=seed,
        )
    raise ValueError(f"Unknown evaluation opponent {name!r}")


def _run_curriculum_evaluation(
    *,
    args: argparse.Namespace,
    rules: BattleRulesConfig,
    rewards: BattleRewardConfig,
    agent: BattleDQNAgent,
    pool: OpponentPool,
    cache: _PolicyCache,
    curriculum: CurriculumState,
    global_step: int,
    wall_seconds: float,
    promote: bool,
) -> tuple[dict[str, dict[str, object]], bool]:
    learner = BattleDQNPolicy(
        agent,
        identifier=f"learner_step_{global_step}",
        kind="current",
        deterministic=True,
        epsilon=0.0,
        seed=args.eval_seed,
    )
    evaluated_stage = curriculum.current.name
    requirements = curriculum.current.promotion if promote else ()
    names = [requirement.opponent for requirement in requirements]
    if not names:
        names = ["random", "cold_clear", "frozen"]
    minimum_matches = {
        requirement.opponent: requirement.min_matches for requirement in requirements
    }
    seed_offsets = {"random": 0, "cold_clear": 10_000, "heuristic": 10_000, "frozen": 20_000, "current": 30_000}
    reports: dict[str, dict[str, object]] = {}
    output_dir = args.log_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        opponent = _evaluation_opponent(
            name,
            pool=pool,
            cache=cache,
            seed=args.eval_seed + seed_offsets[name],
        )
        match_count = max(args.eval_matches, minimum_matches.get(name, 0))
        if match_count % 2:
            match_count += 1
        matches, summary = evaluate_paired_battles(
            learner,
            opponent,
            env_factory=lambda: _make_env(
                args,
                rules,
                rewards,
                seed=args.eval_seed + seed_offsets[name],
            ),
            matches=match_count,
            seed=args.eval_seed + seed_offsets[name],
        )
        reports[name] = summary
        report = {
            "schema_version": 1,
            "global_step": global_step,
            "training_steps": global_step,
            "wall_clock_training_time": wall_seconds,
            "wall_clock_training_time_units": "seconds",
            "stage": evaluated_stage,
            "opponent": name,
            "configuration": {
                "matches": match_count,
                "seed": args.eval_seed + seed_offsets[name],
                "paired_seats": True,
                "deterministic": not (
                    name in {"cold_clear", "heuristic"}
                    and int(args.cold_clear_think_ms) > 0
                ),
            },
            "matches": [match.to_dict() for match in matches],
            "summary": summary,
        }
        path = output_dir / f"step_{global_step:012d}_{name}.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    promoted = (
        curriculum.maybe_promote(reports, global_step=global_step)
        if promote
        else False
    )
    append_jsonl(
        args.log_dir / "evaluations.jsonl",
        [
            {
                "global_step": global_step,
                "training_steps": global_step,
                "wall_clock_training_time": wall_seconds,
                "stage": evaluated_stage,
                "next_stage": curriculum.current.name,
                "promoted": promoted,
                "reports": reports,
            }
        ],
    )
    return reports, promoted


def run(args: argparse.Namespace) -> int:
    if args.init_checkpoint is not None and args.resume_checkpoint is not None:
        raise SystemExit("Set only one of --init-checkpoint and --resume-checkpoint")
    if args.batch_size > args.buffer_size:
        raise SystemExit("--batch-size cannot exceed --buffer-size")
    if not args.disable_curriculum and args.cold_clear_think_ms > 0:
        raise SystemExit(
            "Curriculum promotion requires deterministic Cold Clear; "
            "set --cold-clear-think-ms 0."
        )
    rules = _rules(args)
    reward_config = _reward_config(args)
    manual_mix = _manual_mix(args)
    configured_curriculum = _configured_curriculum(args.curriculum_config)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    pool_dir = args.log_dir / "opponent_pool"
    training_config = {
        "arguments": _serializable_args(args),
        "rules": rules.to_dict(),
        "rewards": reward_config.to_dict(),
        "curriculum_stages": configured_curriculum.state_dict()["stages"],
        "observation_dim": BATTLE_OBSERVATION_DIM,
    }

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    trainer_rng = np.random.default_rng(args.seed + 1)
    epsilon_schedule = LinearSchedule(
        args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps
    )
    lr_schedule = LinearSchedule(
        args.learning_rate_start,
        args.learning_rate_end,
        args.learning_rate_decay_steps,
    )
    curriculum = configured_curriculum
    pool_generation = 0
    global_step = 0
    episode_index = 0
    wall_seconds_before = 0.0

    if args.resume_checkpoint is not None:
        bundle = load_battle_training_checkpoint(
            args.resume_checkpoint,
            device=args.device,
        )
        _validate_resume_config(bundle.training_config, training_config)
        agent = bundle.agent
        replay = bundle.replay
        pool = bundle.opponent_pool
        global_step = bundle.global_step
        episode_index = bundle.episode_index
        epsilon_schedule = bundle.epsilon_schedule
        lr_schedule = bundle.learning_rate_schedule
        extra = bundle.extra
        if "trainer_rng_state" not in extra or "curriculum" not in extra:
            raise ValueError(
                "Resume checkpoint lacks trainer RNG or curriculum state."
            )
        trainer_rng.bit_generator.state = extra["trainer_rng_state"]  # type: ignore[assignment]
        curriculum = CurriculumState.from_state_dict(extra["curriculum"])  # type: ignore[arg-type]
        if (
            bundle.training_config.get("curriculum_stages") is None
            and curriculum.state_dict()["stages"]
            != training_config["curriculum_stages"]
        ):
            raise ValueError(
                "Resume curriculum_stages differ from the checkpoint's live "
                "curriculum; use the original settings."
            )
        pool_generation = int(extra.get("pool_generation", 0))
        wall_seconds_before = float(extra.get("wall_seconds", 0.0))
    else:
        agent = BattleDQNAgent(
            BattleDQNConfig(
                learning_rate=args.learning_rate_start,
                gamma=args.gamma,
                target_sync_interval=args.target_sync_interval,
                max_grad_norm=args.max_grad_norm,
                seed=args.seed,
                device=args.device,
            )
        )
        if args.init_checkpoint is not None:
            agent.warm_start_from_checkpoint(args.init_checkpoint)
        replay = PackedBattleReplayBuffer(
            args.buffer_size,
            seed=args.seed + 2,
        )
        pool = OpponentPool(args.opponent_pool_size, seed=args.seed + 3)
        _save_pool_snapshot(
            agent,
            pool,
            pool_dir=pool_dir,
            global_step=0,
            generation=pool_generation,
            wall_seconds=0.0,
            rules=rules,
            rewards=reward_config,
        )

    env = _make_env(args, rules, reward_config, seed=args.seed)
    cache = _PolicyCache(
        agent,
        pool=pool,
        device=args.device,
        think_ms=args.cold_clear_think_ms,
    )
    episode_path = args.log_dir / "episodes.jsonl"
    recent_returns: deque[float] = deque(maxlen=100)
    recent_losses: deque[float] = deque(maxlen=1000)
    next_log = _next_boundary(global_step, args.log_frequency)
    next_checkpoint = _next_boundary(global_step, args.checkpoint_frequency)
    next_pool = _next_boundary(global_step, args.pool_checkpoint_frequency)
    next_eval = _next_boundary(global_step, args.eval_frequency)
    last_loss = 0.0
    started = perf_counter()

    try:
        while global_step < args.total_timesteps:
            episode_index += 1
            match_seed = int(trainer_rng.integers(0, 2**32, dtype=np.uint32))
            learner_seat = int(trainer_rng.integers(0, 2))
            mix = manual_mix if args.disable_curriculum else dict(curriculum.current.opponent_mix)
            selection = pool.sample_selection(mix)
            descriptor = selection.descriptor
            epsilon = epsilon_schedule.value(global_step)
            opponent = cache.policy(
                descriptor,
                seed=match_seed + 17,
                epsilon=epsilon,
            )
            observations, masks, info = env.reset(seed=match_seed)
            terminated = False
            truncated = False
            learner_return = 0.0
            components: dict[str, float] = {}
            episode_losses: list[float] = []

            # Checkpoints are episode-boundary snapshots. Finish a match after
            # crossing the requested budget so every saved continuation is exact.
            while not (terminated or truncated):
                learner_action = agent.select_action(
                    observations[learner_seat],
                    action_mask=masks[learner_seat],
                    epsilon=epsilon,
                    deterministic=False,
                    rng=trainer_rng,
                )
                other_seat = 1 - learner_seat
                opponent_action = opponent.select_action(
                    observations[other_seat],
                    masks[other_seat],
                    player=other_seat,
                    env=env,
                )
                actions = [0, 0]
                actions[learner_seat] = learner_action
                actions[other_seat] = opponent_action
                next_observations, rewards, terminated, truncated, next_info = env.step(
                    (actions[0], actions[1])
                )
                next_masks = next_info["action_masks"]
                replay.add(
                    observations[learner_seat],
                    masks[learner_seat],
                    learner_action,
                    rewards[learner_seat],
                    next_observations[learner_seat],
                    next_masks[learner_seat],
                    terminated,
                    truncated,
                )
                if descriptor.kind == "current":
                    replay.add(
                        observations[other_seat],
                        masks[other_seat],
                        opponent_action,
                        rewards[other_seat],
                        next_observations[other_seat],
                        next_masks[other_seat],
                        terminated,
                        truncated,
                    )
                learner_return += float(rewards[learner_seat])
                _sum_components(components, next_info["reward_components"][learner_seat])
                observations, masks, info = next_observations, next_masks, next_info
                global_step += 1
                epsilon = epsilon_schedule.value(global_step)
                agent.apply_learning_rate_schedule(global_step, lr_schedule)

                if (
                    global_step >= args.warmup_steps
                    and global_step % args.train_frequency == 0
                    and len(replay) >= args.batch_size
                ):
                    for _ in range(args.gradient_steps):
                        metrics = agent.update(replay.sample(args.batch_size))
                        last_loss = float(metrics["td_loss"])
                        recent_losses.append(last_loss)
                        episode_losses.append(last_loss)

            recent_returns.append(learner_return)
            elapsed = wall_seconds_before + (perf_counter() - started)
            append_jsonl(
                episode_path,
                [
                    _episode_log(
                        episode=episode_index,
                        match_seed=match_seed,
                        learner_seat=learner_seat,
                        descriptor=descriptor,
                        selection=selection.to_dict(),
                        info=info,
                        learner_return=learner_return,
                        epsilon=epsilon,
                        learning_rate=agent.current_learning_rate,
                        loss=(
                            float(np.mean(episode_losses))
                            if episode_losses
                            else 0.0
                        ),
                        last_loss=(episode_losses[-1] if episode_losses else 0.0),
                        optimizer_updates=len(episode_losses),
                        reward_components=components,
                        global_step=global_step,
                        wall_seconds=elapsed,
                    )
                ],
            )

            if global_step >= next_pool:
                pool_generation += 1
                _save_pool_snapshot(
                    agent,
                    pool,
                    pool_dir=pool_dir,
                    global_step=global_step,
                    generation=pool_generation,
                    wall_seconds=elapsed,
                    rules=rules,
                    rewards=reward_config,
                )
                next_pool = _next_boundary(global_step, args.pool_checkpoint_frequency)

            if global_step >= next_eval:
                _, promoted = _run_curriculum_evaluation(
                    args=args,
                    rules=rules,
                    rewards=reward_config,
                    agent=agent,
                    pool=pool,
                    cache=cache,
                    curriculum=curriculum,
                    global_step=global_step,
                    wall_seconds=elapsed,
                    promote=not args.disable_curriculum,
                )
                if promoted:
                    pool_generation += 1
                    _save_pool_snapshot(
                        agent,
                        pool,
                        pool_dir=pool_dir,
                        global_step=global_step,
                        generation=pool_generation,
                        wall_seconds=elapsed,
                        rules=rules,
                        rewards=reward_config,
                    )
                next_eval = _next_boundary(global_step, args.eval_frequency)

            if global_step >= next_checkpoint:
                checkpoint = args.log_dir / f"battle_training_step_{global_step:012d}.pt"
                save_battle_training_checkpoint(
                    checkpoint,
                    agent=agent,
                    replay=replay,
                    opponent_pool=pool,
                    global_step=global_step,
                    episode_index=episode_index,
                    epsilon_schedule=epsilon_schedule,
                    learning_rate_schedule=lr_schedule,
                    training_config=training_config,
                    at_episode_boundary=True,
                    extra={
                        "trainer_rng_state": trainer_rng.bit_generator.state,
                        "curriculum": curriculum.state_dict(),
                        "pool_generation": pool_generation,
                        "wall_seconds": elapsed,
                    },
                )
                next_checkpoint = _next_boundary(global_step, args.checkpoint_frequency)

            if global_step >= next_log:
                print(
                    f"[battle dqn step {global_step:,}] episode={episode_index:,} "
                    f"stage={curriculum.current.name} return={np.mean(recent_returns):.3f} "
                    f"loss={np.mean(recent_losses) if recent_losses else 0.0:.5f} "
                    f"epsilon={epsilon:.3f} replay={len(replay):,} pool={len(pool.frozen)}"
                )
                next_log = _next_boundary(global_step, args.log_frequency)
    finally:
        env.close()

    elapsed = wall_seconds_before + (perf_counter() - started)
    final_checkpoint = args.log_dir / "battle_training_final.pt"
    save_battle_training_checkpoint(
        final_checkpoint,
        agent=agent,
        replay=replay,
        opponent_pool=pool,
        global_step=global_step,
        episode_index=episode_index,
        epsilon_schedule=epsilon_schedule,
        learning_rate_schedule=lr_schedule,
        training_config=training_config,
        at_episode_boundary=True,
        extra={
            "trainer_rng_state": trainer_rng.bit_generator.state,
            "curriculum": curriculum.state_dict(),
            "pool_generation": pool_generation,
            "wall_seconds": elapsed,
        },
    )
    final_policy = args.log_dir / "battle_dqn_final.pt"
    agent.save_frozen(
        final_policy,
        metadata={
            "identifier": "battle_dqn_final",
            "global_step": global_step,
            "episode_index": episode_index,
            "wall_seconds": elapsed,
            "rules": rules.to_dict(),
            "rewards": reward_config.to_dict(),
        },
    )
    print(f"Saved Battle-DQN policy to {final_policy}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "run"]
