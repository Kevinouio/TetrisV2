"""Short real-native smoke test for the battle environment and Battle-DQN."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from tetris_v2.rl.actions import PLACEMENT_ACTION_DIM
from tetris_v2.rl.battle.cli import (
    load_evaluation_policy,
    make_env_factory,
    nonnegative_int,
    policy_checkpoint_metadata,
    positive_even_int,
    positive_int,
    write_json_report,
)
from tetris_v2.rl.battle.config import BattleRewardConfig, BattleRulesConfig
from tetris_v2.rl.battle.dqn import (
    BATTLE_OBSERVATION_DIM,
    BATTLE_OBSERVATION_SCHEMA,
    BattleDQNAgent,
    BattleDQNConfig,
)
from tetris_v2.rl.battle.env import BattleEnv
from tetris_v2.rl.battle.evaluation import (
    compare_repeated_matches,
    evaluate_paired_battles,
)
from tetris_v2.rl.battle.policies import (
    BattleDQNPolicy,
    ColdClearBattlePolicy,
    RandomBattlePolicy,
)
from tetris_v2.rl.battle.replay import PackedBattleReplayBuffer


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise the real native battle ABI with a very short match."
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--opponent", choices=("random", "cold_clear"), default="random"
    )
    parser.add_argument("--matches", type=positive_even_int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=positive_int, default=8)
    parser.add_argument(
        "--cold-clear-think-ms",
        type=nonnegative_int,
        default=0,
        help="0 uses deterministic fixed-work Cold Clear evaluation.",
    )
    parser.add_argument("--repeat-determinism", action="store_true")
    parser.add_argument(
        "--train-updates",
        type=nonnegative_int,
        default=2,
        help="Tiny CPU backprop updates to run after the native match (default: 2).",
    )
    parser.add_argument("--train-batch-size", type=positive_int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lib", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args(argv)


def _native_contract_probe(
    args: argparse.Namespace,
    *,
    rules: BattleRulesConfig,
    rewards: BattleRewardConfig,
) -> dict[str, object]:
    env = BattleEnv(
        seed=args.seed,
        lib_path=args.lib,
        rules=rules,
        reward_config=rewards,
    )
    try:
        observations, masks, info = env.reset(seed=args.seed)
        initial_shapes = [list(value.shape) for value in observations]
        mask_shapes = [list(value.shape) for value in masks]
        observations, masks, queued_info = env.enqueue_garbage(0, [4], delay=0)
        actions = tuple(int(np.flatnonzero(mask > 0.5)[0]) for mask in masks)
        _, step_rewards, terminated, truncated, step_info = env.step(actions)
        checks = {
            "observation_shapes": initial_shapes,
            "action_mask_shapes": mask_shapes,
            "observation_schema": info["observation_schema"],
            "initial_legal_action_counts": list(info["legal_action_counts"]),
            "scripted_garbage_queued": int(
                queued_info["players"][0]["incoming_garbage"]
            ),
            "first_joint_actions": list(actions),
            "antisymmetric_reward": bool(
                np.isclose(float(step_rewards[0]), -float(step_rewards[1]))
            ),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "step": int(step_info["step"]),
        }
        checks["passed"] = bool(
            initial_shapes == [[BATTLE_OBSERVATION_DIM], [BATTLE_OBSERVATION_DIM]]
            and mask_shapes == [[PLACEMENT_ACTION_DIM], [PLACEMENT_ACTION_DIM]]
            and info["observation_schema"] == BATTLE_OBSERVATION_SCHEMA
            and all(int(value) > 0 for value in info["legal_action_counts"])
            and checks["scripted_garbage_queued"] >= 1
            and checks["antisymmetric_reward"]
            and checks["step"] == 1
        )
        return checks
    finally:
        env.close()


def _tiny_training_smoke(
    args: argparse.Namespace,
    *,
    rules: BattleRulesConfig,
    rewards: BattleRewardConfig,
    agent: BattleDQNAgent,
    initialized_from: str | None,
) -> dict[str, object] | None:
    requested = int(args.train_updates)
    if requested == 0:
        return None
    batch_size = int(args.train_batch_size)
    replay = PackedBattleReplayBuffer(
        max(batch_size + requested + 2, 8),
        seed=args.seed + 21,
    )
    learner_actions = RandomBattlePolicy(seed=args.seed + 22)
    opponent_actions = RandomBattlePolicy(seed=args.seed + 23)
    env = BattleEnv(
        seed=args.seed + 24,
        lib_path=args.lib,
        rules=rules,
        reward_config=rewards,
    )
    losses: list[float] = []
    transitions = 0
    episodes = 0
    try:
        while len(losses) < requested:
            episodes += 1
            match_seed = args.seed + 10_000 + episodes
            learner_actions.reset(match_seed * 2)
            opponent_actions.reset(match_seed * 2 + 1)
            observations, masks, _ = env.reset(seed=match_seed)
            terminated = False
            truncated = False
            while not (terminated or truncated) and len(losses) < requested:
                learner_action = learner_actions.select_action(
                    observations[0], masks[0], player=0, env=env
                )
                opponent_action = opponent_actions.select_action(
                    observations[1], masks[1], player=1, env=env
                )
                next_observations, step_rewards, terminated, truncated, info = env.step(
                    (learner_action, opponent_action)
                )
                next_masks = tuple(
                    np.asarray(value, dtype=np.float32)
                    for value in info["action_masks"]
                )
                replay.add(
                    observations[0],
                    masks[0],
                    learner_action,
                    step_rewards[0],
                    next_observations[0],
                    next_masks[0],
                    terminated,
                    truncated,
                )
                transitions += 1
                observations = next_observations
                masks = next_masks
                if len(replay) >= batch_size:
                    metrics = agent.update(replay.sample(batch_size))
                    loss = float(metrics["td_loss"])
                    if not math.isfinite(loss):
                        raise RuntimeError("Tiny Battle-DQN smoke produced a non-finite loss.")
                    losses.append(loss)
    finally:
        env.close()
    return {
        "passed": len(losses) == requested and all(math.isfinite(value) for value in losses),
        "requested_updates": requested,
        "completed_updates": len(losses),
        "batch_size": batch_size,
        "transitions": transitions,
        "episodes": episodes,
        "losses": losses,
        "initialized_from": initialized_from,
        "opponent": "random",
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rules = BattleRulesConfig(max_steps=args.max_steps)
    rewards = BattleRewardConfig()
    contract = _native_contract_probe(args, rules=rules, rewards=rewards)
    if args.checkpoint is None:
        agent = BattleDQNAgent(
            BattleDQNConfig(seed=args.seed + 20, device=args.device)
        )
        learner = BattleDQNPolicy(
            agent=agent,
            identifier="fresh_battle_dqn",
            kind="learner",
            deterministic=True,
            epsilon=0.0,
        )
        learner_metadata: dict[str, object] = {}
        initialized_from = None
    else:
        learner = load_evaluation_policy(
            args.checkpoint,
            device=args.device,
            identifier=Path(args.checkpoint).stem,
            kind="learner",
        )
        agent = learner.agent
        learner_metadata = policy_checkpoint_metadata(learner)
        initialized_from = str(args.checkpoint)
    opponent = (
        RandomBattlePolicy(seed=args.seed + 1)
        if args.opponent == "random"
        else ColdClearBattlePolicy(think_ms=args.cold_clear_think_ms)
    )
    env_factory = make_env_factory(
        seed=args.seed + 1,
        lib_path=args.lib,
        rules=rules,
        rewards=rewards,
    )
    matches, summary = evaluate_paired_battles(
        learner,
        opponent,
        env_factory=env_factory,
        matches=args.matches,
        seed=args.seed + 1,
    )
    audit = None
    if args.repeat_determinism:
        repeated, _ = evaluate_paired_battles(
            learner,
            opponent,
            env_factory=env_factory,
            matches=args.matches,
            seed=args.seed + 1,
        )
        audit = compare_repeated_matches(matches, repeated)
    training = _tiny_training_smoke(
        args,
        rules=rules,
        rewards=rewards,
        agent=agent,
        initialized_from=initialized_from,
    )
    illegal = sum(
        sum(int(value) for value in match.illegal_actions) for match in matches
    )
    passed = bool(contract["passed"]) and illegal == 0
    if audit is not None:
        passed = passed and bool(audit["passed"])
    if training is not None:
        passed = passed and bool(training["passed"])
    return {
        "schema_version": 1,
        "algorithm": "battle_dqn",
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "training_steps": learner_metadata.get("training_steps"),
        "wall_clock_training_time": learner_metadata.get(
            "wall_clock_training_time"
        ),
        "checkpoint_metadata": learner_metadata,
        "configuration": {
            "matches": int(args.matches),
            "seed": int(args.seed),
            "max_steps": int(args.max_steps),
            "opponent": args.opponent,
            "learner": learner.identifier,
            "cold_clear_think_ms": int(args.cold_clear_think_ms),
            "deterministic": not (
                args.opponent == "cold_clear" and args.cold_clear_think_ms > 0
            ),
            "device": args.device,
            "native_library": None if args.lib is None else str(args.lib),
        },
        "native_contract": contract,
        "matches": [match.to_dict() for match in matches],
        "summary": summary,
        "determinism_audit": audit,
        "tiny_training": training,
        "gate": {
            "passed": passed,
            "illegal_action_count": illegal,
            "max_illegal_actions": 0,
        },
    }


def _print_human_report(report: dict[str, Any]) -> None:
    print(
        f"Native contract: {'PASS' if report['native_contract']['passed'] else 'FAIL'}"
    )
    summary = report["summary"]
    print(
        f"Matches: {summary['match_count']} "
        f"wins={summary['wins']} losses={summary['losses']} draws={summary['draws']}"
    )
    if report["determinism_audit"] is not None:
        print(
            "Determinism: "
            f"{'PASS' if report['determinism_audit']['passed'] else 'FAIL'}"
        )
    if report["tiny_training"] is not None:
        training = report["tiny_training"]
        print(
            f"Tiny training: {'PASS' if training['passed'] else 'FAIL'} "
            f"({training['completed_updates']} updates)"
        )
    print(f"Smoke: {'PASS' if report['gate']['passed'] else 'FAIL'}")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
        if args.json_output is not None:
            write_json_report(args.json_output, report)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        _print_human_report(report)
        if args.json_output is not None:
            print(f"Report: {args.json_output}")
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
