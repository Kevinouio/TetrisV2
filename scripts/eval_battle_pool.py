"""Evaluate a Battle-DQN learner against every retained pool snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import numpy as np
import torch

from tetris_v2.rl.battle.cli import (
    load_evaluation_policy,
    load_training_checkpoint_view,
    make_env_factory,
    policy_checkpoint_metadata,
    positive_even_int,
    positive_int,
    resolve_pool_checkpoint,
    rules_and_rewards_from_training_config,
    unit_interval,
    write_json_report,
)
from tetris_v2.rl.battle.evaluation import evaluate_paired_battles
from tetris_v2.rl.battle.metrics import evaluate_battle_gate
from tetris_v2.rl.battle.policies import (
    BattleDQNPolicy,
    load_embedded_battle_dqn_policy,
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the current learner against retained self-play snapshots."
    )
    parser.add_argument("training_checkpoint", type=Path)
    parser.add_argument(
        "--learner-checkpoint",
        type=Path,
        default=None,
        help="Optional frozen or training policy; defaults to the trainer checkpoint agent.",
    )
    parser.add_argument(
        "--snapshot",
        action="append",
        default=None,
        metavar="IDENTIFIER",
        help="Evaluate only the named retained snapshot; may be repeated.",
    )
    parser.add_argument("--matches-per-snapshot", type=positive_even_int, default=20)
    parser.add_argument("--seed", type=int, default=950_000)
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=None,
        help="Override the match limit stored in the training checkpoint.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lib", type=Path, default=None)
    parser.add_argument("--min-win-rate", type=unit_interval, default=None)
    fairness = parser.add_mutually_exclusive_group()
    fairness.add_argument(
        "--max-seat-win-rate-gap",
        type=unit_interval,
        default=0.05,
    )
    fairness.add_argument(
        "--no-seat-fairness-gate",
        dest="max_seat_win_rate_gap",
        action="store_const",
        const=None,
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args(argv)


def _select_snapshots(args: argparse.Namespace, frozen: tuple[Any, ...]) -> list[Any]:
    if not frozen:
        raise ValueError("The training checkpoint opponent pool has no frozen snapshots.")
    if not args.snapshot:
        return list(frozen)
    requested = set(args.snapshot)
    selected = [item for item in frozen if item.identifier in requested]
    missing = requested - {item.identifier for item in selected}
    if missing:
        available = ", ".join(item.identifier for item in frozen)
        raise ValueError(
            f"Unknown pool snapshot(s): {', '.join(sorted(missing))}. "
            f"Available: {available}"
        )
    return selected


def run(args: argparse.Namespace) -> dict[str, object]:
    view = load_training_checkpoint_view(
        args.training_checkpoint,
        device=args.device,
    )
    if args.learner_checkpoint is None:
        learner = BattleDQNPolicy(
            agent=view.agent,
            identifier=f"learner_step_{view.global_step:012d}",
            kind="learner",
            deterministic=True,
            epsilon=0.0,
            checkpoint=str(args.training_checkpoint),
        )
        learner_metadata = dict(view.checkpoint_metadata)
    else:
        learner = load_evaluation_policy(
            args.learner_checkpoint,
            device=args.device,
            identifier=Path(args.learner_checkpoint).stem,
            kind="learner",
        )
        learner_metadata = policy_checkpoint_metadata(learner)
    snapshots = _select_snapshots(args, view.opponent_pool.frozen)
    ordered_frozen = view.opponent_pool.frozen
    recent_start = max(0, len(ordered_frozen) - view.opponent_pool.recent_window)
    recent_ids = {item.identifier for item in ordered_frozen[recent_start:]}
    rules, rewards = rules_and_rewards_from_training_config(
        view.training_config,
        max_steps=args.max_steps,
    )
    env_factory = make_env_factory(
        seed=args.seed,
        lib_path=args.lib,
        rules=rules,
        rewards=rewards,
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    evaluations = []
    for descriptor in snapshots:
        try:
            checkpoint = resolve_pool_checkpoint(
                descriptor,
                training_checkpoint=args.training_checkpoint,
            )
        except FileNotFoundError:
            embedded = view.opponent_pool.embedded_checkpoint(descriptor.identifier)
            if embedded is None:
                raise
            opponent = load_embedded_battle_dqn_policy(
                embedded,
                device=args.device,
                identifier=descriptor.identifier,
            )
            checkpoint_label = f"embedded:{descriptor.identifier}"
        else:
            opponent = load_evaluation_policy(
                checkpoint,
                device=args.device,
                identifier=descriptor.identifier,
                kind="checkpoint",
            )
            checkpoint_label = str(checkpoint)
        matches, summary = evaluate_paired_battles(
            learner,
            opponent,
            env_factory=env_factory,
            matches=args.matches_per_snapshot,
            seed=args.seed,
        )
        gate = evaluate_battle_gate(
            summary,
            min_win_rate=args.min_win_rate,
            max_seat_win_rate_gap=args.max_seat_win_rate_gap,
            max_illegal_actions=0,
        )
        gate["enabled"] = True
        opponent_metadata = policy_checkpoint_metadata(opponent)
        evaluations.append(
            {
                "snapshot": descriptor.to_dict(),
                "pool_bucket": (
                    "recent" if descriptor.identifier in recent_ids else "older"
                ),
                "resolved_checkpoint": checkpoint_label,
                "training_steps": opponent_metadata.get("training_steps"),
                "wall_clock_training_time": opponent_metadata.get(
                    "wall_clock_training_time"
                ),
                "checkpoint_metadata": opponent_metadata,
                "matches": [match.to_dict() for match in matches],
                "summary": summary,
                "gate": gate,
            }
        )

    win_rates = [float(item["summary"]["win_rate"]) for item in evaluations]
    score_rates = [float(item["summary"]["score_rate"]) for item in evaluations]
    aggregate = {
        "snapshot_count": len(evaluations),
        "total_matches": len(evaluations) * int(args.matches_per_snapshot),
        "minimum_win_rate": min(win_rates),
        "mean_win_rate": mean(win_rates),
        "minimum_score_rate": min(score_rates),
        "mean_score_rate": mean(score_rates),
        "passed_snapshot_count": sum(bool(item["gate"]["passed"]) for item in evaluations),
        "all_gates_passed": all(bool(item["gate"]["passed"]) for item in evaluations),
    }
    return {
        "schema_version": 1,
        "algorithm": "battle_dqn",
        "training_checkpoint": str(args.training_checkpoint),
        "learner_checkpoint": str(
            args.learner_checkpoint or args.training_checkpoint
        ),
        "learner_identifier": learner.identifier,
        "training_steps": learner_metadata.get("training_steps"),
        "wall_clock_training_time": learner_metadata.get(
            "wall_clock_training_time"
        ),
        "learner_checkpoint_metadata": learner_metadata,
        "training_state": {
            "global_step": view.global_step,
            "episode_index": view.episode_index,
            "replay_loaded": False,
        },
        "configuration": {
            "matches_per_snapshot": int(args.matches_per_snapshot),
            "seed": int(args.seed),
            "paired_seats": True,
            "same_seed_block_per_snapshot": True,
            "deterministic": True,
            "rules": rules.to_dict(),
            "rewards": rewards.to_dict(),
        },
        "evaluations": evaluations,
        "summary": aggregate,
        "gate": {
            "passed": aggregate["all_gates_passed"],
            "min_win_rate": args.min_win_rate,
            "max_seat_win_rate_gap": args.max_seat_win_rate_gap,
            "max_illegal_actions": 0,
        },
    }


def _print_human_report(report: dict[str, Any]) -> None:
    for item in report["evaluations"]:
        summary = item["summary"]
        identifier = item["snapshot"]["identifier"]
        status = "PASS" if item["gate"]["passed"] else "FAIL"
        print(
            f"{identifier}: win={float(summary['win_rate']):.2%} "
            f"score={float(summary['score_rate']):.2%} "
            f"seat_gap={float(summary['seat_win_rate_gap']):.2%} {status}"
        )
    summary = report["summary"]
    print(
        f"{summary['snapshot_count']} snapshots, {summary['total_matches']} matches: "
        f"{'PASS' if summary['all_gates_passed'] else 'FAIL'}"
    )


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
