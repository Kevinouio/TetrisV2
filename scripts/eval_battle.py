"""Evaluate a Battle-DQN checkpoint with paired physical seats."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from tetris_v2.rl.battle.cli import (
    DEFAULT_BATTLE_GATE_WIN_RATES,
    load_evaluation_policy,
    make_env_factory,
    make_opponent_policy,
    nonnegative_int,
    policy_checkpoint_metadata,
    positive_even_int,
    positive_int,
    unit_interval,
    write_json_report,
)
from tetris_v2.rl.battle.config import BattleRewardConfig, BattleRulesConfig
from tetris_v2.rl.battle.evaluation import (
    ScriptedGarbagePressure,
    compare_repeated_matches,
    evaluate_paired_battles,
)
from tetris_v2.rl.battle.metrics import (
    BattleMatchMetrics,
    evaluate_battle_gate,
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one Battle-DQN policy from both physical seats."
    )
    parser.add_argument("checkpoint", type=Path, help="Frozen or training checkpoint")
    parser.add_argument(
        "--opponent",
        choices=("random", "cold_clear", "checkpoint", "self"),
        default="random",
    )
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--matches", type=positive_even_int, default=500)
    parser.add_argument("--seed", type=int, default=900_000)
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=None,
        help="Override the checkpoint rule (default: stored value, otherwise 500).",
    )
    parser.add_argument(
        "--garbage-delay",
        type=nonnegative_int,
        default=None,
        help="Override the checkpoint rule (default: stored value, otherwise 1).",
    )
    parser.add_argument(
        "--attack-table",
        type=nonnegative_int,
        nargs=5,
        default=None,
        metavar=("ZERO", "SINGLE", "DOUBLE", "TRIPLE", "TETRIS"),
        help="Override the checkpoint attack table.",
    )
    piece_seeds = parser.add_mutually_exclusive_group()
    piece_seeds.add_argument(
        "--independent-piece-seeds",
        dest="mirrored_piece_seeds",
        action="store_false",
        help="Override the checkpoint and give the seats independent piece streams.",
    )
    piece_seeds.add_argument(
        "--mirrored-piece-seeds",
        dest="mirrored_piece_seeds",
        action="store_true",
        help="Override the checkpoint and give the seats mirrored piece streams.",
    )
    parser.set_defaults(mirrored_piece_seeds=None)
    parser.add_argument(
        "--cold-clear-think-ms",
        type=nonnegative_int,
        default=0,
        help="0 uses deterministic fixed-work Cold Clear evaluation.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lib", type=Path, default=None)

    parser.add_argument("--pressure-rows", type=nonnegative_int, default=0)
    parser.add_argument("--pressure-interval", type=nonnegative_int, default=0)
    parser.add_argument("--pressure-hole", type=int, choices=range(10), default=4)
    parser.add_argument("--pressure-delay", type=nonnegative_int, default=0)
    parser.add_argument(
        "--repeat-determinism",
        action="store_true",
        help="Repeat the full fixed-seed pass and compare every non-timing field.",
    )

    parser.add_argument("--min-win-rate", type=unit_interval, default=None)
    parser.add_argument(
        "--no-default-win-gate",
        action="store_true",
        help="Do not apply the 95%% random / 65%% Cold Clear default gate.",
    )
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
    parser.add_argument("--max-illegal-actions", type=nonnegative_int, default=0)
    parser.add_argument("--min-survival-steps", type=nonnegative_int, default=None)

    parser.add_argument("--json", action="store_true", help="Print only JSON")
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args(argv)


def _minimum_win_rate(args: argparse.Namespace) -> float | None:
    if args.min_win_rate is not None:
        return float(args.min_win_rate)
    if args.no_default_win_gate:
        return None
    return DEFAULT_BATTLE_GATE_WIN_RATES.get(str(args.opponent))


def resolve_battle_configuration(
    args: argparse.Namespace,
    checkpoint_metadata: Mapping[str, object],
) -> tuple[BattleRulesConfig, BattleRewardConfig]:
    """Inherit a frozen policy's environment contract unless explicitly overridden."""

    default_rules = BattleRulesConfig().to_dict()
    stored_rules = checkpoint_metadata.get("rules", {})
    if not isinstance(stored_rules, Mapping):
        raise ValueError("Battle checkpoint rules configuration is malformed.")
    rule_values = {**default_rules, **dict(stored_rules)}
    if args.attack_table is not None:
        rule_values["attack_table"] = tuple(int(value) for value in args.attack_table)
    if args.garbage_delay is not None:
        rule_values["garbage_delay"] = int(args.garbage_delay)
    if args.max_steps is not None:
        rule_values["max_steps"] = int(args.max_steps)
    if args.mirrored_piece_seeds is not None:
        rule_values["mirrored_piece_seeds"] = bool(args.mirrored_piece_seeds)
    rules = BattleRulesConfig(**rule_values)

    default_rewards = BattleRewardConfig().to_dict()
    stored_rewards = checkpoint_metadata.get("rewards", {})
    if not isinstance(stored_rewards, Mapping):
        raise ValueError("Battle checkpoint reward configuration is malformed.")
    rewards = BattleRewardConfig(**{**default_rewards, **dict(stored_rewards)})
    return rules, rewards


def _survival_summary(matches: Sequence[BattleMatchMetrics]) -> dict[str, object]:
    steps = [int(match.match_steps) for match in matches]
    learner_placements = [
        int(match.placements[match.learner_seat]) for match in matches
    ]
    return {
        "minimum_match_steps": min(steps),
        "mean_match_steps": mean(steps),
        "minimum_learner_placements": min(learner_placements),
        "mean_learner_placements": mean(learner_placements),
    }


def build_report(
    args: argparse.Namespace,
    *,
    matches: Sequence[BattleMatchMetrics],
    summary: dict[str, object],
    determinism: Optional[dict[str, object]],
    learner_identifier: str,
    opponent_identifier: str,
    learner_checkpoint_metadata: Optional[dict[str, object]] = None,
    opponent_checkpoint_metadata: Optional[dict[str, object]] = None,
    rules: BattleRulesConfig | None = None,
    rewards: BattleRewardConfig | None = None,
) -> dict[str, object]:
    gate = evaluate_battle_gate(
        summary,
        min_win_rate=_minimum_win_rate(args),
        max_seat_win_rate_gap=args.max_seat_win_rate_gap,
        max_illegal_actions=int(args.max_illegal_actions),
    )
    # The illegal-action threshold is always active, even when the optional
    # win-rate and seat-fairness thresholds are disabled.
    gate["enabled"] = True
    survival = _survival_summary(matches)
    minimum_survival = args.min_survival_steps
    if minimum_survival is not None:
        gate["enabled"] = True
        if int(survival["minimum_match_steps"]) < int(minimum_survival):
            gate["failures"].append(
                "minimum_match_steps "
                f"{survival['minimum_match_steps']} < {int(minimum_survival)}"
            )
            gate["passed"] = False
    gate["min_survival_steps"] = minimum_survival
    if determinism is not None and not bool(determinism.get("passed", False)):
        gate["failures"].append("determinism audit failed")
        gate["passed"] = False
    execution_deterministic = not (
        args.opponent == "cold_clear" and int(args.cold_clear_think_ms) > 0
    )
    if not execution_deterministic and _minimum_win_rate(args) is not None:
        gate["failures"].append(
            "Cold Clear win-rate gates require --cold-clear-think-ms 0"
        )
        gate["passed"] = False
    gate["repeat_determinism_required"] = determinism is not None
    pressure = ScriptedGarbagePressure(
        rows=args.pressure_rows,
        interval=args.pressure_interval,
        hole_column=args.pressure_hole,
        delay=args.pressure_delay,
    )
    learner_metadata = dict(learner_checkpoint_metadata or {})
    opponent_metadata = dict(opponent_checkpoint_metadata or {})
    effective_rules = rules or BattleRulesConfig()
    effective_rewards = rewards or BattleRewardConfig()
    return {
        "schema_version": 1,
        "algorithm": "battle_dqn",
        "checkpoint": str(args.checkpoint),
        "training_steps": learner_metadata.get("training_steps"),
        "wall_clock_training_time": learner_metadata.get(
            "wall_clock_training_time"
        ),
        "checkpoint_metadata": learner_metadata,
        "learner_identifier": learner_identifier,
        "opponent": {
            "mode": args.opponent,
            "identifier": opponent_identifier,
            "checkpoint": (
                str(args.opponent_checkpoint)
                if args.opponent_checkpoint is not None
                else (str(args.checkpoint) if args.opponent == "self" else None)
            ),
            "checkpoint_metadata": opponent_metadata,
        },
        "configuration": {
            "matches": int(args.matches),
            "seed": int(args.seed),
            "paired_seats": True,
            "deterministic": execution_deterministic,
            "device": str(args.device),
            "rules": effective_rules.to_dict(),
            "rewards": effective_rewards.to_dict(),
            "scripted_pressure": asdict(pressure),
            "repeat_determinism": bool(args.repeat_determinism),
            "cold_clear_think_ms": int(args.cold_clear_think_ms),
        },
        "matches": [match.to_dict() for match in matches],
        "summary": summary,
        "survival": survival,
        "gate": gate,
        "determinism_audit": determinism,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.opponent == "checkpoint" and args.opponent_checkpoint is None:
        raise ValueError("--opponent-checkpoint is required with --opponent checkpoint.")
    if args.opponent != "checkpoint" and args.opponent_checkpoint is not None:
        raise ValueError("--opponent-checkpoint is only valid with --opponent checkpoint.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    learner = load_evaluation_policy(
        args.checkpoint,
        device=args.device,
        identifier=Path(args.checkpoint).stem,
        kind="learner",
    )
    learner_metadata = policy_checkpoint_metadata(learner)
    opponent = make_opponent_policy(
        args.opponent,
        learner_checkpoint=args.checkpoint,
        opponent_checkpoint=args.opponent_checkpoint,
        device=args.device,
        cold_clear_think_ms=args.cold_clear_think_ms,
    )
    rules, rewards = resolve_battle_configuration(
        args,
        learner_metadata,
    )
    pressure = ScriptedGarbagePressure(
        rows=args.pressure_rows,
        interval=args.pressure_interval,
        hole_column=args.pressure_hole,
        delay=args.pressure_delay,
    )
    env_factory = make_env_factory(
        seed=args.seed,
        lib_path=args.lib,
        rules=rules,
        rewards=rewards,
    )
    matches, summary = evaluate_paired_battles(
        learner,
        opponent,
        env_factory=env_factory,
        matches=args.matches,
        seed=args.seed,
        pressure=pressure,
    )
    determinism = None
    if args.repeat_determinism:
        repeated, _ = evaluate_paired_battles(
            learner,
            opponent,
            env_factory=env_factory,
            matches=args.matches,
            seed=args.seed,
            pressure=pressure,
        )
        determinism = compare_repeated_matches(matches, repeated)
    return build_report(
        args,
        matches=matches,
        summary=summary,
        determinism=determinism,
        learner_identifier=learner.identifier,
        opponent_identifier=opponent.identifier,
        learner_checkpoint_metadata=learner_metadata,
        opponent_checkpoint_metadata=policy_checkpoint_metadata(opponent),
        rules=rules,
        rewards=rewards,
    )


def _print_human_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    gate = report["gate"]
    print(
        f"{summary['match_count']} paired-seat matches: "
        f"wins={summary['wins']} losses={summary['losses']} draws={summary['draws']}"
    )
    print(
        f"win_rate={float(summary['win_rate']):.2%} "
        f"score_rate={float(summary['score_rate']):.2%} "
        f"seat_gap={float(summary['seat_win_rate_gap']):.2%}"
    )
    print(
        f"illegal_actions={sum(int(value) for value in summary['illegal_action_count'])} "
        f"minimum_match_steps={report['survival']['minimum_match_steps']}"
    )
    if gate["enabled"]:
        print(f"Gate: {'PASS' if gate['passed'] else 'FAIL'}")
        for failure in gate["failures"]:
            print(f"  {failure}")
    audit = report["determinism_audit"]
    if audit is not None:
        print(f"Determinism: {'PASS' if audit['passed'] else 'FAIL'}")


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
    audit = report["determinism_audit"]
    deterministic = audit is None or bool(audit["passed"])
    return 0 if report["gate"]["passed"] and deterministic else 1


if __name__ == "__main__":
    raise SystemExit(main())
