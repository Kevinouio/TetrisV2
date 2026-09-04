"""Build a paired-seat round-robin matrix for Battle-DQN checkpoints."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from tetris_v2.rl.battle.cli import (
    load_evaluation_policy,
    make_env_factory,
    nonnegative_int,
    policy_checkpoint_metadata,
    positive_even_int,
    positive_int,
    write_json_report,
)
from tetris_v2.rl.battle.config import BattleRulesConfig
from tetris_v2.rl.battle.evaluation import evaluate_paired_battles
from tetris_v2.rl.battle.metrics import (
    BattleMatchMetrics,
    outcome_matrices,
    win_matrix,
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate every unordered Battle-DQN checkpoint pairing."
    )
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional unique labels in the same order as the checkpoints.",
    )
    parser.add_argument("--matches-per-pair", type=positive_even_int, default=20)
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--seed-stride", type=positive_int, default=100_000)
    parser.add_argument("--max-steps", type=positive_int, default=500)
    parser.add_argument("--garbage-delay", type=nonnegative_int, default=1)
    parser.add_argument(
        "--attack-table",
        type=nonnegative_int,
        nargs=5,
        default=[0, 0, 1, 2, 4],
        metavar=("ZERO", "SINGLE", "DOUBLE", "TRIPLE", "TETRIS"),
    )
    parser.add_argument("--independent-piece-seeds", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lib", type=Path, default=None)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("runs/battle_matrix/matrix.json"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("runs/battle_matrix/matrix.csv"),
    )
    parser.add_argument("--json", action="store_true", help="Also print JSON")
    return parser.parse_args(argv)


def _policy_labels(args: argparse.Namespace) -> list[str]:
    if len(args.checkpoints) < 2:
        raise ValueError("Battle matrix evaluation requires at least two checkpoints.")
    labels = (
        [str(value) for value in args.labels]
        if args.labels is not None
        else [Path(value).stem for value in args.checkpoints]
    )
    if len(labels) != len(args.checkpoints):
        raise ValueError("--labels must contain one label per checkpoint.")
    if len(set(labels)) != len(labels):
        raise ValueError("Battle matrix policy labels must be unique; use --labels.")
    return labels


def _relative_winner(match: BattleMatchMetrics) -> int | None:
    if match.result == "draw":
        return None
    return 0 if match.result == "win" else 1


def build_matrix_report(
    *,
    labels: Sequence[str],
    checkpoints: Sequence[Path],
    pair_results: Sequence[tuple[str, str, Sequence[BattleMatchMetrics], dict[str, object]]],
    configuration: dict[str, object],
    policy_metadata: Optional[Sequence[dict[str, object]]] = None,
) -> dict[str, object]:
    matrix_rows: list[tuple[str, str, int | None]] = []
    pairs = []
    illegal_actions = 0
    for row_id, column_id, matches, summary in pair_results:
        raw_matches = []
        for match in matches:
            winner = _relative_winner(match)
            reverse_winner = None if winner is None else 1 - winner
            matrix_rows.append((row_id, column_id, winner))
            matrix_rows.append((column_id, row_id, reverse_winner))
            illegal_actions += sum(int(value) for value in match.illegal_actions)
            raw_matches.append(match.to_dict())
        pairs.append(
            {
                "row_policy": row_id,
                "column_policy": column_id,
                "matches": raw_matches,
                "summary_from_row_perspective": summary,
            }
        )
    rates, counts = win_matrix(labels, matrix_rows)
    win_rates, wins, losses, draws = outcome_matrices(labels, matrix_rows)
    metadata_rows = (
        list(policy_metadata)
        if policy_metadata is not None
        else [{} for _ in checkpoints]
    )
    if len(metadata_rows) != len(checkpoints):
        raise ValueError("Policy checkpoint metadata must align with checkpoints.")
    return {
        "schema_version": 1,
        "algorithm": "battle_dqn",
        "policies": [
            {
                "identifier": label,
                "checkpoint": str(checkpoint),
                "training_steps": metadata.get("training_steps"),
                "wall_clock_training_time": metadata.get(
                    "wall_clock_training_time"
                ),
                "checkpoint_metadata": metadata,
            }
            for label, checkpoint, metadata in zip(
                labels, checkpoints, metadata_rows, strict=True
            )
        ],
        "configuration": configuration,
        "pairs": pairs,
        "matrix": {
            "policy_order": list(labels),
            "win_rates": win_rates,
            "score_rates": rates,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "match_counts": counts,
        },
        "gate": {
            "passed": illegal_actions == 0,
            "max_illegal_actions": 0,
            "illegal_action_count": illegal_actions,
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    labels = _policy_labels(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policies = [
        load_evaluation_policy(
            checkpoint,
            device=args.device,
            identifier=label,
            kind="checkpoint",
        )
        for label, checkpoint in zip(labels, args.checkpoints, strict=True)
    ]
    rules = BattleRulesConfig(
        attack_table=tuple(args.attack_table),
        garbage_delay=args.garbage_delay,
        max_steps=args.max_steps,
        mirrored_piece_seeds=not args.independent_piece_seeds,
    )
    env_factory = make_env_factory(
        seed=args.seed,
        lib_path=args.lib,
        rules=rules,
    )
    pair_results = []
    for pair_index, (row, column) in enumerate(combinations(range(len(labels)), 2)):
        pair_seed = int(args.seed) + pair_index * int(args.seed_stride)
        matches, summary = evaluate_paired_battles(
            policies[row],
            policies[column],
            env_factory=env_factory,
            matches=args.matches_per_pair,
            seed=pair_seed,
        )
        pair_results.append((labels[row], labels[column], matches, summary))
    return build_matrix_report(
        labels=labels,
        checkpoints=args.checkpoints,
        pair_results=pair_results,
        configuration={
            "matches_per_pair": int(args.matches_per_pair),
            "seed": int(args.seed),
            "seed_stride": int(args.seed_stride),
            "paired_seats": True,
            "deterministic": True,
            "rules": rules.to_dict(),
        },
        policy_metadata=[policy_checkpoint_metadata(policy) for policy in policies],
    )


def write_matrix_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = report["matrix"]
    labels = matrix["policy_order"]
    win_rates = matrix["win_rates"]
    score_rates = matrix["score_rates"]
    wins = matrix["wins"]
    losses = matrix["losses"]
    draws = matrix["draws"]
    counts = matrix["match_counts"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["policy", *labels])
        rows = zip(
            labels,
            win_rates,
            score_rates,
            wins,
            losses,
            draws,
            counts,
            strict=True,
        )
        for label, win_rate_row, score_rate_row, win_row, loss_row, draw_row, count_row in rows:
            cells = []
            cell_rows = zip(
                win_rate_row,
                score_rate_row,
                win_row,
                loss_row,
                draw_row,
                count_row,
                strict=True,
            )
            for win_rate, score_rate, wins_cell, losses_cell, draws_cell, count in cell_rows:
                cells.append(
                    ""
                    if win_rate is None
                    else (
                        f"win={float(win_rate):.6f} "
                        f"score={float(score_rate):.6f} "
                        f"W-L-D={int(wins_cell)}-{int(losses_cell)}-{int(draws_cell)} "
                        f"(n={int(count)})"
                    )
                )
            writer.writerow([label, *cells])


def _print_human_report(report: dict[str, Any]) -> None:
    labels = report["matrix"]["policy_order"]
    win_rates = report["matrix"]["win_rates"]
    wins = report["matrix"]["wins"]
    losses = report["matrix"]["losses"]
    draws = report["matrix"]["draws"]
    counts = report["matrix"]["match_counts"]
    print("win-rate matrix (row perspective; W-L-D/n)")
    print("policy\t" + "\t".join(labels))
    rows = zip(labels, win_rates, wins, losses, draws, counts, strict=True)
    for label, rate_row, win_row, loss_row, draw_row, count_row in rows:
        cell_rows = zip(
            rate_row, win_row, loss_row, draw_row, count_row, strict=True
        )
        cells = [
            "-"
            if rate is None
            else (
                f"{float(rate):.3f} "
                f"[{int(win)}-{int(loss)}-{int(draw)}/{int(count)}]"
            )
            for rate, win, loss, draw, count in cell_rows
        ]
        print(label + "\t" + "\t".join(cells))

    rates = report["matrix"]["score_rates"]
    print("score-rate matrix (row perspective)")
    print("policy\t" + "\t".join(labels))
    for label, row in zip(labels, rates, strict=True):
        cells = ["-" if value is None else f"{float(value):.3f}" for value in row]
        print(label + "\t" + "\t".join(cells))
    print(
        f"Illegal actions: {report['gate']['illegal_action_count']} "
        f"({'PASS' if report['gate']['passed'] else 'FAIL'})"
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
        write_json_report(args.json_output, report)
        write_matrix_csv(args.csv_output, report)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        _print_human_report(report)
        print(f"JSON: {args.json_output}")
        print(f"CSV: {args.csv_output}")
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
