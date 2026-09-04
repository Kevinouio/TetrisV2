"""Machine-readable metrics for deterministic two-player battle evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np


BattleResult = Literal["win", "loss", "draw"]


@dataclass(frozen=True)
class BattleMatchMetrics:
    """One match, recorded from the evaluated learner's perspective."""

    match: int
    seed: int
    learner_seat: int
    opponent_type: str
    opponent_checkpoint: str | None
    result: BattleResult
    winner: int | None
    match_steps: int
    placements: tuple[int, int]
    lines_cleared: tuple[int, int]
    attack_sent: tuple[int, int]
    garbage_received: tuple[int, int]
    garbage_cancelled: tuple[int, int]
    top_out: tuple[bool, bool]
    average_height: tuple[float, float]
    maximum_height: tuple[int, int]
    average_holes: tuple[float, float]
    illegal_actions: tuple[int, int]
    inference_ms: tuple[float, float]
    returns: tuple[float, float]
    reward_components: Mapping[str, float] = field(default_factory=dict)
    trace_hash: str | None = None
    garbage_applied: tuple[int, int] = (0, 0)
    attack_generated: tuple[int, int] = (0, 0)
    inference_decisions: tuple[int, int] = (0, 0)
    board_samples: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.learner_seat not in (0, 1):
            raise ValueError("learner_seat must be 0 or 1")
        if self.winner not in (None, 0, 1):
            raise ValueError("winner must be 0, 1, or None")
        expected = "draw" if self.winner is None else (
            "win" if self.winner == self.learner_seat else "loss"
        )
        if self.result != expected:
            raise ValueError(
                f"result={self.result!r} is inconsistent with learner_seat={self.learner_seat} "
                f"and winner={self.winner}"
            )

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["reward_components"] = dict(self.reward_components)
        # ``attack_sent`` is retained for report compatibility. Its value is
        # post-cancellation garbage sent, which is also exposed under the
        # unambiguous name below.
        value["garbage_sent"] = value["attack_sent"]
        return value


def _seat_mean(matches: Sequence[BattleMatchMetrics], name: str) -> list[float]:
    values = [getattr(match, name) for match in matches]
    return [float(np.mean([value[seat] for value in values])) for seat in (0, 1)]


def _perspective_mean(
    matches: Sequence[BattleMatchMetrics],
    name: str,
) -> tuple[float, float]:
    values = [getattr(match, name) for match in matches]
    learner = [value[match.learner_seat] for match, value in zip(matches, values, strict=True)]
    opponent = [
        value[1 - match.learner_seat]
        for match, value in zip(matches, values, strict=True)
    ]
    return float(np.mean(learner)), float(np.mean(opponent))


def _weighted_seat_mean(
    matches: Sequence[BattleMatchMetrics],
    name: str,
    weight_name: str,
) -> list[float]:
    values = [getattr(match, name) for match in matches]
    weights = [getattr(match, weight_name) for match in matches]
    output = []
    for seat in (0, 1):
        seat_weights = np.asarray([value[seat] for value in weights], dtype=np.float64)
        seat_values = np.asarray([value[seat] for value in values], dtype=np.float64)
        total = float(seat_weights.sum())
        output.append(
            float(np.dot(seat_values, seat_weights) / total)
            if total > 0.0
            else float(np.mean(seat_values))
        )
    return output


def _weighted_perspective_mean(
    matches: Sequence[BattleMatchMetrics],
    name: str,
    weight_name: str,
) -> tuple[float, float]:
    learner_values: list[float] = []
    learner_weights: list[float] = []
    opponent_values: list[float] = []
    opponent_weights: list[float] = []
    for match in matches:
        values = getattr(match, name)
        weights = getattr(match, weight_name)
        learner_seat = match.learner_seat
        opponent_seat = 1 - learner_seat
        learner_values.append(float(values[learner_seat]))
        learner_weights.append(float(weights[learner_seat]))
        opponent_values.append(float(values[opponent_seat]))
        opponent_weights.append(float(weights[opponent_seat]))

    def weighted(values: list[float], weights: list[float]) -> float:
        total = float(sum(weights))
        if total <= 0.0:
            return float(np.mean(values))
        return float(np.dot(values, weights) / total)

    return (
        weighted(learner_values, learner_weights),
        weighted(opponent_values, opponent_weights),
    )


def summarize_battle_matches(matches: Sequence[BattleMatchMetrics]) -> dict[str, object]:
    """Aggregate raw battle matches without hiding seat-specific behavior."""

    if not matches:
        raise ValueError("At least one battle match is required.")

    count = len(matches)
    wins = sum(match.result == "win" for match in matches)
    losses = sum(match.result == "loss" for match in matches)
    draws = sum(match.result == "draw" for match in matches)
    p1_wins = sum(match.winner == 0 for match in matches)
    p2_wins = sum(match.winner == 1 for match in matches)
    total_placements = [
        sum(match.placements[seat] for match in matches) for seat in (0, 1)
    ]
    total_attack_generated = [
        sum(match.attack_generated[seat] for match in matches) for seat in (0, 1)
    ]
    total_garbage_sent = [
        sum(match.attack_sent[seat] for match in matches) for seat in (0, 1)
    ]
    learner_inference, opponent_inference = _weighted_perspective_mean(
        matches, "inference_ms", "inference_decisions"
    )
    learner_placements, opponent_placements = _perspective_mean(matches, "placements")
    learner_attack_generated, opponent_attack_generated = _perspective_mean(
        matches, "attack_generated"
    )
    learner_garbage_sent, opponent_garbage_sent = _perspective_mean(
        matches, "attack_sent"
    )
    learner_lines, opponent_lines = _perspective_mean(matches, "lines_cleared")
    learner_received, opponent_received = _perspective_mean(
        matches, "garbage_received"
    )
    learner_applied, opponent_applied = _perspective_mean(matches, "garbage_applied")
    learner_cancelled, opponent_cancelled = _perspective_mean(
        matches, "garbage_cancelled"
    )
    learner_height, opponent_height = _weighted_perspective_mean(
        matches, "average_height", "board_samples"
    )
    learner_holes, opponent_holes = _weighted_perspective_mean(
        matches, "average_holes", "board_samples"
    )
    learner_return, opponent_return = _perspective_mean(matches, "returns")
    learner_total_placements = sum(
        match.placements[match.learner_seat] for match in matches
    )
    opponent_total_placements = sum(
        match.placements[1 - match.learner_seat] for match in matches
    )
    learner_total_attack_generated = sum(
        match.attack_generated[match.learner_seat] for match in matches
    )
    opponent_total_attack_generated = sum(
        match.attack_generated[1 - match.learner_seat] for match in matches
    )
    learner_total_garbage_sent = sum(
        match.attack_sent[match.learner_seat] for match in matches
    )
    opponent_total_garbage_sent = sum(
        match.attack_sent[1 - match.learner_seat] for match in matches
    )

    return {
        "match_count": count,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / count,
        "loss_rate": losses / count,
        "draw_rate": draws / count,
        "score_rate": (wins + 0.5 * draws) / count,
        "player_1_win_rate": p1_wins / count,
        "player_2_win_rate": p2_wins / count,
        "seat_win_rate_gap": abs(p1_wins - p2_wins) / count,
        "average_placements": _seat_mean(matches, "placements"),
        "median_match_length": float(median(match.match_steps for match in matches)),
        "average_lines_cleared": _seat_mean(matches, "lines_cleared"),
        "average_attack_generated": _seat_mean(matches, "attack_generated"),
        "average_attack_sent": _seat_mean(matches, "attack_sent"),
        "average_garbage_sent": _seat_mean(matches, "attack_sent"),
        "average_garbage_received": _seat_mean(matches, "garbage_received"),
        "average_garbage_applied": _seat_mean(matches, "garbage_applied"),
        "average_garbage_cancelled": _seat_mean(matches, "garbage_cancelled"),
        "attack_per_100_pieces": [
            100.0 * total_attack_generated[seat] / max(1, total_placements[seat])
            for seat in (0, 1)
        ],
        "garbage_sent_per_100_pieces": [
            100.0 * total_garbage_sent[seat] / max(1, total_placements[seat])
            for seat in (0, 1)
        ],
        "top_out_count": [
            sum(match.top_out[seat] for match in matches) for seat in (0, 1)
        ],
        "average_height": _weighted_seat_mean(
            matches, "average_height", "board_samples"
        ),
        "maximum_height": [
            max(match.maximum_height[seat] for match in matches) for seat in (0, 1)
        ],
        "average_holes": _weighted_seat_mean(
            matches, "average_holes", "board_samples"
        ),
        "illegal_action_count": [
            sum(match.illegal_actions[seat] for match in matches) for seat in (0, 1)
        ],
        "mean_inference_ms": _weighted_seat_mean(
            matches, "inference_ms", "inference_decisions"
        ),
        "learner_mean_inference_ms": learner_inference,
        "opponent_mean_inference_ms": opponent_inference,
        "learner_average_placements": learner_placements,
        "opponent_average_placements": opponent_placements,
        "learner_average_lines_cleared": learner_lines,
        "opponent_average_lines_cleared": opponent_lines,
        "learner_average_attack_generated": learner_attack_generated,
        "opponent_average_attack_generated": opponent_attack_generated,
        "learner_average_attack_sent": learner_garbage_sent,
        "opponent_average_attack_sent": opponent_garbage_sent,
        "learner_average_garbage_sent": learner_garbage_sent,
        "opponent_average_garbage_sent": opponent_garbage_sent,
        "learner_attack_per_100_pieces": (
            100.0
            * learner_total_attack_generated
            / max(1, learner_total_placements)
        ),
        "opponent_attack_per_100_pieces": (
            100.0
            * opponent_total_attack_generated
            / max(1, opponent_total_placements)
        ),
        "learner_garbage_sent_per_100_pieces": (
            100.0 * learner_total_garbage_sent / max(1, learner_total_placements)
        ),
        "opponent_garbage_sent_per_100_pieces": (
            100.0 * opponent_total_garbage_sent / max(1, opponent_total_placements)
        ),
        "learner_average_garbage_received": learner_received,
        "opponent_average_garbage_received": opponent_received,
        "learner_average_garbage_applied": learner_applied,
        "opponent_average_garbage_applied": opponent_applied,
        "learner_average_garbage_cancelled": learner_cancelled,
        "opponent_average_garbage_cancelled": opponent_cancelled,
        "learner_top_out_count": sum(
            match.top_out[match.learner_seat] for match in matches
        ),
        "opponent_top_out_count": sum(
            match.top_out[1 - match.learner_seat] for match in matches
        ),
        "learner_average_height": learner_height,
        "opponent_average_height": opponent_height,
        "learner_maximum_height": max(
            match.maximum_height[match.learner_seat] for match in matches
        ),
        "opponent_maximum_height": max(
            match.maximum_height[1 - match.learner_seat] for match in matches
        ),
        "learner_average_holes": learner_holes,
        "opponent_average_holes": opponent_holes,
        "learner_illegal_action_count": sum(
            match.illegal_actions[match.learner_seat] for match in matches
        ),
        "opponent_illegal_action_count": sum(
            match.illegal_actions[1 - match.learner_seat] for match in matches
        ),
        "learner_mean_return": learner_return,
        "opponent_mean_return": opponent_return,
        "mean_return": _seat_mean(matches, "returns"),
    }


def evaluate_battle_gate(
    summary: Mapping[str, object],
    *,
    min_win_rate: float | None = None,
    max_seat_win_rate_gap: float | None = 0.05,
    max_illegal_actions: int = 0,
) -> dict[str, object]:
    """Evaluate explicit battle acceptance thresholds against a raw summary."""

    failures: list[str] = []
    win_rate = float(summary["win_rate"])
    seat_gap = float(summary["seat_win_rate_gap"])
    illegal_raw = summary["illegal_action_count"]
    if isinstance(illegal_raw, Sequence) and not isinstance(illegal_raw, (str, bytes)):
        illegal = sum(int(value) for value in illegal_raw)
    else:
        illegal = int(illegal_raw)
    if min_win_rate is not None and win_rate < min_win_rate:
        failures.append(f"win_rate {win_rate:.6f} < {min_win_rate:.6f}")
    if max_seat_win_rate_gap is not None and seat_gap > max_seat_win_rate_gap:
        failures.append(
            f"seat_win_rate_gap {seat_gap:.6f} > {max_seat_win_rate_gap:.6f}"
        )
    if illegal > max_illegal_actions:
        failures.append(f"illegal_actions {illegal} > {max_illegal_actions}")
    return {
        "enabled": min_win_rate is not None or max_seat_win_rate_gap is not None,
        "passed": not failures,
        "min_win_rate": min_win_rate,
        "max_seat_win_rate_gap": max_seat_win_rate_gap,
        "max_illegal_actions": max_illegal_actions,
        "failures": failures,
    }


def append_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Append JSON objects one per line, creating the parent directory."""

    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), allow_nan=False, sort_keys=True))
            stream.write("\n")


def win_matrix(
    policy_ids: Sequence[str],
    matches: Sequence[tuple[str, str, int | None]],
) -> tuple[list[list[float | None]], list[list[int]]]:
    """Return score rates and raw counts for ordered policy pairs.

    ``winner`` is ``0`` for the row policy, ``1`` for the column policy, and
    ``None`` for a draw. Diagonal cells are intentionally left empty.
    """

    if len(set(policy_ids)) != len(policy_ids):
        raise ValueError("policy_ids must be unique")
    index = {name: idx for idx, name in enumerate(policy_ids)}
    scores = [[0.0 for _ in policy_ids] for _ in policy_ids]
    counts = [[0 for _ in policy_ids] for _ in policy_ids]
    for row_id, column_id, winner in matches:
        if row_id not in index or column_id not in index:
            raise ValueError("matrix match references an unknown policy")
        if winner not in (None, 0, 1):
            raise ValueError("winner must be 0, 1, or None")
        row = index[row_id]
        column = index[column_id]
        counts[row][column] += 1
        scores[row][column] += 0.5 if winner is None else float(winner == 0)

    rates: list[list[float | None]] = []
    for row in range(len(policy_ids)):
        rate_row: list[float | None] = []
        for column in range(len(policy_ids)):
            if row == column or counts[row][column] == 0:
                rate_row.append(None)
            else:
                rate_row.append(scores[row][column] / counts[row][column])
        rates.append(rate_row)
    return rates, counts


def outcome_matrices(
    policy_ids: Sequence[str],
    matches: Sequence[tuple[str, str, int | None]],
) -> tuple[
    list[list[float | None]],
    list[list[int]],
    list[list[int]],
    list[list[int]],
]:
    """Return directed raw win rates and W/L/D counts for policy pairs.

    ``winner`` is relative to the ordered pair: ``0`` means the row policy
    won, ``1`` means the column policy won, and ``None`` means a draw.
    Diagonal and unevaluated cells have a ``None`` win rate.
    """

    if len(set(policy_ids)) != len(policy_ids):
        raise ValueError("policy_ids must be unique")
    index = {name: idx for idx, name in enumerate(policy_ids)}
    wins = [[0 for _ in policy_ids] for _ in policy_ids]
    losses = [[0 for _ in policy_ids] for _ in policy_ids]
    draws = [[0 for _ in policy_ids] for _ in policy_ids]
    for row_id, column_id, winner in matches:
        if row_id not in index or column_id not in index:
            raise ValueError("matrix match references an unknown policy")
        if winner not in (None, 0, 1):
            raise ValueError("winner must be 0, 1, or None")
        row = index[row_id]
        column = index[column_id]
        if winner is None:
            draws[row][column] += 1
        elif winner == 0:
            wins[row][column] += 1
        else:
            losses[row][column] += 1

    win_rates: list[list[float | None]] = []
    for row in range(len(policy_ids)):
        rate_row: list[float | None] = []
        for column in range(len(policy_ids)):
            count = wins[row][column] + losses[row][column] + draws[row][column]
            rate_row.append(
                None
                if row == column or count == 0
                else wins[row][column] / count
            )
        win_rates.append(rate_row)
    return win_rates, wins, losses, draws


__all__ = [
    "BattleMatchMetrics",
    "BattleResult",
    "append_jsonl",
    "evaluate_battle_gate",
    "outcome_matrices",
    "summarize_battle_matches",
    "win_matrix",
]
