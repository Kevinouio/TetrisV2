from __future__ import annotations

import json

import pytest

from tetris_v2.rl.battle.metrics import (
    BattleMatchMetrics,
    append_jsonl,
    evaluate_battle_gate,
    outcome_matrices,
    summarize_battle_matches,
    win_matrix,
)


def _match(*, match: int, learner_seat: int, winner: int | None) -> BattleMatchMetrics:
    return BattleMatchMetrics(
        match=match,
        seed=100 + match,
        learner_seat=learner_seat,
        opponent_type="random",
        opponent_checkpoint=None,
        result="draw" if winner is None else ("win" if winner == learner_seat else "loss"),
        winner=winner,
        match_steps=10 + match,
        placements=(10, 11),
        lines_cleared=(4, 3),
        attack_generated=(3, 1),
        attack_sent=(2, 1),
        garbage_received=(1, 2),
        garbage_cancelled=(1, 0),
        top_out=(winner == 1, winner == 0),
        average_height=(3.0, 4.0),
        maximum_height=(6, 7),
        average_holes=(1.0, 2.0),
        illegal_actions=(0, 0),
        inference_ms=(0.2, 0.1),
        inference_decisions=(10, 11),
        board_samples=(10, 11),
        returns=(20.0 if winner == 0 else 0.0, -20.0 if winner == 0 else 0.0),
        reward_components={"terminal": 20.0 if winner == learner_seat else 0.0},
    )


def test_battle_summary_keeps_seat_and_offense_metrics() -> None:
    matches = [_match(match=1, learner_seat=0, winner=0), _match(match=2, learner_seat=1, winner=None)]
    summary = summarize_battle_matches(matches)

    assert summary["wins"] == 1
    assert summary["draws"] == 1
    assert summary["win_rate"] == pytest.approx(0.5)
    assert summary["player_1_win_rate"] == pytest.approx(0.5)
    assert summary["average_placements"] == pytest.approx([10.0, 11.0])
    assert summary["average_attack_generated"] == pytest.approx([3.0, 1.0])
    assert summary["average_attack_sent"] == pytest.approx([2.0, 1.0])
    assert summary["average_garbage_sent"] == pytest.approx([2.0, 1.0])
    assert summary["attack_per_100_pieces"] == pytest.approx([30.0, 100.0 / 11.0])
    assert summary["garbage_sent_per_100_pieces"] == pytest.approx(
        [20.0, 100.0 / 11.0]
    )
    assert summary["illegal_action_count"] == [0, 0]
    assert summary["learner_mean_return"] == pytest.approx(10.0)
    assert summary["opponent_mean_return"] == pytest.approx(-10.0)
    assert summary["learner_top_out_count"] == 0
    assert summary["opponent_top_out_count"] == 1
    assert summary["learner_average_lines_cleared"] == pytest.approx(3.5)
    assert summary["opponent_average_lines_cleared"] == pytest.approx(3.5)
    assert summary["learner_attack_per_100_pieces"] == pytest.approx(
        400.0 / 21.0
    )
    assert summary["learner_average_attack_generated"] == pytest.approx(2.0)
    assert summary["learner_average_garbage_sent"] == pytest.approx(1.5)
    assert summary["learner_garbage_sent_per_100_pieces"] == pytest.approx(
        300.0 / 21.0
    )


def test_battle_summary_weights_latency_and_board_averages_by_sample_counts() -> None:
    short = _match(match=1, learner_seat=0, winner=None)
    long = _match(match=2, learner_seat=1, winner=None)
    matches = [
        BattleMatchMetrics(
            **{
                **short.__dict__,
                "average_height": (10.0, 20.0),
                "average_holes": (4.0, 8.0),
                "inference_ms": (1.0, 3.0),
                "inference_decisions": (2, 4),
                "board_samples": (2, 2),
            }
        ),
        BattleMatchMetrics(
            **{
                **long.__dict__,
                "average_height": (2.0, 4.0),
                "average_holes": (0.0, 2.0),
                "inference_ms": (9.0, 11.0),
                "inference_decisions": (6, 2),
                "board_samples": (6, 6),
            }
        ),
    ]

    summary = summarize_battle_matches(matches)

    assert summary["mean_inference_ms"] == pytest.approx([7.0, 34.0 / 6.0])
    assert summary["learner_mean_inference_ms"] == pytest.approx(6.0)
    assert summary["opponent_mean_inference_ms"] == pytest.approx(6.6)
    assert summary["average_height"] == pytest.approx([4.0, 8.0])
    assert summary["average_holes"] == pytest.approx([1.0, 3.5])
    assert summary["learner_average_height"] == pytest.approx(5.5)
    assert summary["opponent_average_height"] == pytest.approx(6.5)
    assert summary["learner_average_holes"] == pytest.approx(2.5)
    assert summary["opponent_average_holes"] == pytest.approx(2.0)


def test_match_rejects_inconsistent_perspective_result() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        BattleMatchMetrics(
            **{
                **_match(match=1, learner_seat=0, winner=0).__dict__,
                "result": "loss",
            }
        )


def test_jsonl_and_win_matrix_preserve_raw_counts(tmp_path) -> None:
    output = tmp_path / "matches.jsonl"
    append_jsonl(output, [_match(match=1, learner_seat=0, winner=0).to_dict()])
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["winner"] == 0
    assert row["placements"] == [10, 11]
    assert row["attack_generated"] == [3, 1]
    assert row["attack_sent"] == [2, 1]
    assert row["garbage_sent"] == [2, 1]

    rates, counts = win_matrix(
        ["initial", "trained"],
        [
            ("initial", "trained", 1),
            ("initial", "trained", None),
            ("trained", "initial", 0),
        ],
    )
    assert rates == [[None, pytest.approx(0.25)], [pytest.approx(1.0), None]]
    assert counts == [[0, 2], [1, 0]]

    win_rates, wins, losses, draws = outcome_matrices(
        ["initial", "trained"],
        [
            ("initial", "trained", 1),
            ("initial", "trained", None),
            ("trained", "initial", 0),
        ],
    )
    assert win_rates == [[None, 0.0], [1.0, None]]
    assert wins == [[0, 0], [1, 0]]
    assert losses == [[0, 1], [0, 0]]
    assert draws == [[0, 1], [0, 0]]


def test_battle_gate_never_hides_illegal_actions_or_seat_bias() -> None:
    summary = summarize_battle_matches(
        [_match(match=1, learner_seat=0, winner=0), _match(match=2, learner_seat=1, winner=0)]
    )
    gate = evaluate_battle_gate(summary, min_win_rate=0.95, max_seat_win_rate_gap=0.05)
    assert not gate["passed"]
    assert any("win_rate" in failure for failure in gate["failures"])
    assert any("seat_win_rate_gap" in failure for failure in gate["failures"])
