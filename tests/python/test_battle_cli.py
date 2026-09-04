from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import battle_matrix, battle_smoke, eval_battle, eval_battle_pool
from tetris_v2.rl.battle import cli as battle_cli
from tetris_v2.rl.battle.checkpoint import BATTLE_TRAINING_CHECKPOINT_VERSION
from tetris_v2.rl.battle.dqn import (
    BATTLE_ACTION_ORDER,
    BATTLE_OBSERVATION_SCHEMA,
)
from tetris_v2.rl.battle.metrics import BattleMatchMetrics, summarize_battle_matches
from tetris_v2.rl.battle.opponents import OpponentDescriptor, OpponentPool


def _match(
    *,
    match: int,
    learner_seat: int,
    winner: int | None,
    steps: int = 5,
) -> BattleMatchMetrics:
    return BattleMatchMetrics(
        match=match,
        seed=100 + match,
        learner_seat=learner_seat,
        opponent_type="checkpoint",
        opponent_checkpoint="opponent.pt",
        result=(
            "draw"
            if winner is None
            else ("win" if winner == learner_seat else "loss")
        ),
        winner=winner,
        match_steps=steps,
        placements=(steps, steps),
        lines_cleared=(1, 1),
        attack_sent=(1, 1),
        garbage_received=(1, 1),
        garbage_cancelled=(0, 0),
        top_out=(winner == 1, winner == 0),
        average_height=(2.0, 2.0),
        maximum_height=(4, 4),
        average_holes=(0.0, 0.0),
        illegal_actions=(0, 0),
        inference_ms=(0.1, 0.1),
        returns=(1.0, -1.0),
        trace_hash=f"trace-{match}",
    )


def test_eval_battle_defaults_and_report_gates() -> None:
    args = eval_battle.parse_args(
        ["learner.pt", "--min-survival-steps", "6", "--pressure-rows", "2"]
    )
    assert args.matches == 500
    assert args.cold_clear_think_ms == 0
    with pytest.raises(SystemExit):
        eval_battle.parse_args(["learner.pt", "--matches", "3"])

    matches = [
        _match(match=1, learner_seat=0, winner=0),
        _match(match=2, learner_seat=1, winner=1),
    ]
    summary = summarize_battle_matches(matches)
    report = eval_battle.build_report(
        args,
        matches=matches,
        summary=summary,
        determinism={"passed": True, "mismatches": []},
        learner_identifier="learner",
        opponent_identifier="random",
        learner_checkpoint_metadata={
            "training_steps": 123,
            "wall_clock_training_time": 45.5,
        },
    )
    assert report["configuration"]["scripted_pressure"]["rows"] == 2
    assert report["configuration"]["cold_clear_think_ms"] == 0
    assert report["gate"]["min_win_rate"] == pytest.approx(0.95)
    assert report["training_steps"] == 123
    assert report["wall_clock_training_time"] == pytest.approx(45.5)
    assert not report["gate"]["passed"]
    assert any("minimum_match_steps" in value for value in report["gate"]["failures"])
    assert len(report["matches"]) == 2

    wall_clock_args = eval_battle.parse_args(
        [
            "learner.pt",
            "--opponent",
            "cold_clear",
            "--cold-clear-think-ms",
            "5",
        ]
    )
    wall_clock_report = eval_battle.build_report(
        wall_clock_args,
        matches=matches,
        summary=summary,
        determinism=None,
        learner_identifier="learner",
        opponent_identifier="cold_clear",
    )
    assert not wall_clock_report["configuration"]["deterministic"]
    assert any(
        "think-ms 0" in failure
        for failure in wall_clock_report["gate"]["failures"]
    )


def test_eval_inherits_checkpoint_rules_and_rewards_until_overridden() -> None:
    metadata = {
        "rules": {
            "attack_table": [0, 1, 2, 3, 5],
            "garbage_delay": 3,
            "max_steps": 77,
            "mirrored_piece_seeds": False,
        },
        "rewards": {"terminal": 31.0, "attack": 0.4},
    }
    inherited_args = eval_battle.parse_args(["learner.pt"])
    inherited_rules, inherited_rewards = eval_battle.resolve_battle_configuration(
        inherited_args,
        metadata,
    )
    assert inherited_rules.attack_table == (0, 1, 2, 3, 5)
    assert inherited_rules.garbage_delay == 3
    assert inherited_rules.max_steps == 77
    assert not inherited_rules.mirrored_piece_seeds
    assert inherited_rewards.terminal == pytest.approx(31.0)
    assert inherited_rewards.attack == pytest.approx(0.4)
    assert inherited_rewards.cancellation == pytest.approx(0.03)

    override_args = eval_battle.parse_args(
        [
            "learner.pt",
            "--attack-table",
            "0",
            "0",
            "1",
            "2",
            "4",
            "--garbage-delay",
            "1",
            "--max-steps",
            "500",
            "--mirrored-piece-seeds",
        ]
    )
    overridden_rules, _ = eval_battle.resolve_battle_configuration(
        override_args,
        metadata,
    )
    assert overridden_rules.attack_table == (0, 0, 1, 2, 4)
    assert overridden_rules.garbage_delay == 1
    assert overridden_rules.max_steps == 500
    assert overridden_rules.mirrored_piece_seeds


def test_paired_evaluation_seeds_policies_by_logical_role() -> None:
    class RecordingPolicy:
        def __init__(self, identifier: str) -> None:
            self.identifier = identifier
            self.kind = identifier
            self.reset_seeds: list[int] = []

        def reset(self, seed: int) -> None:
            self.reset_seeds.append(seed)

        def select_action(self, observation, action_mask, *, player, env) -> int:
            return 0

    class OneStepEnv:
        def __init__(self) -> None:
            self.mask = np.zeros(3200, dtype=np.float32)
            self.mask[0] = 1.0

        def _info(self, step: int) -> dict[str, object]:
            player = {
                "placements": step,
                "lines_cleared": 0,
                "attack_generated": 0,
                "garbage_sent": 0,
                "garbage_received": 0,
                "garbage_cancelled": 0,
                "top_out": False,
            }
            board = {"max_height": 0, "holes": 0}
            return {
                "step": step,
                "winner": None,
                "result": "draw",
                "players": (dict(player), dict(player)),
                "board_stats": (dict(board), dict(board)),
                "step_stats": ({}, {}),
                "action_masks": (self.mask.copy(), self.mask.copy()),
                "reward_components": ({}, {}),
            }

        def reset(self, *, seed: int):
            observations = (
                np.zeros(470, dtype=np.float32),
                np.zeros(470, dtype=np.float32),
            )
            return observations, (self.mask.copy(), self.mask.copy()), self._info(0)

        def step(self, actions):
            observations = (
                np.zeros(470, dtype=np.float32),
                np.zeros(470, dtype=np.float32),
            )
            return observations, (0.0, 0.0), False, True, self._info(1)

        def close(self) -> None:
            pass

    from tetris_v2.rl.battle.evaluation import evaluate_paired_battles

    learner = RecordingPolicy("learner")
    opponent = RecordingPolicy("opponent")
    evaluate_paired_battles(
        learner,
        opponent,
        env_factory=OneStepEnv,
        matches=2,
        seed=7,
    )
    assert learner.reset_seeds == [29, 29]
    assert opponent.reset_seeds == [30, 30]


def test_malformed_checkpoint_exits_without_loader_traceback(tmp_path: Path) -> None:
    malformed = tmp_path / "bad.pt"
    malformed.write_bytes(b"not a torch checkpoint")
    with pytest.raises(SystemExit, match="Could not read battle checkpoint"):
        eval_battle.main([str(malformed), "--matches", "2"])


def test_training_checkpoint_view_does_not_require_replay_and_resolves_moved_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = Path("/old/machine/run/opponent_pool/snapshot.pt")
    descriptor = OpponentDescriptor(
        identifier="snapshot",
        kind="checkpoint",
        checkpoint=str(stale),
        generation=3,
        created_step=30,
    )
    pool = OpponentPool(max_frozen=2, seed=5)
    pool.add(descriptor)
    checkpoint = tmp_path / "battle_training.pt"
    torch.save(
        {
            "format_version": BATTLE_TRAINING_CHECKPOINT_VERSION,
            "algo": "battle_dqn_training",
            "observation_schema": BATTLE_OBSERVATION_SCHEMA,
            "action_order": BATTLE_ACTION_ORDER,
            "episode_boundary": True,
            "agent": {"sentinel": True},
            "opponent_pool": pool.state_dict(),
            "counters": {"global_step": 30, "episode_index": 4},
            "training_config": {"rules": {}, "rewards": {}},
            "extra": {"stage": "smoke", "wall_seconds": 12.5},
        },
        checkpoint,
    )
    sentinel_agent = object()
    monkeypatch.setattr(
        battle_cli.BattleDQNAgent,
        "from_state_dict",
        classmethod(lambda cls, state, device=None: sentinel_agent),
    )
    view = battle_cli.load_training_checkpoint_view(checkpoint, device="cpu")
    assert view.agent is sentinel_agent
    assert view.global_step == 30
    assert view.checkpoint_metadata["training_steps"] == 30
    assert view.checkpoint_metadata["wall_clock_training_time"] == pytest.approx(12.5)
    assert view.opponent_pool.frozen[0].identifier == "snapshot"
    assert not checkpoint.with_suffix(".npz").exists()
    policy = battle_cli.load_evaluation_policy(checkpoint, device="cpu")
    policy_metadata = battle_cli.policy_checkpoint_metadata(policy)
    assert policy.agent is sentinel_agent
    assert policy_metadata["training_steps"] == 30
    assert policy_metadata["wall_clock_training_time"] == pytest.approx(12.5)
    assert policy_metadata["rules"] == {}
    assert policy_metadata["rewards"] == {}

    moved = tmp_path / "opponent_pool" / stale.name
    moved.parent.mkdir()
    moved.write_bytes(b"policy")
    assert battle_cli.resolve_pool_checkpoint(
        descriptor, training_checkpoint=checkpoint
    ) == moved.resolve()


def test_pool_snapshot_filter_is_explicit() -> None:
    frozen = (
        OpponentDescriptor("first", "checkpoint", checkpoint="first.pt"),
        OpponentDescriptor("second", "checkpoint", checkpoint="second.pt"),
    )
    args = eval_battle_pool.parse_args(
        ["training.pt", "--snapshot", "second", "--matches-per-snapshot", "2"]
    )
    assert [value.identifier for value in eval_battle_pool._select_snapshots(args, frozen)] == [
        "second"
    ]
    args.snapshot = ["missing"]
    with pytest.raises(ValueError, match="Available"):
        eval_battle_pool._select_snapshots(args, frozen)


def test_matrix_converts_physical_winner_and_writes_json_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Row policy is the learner in physical seat 1 and wins there. The matrix
    # must record a row-policy win, not blindly treat physical winner 1 as column.
    match = _match(match=1, learner_seat=1, winner=1)
    draw = _match(match=2, learner_seat=0, winner=None)
    matches = [match, draw]
    report = battle_matrix.build_matrix_report(
        labels=["row", "column"],
        checkpoints=[Path("row.pt"), Path("column.pt")],
        pair_results=[
            ("row", "column", matches, summarize_battle_matches(matches))
        ],
        configuration={"matches_per_pair": 2},
    )
    assert report["matrix"]["win_rates"] == [[None, 0.5], [0.0, None]]
    assert report["matrix"]["score_rates"] == [[None, 0.75], [0.25, None]]
    assert report["matrix"]["wins"] == [[0, 1], [0, 0]]
    assert report["matrix"]["losses"] == [[0, 0], [1, 0]]
    assert report["matrix"]["draws"] == [[0, 1], [1, 0]]
    assert report["matrix"]["match_counts"] == [[0, 2], [2, 0]]

    json_path = tmp_path / "nested" / "matrix.json"
    csv_path = tmp_path / "nested" / "matrix.csv"
    battle_cli.write_json_report(json_path, report)
    battle_matrix.write_matrix_csv(csv_path, report)
    assert json.loads(json_path.read_text(encoding="utf-8"))["pairs"][0][
        "row_policy"
    ] == "row"
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "win=0.500000 score=0.750000 W-L-D=1-0-1 (n=2)" in csv_text
    assert "win=0.000000 score=0.250000 W-L-D=0-1-1 (n=2)" in csv_text

    battle_matrix._print_human_report(report)
    human = capsys.readouterr().out
    assert "win-rate matrix (row perspective; W-L-D/n)" in human
    assert "0.500 [1-0-1/2]" in human
    assert "score-rate matrix (row perspective)" in human
    assert "0.750" in human


def test_real_native_battle_smoke_cli_with_one_cpu_update(tmp_path: Path) -> None:
    assert battle_smoke.parse_args([]).train_updates == 2
    output = tmp_path / "battle-smoke.json"
    exit_code = battle_smoke.main(
        [
            "--matches",
            "2",
            "--max-steps",
            "1",
            "--train-updates",
            "1",
            "--train-batch-size",
            "2",
            "--json-output",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["native_contract"]["passed"]
    assert report["summary"]["match_count"] == 2
    assert report["tiny_training"]["completed_updates"] == 1
    assert report["gate"]["passed"]
