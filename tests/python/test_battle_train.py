from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tetris_v2.rl.battle.checkpoint import load_battle_training_checkpoint
from tetris_v2.rl.battle.dqn import BattleDQNAgent
from tetris_v2.rl.battle.train import _validate_resume_config, parse_args, run


def _training_args(
    log_dir: Path,
    *,
    total_timesteps: int,
    resume_checkpoint: Path | None = None,
):
    argv = [
        "--total-timesteps",
        str(total_timesteps),
        "--buffer-size",
        "4",
        "--warmup-steps",
        "0",
        "--batch-size",
        "1",
        "--train-frequency",
        "1",
        "--gradient-steps",
        "1",
        "--learning-rate-decay-steps",
        "10",
        "--epsilon-decay-steps",
        "10",
        "--target-sync-interval",
        "2",
        "--max-match-steps",
        "1",
        "--opponent-pool-size",
        "2",
        "--random-opponent-weight",
        "1",
        "--heuristic-opponent-weight",
        "0",
        "--frozen-opponent-weight",
        "0",
        "--current-opponent-weight",
        "0",
        "--disable-curriculum",
        "--log-frequency",
        "100",
        "--eval-frequency",
        "100",
        "--checkpoint-frequency",
        "100",
        "--pool-checkpoint-frequency",
        "100",
        "--log-dir",
        str(log_dir),
        "--device",
        "cpu",
        "--seed",
        "811",
    ]
    if resume_checkpoint is not None:
        argv.extend(("--resume-checkpoint", str(resume_checkpoint)))
    return parse_args(argv)


def _assert_agent_equal(left, right) -> None:
    for section in (left.online, left.target):
        matching = right.online if section is left.online else right.target
        for name, value in section.state_dict().items():
            torch.testing.assert_close(value, matching.state_dict()[name], rtol=0, atol=0)
    assert left.update_steps == right.update_steps
    assert left.current_learning_rate == right.current_learning_rate


def test_resume_rejects_rule_and_curriculum_schedule_changes() -> None:
    stored = {
        "rules": {"garbage_delay": 1},
        "rewards": {"terminal": 20.0},
        "observation_dim": 470,
        "arguments": {"eval_frequency": 50_000},
    }
    changed_eval = {
        **stored,
        "arguments": {"eval_frequency": 10_000},
    }
    with pytest.raises(ValueError, match="eval_frequency"):
        _validate_resume_config(stored, changed_eval)

    changed_rules = {**stored, "rules": {"garbage_delay": 0}}
    with pytest.raises(ValueError, match="rules"):
        _validate_resume_config(stored, changed_rules)

    current_with_stages = {
        **stored,
        "curriculum_stages": [{"name": "random"}],
    }
    _validate_resume_config(stored, current_with_stages)

    stored_with_stages = {
        **stored,
        "curriculum_stages": [{"name": "random"}],
    }
    changed_stages = {
        **stored,
        "curriculum_stages": [{"name": "heuristic"}],
    }
    with pytest.raises(ValueError, match="curriculum_stages"):
        _validate_resume_config(stored_with_stages, changed_stages)


def test_curriculum_rejects_wall_clock_cold_clear(tmp_path: Path) -> None:
    args = _training_args(tmp_path / "nondeterministic", total_timesteps=1)
    args.disable_curriculum = False
    args.cold_clear_think_ms = 1
    with pytest.raises(SystemExit, match="deterministic Cold Clear"):
        run(args)


def test_episode_without_optimizer_update_logs_zero_loss(tmp_path: Path) -> None:
    args = _training_args(tmp_path / "no_update", total_timesteps=1)
    args.warmup_steps = 10

    assert run(args) == 0
    row = json.loads(
        (args.log_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert row["optimizer_updates"] == 0
    assert row["loss"] == 0.0
    assert row["episode_mean_loss"] == 0.0
    assert row["last_loss"] == 0.0


def test_episode_logs_mean_and_last_loss_across_multiple_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    losses = iter((1.0, 4.0, 10.0))

    def update(self: BattleDQNAgent, batch) -> dict[str, float]:
        del batch
        self.update_steps += 1
        return {"td_loss": next(losses)}

    monkeypatch.setattr(BattleDQNAgent, "update", update)
    args = _training_args(tmp_path / "three_updates", total_timesteps=1)
    args.gradient_steps = 3

    assert run(args) == 0
    row = json.loads(
        (args.log_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert row["optimizer_updates"] == 3
    assert row["loss"] == pytest.approx(5.0)
    assert row["episode_mean_loss"] == pytest.approx(5.0)
    assert row["last_loss"] == pytest.approx(10.0)


def test_curriculum_evaluation_logs_training_steps_and_wall_time(tmp_path: Path) -> None:
    curriculum_path = tmp_path / "curriculum.yaml"
    curriculum_path.write_text(
        """\
stages:
  - name: audit
    opponent_mix:
      random: 1.0
    promotion:
      - opponent: random
        min_win_rate: 0.0
        min_matches: 2
        max_illegal_actions: 0
  - name: complete
    opponent_mix:
      random: 1.0
""",
        encoding="utf-8",
    )
    args = _training_args(tmp_path / "curriculum_eval", total_timesteps=1)
    args.disable_curriculum = False
    args.curriculum_config = curriculum_path
    args.eval_frequency = 1
    args.eval_matches = 2

    assert run(args) == 0
    report_path = args.log_dir / "evaluation" / "step_000000000001_random.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["training_steps"] == 1
    assert report["wall_clock_training_time"] >= 0.0
    assert report["wall_clock_training_time_units"] == "seconds"
    evaluation_row = json.loads(
        (args.log_dir / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert evaluation_row["training_steps"] == 1
    assert evaluation_row["wall_clock_training_time"] == pytest.approx(
        report["wall_clock_training_time"]
    )


def test_native_self_play_smoke_resume_matches_uninterrupted_training(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "split"
    assert run(_training_args(split_dir, total_timesteps=1)) == 0
    episode_row = json.loads(
        (split_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert episode_row["learner_checkpoint_identifier"] == "learner_step_000000000001"
    assert episode_row["attack_generated"] >= episode_row["garbage_sent"]
    assert episode_row["attack_sent"] == episode_row["garbage_sent"]
    assert episode_row["episode_mean_loss"] == episode_row["loss"]
    assert episode_row["optimizer_updates"] == 1
    assert episode_row["last_loss"] == episode_row["loss"]
    first_checkpoint = split_dir / "battle_training_final.pt"
    assert first_checkpoint.is_file()
    legacy_payload = torch.load(first_checkpoint, map_location="cpu", weights_only=False)
    legacy_payload["training_config"].pop("curriculum_stages")
    legacy_checkpoint = split_dir / "legacy_training.pt"
    torch.save(legacy_payload, legacy_checkpoint)
    custom_curriculum = tmp_path / "different_curriculum.yaml"
    custom_curriculum.write_text(
        """\
stages:
  - name: custom
    opponent_mix:
      random: 1.0
""",
        encoding="utf-8",
    )
    legacy_resume_args = _training_args(
        tmp_path / "legacy_resume",
        total_timesteps=2,
        resume_checkpoint=legacy_checkpoint,
    )
    legacy_resume_args.curriculum_config = custom_curriculum
    with pytest.raises(ValueError, match="live curriculum"):
        run(legacy_resume_args)

    initial = load_battle_training_checkpoint(first_checkpoint, device="cpu")
    initial_snapshot = Path(initial.opponent_pool.frozen[0].checkpoint or "")
    snapshot_payload = torch.load(initial_snapshot, map_location="cpu", weights_only=False)
    assert snapshot_payload["metadata"]["wall_seconds"] == 0.0
    assert snapshot_payload["metadata"]["rules"]["max_steps"] == 1
    assert snapshot_payload["metadata"]["rewards"]["terminal"] == 20.0
    initial_snapshot.unlink()
    final_policy_payload = torch.load(
        split_dir / "battle_dqn_final.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert final_policy_payload["metadata"]["rules"]["max_steps"] == 1
    assert final_policy_payload["metadata"]["rewards"]["attack"] == 0.05

    resumed_dir = tmp_path / "resumed"
    assert run(
        _training_args(
            resumed_dir,
            total_timesteps=2,
            resume_checkpoint=first_checkpoint,
        )
    ) == 0
    resumed = load_battle_training_checkpoint(
        resumed_dir / "battle_training_final.pt",
        device="cpu",
    )
    assert resumed.global_step == 2
    assert resumed.episode_index == 2
    assert len(resumed.replay) == 2
    assert len(resumed.opponent_pool.frozen) == 1

    full_dir = tmp_path / "full"
    assert run(_training_args(full_dir, total_timesteps=2)) == 0
    uninterrupted = load_battle_training_checkpoint(
        full_dir / "battle_training_final.pt",
        device="cpu",
    )
    _assert_agent_equal(resumed.agent, uninterrupted.agent)
    for index in range(2):
        left = resumed.replay.transition(index)
        right = uninterrupted.replay.transition(index)
        assert left.keys() == right.keys()
        for name in left:
            if isinstance(left[name], np.ndarray):
                np.testing.assert_array_equal(left[name], right[name])
            else:
                assert left[name] == right[name]
