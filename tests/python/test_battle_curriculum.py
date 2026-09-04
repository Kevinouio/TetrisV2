from __future__ import annotations

from pathlib import Path

from tetris_v2.rl.battle.curriculum import CurriculumState
from tetris_v2.rl.battle.train import _configured_curriculum


def test_packaged_curriculum_matches_typed_defaults() -> None:
    configured = _configured_curriculum(None)
    expected = CurriculumState()

    assert configured.state_dict()["stages"] == expected.state_dict()["stages"]
    assert configured.stage_index == 0
    assert configured.entered_at_step == 0
    assert configured.promotion_history == []


def test_custom_curriculum_yaml_changes_mixes_and_promotion_gates(
    tmp_path: Path,
) -> None:
    config = tmp_path / "custom_battle_curriculum.yaml"
    config.write_text(
        """\
stages:
  - name: warmup
    opponent_mix:
      random: 0.25
      cold_clear: 0.75
    promotion:
      - opponent: cold_clear
        min_win_rate: 0.42
        min_matches: 12
        max_illegal_actions: 1
  - name: self_play
    opponent_mix:
      frozen: 0.80
      current: 0.20
""",
        encoding="utf-8",
    )

    state = _configured_curriculum(config)

    assert [stage.name for stage in state.stages] == ["warmup", "self_play"]
    assert state.current.opponent_mix == {"random": 0.25, "cold_clear": 0.75}
    requirement = state.current.promotion[0]
    assert requirement.opponent == "cold_clear"
    assert requirement.min_win_rate == 0.42
    assert requirement.min_matches == 12
    assert requirement.max_illegal_actions == 1

    assert not state.maybe_promote(
        {
            "cold_clear": {
                "match_count": 12,
                "win_rate": 0.41,
                "illegal_action_count": 0,
            }
        },
        global_step=100,
    )
    assert state.current.name == "warmup"
    assert state.maybe_promote(
        {
            "cold_clear": {
                "match_count": 12,
                "win_rate": 0.42,
                "illegal_action_count": 1,
            }
        },
        global_step=200,
    )
    assert state.current.name == "self_play"
    assert state.current.opponent_mix == {"frozen": 0.80, "current": 0.20}


def test_curriculum_promotes_only_from_fixed_evaluation_gate() -> None:
    state = CurriculumState()
    assert state.current.name == "random"

    passed = state.maybe_promote(
        {"random": {"match_count": 100, "win_rate": 0.89, "illegal_action_count": [0, 0]}},
        global_step=10_000,
    )
    assert not passed
    assert state.current.name == "random"

    passed = state.maybe_promote(
        {"random": {"match_count": 100, "win_rate": 0.95, "illegal_action_count": [0, 0]}},
        global_step=20_000,
    )
    assert passed
    assert state.current.name == "heuristic"


def test_curriculum_round_trip_preserves_history_and_mix() -> None:
    state = CurriculumState()
    state.maybe_promote(
        {"random": {"match_count": 100, "win_rate": 0.95, "illegal_action_count": 0}},
        global_step=123,
    )

    loaded = CurriculumState.from_state_dict(state.state_dict())
    assert loaded.stage_index == 1
    assert loaded.entered_at_step == 123
    assert loaded.current.opponent_mix == {"random": 0.30, "cold_clear": 0.70}
    assert loaded.promotion_history == state.promotion_history
