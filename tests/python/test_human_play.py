from __future__ import annotations

from typing import Any

import pytest

from scripts import play_human
from tetris_v2.play.handling import HandlingConfig, InputKey, RealtimeInputScheduler
from tetris_v2.rl.runtime import ACTION_HARD_DROP, ACTION_LEFT, ACTION_RIGHT


class FakeRenderer:
    def __init__(self) -> None:
        self.events: list[tuple[dict[str, Any], int]] = []
        self.resets = 0

    def on_step(self, _before, outcome, action, _after, *, now_ms) -> None:
        self.events.append((dict(outcome), int(action)))

    def reset_effects(self) -> None:
        self.resets += 1


class FakeEnv:
    def __init__(self) -> None:
        self.game_over = False
        self.lines = 0
        self.seed = 3

    def meta(self) -> dict[str, Any]:
        return {
            "game_over": self.game_over,
            "top_out": self.game_over,
            "combo": -1,
            "b2b": False,
            "lines": self.lines,
            "lock_timer": 0,
            "lock_resets": 0,
        }

    def board_piece_ids(self, *, include_active: bool) -> list[list[int]]:
        assert not include_active
        return [[255] * 10 for _ in range(20)]

    def active(self) -> dict[str, int]:
        return {"piece": 2, "rotation": 0, "x": 4, "y": 19}

    def ghost(self) -> dict[str, int]:
        return {"piece": 2, "rotation": 0, "x": 4, "y": 0}

    def hold_info(self) -> dict[str, Any]:
        return {"has_hold": False, "hold_piece": 7, "hold_available": True}

    def queue(self) -> list[int]:
        return [0, 1, 2, 3, 4]

    def input(self, action: int) -> dict[str, Any]:
        assert action == ACTION_HARD_DROP
        self.game_over = True
        return {
            "action_succeeded": True,
            "piece_locked": True,
            "hold_used": False,
            "lines_cleared": 0,
            "spin_clear": False,
            "spin_type": 0,
            "difficult_clear": False,
            "b2b_bonus_applied": False,
            "combo": -1,
            "back_to_back": False,
            "reward": 38.0,
            "game_over": True,
            "top_out": True,
        }

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        self.game_over = False
        self.lines = 0


def test_parse_args_exposes_fast_handling_defaults() -> None:
    args = play_human.parse_args([])
    assert args.fps == 144
    assert args.das_ms == 100.0
    assert args.arr_ms == 16.0
    assert args.sdf == 6


@pytest.mark.parametrize(
    "argv",
    [
        ["--fps", "29"],
        ["--das-ms", "-1"],
        ["--arr-ms", "-1"],
        ["--arr-ms", "nan"],
        ["--das-ms", "inf"],
        ["--sdf", "0"],
    ],
)
def test_parse_args_rejects_unsafe_timing_values(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        play_human.parse_args(argv)


def test_game_session_tracks_native_lock_and_topout() -> None:
    env = FakeEnv()
    renderer = FakeRenderer()
    session = play_human.GameSession(env, renderer, seed=3)

    outcome = session.apply_input(ACTION_HARD_DROP, now_ms=12.0)

    assert outcome is not None
    assert session.score == 38.0
    assert session.pieces == 1
    assert session.state == "game_over"
    assert session.snapshot["active"] is None
    assert renderer.events[-1][1] == ACTION_HARD_DROP

    session.reset(seed=4, now_ms=20.0)
    assert session.state == "playing"
    assert session.score == 0.0
    assert session.pieces == 0
    assert env.seed == 4
    assert renderer.resets == 1


class RecordingSession:
    def __init__(self) -> None:
        self.state = "playing"
        self.elapsed_ms = 0.0
        self.piece = 0
        self.events: list[tuple[str, int, int]] = []
        self.event_times: list[float] = []

    def apply_input(self, action: int, now_ms: float) -> dict[str, bool]:
        self.events.append(("input", int(action), self.piece))
        self.event_times.append(float(now_ms))
        if action == ACTION_HARD_DROP:
            self.piece += 1
        return {"game_over": False}

    def tick(self, now_ms: float) -> dict[str, bool]:
        self.events.append(("tick", 0, self.piece))
        self.event_times.append(float(now_ms))
        return {"game_over": False}


def _recording_driver(
    *, das_ms: float = 10.0, arr_ms: float = 10.0, max_ticks: int = 5
) -> tuple[RecordingSession, RealtimeInputScheduler, play_human.FixedStepDriver]:
    session = RecordingSession()
    handling = RealtimeInputScheduler(
        HandlingConfig(
            das_ms=das_ms,
            arr_ms=arr_ms,
            soft_drop_ms=10.0,
            sdf=1,
            max_catchup_actions=64,
        )
    )
    handling.reset(0)
    driver = play_human.FixedStepDriver(
        session, handling, 0, max_catchup_ticks=max_ticks
    )
    return session, handling, driver


def test_fixed_step_driver_interleaves_repeats_before_ticks_and_hard_drop() -> None:
    session, handling, driver = _recording_driver()
    handling.press(InputKey.LEFT, 0)
    driver.drain_inputs(0)

    # This render frame owes two fixed ticks and several held-key repeats.
    driver.advance_to(40)
    session.apply_input(ACTION_HARD_DROP, 40)

    assert session.events == [
        ("input", ACTION_LEFT, 0),
        ("input", ACTION_LEFT, 0),
        ("tick", 0, 0),
        ("input", ACTION_LEFT, 0),
        ("input", ACTION_LEFT, 0),
        ("tick", 0, 0),
        ("input", ACTION_LEFT, 0),
        ("input", ACTION_HARD_DROP, 0),
    ]
    assert all(piece == 0 for kind, _action, piece in session.events if kind == "tick")


def test_release_takeover_is_drained_before_same_timestamp_hard_drop() -> None:
    session, handling, driver = _recording_driver(das_ms=100.0, arr_ms=20.0)
    handling.press(InputKey.RIGHT, 0)
    driver.drain_inputs(0)
    handling.press(InputKey.LEFT, 5)
    driver.drain_inputs(5)

    driver.advance_to(20)
    handling.release(InputKey.LEFT, 20)
    driver.drain_inputs(20)
    session.apply_input(ACTION_HARD_DROP, 20)

    assert session.events[-2:] == [
        ("input", ACTION_RIGHT, 0),
        ("input", ACTION_HARD_DROP, 0),
    ]


def test_direction_event_replaces_repeat_due_at_exact_deadline() -> None:
    session, handling, driver = _recording_driver(das_ms=100.0, arr_ms=25.0)
    handling.press(InputKey.LEFT, 0)
    driver.drain_inputs(0)
    driver.advance_to(50)

    driver.advance_before(100)
    handling.press(InputKey.RIGHT, 100)
    driver.drain_inputs(100)
    driver.advance_to(100)

    inputs = [action for kind, action, _piece in session.events if kind == "input"]
    assert inputs == [ACTION_LEFT, ACTION_RIGHT]


def test_direction_event_preempts_arr_zero_burst_at_same_timestamp() -> None:
    session, handling, driver = _recording_driver(das_ms=10.0, arr_ms=0.0)
    handling.press(InputKey.LEFT, 0)
    driver.drain_inputs(0)
    driver.advance_to(10)
    event_count = len(session.events)

    driver.advance_before(11)
    handling.press(InputKey.RIGHT, 11)
    driver.drain_inputs(11)

    assert session.events[event_count:] == [("input", ACTION_RIGHT, 0)]


def test_edge_input_at_tick_deadline_runs_before_that_tick() -> None:
    session, _handling, driver = _recording_driver()
    deadline = 1000.0 / 60.0

    driver.advance_before(deadline)
    session.apply_input(ACTION_HARD_DROP, deadline)
    driver.advance_to(deadline)

    assert session.events == [
        ("input", ACTION_HARD_DROP, 0),
        ("tick", 0, 1),
    ]


def test_exact_six_tick_boundary_never_advances_input_clock_past_target() -> None:
    session, handling, driver = _recording_driver(max_ticks=6)
    handling.press(InputKey.LEFT, 0)
    driver.drain_inputs(0)

    driver.advance_to(100)

    assert sum(kind == "tick" for kind, _action, _piece in session.events) == 6
    assert driver.current_ms == 100


def test_deferred_float_drift_tick_is_clamped_monotonic() -> None:
    session, handling, _driver = _recording_driver()
    handling.reset(0)
    driver = play_human.FixedStepDriver(
        session,
        handling,
        0,
        simulation_hz=10_000.0,
        max_catchup_ticks=20,
    )

    driver.advance_before(1.0)
    assert driver.next_tick_ms < driver.current_ms
    driver.advance_to(1.0)

    assert sum(kind == "tick" for kind, _action, _piece in session.events) == 10
    assert session.event_times[-1] == 1.0


def test_reset_clock_does_not_transfer_old_frame_debt() -> None:
    session, handling, driver = _recording_driver()
    driver.advance_to(80)
    session.events.clear()
    session.elapsed_ms = 0.0

    handling.reset(1000)
    driver.reset_clock(1000)
    driver.advance_to(1005)

    assert session.events == []
    assert session.elapsed_ms == pytest.approx(5.0)


def test_long_hitch_caps_input_and_physics_to_the_same_recent_window() -> None:
    session, handling, driver = _recording_driver(
        das_ms=5.0, arr_ms=5.0, max_ticks=2
    )
    handling.press(InputKey.LEFT, 0)
    driver.drain_inputs(0)
    session.events.clear()

    driver.advance_to(1000)

    assert sum(kind == "tick" for kind, _action, _piece in session.events) == 2
    assert sum(kind == "input" for kind, _action, _piece in session.events) <= 8
    assert session.elapsed_ms == pytest.approx(2000.0 / 60.0)
