from __future__ import annotations

import pytest

from tetris_v2.play.handling import (
    HandlingConfig,
    InputAction,
    InputKey,
    RealtimeInputScheduler,
)


LEFT = InputAction.MOVE_LEFT
RIGHT = InputAction.MOVE_RIGHT
DOWN = InputAction.SOFT_DROP


def scheduler(**overrides) -> RealtimeInputScheduler:
    values = {
        "das_ms": 100.0,
        "arr_ms": 25.0,
        "soft_drop_ms": 20.0,
        "sdf": 1,
        "arr_zero_burst": 10,
        "max_catchup_actions": 64,
    }
    values.update(overrides)
    return RealtimeInputScheduler(HandlingConfig(**values))


def test_horizontal_press_moves_immediately_then_uses_das_and_arr() -> None:
    handling = scheduler()

    handling.press(InputKey.LEFT, 0)
    assert handling.actions(0) == (LEFT,)
    assert handling.actions(99) == ()
    assert handling.actions(100) == (LEFT,)
    assert handling.actions(124) == ()
    assert handling.actions(150) == (LEFT, LEFT)
    assert handling.actions(150) == ()


def test_last_pressed_direction_wins_and_release_immediately_takes_over() -> None:
    handling = scheduler()

    handling.press("left", 0)
    assert handling.actions(0) == (LEFT,)

    handling.press("right", 30)
    assert handling.active_direction is InputKey.RIGHT
    assert handling.held_keys == frozenset((InputKey.LEFT, InputKey.RIGHT))
    assert handling.actions(30) == (RIGHT,)

    # OS-generated duplicate keydown must not reset DAS or steal priority.
    handling.press("left", 40)
    assert handling.active_direction is InputKey.RIGHT
    assert handling.actions(40) == ()

    handling.release("right", 60)
    assert handling.active_direction is InputKey.LEFT
    assert handling.actions(60) == (LEFT,)
    assert handling.actions(159) == ()
    assert handling.actions(160) == (LEFT,)


def test_releasing_inactive_direction_does_not_reset_active_das() -> None:
    handling = scheduler()
    handling.press("left", 0)
    handling.press("right", 10)
    assert handling.actions(10) == (LEFT, RIGHT)

    handling.release("left", 50)
    assert handling.actions(50) == ()
    assert handling.actions(110) == (RIGHT,)


def test_soft_drop_rate_and_sdf_are_independent_of_horizontal_repeat() -> None:
    handling = scheduler(soft_drop_ms=20.0, sdf=3)
    handling.press("left", 0)
    handling.press("soft_drop", 5)

    assert handling.actions(5) == (LEFT, DOWN, DOWN, DOWN)
    assert handling.actions(24) == ()
    assert handling.actions(25) == (DOWN, DOWN, DOWN)
    # Catch up repeats at 45 and 65 in timestamp order.
    assert handling.actions(65) == (DOWN, DOWN, DOWN, DOWN, DOWN, DOWN)

    handling.release("soft_drop", 70)
    assert handling.actions(200) == (LEFT, LEFT, LEFT, LEFT, LEFT)


def test_arr_zero_emits_a_bounded_burst_once_per_poll_timestamp() -> None:
    handling = scheduler(das_ms=50.0, arr_ms=0.0, arr_zero_burst=4)
    handling.press("right", 0)

    assert handling.actions(0) == (RIGHT,)
    assert handling.actions(49) == ()
    assert handling.actions(50) == (RIGHT,) * 4
    assert handling.actions(50) == ()
    assert handling.actions(51) == (RIGHT,) * 4
    assert handling.actions(5000) == (RIGHT,) * 4


def test_actions_before_is_strict_for_immediate_and_arr_zero_actions() -> None:
    handling = scheduler(das_ms=50.0, arr_ms=0.0, arr_zero_burst=4)
    handling.press("right", 0)

    # The queued press belongs to t=0, not to the interval strictly before it.
    assert handling.actions_before(0) == ()
    assert handling.actions(0) == (RIGHT,)
    assert handling.actions(50) == (RIGHT,) * 4

    # An ongoing ARR=0 burst is synthesized at its poll timestamp, so a
    # physical transition at t=51 must take priority over that burst.
    assert handling.actions_before(51) == ()
    handling.press("left", 51)
    assert handling.actions(51) == (LEFT,)


def test_long_stall_catchup_is_capped_and_stale_repeats_are_skipped() -> None:
    handling = scheduler(das_ms=10.0, arr_ms=5.0, max_catchup_actions=3)
    handling.press("left", 0)
    assert handling.actions(0) == (LEFT,)

    assert handling.actions(1000) == (LEFT,) * 3
    assert handling.actions(1000) == ()
    assert handling.actions(1004) == ()
    assert handling.actions(1005) == (LEFT,)


def test_discard_until_keeps_hold_state_without_emitting_stale_repeats() -> None:
    handling = scheduler(das_ms=10.0, arr_ms=5.0)
    handling.press("left", 0)
    assert handling.actions(0) == (LEFT,)

    handling.discard_until(1000)
    assert handling.active_direction is InputKey.LEFT
    assert handling.actions(1000) == ()
    assert handling.actions(1004) == ()
    assert handling.actions(1005) == (LEFT,)


def test_discard_until_suppresses_arr_zero_burst_at_sync_point() -> None:
    handling = scheduler(das_ms=10.0, arr_ms=0.0, arr_zero_burst=3)
    handling.press("right", 0)
    assert handling.actions(0) == (RIGHT,)

    handling.discard_until(1000)
    assert handling.actions(1000) == ()
    assert handling.actions(1001) == (RIGHT,) * 3


def test_events_at_repeat_deadline_take_effect_before_that_repeat() -> None:
    handling = scheduler(das_ms=100.0, arr_ms=25.0)
    handling.press("left", 0)
    assert handling.actions(0) == (LEFT,)

    handling.press("right", 100)
    # The new immediate right movement replaces the left repeat due at 100.
    assert handling.actions(100) == (RIGHT,)

    handling.release("right", 200)
    assert handling.actions(200) == (LEFT,)


def test_catchup_cap_before_event_does_not_skip_repeat_at_event_timestamp() -> None:
    handling = scheduler(das_ms=10.0, arr_ms=5.0, max_catchup_actions=1)
    handling.press("left", 0)
    assert handling.actions(0) == (LEFT,)

    # This duplicate OS keydown advances old repeats but must otherwise have no
    # effect.  The cap keeps only t=10; t=15 is stale, while t=20 remains due.
    handling.press("left", 20)
    assert handling.actions(20) == (LEFT, LEFT)


def test_reset_clears_all_state_and_optionally_starts_a_new_clock() -> None:
    handling = scheduler()
    handling.press("left", 100)
    handling.press("soft_drop", 101)

    handling.reset()
    assert handling.held_keys == frozenset()
    assert handling.active_direction is None
    assert handling.actions(0) == ()

    handling.press("right", 1)
    handling.reset(50)
    assert handling.actions(50) == ()
    with pytest.raises(ValueError, match="monotonic"):
        handling.actions(49)


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"das_ms": -1}, ValueError),
        ({"arr_ms": float("inf")}, ValueError),
        ({"soft_drop_ms": 0}, ValueError),
        ({"sdf": 0}, ValueError),
        ({"arr_zero_burst": 1.5}, TypeError),
        ({"max_catchup_actions": 0}, ValueError),
    ],
)
def test_config_rejects_invalid_values(kwargs, exception) -> None:
    with pytest.raises(exception):
        HandlingConfig(**kwargs)


def test_invalid_key_and_clock_inputs_are_rejected() -> None:
    handling = scheduler()
    with pytest.raises(ValueError, match="unknown timed input key"):
        handling.press("rotate", 0)
    with pytest.raises(TypeError, match="now_ms"):
        handling.actions(True)
    with pytest.raises(ValueError, match="finite"):
        handling.actions(float("nan"))

    handling.actions(20)
    with pytest.raises(ValueError, match="monotonic"):
        handling.release("left", 19)
