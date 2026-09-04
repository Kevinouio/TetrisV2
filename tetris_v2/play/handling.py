"""Deterministic realtime key handling for human Tetris play.

The scheduler deliberately knows nothing about pygame (or any other windowing
library).  A frontend forwards physical key transitions through :meth:`press`
and :meth:`release`, then consumes logical movement actions once per game-loop
iteration with :meth:`actions`.

All timestamps are caller-supplied monotonic milliseconds.  Consequently OS
key-repeat settings and render-frame jitter do not change DAS/ARR behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TypeAlias


Milliseconds: TypeAlias = int | float


class InputKey(str, Enum):
    """Physical controls whose held state is timed by the scheduler."""

    LEFT = "left"
    RIGHT = "right"
    SOFT_DROP = "soft_drop"


class InputAction(str, Enum):
    """Logical actions emitted for the game runtime."""

    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    SOFT_DROP = "soft_drop"


_HORIZONTAL_ACTION = {
    InputKey.LEFT: InputAction.MOVE_LEFT,
    InputKey.RIGHT: InputAction.MOVE_RIGHT,
}
_OPPOSITE = {
    InputKey.LEFT: InputKey.RIGHT,
    InputKey.RIGHT: InputKey.LEFT,
}


@dataclass(frozen=True, slots=True)
class HandlingConfig:
    """Timing and safety limits for :class:`RealtimeInputScheduler`.

    The defaults correspond to common TETR.IO-style 60 Hz settings: 10-frame
    DAS, 2-frame ARR, and 6-cell soft drop.  ``sdf`` is the number of downward
    actions generated per soft-drop repeat.  Setting ``arr_ms`` to zero emits
    ``arr_zero_burst`` horizontal actions on every distinct poll after DAS;
    this models instant auto-shift without an unbounded loop.

    ``max_catchup_actions`` limits timer-generated actions during any one
    scheduler advance.  If a client stalls longer than the limit can cover,
    stale repetitions are discarded and the timer is advanced to the present.
    Explicit first-press/takeover actions are never discarded.
    """

    das_ms: float = 1000.0 * 10.0 / 60.0
    arr_ms: float = 1000.0 * 2.0 / 60.0
    soft_drop_ms: float = 1000.0 / 60.0
    sdf: int = 6
    arr_zero_burst: int = 10
    max_catchup_actions: int = 64

    def __post_init__(self) -> None:
        _validate_nonnegative_finite("das_ms", self.das_ms)
        _validate_nonnegative_finite("arr_ms", self.arr_ms)
        _validate_positive_finite("soft_drop_ms", self.soft_drop_ms)
        _validate_positive_int("sdf", self.sdf)
        _validate_positive_int("arr_zero_burst", self.arr_zero_burst)
        _validate_positive_int("max_catchup_actions", self.max_catchup_actions)


class RealtimeInputScheduler:
    """Convert explicit key transitions into deterministic movement actions.

    ``press`` and ``release`` intentionally return nothing.  Call ``actions``
    with the same timestamp after processing input events for a frame; the
    immediate movement and any due repeats are returned together.  Duplicate
    presses for an already-held key are ignored, which isolates handling from
    operating-system key repeat.
    """

    def __init__(self, config: HandlingConfig | None = None) -> None:
        self.config = config or HandlingConfig()
        self.reset()

    @property
    def held_keys(self) -> frozenset[InputKey]:
        """Currently held timed keys."""

        return frozenset(self._held)

    @property
    def active_direction(self) -> InputKey | None:
        """The held horizontal direction currently receiving repeats."""

        return self._active_horizontal

    def is_held(self, key: InputKey | str) -> bool:
        """Return whether ``key`` is currently held."""

        return _coerce_key(key) in self._held

    def press(self, key: InputKey | str, now_ms: Milliseconds) -> None:
        """Record a physical key-down transition.

        A horizontal press immediately becomes the active direction and queues
        one move.  If both horizontal keys are held, the most recently pressed
        one has priority.  Re-pressing a held key has no effect.
        """

        normalized = _coerce_key(key)
        now = self._observe_time(now_ms)
        # Preserve repeats that became due before this input transition.  A
        # repeat exactly at ``now`` is decided after the transition instead.
        self._advance(now, inclusive=False)

        if normalized in self._held:
            return

        self._held.add(normalized)
        if normalized in _HORIZONTAL_ACTION:
            self._activate_horizontal(normalized, now)
        else:
            self._enqueue_soft_drop(now)
            self._soft_drop_next_ms = now + self.config.soft_drop_ms

    def release(self, key: InputKey | str, now_ms: Milliseconds) -> None:
        """Record a physical key-up transition.

        Releasing the active horizontal direction immediately transfers
        priority to the still-held opposite direction and queues its first
        movement.  Releasing a key that is not held is harmless.
        """

        normalized = _coerce_key(key)
        now = self._observe_time(now_ms)
        self._advance(now, inclusive=False)

        if normalized not in self._held:
            return

        self._held.remove(normalized)
        if normalized is InputKey.SOFT_DROP:
            self._soft_drop_next_ms = None
            return

        if normalized is not self._active_horizontal:
            return

        opposite = _OPPOSITE[normalized]
        if opposite in self._held:
            self._activate_horizontal(opposite, now)
        else:
            self._active_horizontal = None
            self._horizontal_next_ms = None
            self._arr_zero_last_advance_ms = None

    def actions(self, now_ms: Milliseconds) -> tuple[InputAction, ...]:
        """Return all immediate and repeat actions due through ``now_ms``.

        Calling this more than once with the same timestamp is idempotent.  At
        most ``max_catchup_actions`` timer actions are created by each call;
        ARR=0 additionally remains bounded by ``arr_zero_burst``.
        """

        now = self._observe_time(now_ms)
        self._advance(now, inclusive=True)
        return self._take_pending()

    def actions_before(self, now_ms: Milliseconds) -> tuple[InputAction, ...]:
        """Return actions strictly older than ``now_ms``.

        Frontends use this immediately before applying a physical transition at
        the same timestamp.  The transition can then replace/cancel a repeat at
        the exact deadline, matching :meth:`press` and :meth:`release` semantics.
        """

        now = self._observe_time(now_ms)
        self._advance(now, inclusive=False)
        return self._take_pending(before_ms=now)

    def _take_pending(self, *, before_ms: float | None = None) -> tuple[InputAction, ...]:
        if not self._pending:
            return ()

        if before_ms is None:
            pending = self._pending
            self._pending = []
        else:
            pending = [event for event in self._pending if event[0] < before_ms]
            self._pending = [event for event in self._pending if event[0] >= before_ms]
        if not pending:
            return ()

        # Input transitions and repeat timers can be advanced by separate API
        # calls.  Sorting restores their actual timestamp order.  The sequence
        # number makes simultaneous actions stable (horizontal before soft drop
        # for timer ties, and caller order for explicit transitions).
        pending.sort(key=lambda event: (event[0], event[1]))
        ready = tuple(event[2] for event in pending)
        return ready

    def discard_until(self, now_ms: Milliseconds) -> None:
        """Advance held-key timers to ``now_ms`` without emitting stale input.

        A realtime frontend uses this after a long suspension or render hitch
        when it intentionally drops old simulation ticks.  Keeping input and
        physics on the same retained time window prevents a stall from turning
        into a burst of zero-time movement.  Held keys remain held; finite
        repeat timers resume on the first interval after ``now_ms``.

        Callers should consume explicit press/release actions before discarding,
        because all already-queued actions are intentionally removed.
        """

        now = self._observe_time(now_ms)
        self._pending.clear()
        self._skip_stale_finite_repeats(now, inclusive=True)
        if (
            self.config.arr_ms == 0.0
            and self._active_horizontal is not None
            and self._horizontal_next_ms is not None
            and self._horizontal_next_ms <= now
        ):
            # DAS has elapsed, but this synchronization point must not itself
            # generate an ARR=0 wall burst.  The next distinct poll may do so.
            self._arr_zero_last_advance_ms = now

    def reset(self, now_ms: Milliseconds | None = None) -> None:
        """Clear held keys, timers, and queued actions.

        With no timestamp, a new time origin may be used.  Passing ``now_ms``
        anchors the reset so later calls must use that time or a newer one.
        """

        anchor = None if now_ms is None else _coerce_time(now_ms)
        self._held: set[InputKey] = set()
        self._active_horizontal: InputKey | None = None
        self._horizontal_next_ms: float | None = None
        self._soft_drop_next_ms: float | None = None
        self._arr_zero_last_advance_ms: float | None = None
        self._pending: list[tuple[float, int, InputAction]] = []
        self._sequence = 0
        self._last_now_ms = anchor

    def _activate_horizontal(self, key: InputKey, now: float) -> None:
        self._active_horizontal = key
        self._enqueue(now, _HORIZONTAL_ACTION[key])
        self._horizontal_next_ms = now + self.config.das_ms
        self._arr_zero_last_advance_ms = None

    def _enqueue_soft_drop(self, timestamp: float) -> None:
        for _ in range(self.config.sdf):
            self._enqueue(timestamp, InputAction.SOFT_DROP)

    def _enqueue(self, timestamp: float, action: InputAction) -> None:
        self._pending.append((timestamp, self._sequence, action))
        self._sequence += 1

    def _observe_time(self, now_ms: Milliseconds) -> float:
        now = _coerce_time(now_ms)
        if self._last_now_ms is not None and now < self._last_now_ms:
            raise ValueError(
                f"now_ms must be monotonic (got {now!r} after {self._last_now_ms!r})"
            )
        self._last_now_ms = now
        return now

    def _advance(self, now: float, *, inclusive: bool) -> None:
        budget = self.config.max_catchup_actions

        while budget > 0:
            horizontal_due = self._horizontal_due_time(now, inclusive=inclusive)
            soft_drop_due = self._soft_drop_due_time(now, inclusive=inclusive)
            if horizontal_due is None and soft_drop_due is None:
                break

            # Stable tie breaking makes simultaneous horizontal/vertical input
            # deterministic and is usually the most natural order for slides.
            if horizontal_due is not None and (
                soft_drop_due is None or horizontal_due <= soft_drop_due
            ):
                if self.config.arr_ms == 0.0:
                    count = min(self.config.arr_zero_burst, budget)
                    for _ in range(count):
                        self._enqueue(
                            horizontal_due,
                            _HORIZONTAL_ACTION[self._active_horizontal],
                        )
                    budget -= count
                    self._arr_zero_last_advance_ms = now
                else:
                    self._enqueue(horizontal_due, _HORIZONTAL_ACTION[self._active_horizontal])
                    budget -= 1
                    self._horizontal_next_ms = horizontal_due + self.config.arr_ms
            else:
                count = min(self.config.sdf, budget)
                for _ in range(count):
                    self._enqueue(soft_drop_due, InputAction.SOFT_DROP)
                budget -= count
                self._soft_drop_next_ms = soft_drop_due + self.config.soft_drop_ms

        # A long suspension must not leave minutes of stale repeats queued for
        # later frames.  Once the per-advance budget is consumed, advance each
        # finite timer beyond the boundary and intentionally drop the excess.
        self._skip_stale_finite_repeats(now, inclusive=inclusive)

    def _horizontal_due_time(self, now: float, *, inclusive: bool) -> float | None:
        if self._active_horizontal is None or self._horizontal_next_ms is None:
            return None
        if self.config.arr_ms == 0.0:
            if self._arr_zero_last_advance_ms == now:
                return None
            if _is_due(self._horizontal_next_ms, now, inclusive=inclusive):
                # The first burst is anchored at the DAS deadline.  Later
                # bursts represent this poll's ongoing instant auto-shift.
                if self._arr_zero_last_advance_ms is None:
                    return self._horizontal_next_ms
                if not inclusive:
                    return None
                return now
            return None
        if _is_due(self._horizontal_next_ms, now, inclusive=inclusive):
            return self._horizontal_next_ms
        return None

    def _soft_drop_due_time(self, now: float, *, inclusive: bool) -> float | None:
        if InputKey.SOFT_DROP not in self._held or self._soft_drop_next_ms is None:
            return None
        if _is_due(self._soft_drop_next_ms, now, inclusive=inclusive):
            return self._soft_drop_next_ms
        return None

    def _skip_stale_finite_repeats(self, now: float, *, inclusive: bool) -> None:
        if self.config.arr_ms > 0.0 and self._horizontal_next_ms is not None:
            self._horizontal_next_ms = _first_time_after_boundary(
                self._horizontal_next_ms,
                self.config.arr_ms,
                now,
                inclusive=inclusive,
            )
        if self._soft_drop_next_ms is not None:
            self._soft_drop_next_ms = _first_time_after_boundary(
                self._soft_drop_next_ms,
                self.config.soft_drop_ms,
                now,
                inclusive=inclusive,
            )


def _coerce_key(key: InputKey | str) -> InputKey:
    try:
        return InputKey(key)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in InputKey)
        raise ValueError(f"unknown timed input key {key!r}; expected one of: {choices}") from exc


def _coerce_time(value: Milliseconds) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("now_ms must be a finite int or float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("now_ms must be finite")
    return result


def _is_due(deadline: float, now: float, *, inclusive: bool) -> bool:
    return deadline <= now if inclusive else deadline < now


def _first_time_after_boundary(
    deadline: float,
    interval: float,
    boundary: float,
    *,
    inclusive: bool,
) -> float:
    if not _is_due(deadline, boundary, inclusive=inclusive):
        return deadline
    elapsed = boundary - deadline
    if inclusive:
        periods = math.floor(elapsed / interval) + 1
    else:
        # A transition at the boundary takes precedence over a repeat at that
        # exact timestamp, but the subsequent ``actions(boundary)`` call may
        # still emit the repeat for a key whose state did not change.
        periods = math.ceil(elapsed / interval)
    return deadline + periods * interval


def _validate_nonnegative_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")


def _validate_positive_finite(name: str, value: object) -> None:
    _validate_nonnegative_finite(name, value)
    if float(value) == 0.0:
        raise ValueError(f"{name} must be > 0")


def _validate_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


__all__ = [
    "HandlingConfig",
    "InputAction",
    "InputKey",
    "Milliseconds",
    "RealtimeInputScheduler",
]
