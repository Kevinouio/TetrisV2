"""Responsive real-time Tetris client backed by the native rules engine."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Optional

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from tetris_v2.play import (
    HandlingConfig,
    InputAction,
    InputKey,
    RealtimeInputScheduler,
)
from tetris_v2.play.rendering import Renderer
from tetris_v2.rl.runtime import (
    ACTION_CCW,
    ACTION_CW,
    ACTION_HARD_DROP,
    ACTION_HOLD,
    ACTION_LEFT,
    ACTION_NONE,
    ACTION_RIGHT,
    ACTION_ROTATE_180,
    ACTION_SOFT_DROP,
    EnvCtypes,
    find_library,
)


SIMULATION_HZ = 60.0
MAX_CATCHUP_TICKS = 5
DEFAULT_WINDOW_SIZE = (960, 900)

TIMED_KEYS = {
    pygame.K_LEFT: InputKey.LEFT,
    pygame.K_RIGHT: InputKey.RIGHT,
    pygame.K_DOWN: InputKey.SOFT_DROP,
}
TIMED_ACTIONS = {
    InputAction.MOVE_LEFT: ACTION_LEFT,
    InputAction.MOVE_RIGHT: ACTION_RIGHT,
    InputAction.SOFT_DROP: ACTION_SOFT_DROP,
}
EDGE_ACTIONS = {
    pygame.K_z: ACTION_CCW,
    pygame.K_x: ACTION_CW,
    pygame.K_UP: ACTION_CW,
    pygame.K_a: ACTION_ROTATE_180,
    pygame.K_SPACE: ACTION_HARD_DROP,
    pygame.K_c: ACTION_HOLD,
    pygame.K_LSHIFT: ACTION_HOLD,
    pygame.K_RSHIFT: ACTION_HOLD,
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play responsive, real-time Tetris using the native TetrisV2 engine."
    )
    parser.add_argument("--lib", type=Path, default=None, help="Path to the native C API library.")
    parser.add_argument("--seed", type=int, default=None, help="Seven-bag seed (random by default).")
    parser.add_argument(
        "--fps",
        type=int,
        default=144,
        help="Render cap; simulation remains fixed at 60 Hz (default: 144).",
    )
    parser.add_argument("--das-ms", type=float, default=100.0, help="Delayed auto-shift delay in ms.")
    parser.add_argument("--arr-ms", type=float, default=16.0, help="Auto-repeat interval in ms; 0 is instant.")
    parser.add_argument(
        "--sdf",
        type=int,
        default=6,
        help="Soft-drop cells per 60 Hz input pulse (default: 6).",
    )
    parser.add_argument(
        "--reduced-motion",
        action="store_true",
        help="Disable shake and reduce particles while keeping gameplay feedback.",
    )
    parser.add_argument("--mute", action="store_true", help="Disable synthesized gameplay sounds.")
    parser.add_argument(
        "--vsync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request display synchronization when supported (default: on).",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Save the final logical frame on exit (useful for visual regression checks).",
    )
    parser.add_argument("--max-frames", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.fps < 30:
        parser.error("--fps must be at least 30")
    if not math.isfinite(args.das_ms) or args.das_ms < 0:
        parser.error("--das-ms must be nonnegative")
    if not math.isfinite(args.arr_ms) or args.arr_ms < 0:
        parser.error("--arr-ms must be nonnegative")
    if args.sdf <= 0:
        parser.error("--sdf must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    return args


def _snapshot(env: EnvCtypes) -> dict[str, Any]:
    meta = env.meta()
    game_over = bool(meta["game_over"])
    return {
        "board": env.board_piece_ids(include_active=False),
        "active": None if game_over else env.active(),
        "ghost": None if game_over else env.ghost(),
        "hold": env.hold_info(),
        "queue": env.queue(),
        "meta": meta,
    }


class GameSession:
    """Small stateful adapter between input, native simulation, and presentation."""

    def __init__(self, env: EnvCtypes, renderer: Renderer, seed: int) -> None:
        self.env = env
        self.renderer = renderer
        self.seed = int(seed)
        self.state = "playing"
        self.snapshot = _snapshot(env)
        self.score = 0.0
        self.pieces = 0
        self.elapsed_ms = 0.0
        self.last_action = ACTION_NONE

    @property
    def stats(self) -> dict[str, Any]:
        elapsed_s = self.elapsed_ms / 1000.0
        meta = self.snapshot["meta"]
        return {
            "score": int(round(self.score)),
            "pieces": self.pieces,
            "elapsed_s": elapsed_s,
            "pps": self.pieces / elapsed_s if elapsed_s > 0.0 else 0.0,
            "lines": int(meta["lines"]),
            "combo": int(meta["combo"]),
            "b2b": bool(meta["b2b"]),
            "seed": self.seed,
        }

    def reset(self, seed: int, now_ms: float) -> None:
        self.seed = int(seed)
        self.env.reset(self.seed)
        self.state = "playing"
        self.snapshot = _snapshot(self.env)
        self.score = 0.0
        self.pieces = 0
        self.elapsed_ms = 0.0
        self.last_action = ACTION_NONE
        self.renderer.reset_effects()

    def apply_input(self, action: int, now_ms: float) -> dict[str, Any] | None:
        if self.state != "playing":
            return None
        before = self.snapshot
        outcome = self.env.input(action)
        after = _snapshot(self.env)
        self._accept_step(before, outcome, action, after, now_ms)
        return outcome

    def tick(self, now_ms: float) -> dict[str, Any] | None:
        if self.state != "playing":
            return None
        before = self.snapshot
        outcome = self.env.tick()
        after = _snapshot(self.env)
        self._accept_step(before, outcome, ACTION_NONE, after, now_ms)
        return outcome

    def _accept_step(
        self,
        before: dict[str, Any],
        outcome: dict[str, Any],
        action: int,
        after: dict[str, Any],
        now_ms: float,
    ) -> None:
        self.last_action = action
        self.score += float(outcome["reward"])
        if outcome["piece_locked"]:
            self.pieces += 1
        self.snapshot = after
        self.renderer.on_step(before, outcome, action, after, now_ms=now_ms)
        if outcome["game_over"]:
            self.state = "game_over"


class FixedStepDriver:
    """Interleave timed handling with a fixed-rate native simulation.

    Input due at a tick boundary is applied before that tick.  Old wall time is
    dropped for both clocks together after a long hitch, so repeats can never
    run far ahead of gravity or lock delay.
    """

    def __init__(
        self,
        session: GameSession,
        handling: RealtimeInputScheduler,
        start_ms: float,
        *,
        simulation_hz: float = SIMULATION_HZ,
        max_catchup_ticks: int = MAX_CATCHUP_TICKS,
    ) -> None:
        self.session = session
        self.handling = handling
        self.tick_ms = 1000.0 / simulation_hz
        self.max_catchup_ticks = max_catchup_ticks
        self.current_ms = float(start_ms)
        self.next_tick_ms = self.current_ms + self.tick_ms

    def reset_clock(self, now_ms: float) -> None:
        """Start a fresh fixed-step interval after pause, focus, or reset."""

        self.current_ms = float(now_ms)
        self.next_tick_ms = self.current_ms + self.tick_ms

    def drain_inputs(self, now_ms: float) -> None:
        """Apply all handling actions due through ``now_ms`` in stable order."""

        self._apply_logical_actions(self.handling.actions(now_ms), now_ms)

    def drain_inputs_before(self, now_ms: float) -> None:
        """Apply handling actions strictly older than an input event."""

        self._apply_logical_actions(self.handling.actions_before(now_ms), now_ms)

    def _apply_logical_actions(
        self, logical_actions: tuple[InputAction, ...], now_ms: float
    ) -> None:
        if self.session.state != "playing":
            return
        for logical_action in logical_actions:
            outcome = self.session.apply_input(TIMED_ACTIONS[logical_action], now_ms)
            if outcome is not None and outcome["game_over"]:
                break

    def advance_before(self, now_ms: float) -> None:
        """Advance strictly before an event, leaving its timestamp input-first."""

        self._advance(now_ms, inclusive=False)

    def advance_to(self, now_ms: float) -> None:
        """Advance input and physics chronologically through ``now_ms``."""

        self._advance(now_ms, inclusive=True)

    def _advance(self, now_ms: float, *, inclusive: bool) -> None:
        target_ms = float(now_ms)
        if target_ms < self.current_ms:
            raise ValueError("simulation time must be monotonic")

        if self.session.state != "playing":
            self.reset_clock(target_ms)
            return

        retained_span_ms = self.tick_ms * self.max_catchup_ticks
        if target_ms - self.current_ms > retained_span_ms:
            # Retain the most recent bounded interval.  Discarding both input
            # repeats and physics debt together avoids hitch-dependent bursts.
            retained_start_ms = target_ms - retained_span_ms
            self.handling.discard_until(retained_start_ms)
            self.current_ms = retained_start_ms
            self.next_tick_ms = retained_start_ms + self.tick_ms

        cursor_ms = self.current_ms
        ticks = 0
        epsilon = 1e-7
        def tick_is_due() -> bool:
            if inclusive:
                return self.next_tick_ms <= target_ms + epsilon
            return self.next_tick_ms < target_ms - epsilon

        while ticks < self.max_catchup_ticks and self.session.state == "playing" and tick_is_due():
            # A mathematically exact boundary can accumulate a tiny positive
            # floating error (e.g. six 1/60 s ticks).  Never advance the input
            # scheduler beyond the caller's target.
            # Repeated 1/60 s additions can place a mathematically equal
            # boundary a few ulps behind an integer SDL timestamp.  Clamp the
            # boundary into the already-processed interval as well as to the
            # target so the handling clock never moves backwards.
            tick_time_ms = min(target_ms, max(cursor_ms, self.next_tick_ms))
            self.drain_inputs(tick_time_ms)
            if self.session.state != "playing":
                break
            self.session.elapsed_ms += max(0.0, tick_time_ms - cursor_ms)
            cursor_ms = tick_time_ms
            self.session.tick(tick_time_ms)
            self.next_tick_ms += self.tick_ms
            ticks += 1

        if self.session.state == "playing":
            if inclusive:
                self.drain_inputs(target_ms)
            else:
                self.drain_inputs_before(target_ms)
            if self.session.state == "playing":
                self.session.elapsed_ms += max(0.0, target_ms - cursor_ms)

        self.current_ms = target_ms
        if ticks == self.max_catchup_ticks and self.next_tick_ms <= target_ms + epsilon:
            self.next_tick_ms = target_ms + self.tick_ms


def _event_time_ms(event: pygame.event.Event, floor_ms: float, ceiling_ms: float) -> float:
    """Return an SDL event timestamp clamped to this processed frame."""

    raw_timestamp = getattr(event, "timestamp", ceiling_ms)
    try:
        timestamp = float(raw_timestamp)
    except (TypeError, ValueError):
        timestamp = ceiling_ms
    if not math.isfinite(timestamp):
        timestamp = ceiling_ms
    return min(ceiling_ms, max(floor_ms, timestamp))


def _open_window(size: tuple[int, int], *, fullscreen: bool, vsync: bool) -> pygame.Surface:
    flags = pygame.FULLSCREEN if fullscreen else pygame.RESIZABLE
    requested_size = (0, 0) if fullscreen else size
    try:
        return pygame.display.set_mode(requested_size, flags, vsync=1 if vsync else 0)
    except (pygame.error, TypeError):
        return pygame.display.set_mode(requested_size, flags)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    seed = int(args.seed if args.seed is not None else random.SystemRandom().randrange(1, 2**31))
    lib_path = find_library(args.lib)

    try:
        env = EnvCtypes(lib_path, seed=seed, play_mode=True)
    except AttributeError as exc:
        raise SystemExit(
            "The native library is missing real-time play symbols. Rebuild it with "
            "`cmake --build build --parallel`, then try again."
        ) from exc

    pygame_started = False
    final_canvas: pygame.Surface | None = None
    try:
        pygame.mixer.pre_init(48_000, -16, 2, 256)
        pygame.init()
        pygame_started = True
        pygame.key.set_repeat()
        pygame.display.set_caption("TetrisV2 — Flow")
        pygame.event.set_blocked(None)
        pygame.event.set_allowed(
            [
                pygame.QUIT,
                pygame.KEYDOWN,
                pygame.KEYUP,
                pygame.WINDOWFOCUSLOST,
                pygame.WINDOWFOCUSGAINED,
                pygame.WINDOWRESIZED,
            ]
        )

        fullscreen = False
        windowed_size = DEFAULT_WINDOW_SIZE
        screen = _open_window(windowed_size, fullscreen=fullscreen, vsync=args.vsync)
        clock = pygame.time.Clock()
        renderer = Renderer(
            logical_size=DEFAULT_WINDOW_SIZE,
            enable_audio=not args.mute,
            reduced_motion=args.reduced_motion,
        )
        renderer.warm_up_audio()
        session = GameSession(env, renderer, seed)
        handling = RealtimeInputScheduler(
            HandlingConfig(
                das_ms=float(args.das_ms),
                arr_ms=float(args.arr_ms),
                soft_drop_ms=1000.0 / SIMULATION_HZ,
                sdf=int(args.sdf),
                arr_zero_burst=10,
                max_catchup_actions=64,
            )
        )

        start_ms = float(pygame.time.get_ticks())
        handling.reset(start_ms)
        driver = FixedStepDriver(session, handling, start_ms)
        physical_keys: set[int] = set()
        running = True
        frames = 0
        last_time = time.perf_counter()

        def clear_inputs(now_ms: float, *, preserve_key: int | None = None) -> None:
            physical_keys.clear()
            if preserve_key is not None:
                physical_keys.add(preserve_key)
            handling.reset(now_ms)

        while running:
            current_time = time.perf_counter()
            frame_dt = min(0.1, max(0.0, current_time - last_time))
            last_time = current_time
            now_ms = max(driver.current_ms, float(pygame.time.get_ticks()))

            for event in pygame.event.get():
                event_ms = _event_time_ms(event, driver.current_ms, now_ms)
                if event.type == pygame.QUIT:
                    running = False
                    continue
                if event.type == pygame.WINDOWFOCUSLOST:
                    driver.advance_before(event_ms)
                    if session.state == "playing":
                        session.state = "paused"
                    clear_inputs(event_ms)
                    driver.reset_clock(event_ms)
                    continue
                if event.type == pygame.WINDOWRESIZED and not fullscreen:
                    windowed_size = (max(480, event.x), max(540, event.y))
                    continue
                if event.type == pygame.KEYUP:
                    driver.advance_before(event_ms)
                    physical_keys.discard(event.key)
                    timed_key = TIMED_KEYS.get(event.key)
                    if timed_key is not None:
                        handling.release(timed_key, event_ms)
                        driver.drain_inputs(event_ms)
                    continue
                if event.type != pygame.KEYDOWN or event.key in physical_keys:
                    continue

                physical_keys.add(event.key)
                if event.key == pygame.K_ESCAPE:
                    running = False
                    continue
                if event.key == pygame.K_F11:
                    if not fullscreen:
                        windowed_size = screen.get_size()
                    fullscreen = not fullscreen
                    screen = _open_window(windowed_size, fullscreen=fullscreen, vsync=args.vsync)
                    continue
                if event.key == pygame.K_p:
                    driver.advance_before(event_ms)
                    if session.state == "playing":
                        session.state = "paused"
                    elif session.state == "paused":
                        session.state = "playing"
                    clear_inputs(event_ms, preserve_key=event.key)
                    driver.reset_clock(event_ms)
                    continue
                if event.key in (pygame.K_r, pygame.K_RETURN) and session.state == "game_over":
                    driver.advance_before(event_ms)
                    if event.key == pygame.K_RETURN:
                        seed += 1
                    session.reset(seed, event_ms)
                    clear_inputs(event_ms, preserve_key=event.key)
                    driver.reset_clock(event_ms)
                    continue
                if event.key == pygame.K_r and session.state != "game_over":
                    driver.advance_before(event_ms)
                    session.reset(seed, event_ms)
                    clear_inputs(event_ms, preserve_key=event.key)
                    driver.reset_clock(event_ms)
                    continue
                if session.state != "playing":
                    continue

                # Advance old repeats and fixed ticks before this event.  SDL's
                # event timestamp preserves sub-frame ordering where available.
                driver.advance_before(event_ms)
                timed_key = TIMED_KEYS.get(event.key)
                if timed_key is not None:
                    handling.press(timed_key, event_ms)
                    driver.drain_inputs(event_ms)
                    continue

                native_action = EDGE_ACTIONS.get(event.key)
                if native_action is not None:
                    # Movement due at this exact timestamp belongs to the old
                    # piece; physics at the boundary runs after all SDL input.
                    driver.drain_inputs(event_ms)
                    session.apply_input(native_action, event_ms)

            driver.advance_to(now_ms)

            renderer.update(0.0 if session.state == "paused" else frame_dt * 1000.0)
            final_canvas = renderer.render(
                None,
                session.snapshot,
                stats=session.stats,
                state=session.state,
                now_ms=now_ms,
            )
            renderer.present(screen)
            pygame.display.flip()

            frames += 1
            if args.max_frames is not None and frames >= args.max_frames:
                running = False
            clock.tick_busy_loop(int(args.fps))
    finally:
        try:
            if args.screenshot is not None and final_canvas is not None:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                pygame.image.save(final_canvas, args.screenshot)
        finally:
            env.close()
            if pygame_started:
                pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
