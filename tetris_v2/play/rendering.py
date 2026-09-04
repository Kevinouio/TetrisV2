"""Polished, fixed-canvas pygame rendering for human Tetris play.

The module deliberately has no dependency on the input/handling layer.  The
client loop can pass plain dictionaries obtained from the runtime bridge::

    renderer = Renderer(enable_audio=True)
    renderer.update(frame_delta_ms)
    renderer.render(window, snapshot, stats, game_state, pygame.time.get_ticks())

``snapshot`` may contain ``board``, ``active``, ``ghost``, ``hold``, ``queue``
and ``meta``.  Board rows are top-to-bottom, matching ``EnvCtypes``' visible
board methods; active-piece ``y`` coordinates are bottom-up, matching the C
API.  ``on_step`` consumes before/after snapshots plus the mapping returned by
``step_ex`` and turns game events into bounded visual and audio feedback.

Nothing initializes pygame's display or mixer at import time, which keeps the
geometry and renderer usable with SDL's dummy video driver in tests.
"""

from __future__ import annotations

from array import array
from collections import OrderedDict
from dataclasses import dataclass
import math
import random
from typing import Any, Iterable, Mapping, Sequence

import pygame


BOARD_COLS = 10
BOARD_ROWS = 20
EMPTY_CELL_ID = 255
GARBAGE_CELL_ID = 7

ACTION_NONE = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_SOFT_DROP = 3
ACTION_HARD_DROP = 4
ACTION_CW = 5
ACTION_CCW = 6
ACTION_180 = 7
ACTION_HOLD = 8

DEFAULT_LOGICAL_SIZE = (960, 900)


# These cells intentionally mirror src/piece_defs.cpp.  Coordinates use the
# engine convention: positive y points upward.
_BASE_CELLS: dict[int, tuple[tuple[int, int], ...]] = {
    0: ((-1, 0), (0, 0), (1, 0), (2, 0)),  # I
    1: ((0, 0), (1, 0), (0, 1), (1, 1)),  # O
    2: ((-1, 0), (0, 0), (1, 0), (0, 1)),  # T
    3: ((-1, 0), (0, 0), (1, 0), (1, 1)),  # L
    4: ((-1, 0), (0, 0), (1, 0), (-1, 1)),  # J
    5: ((-1, 0), (0, 0), (0, 1), (1, 1)),  # S
    6: ((-1, 1), (0, 1), (0, 0), (1, 0)),  # Z
}


PIECE_COLORS: dict[int, tuple[int, int, int]] = {
    0: (42, 220, 232),
    1: (247, 210, 64),
    2: (176, 94, 255),
    3: (255, 151, 56),
    4: (67, 111, 255),
    5: (70, 220, 111),
    6: (247, 80, 101),
    GARBAGE_CELL_ID: (105, 116, 136),
}


@dataclass(frozen=True)
class PiecePose:
    """A piece pose in engine coordinates.

    ``x`` and ``y`` are floats so a caller may pass a visually interpolated
    position without changing authoritative gameplay state.
    """

    piece: int
    rotation: int
    x: float
    y: float


def piece_cells(piece: int, rotation: int = 0) -> tuple[tuple[int, int], ...]:
    """Return the four local cells for ``piece`` at ``rotation``.

    Unknown piece IDs (including the hold sentinel/garbage ID) return an empty
    tuple.  Rotation values wrap modulo four.
    """

    cells = _BASE_CELLS.get(int(piece))
    if cells is None:
        return ()
    result = cells
    for _ in range(int(rotation) % 4):
        result = tuple((y, -x) for x, y in result)
    return result


def visible_cells(pose: PiecePose) -> tuple[tuple[int, int], ...]:
    """Return visible ``(column, top_down_row)`` cells for a piece pose."""

    cells: list[tuple[int, int]] = []
    for dx, dy in piece_cells(pose.piece, pose.rotation):
        x = int(round(pose.x + dx))
        y = int(round(pose.y + dy))
        if 0 <= x < BOARD_COLS and 0 <= y < BOARD_ROWS:
            cells.append((x, BOARD_ROWS - 1 - y))
    return tuple(cells)


def landing_y(
    board: Sequence[Sequence[int]],
    piece: int,
    rotation: int,
    x: int,
    y: int,
) -> int:
    """Compute a visible-board ghost landing y without mutating game state.

    The native ghost pose should be preferred when hidden-row blocks are
    possible.  This helper is exact for the normal visible playfield and is a
    useful graceful fallback for older native libraries.
    """

    shape = piece_cells(piece, rotation)
    if not shape:
        return int(y)

    normalized = _normalize_board(board)

    def collides(candidate_y: int) -> bool:
        for dx, dy in shape:
            cell_x = int(x) + dx
            cell_y = candidate_y + dy
            if cell_x < 0 or cell_x >= BOARD_COLS or cell_y < 0:
                return True
            if cell_y >= BOARD_ROWS:
                continue
            value = normalized[BOARD_ROWS - 1 - cell_y][cell_x]
            if _occupied(value):
                return True
        return False

    candidate = int(y)
    while not collides(candidate - 1):
        candidate -= 1
    return candidate


@dataclass
class PoseTween:
    """Small position-only tween suitable for active-piece presentation.

    Rotations intentionally snap: interpolating a tetromino through occupied
    cells looks muddy and can visually imply illegal geometry.  Call ``reset``
    after hold, lock, spawn, or game reset—even when the new piece has the same
    piece ID as the previous one.
    """

    duration_ms: float = 45.0
    _pose: PiecePose | None = None
    _target: PiecePose | None = None

    @property
    def pose(self) -> PiecePose | None:
        return self._pose

    def reset(self, pose: PiecePose | Mapping[str, Any] | None) -> None:
        parsed = _coerce_pose(pose)
        self._pose = parsed
        self._target = parsed

    def set_target(self, pose: PiecePose | Mapping[str, Any] | None, *, snap: bool = False) -> None:
        parsed = _coerce_pose(pose)
        if parsed is None or snap or self._pose is None:
            self.reset(parsed)
            return
        if parsed.piece != self._pose.piece:
            self.reset(parsed)
            return
        self._target = parsed
        # Geometry snaps while the kicked x/y offset receives the short tween.
        self._pose = PiecePose(parsed.piece, parsed.rotation, self._pose.x, self._pose.y)

    def update(self, dt_ms: float) -> PiecePose | None:
        if self._pose is None or self._target is None:
            return self._pose
        if self.duration_ms <= 0:
            self._pose = self._target
            return self._pose
        # Exponential ease is stable under variable render frame times.
        amount = 1.0 - math.exp(-max(0.0, float(dt_ms)) * 4.6 / self.duration_ms)
        x = self._pose.x + (self._target.x - self._pose.x) * amount
        y = self._pose.y + (self._target.y - self._pose.y) * amount
        if abs(x - self._target.x) < 0.002 and abs(y - self._target.y) < 0.002:
            x, y = self._target.x, self._target.y
        self._pose = PiecePose(self._target.piece, self._target.rotation, x, y)
        return self._pose


@dataclass(frozen=True)
class Palette:
    background_top: tuple[int, int, int] = (7, 11, 21)
    background_bottom: tuple[int, int, int] = (13, 20, 35)
    panel: tuple[int, int, int] = (17, 26, 43)
    panel_border: tuple[int, int, int] = (39, 55, 80)
    board: tuple[int, int, int] = (7, 11, 19)
    board_border: tuple[int, int, int] = (62, 82, 115)
    grid: tuple[int, int, int] = (23, 33, 52)
    text: tuple[int, int, int] = (238, 244, 255)
    muted_text: tuple[int, int, int] = (137, 154, 183)
    accent: tuple[int, int, int] = (105, 168, 255)
    success: tuple[int, int, int] = (106, 231, 164)
    danger: tuple[int, int, int] = (255, 91, 116)
    flash: tuple[int, int, int] = (241, 248, 255)


@dataclass(frozen=True)
class LogicalLayout:
    size: tuple[int, int]
    scale: float
    board_cell: int
    board_rect: pygame.Rect
    hold_rect: pygame.Rect
    stats_rect: pygame.Rect
    next_rect: pygame.Rect
    controls_rect: pygame.Rect

    @classmethod
    def for_size(cls, size: tuple[int, int]) -> "LogicalLayout":
        width, height = max(1, int(size[0])), max(1, int(size[1]))
        scale = min(width / 960.0, height / 900.0)
        offset_x = (width - 960.0 * scale) / 2.0
        offset_y = (height - 900.0 * scale) / 2.0

        def rect(x: float, y: float, w: float, h: float) -> pygame.Rect:
            return pygame.Rect(
                round(offset_x + x * scale),
                round(offset_y + y * scale),
                max(1, round(w * scale)),
                max(1, round(h * scale)),
            )

        cell = max(4, round(36 * scale))
        board = rect(300, 90, 360, 720)
        # Keep the playfield an exact 10x20 grid after integer rounding.
        board.width = BOARD_COLS * cell
        board.height = BOARD_ROWS * cell
        board.centerx = width // 2
        return cls(
            size=(width, height),
            scale=scale,
            board_cell=cell,
            board_rect=board,
            hold_rect=rect(48, 90, 220, 190),
            stats_rect=rect(48, 310, 220, 318),
            next_rect=rect(692, 90, 220, 610),
            controls_rect=rect(48, 660, 220, 150),
        )


@dataclass
class _Particle:
    x: float
    y: float
    vx: float
    vy: float
    color: tuple[int, int, int]
    lifetime: float
    radius: float
    gravity: float = 0.0
    age: float = 0.0


@dataclass
class _LineFlash:
    row: int
    duration: float = 0.18
    age: float = 0.0


@dataclass
class _Callout:
    text: str
    color: tuple[int, int, int]
    duration: float = 1.0
    age: float = 0.0


class FeedbackLayer:
    """Bounded, deterministic visual feedback stored in board coordinates."""

    def __init__(self, *, max_particles: int = 220, seed: int = 0x7E7A15) -> None:
        self.max_particles = max(0, int(max_particles))
        self.particles: list[_Particle] = []
        self.line_flashes: list[_LineFlash] = []
        self.callouts: list[_Callout] = []
        self._shake_age = 1.0
        self._shake_duration = 0.0
        self._shake_amplitude = 0.0
        self._rng = random.Random(seed)

    def clear(self) -> None:
        self.particles.clear()
        self.line_flashes.clear()
        self.callouts.clear()
        self._shake_age = 1.0
        self._shake_duration = 0.0
        self._shake_amplitude = 0.0

    def update(self, dt_seconds: float) -> None:
        dt = min(0.1, max(0.0, float(dt_seconds)))
        if dt <= 0:
            return
        for particle in self.particles:
            particle.age += dt
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.vy += particle.gravity * dt
        self.particles = [p for p in self.particles if p.age < p.lifetime]

        for flash in self.line_flashes:
            flash.age += dt
        self.line_flashes = [flash for flash in self.line_flashes if flash.age < flash.duration]

        for callout in self.callouts:
            callout.age += dt
        self.callouts = [callout for callout in self.callouts if callout.age < callout.duration]

        self._shake_age += dt

    def shake(self, amplitude: float, duration: float = 0.12) -> None:
        amplitude = max(0.0, float(amplitude))
        duration = max(0.0, float(duration))
        if amplitude >= self._shake_amplitude or self._shake_age >= self._shake_duration:
            self._shake_amplitude = amplitude
            self._shake_duration = duration
            self._shake_age = 0.0

    def shake_offset(self) -> tuple[float, float]:
        if self._shake_duration <= 0 or self._shake_age >= self._shake_duration:
            return (0.0, 0.0)
        progress = self._shake_age / self._shake_duration
        amplitude = self._shake_amplitude * (1.0 - progress) ** 2
        return (
            math.sin(self._shake_age * 103.0) * amplitude,
            math.sin(self._shake_age * 137.0 + 0.9) * amplitude * 0.72,
        )

    def add_callout(
        self,
        text: str,
        color: tuple[int, int, int],
        *,
        duration: float = 1.0,
    ) -> None:
        if not text:
            return
        self.callouts.append(_Callout(str(text), color, max(0.1, float(duration))))
        self.callouts = self.callouts[-3:]

    def emit_lock(
        self,
        cells: Iterable[tuple[int, int]],
        color: tuple[int, int, int],
    ) -> None:
        new_particles: list[_Particle] = []
        for col, row in cells:
            for _ in range(4):
                new_particles.append(
                    _Particle(
                        x=float(col) + self._rng.uniform(0.15, 0.85),
                        y=float(row) + self._rng.uniform(0.1, 0.75),
                        vx=self._rng.uniform(-1.5, 1.5),
                        vy=self._rng.uniform(-2.4, -0.5),
                        color=color,
                        lifetime=self._rng.uniform(0.18, 0.36),
                        radius=self._rng.uniform(0.05, 0.13),
                        gravity=5.5,
                    )
                )
        self._append_particles(new_particles)

    def emit_hard_drop(
        self,
        cells: Iterable[tuple[int, int]],
        color: tuple[int, int, int],
        distance: float,
    ) -> None:
        if distance <= 0:
            return
        new_particles: list[_Particle] = []
        count = min(28, max(8, int(distance * 1.5)))
        visible = tuple(cells)
        if not visible:
            return
        for _ in range(count):
            col, row = self._rng.choice(visible)
            new_particles.append(
                _Particle(
                    x=float(col) + self._rng.uniform(0.2, 0.8),
                    y=float(row) - self._rng.uniform(0.0, min(8.0, distance)),
                    vx=self._rng.uniform(-0.25, 0.25),
                    vy=self._rng.uniform(2.5, 5.5),
                    color=color,
                    lifetime=self._rng.uniform(0.12, 0.28),
                    radius=self._rng.uniform(0.035, 0.09),
                )
            )
        self._append_particles(new_particles)

    def emit_line_clear(
        self,
        rows: Iterable[int],
        color: tuple[int, int, int],
    ) -> None:
        visible_rows = sorted({int(row) for row in rows if 0 <= int(row) < BOARD_ROWS})
        for row in visible_rows:
            self.line_flashes.append(_LineFlash(row))
        self.line_flashes = self.line_flashes[-8:]

        new_particles: list[_Particle] = []
        for row in visible_rows:
            for col in range(BOARD_COLS):
                new_particles.append(
                    _Particle(
                        x=col + self._rng.uniform(0.25, 0.75),
                        y=row + self._rng.uniform(0.3, 0.7),
                        vx=(col - 4.5) * self._rng.uniform(0.18, 0.34),
                        vy=self._rng.uniform(-2.4, -0.6),
                        color=color,
                        lifetime=self._rng.uniform(0.28, 0.55),
                        radius=self._rng.uniform(0.05, 0.12),
                        gravity=4.2,
                    )
                )
        self._append_particles(new_particles)

    def _append_particles(self, particles: Iterable[_Particle]) -> None:
        if self.max_particles <= 0:
            return
        self.particles.extend(particles)
        if len(self.particles) > self.max_particles:
            self.particles = self.particles[-self.max_particles :]


class SynthSoundBank:
    """Lazy, optional synthesized feedback with a silent failure mode."""

    def __init__(self, *, enabled: bool = True, volume: float = 0.38) -> None:
        self.requested = bool(enabled)
        self.volume = max(0.0, min(1.0, float(volume)))
        self.available = False
        self._attempted = False
        self._sounds: dict[str, pygame.mixer.Sound] = {}

    def warm_up(self) -> bool:
        if self._attempted:
            return self.available
        self._attempted = True
        if not self.requested:
            return False
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.pre_init(44100, -16, 2, 256)
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=256)
            mixer_info = pygame.mixer.get_init()
            if mixer_info is None or int(mixer_info[1]) != -16:
                return False
            frequency, _sample_format, channels = mixer_info
            specifications = {
                "move": (180.0, 205.0, 0.025, 0.22),
                "soft_drop": (115.0, 105.0, 0.022, 0.14),
                "rotate": (310.0, 455.0, 0.055, 0.30),
                "hold": (240.0, 530.0, 0.090, 0.32),
                "lock": (115.0, 72.0, 0.065, 0.34),
                "hard_drop": (92.0, 48.0, 0.095, 0.48),
                "clear1": (430.0, 590.0, 0.130, 0.38),
                "clear2": (470.0, 710.0, 0.155, 0.40),
                "clear3": (520.0, 850.0, 0.175, 0.42),
                "clear4": (610.0, 1120.0, 0.220, 0.45),
                "spin": (380.0, 980.0, 0.230, 0.44),
                "topout": (230.0, 58.0, 0.420, 0.46),
            }
            for name, (start, end, duration, level) in specifications.items():
                pcm = self._tone(
                    int(frequency), int(channels), start, end, duration, level
                )
                sound = pygame.mixer.Sound(buffer=pcm)
                sound.set_volume(self.volume)
                self._sounds[name] = sound
            self.available = True
        except (pygame.error, OSError, ValueError):
            self._sounds.clear()
            self.available = False
        return self.available

    @staticmethod
    def _tone(
        frequency: int,
        channels: int,
        start_hz: float,
        end_hz: float,
        duration: float,
        level: float,
    ) -> bytes:
        frame_count = max(1, int(frequency * duration))
        samples = array("h")
        phase = 0.0
        for index in range(frame_count):
            progress = index / max(1, frame_count - 1)
            hz = start_hz + (end_hz - start_hz) * progress
            phase += math.tau * hz / frequency
            attack = min(1.0, progress / 0.06) if progress < 0.06 else 1.0
            envelope = attack * (1.0 - progress) ** 1.8
            wave = math.sin(phase) + 0.22 * math.sin(phase * 2.0)
            sample = int(max(-1.0, min(1.0, wave * level * envelope)) * 32767)
            for _ in range(max(1, channels)):
                samples.append(sample)
        return samples.tobytes()

    def play(self, name: str) -> bool:
        if not self.warm_up():
            return False
        sound = self._sounds.get(str(name))
        if sound is None:
            return False
        try:
            sound.play()
            return True
        except pygame.error:
            return False

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, float(volume)))
        for sound in self._sounds.values():
            sound.set_volume(self.volume)


class Renderer:
    """Fixed-logical-surface renderer and event-driven feedback coordinator."""

    def __init__(
        self,
        logical_size: tuple[int, int] = DEFAULT_LOGICAL_SIZE,
        *,
        palette: Palette | None = None,
        max_particles: int = 220,
        enable_audio: bool = True,
        audio_volume: float = 0.38,
        reduced_motion: bool = False,
    ) -> None:
        if not pygame.font.get_init():
            pygame.font.init()
        self.palette = palette or Palette()
        self.layout = LogicalLayout.for_size(logical_size)
        self.reduced_motion = bool(reduced_motion)
        particle_limit = min(int(max_particles), 72) if self.reduced_motion else int(max_particles)
        self.canvas = pygame.Surface(self.layout.size)
        self.effects = FeedbackLayer(max_particles=particle_limit)
        self.sound = SynthSoundBank(enabled=enable_audio, volume=audio_volume)
        self.pose_tween = PoseTween()
        self._sprite_cache: dict[tuple[int, int, str], pygame.Surface] = {}
        self._font_cache: dict[tuple[int, bool], pygame.font.Font] = {}
        self._text_cache: OrderedDict[
            tuple[str, int, tuple[int, int, int], bool], pygame.Surface
        ] = OrderedDict()
        self._background = self._make_background()
        self._last_snapshot: Mapping[str, Any] | None = None
        # Construction happens after pygame.init() in the interactive client;
        # paying synthesis cost here prevents a hitch on the first key press.
        if enable_audio:
            self.sound.warm_up()

    @property
    def logical_size(self) -> tuple[int, int]:
        return self.layout.size

    def warm_up_audio(self) -> bool:
        """Initialize and synthesize sounds; safe to call after ``pygame.init``."""

        return self.sound.warm_up()

    def update(self, dt_ms: float) -> None:
        """Advance visual effects by a real-time delta in milliseconds."""

        delta = max(0.0, float(dt_ms))
        self.effects.update(delta / 1000.0)
        self.pose_tween.update(delta)

    def reset_effects(self) -> None:
        self.effects.clear()
        # The game loop calls this for reset as well as presentation cleanup;
        # discarding the visual pose prevents a same-piece restart from easing
        # out of the previous run's final location.
        self.pose_tween.reset(None)

    def render(
        self,
        surface: pygame.Surface | None,
        snapshot: Mapping[str, Any],
        stats: Mapping[str, Any] | None = None,
        state: str | Mapping[str, Any] | None = "playing",
        now_ms: int | float | None = None,
    ) -> pygame.Surface:
        """Draw a complete frame and optionally scale it onto ``surface``.

        The returned object is always the renderer's fixed logical canvas.
        Passing ``None`` is useful for tests or for a loop that performs its own
        scaling.  A differently sized destination receives an aspect-fitted,
        letterboxed copy.
        """

        if not isinstance(snapshot, Mapping):
            raise TypeError("snapshot must be a mapping")
        stats = stats if isinstance(stats, Mapping) else {}
        now = float(pygame.time.get_ticks() if now_ms is None else now_ms)
        board = _normalize_board(
            _get(snapshot, "board", _get(snapshot, "locked_board", ()))
        )
        meta = _mapping(_get(snapshot, "meta", {}))
        authoritative_active = _coerce_pose(_get(snapshot, "active", None))
        active = self._visual_active_pose(authoritative_active)
        paused, game_over, overlay_title, overlay_detail = self._state_flags(
            state, meta
        )

        self.canvas.blit(self._background, (0, 0))
        self._draw_header(now)
        self._draw_side_panels(snapshot, stats, meta, now)

        ghost = self._ghost_pose(snapshot, board, active)
        self._draw_board(board, active, ghost, now, game_over=game_over)
        self._draw_callouts()

        if paused:
            self._draw_overlay("PAUSED", overlay_detail or "Press P to resume")
        elif game_over:
            self._draw_overlay(
                overlay_title or "TOP OUT",
                overlay_detail or "Press R to play again",
                danger=True,
            )
        elif overlay_title:
            self._draw_overlay(overlay_title, overlay_detail)

        self._last_snapshot = snapshot
        if surface is not None and surface is not self.canvas:
            self.present(surface)
        return self.canvas

    def present(self, surface: pygame.Surface, *, smooth: bool = True) -> pygame.Rect:
        """Aspect-fit the latest logical frame onto a destination surface."""

        target_w, target_h = surface.get_size()
        logical_w, logical_h = self.logical_size
        if target_w <= 0 or target_h <= 0:
            return pygame.Rect(0, 0, 0, 0)
        scale = min(target_w / logical_w, target_h / logical_h)
        width = max(1, round(logical_w * scale))
        height = max(1, round(logical_h * scale))
        destination = pygame.Rect(0, 0, width, height)
        destination.center = (target_w // 2, target_h // 2)
        surface.fill(self.palette.background_top)
        if (width, height) == self.logical_size:
            scaled = self.canvas
        elif smooth:
            scaled = pygame.transform.smoothscale(self.canvas, (width, height))
        else:
            scaled = pygame.transform.scale(self.canvas, (width, height))
        surface.blit(scaled, destination)
        return destination

    def on_step(
        self,
        before_snapshot: Mapping[str, Any],
        outcome: Mapping[str, Any] | Any,
        action: int | str,
        after_snapshot: Mapping[str, Any],
        now_ms: int | float | None = None,
    ) -> None:
        """Translate one authoritative step into feedback events.

        ``now_ms`` is accepted for loop symmetry and future scheduling; effect
        lifetimes advance only through ``update`` so pause behavior remains in
        the caller's control.
        """

        del now_ms
        if not isinstance(before_snapshot, Mapping) or not isinstance(after_snapshot, Mapping):
            return
        action_name = _action_name(action)
        succeeded = bool(
            _get(outcome, "action_succeeded", _get(outcome, "success", False))
        )
        piece_locked = bool(_get(outcome, "piece_locked", False))
        hold_used = bool(_get(outcome, "hold_used", _get(outcome, "used_hold", False)))
        lines = int(_get(outcome, "lines_cleared", _get(outcome, "lines", 0)) or 0)
        spin_type = int(_get(outcome, "spin_type", 0) or 0)
        spin_clear = bool(_get(outcome, "spin_clear", spin_type > 0))
        b2b_bonus = bool(_get(outcome, "b2b_bonus_applied", False))
        combo = int(_get(outcome, "combo", -1) or 0)
        game_over = bool(_get(outcome, "game_over", _get(outcome, "top_out", False)))

        before_active = _coerce_pose(_get(before_snapshot, "active", None))
        after_active = _coerce_pose(_get(after_snapshot, "active", None))
        if piece_locked or hold_used or game_over:
            # Lock/hold changes piece ownership, and consecutive pieces may
            # share the same ID, so event identity—not just piece type—must
            # define the tween boundary.
            self.pose_tween.reset(None if game_over else after_active)
        elif after_active is None:
            self.pose_tween.reset(None)
        else:
            if self.pose_tween.pose is None:
                self.pose_tween.reset(before_active or after_active)
            if before_active is None or before_active.piece != after_active.piece:
                self.pose_tween.reset(after_active)
            else:
                # This also catches gravity movement from a zero-action tick.
                # PoseTween snaps the new rotation geometry while easing the
                # x/y component of movement and rotation kicks.
                self.pose_tween.set_target(after_active)

        if hold_used or (succeeded and action_name == "hold"):
            self.sound.play("hold")
        elif succeeded and action_name in {"left", "right"}:
            self.sound.play("move")
        elif succeeded and action_name == "soft_drop":
            self.sound.play("soft_drop")
        elif succeeded and action_name in {"cw", "ccw", "180"}:
            self.sound.play("rotate")

        final_pose: PiecePose | None = None
        final_cells: tuple[tuple[int, int], ...] = ()
        drop_distance = float(_get(outcome, "drop_distance", 0.0) or 0.0)
        if piece_locked:
            before_board = _normalize_board(_get(before_snapshot, "board", ()))
            active = _coerce_pose(_get(before_snapshot, "active", None))
            final_pose = active
            if active is not None and action_name == "hard_drop":
                ghost = self._ghost_pose(before_snapshot, before_board, active)
                if ghost is not None:
                    final_pose = ghost
                    if drop_distance <= 0:
                        drop_distance = max(0.0, active.y - ghost.y)
            if final_pose is not None:
                final_cells = visible_cells(final_pose)
                color = PIECE_COLORS.get(final_pose.piece, self.palette.muted_text)
                self.effects.emit_lock(final_cells, color)
                if action_name == "hard_drop" or drop_distance > 0:
                    self.effects.emit_hard_drop(final_cells, color, drop_distance)
                    if not self.reduced_motion:
                        self.effects.shake(min(4.0, 1.3 + drop_distance * 0.12), 0.11)
                    self.sound.play("hard_drop")
                else:
                    self.sound.play("lock")

        if lines > 0:
            rows = self._cleared_rows(before_snapshot, outcome, final_pose, lines)
            self.effects.emit_line_clear(rows, self.palette.flash)
            label = _clear_label(lines, spin_clear, spin_type, b2b_bonus, combo)
            self.effects.add_callout(
                label,
                self.palette.success if spin_clear or lines >= 4 else self.palette.accent,
                duration=1.1,
            )
            if not self.reduced_motion:
                self.effects.shake(min(7.0, 2.0 + lines * 1.1), 0.16)
            self.sound.play("spin" if spin_clear else f"clear{min(4, max(1, lines))}")

        if game_over:
            self.effects.add_callout("TOP OUT", self.palette.danger, duration=1.25)
            if not self.reduced_motion:
                self.effects.shake(7.0, 0.30)
            self.sound.play("topout")

    def _state_flags(
        self,
        state: str | Mapping[str, Any] | None,
        meta: Mapping[str, Any],
    ) -> tuple[bool, bool, str, str]:
        paused = False
        game_over = bool(_get(meta, "game_over", _get(meta, "top_out", False)))
        title = ""
        detail = ""
        if isinstance(state, str):
            normalized = state.strip().lower().replace("-", "_")
            paused = normalized in {"paused", "pause"}
            game_over = game_over or normalized in {"game_over", "top_out", "gameover"}
            if normalized in {"ready", "countdown"}:
                title = "READY"
        elif isinstance(state, Mapping):
            paused = bool(_get(state, "paused", False))
            game_over = game_over or bool(
                _get(state, "game_over", _get(state, "top_out", False))
            )
            title = str(_get(state, "overlay_title", _get(state, "title", "")) or "")
            detail = str(_get(state, "overlay_detail", _get(state, "detail", "")) or "")
        return paused, game_over, title, detail

    def _visual_active_pose(self, authoritative: PiecePose | None) -> PiecePose | None:
        """Synchronize an authoritative pose with the renderer-owned tween."""

        if authoritative is None:
            self.pose_tween.reset(None)
            return None
        current = self.pose_tween.pose
        if current is None or current.piece != authoritative.piece:
            self.pose_tween.reset(authoritative)
        else:
            self.pose_tween.set_target(authoritative)
        return self.pose_tween.pose or authoritative

    def _ghost_pose(
        self,
        snapshot: Mapping[str, Any],
        board: Sequence[Sequence[int]],
        active: PiecePose | None,
    ) -> PiecePose | None:
        if active is None or active.piece not in _BASE_CELLS:
            return None
        raw = _get(snapshot, "ghost", None)
        if raw is False:
            return None
        pose = _coerce_pose(raw, fallback=active)
        if pose is not None:
            return pose
        y = landing_y(board, active.piece, active.rotation, round(active.x), round(active.y))
        return PiecePose(active.piece, active.rotation, active.x, float(y))

    def _cleared_rows(
        self,
        before_snapshot: Mapping[str, Any],
        outcome: Mapping[str, Any] | Any,
        final_pose: PiecePose | None,
        line_count: int,
    ) -> list[int]:
        # An explicit top-down list is unambiguous and preferred by clients.
        explicit_top = _get(outcome, "cleared_rows_top_down", None)
        if explicit_top is not None:
            rows = [int(row) for row in explicit_top if 0 <= int(row) < BOARD_ROWS]
            if rows:
                return rows

        # Native row masks index engine y from the bottom.
        mask = _get(outcome, "cleared_rows_mask", _get(outcome, "cleared_row_mask", None))
        if mask is not None:
            value = int(mask)
            rows = [BOARD_ROWS - 1 - y for y in range(BOARD_ROWS) if value & (1 << y)]
            if rows:
                return rows

        board = _normalize_board(_get(before_snapshot, "board", ()))
        composed = [row[:] for row in board]
        if final_pose is not None:
            for col, row in visible_cells(final_pose):
                composed[row][col] = final_pose.piece
        rows = [row for row in range(BOARD_ROWS) if all(_occupied(v) for v in composed[row])]
        if rows:
            return rows
        # Feedback remains visible for older bridges even if exact rows are not
        # available.  Normal play clears low rows, making this a benign fallback.
        count = max(1, min(BOARD_ROWS, int(line_count)))
        return list(range(BOARD_ROWS - count, BOARD_ROWS))

    def _draw_header(self, now_ms: float) -> None:
        del now_ms
        title = self._text("TETRIS", 34, self.palette.text, bold=True)
        subtitle = self._text("// V2", 20, self.palette.accent, bold=True)
        center_x = self.layout.board_rect.centerx
        total_width = title.get_width() + subtitle.get_width() + self._scaled(8)
        x = center_x - total_width // 2
        y = max(4, self.layout.board_rect.top - self._scaled(56))
        self.canvas.blit(title, (x, y))
        self.canvas.blit(subtitle, (x + title.get_width() + self._scaled(8), y + self._scaled(8)))

    def _draw_side_panels(
        self,
        snapshot: Mapping[str, Any],
        stats: Mapping[str, Any],
        meta: Mapping[str, Any],
        now_ms: float,
    ) -> None:
        hold = _get(snapshot, "hold", None)
        hold_mapping = _mapping(hold)
        if hold_mapping:
            held_piece = _get(hold_mapping, "hold_piece", _get(hold_mapping, "piece", None))
            hold_available = bool(_get(hold_mapping, "hold_available", _get(hold_mapping, "available", True)))
            if not bool(_get(hold_mapping, "has_hold", held_piece not in {None, 7})):
                held_piece = None
        else:
            held_piece = hold
            hold_available = bool(_get(snapshot, "hold_available", True))

        queue = _get(snapshot, "queue", ())
        if isinstance(queue, (str, bytes)) or queue is None:
            queue = ()

        self._panel(self.layout.hold_rect)
        self._panel_title(self.layout.hold_rect, "HOLD")
        hold_content = self.layout.hold_rect.inflate(-self._scaled(22), -self._scaled(52))
        hold_content.top += self._scaled(16)
        if held_piece is not None and int(held_piece) in _BASE_CELLS:
            alpha = 255 if hold_available else 95
            self._draw_preview_piece(int(held_piece), hold_content, alpha=alpha, large=True)
        else:
            empty = self._text("EMPTY", 16, self.palette.muted_text, bold=True)
            self.canvas.blit(empty, empty.get_rect(center=hold_content.center))
        if not hold_available:
            badge = self._text("USED", 13, self.palette.danger, bold=True)
            badge_rect = badge.get_rect(
                top=self.layout.hold_rect.top + self._scaled(15),
                right=self.layout.hold_rect.right - self._scaled(14),
            )
            self.canvas.blit(badge, badge_rect)

        self._panel(self.layout.next_rect)
        self._panel_title(self.layout.next_rect, "NEXT")
        slot_top = self.layout.next_rect.top + self._scaled(58)
        slot_height = max(1, (self.layout.next_rect.height - self._scaled(72)) // 5)
        for index, piece in enumerate(tuple(queue)[:5]):
            try:
                piece_id = int(piece)
            except (TypeError, ValueError):
                continue
            slot = pygame.Rect(
                self.layout.next_rect.left + self._scaled(14),
                slot_top + slot_height * index,
                self.layout.next_rect.width - self._scaled(28),
                slot_height,
            )
            if index > 0:
                pygame.draw.line(
                    self.canvas,
                    _mix(self.palette.panel, self.palette.panel_border, 0.55),
                    (slot.left + self._scaled(12), slot.top),
                    (slot.right - self._scaled(12), slot.top),
                    max(1, self._scaled(1)),
                )
            self._draw_preview_piece(piece_id, slot, alpha=255 if index == 0 else 210)

        self._draw_stats(stats, meta, now_ms)
        self._draw_controls()

    def _draw_stats(
        self,
        stats: Mapping[str, Any],
        meta: Mapping[str, Any],
        now_ms: float,
    ) -> None:
        self._panel(self.layout.stats_rect)
        self._panel_title(self.layout.stats_rect, "RUN")
        lines = int(_get(stats, "lines", _get(meta, "lines", 0)) or 0)
        combo = int(_get(stats, "combo", _get(meta, "combo", -1)) or 0)
        b2b = bool(_get(stats, "b2b", _get(meta, "b2b", False)))
        score = _get(stats, "score", _get(stats, "total_score", None))
        elapsed = _get(stats, "elapsed_s", _get(stats, "elapsed_seconds", None))
        if elapsed is None:
            start_ms = _get(stats, "start_ms", None)
            elapsed = max(0.0, (now_ms - float(start_ms)) / 1000.0) if start_ms is not None else 0.0
        pps = _get(stats, "pps", None)

        entries: list[tuple[str, str, tuple[int, int, int]]] = [
            ("LINES", f"{lines:,}", self.palette.text),
            ("COMBO", "--" if combo < 0 else f"x{combo + 1}", self.palette.text),
            ("B2B", "ACTIVE" if b2b else "--", self.palette.success if b2b else self.palette.muted_text),
            ("TIME", _format_time(float(elapsed)), self.palette.text),
        ]
        if score is not None:
            entries.insert(0, ("SCORE", f"{int(score):,}", self.palette.text))
        if pps is not None:
            entries.append(("PPS", f"{float(pps):.2f}", self.palette.text))

        top = self.layout.stats_rect.top + self._scaled(56)
        available = self.layout.stats_rect.height - self._scaled(70)
        row_height = max(1, available // max(1, len(entries)))
        for index, (label, value, color) in enumerate(entries):
            y = top + index * row_height
            label_surface = self._text(label, 13, self.palette.muted_text, bold=True)
            value_surface = self._text(value, 22, color, bold=True)
            self.canvas.blit(label_surface, (self.layout.stats_rect.left + self._scaled(18), y))
            self.canvas.blit(
                value_surface,
                (
                    self.layout.stats_rect.right - self._scaled(18) - value_surface.get_width(),
                    y - self._scaled(4),
                ),
            )

    def _draw_controls(self) -> None:
        self._panel(self.layout.controls_rect)
        self._panel_title(self.layout.controls_rect, "CONTROLS")
        lines = (
            "LEFT / RIGHT  MOVE",
            "DOWN  SOFT DROP",
            "SPACE  HARD DROP",
            "Z / X / A  ROTATE",
            "C / SHIFT  HOLD",
        )
        y = self.layout.controls_rect.top + self._scaled(48)
        for line in lines:
            text = self._text(line, 12, self.palette.muted_text, bold=True)
            self.canvas.blit(text, (self.layout.controls_rect.left + self._scaled(16), y))
            y += self._scaled(18)

    def _draw_board(
        self,
        board: Sequence[Sequence[int]],
        active: PiecePose | None,
        ghost: PiecePose | None,
        now_ms: float,
        *,
        game_over: bool,
    ) -> None:
        shake_x, shake_y = (0.0, 0.0) if self.reduced_motion else self.effects.shake_offset()
        rect = self.layout.board_rect.move(round(shake_x * self.layout.scale), round(shake_y * self.layout.scale))
        cell = self.layout.board_cell
        radius = max(3, self._scaled(7))
        shadow = rect.inflate(self._scaled(18), self._scaled(18)).move(0, self._scaled(7))
        pygame.draw.rect(self.canvas, (2, 4, 9), shadow, border_radius=radius)
        frame = rect.inflate(self._scaled(12), self._scaled(12))
        pygame.draw.rect(self.canvas, self.palette.panel_border, frame, border_radius=radius)
        pygame.draw.rect(self.canvas, self.palette.board, rect)

        old_clip = self.canvas.get_clip()
        self.canvas.set_clip(rect)
        for col in range(BOARD_COLS + 1):
            x = rect.left + col * cell
            pygame.draw.line(self.canvas, self.palette.grid, (x, rect.top), (x, rect.bottom), 1)
        for row in range(BOARD_ROWS + 1):
            y = rect.top + row * cell
            pygame.draw.line(self.canvas, self.palette.grid, (rect.left, y), (rect.right, y), 1)

        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                piece = board[row][col]
                if not _occupied(piece):
                    continue
                piece_id = int(piece)
                sprite = self._block_sprite(piece_id, cell, "locked")
                self.canvas.blit(sprite, (rect.left + col * cell, rect.top + row * cell))

        if ghost is not None and not game_over:
            self._draw_pose(rect, ghost, "ghost", pulse=0.82 + 0.12 * math.sin(now_ms * 0.007))
        if active is not None and not game_over:
            self._draw_pose(rect, active, "active")

        self._draw_line_flashes(rect)
        self._draw_particles(rect)
        self.canvas.set_clip(old_clip)
        pygame.draw.rect(
            self.canvas,
            self.palette.board_border,
            frame,
            width=max(1, self._scaled(2)),
            border_radius=radius,
        )

    def _draw_pose(
        self,
        board_rect: pygame.Rect,
        pose: PiecePose,
        style: str,
        *,
        pulse: float = 1.0,
    ) -> None:
        cell = self.layout.board_cell
        sprite = self._block_sprite(pose.piece, cell, style)
        if pulse != 1.0:
            sprite = sprite.copy()
            sprite.set_alpha(max(0, min(255, round(255 * pulse))))
        for dx, dy in piece_cells(pose.piece, pose.rotation):
            col = pose.x + dx
            engine_y = pose.y + dy
            x = board_rect.left + round(col * cell)
            y = board_rect.top + round((BOARD_ROWS - 1 - engine_y) * cell)
            self.canvas.blit(sprite, (x, y))

    def _draw_line_flashes(self, board_rect: pygame.Rect) -> None:
        cell = self.layout.board_cell
        for flash in self.effects.line_flashes:
            progress = min(1.0, flash.age / max(0.001, flash.duration))
            width_progress = min(1.0, progress / 0.38)
            alpha = round(245 * (1.0 - max(0.0, progress - 0.32) / 0.68))
            width = max(cell, round(board_rect.width * _ease_out_cubic(width_progress)))
            overlay = pygame.Surface((width, cell), pygame.SRCALPHA)
            overlay.fill((*self.palette.flash, max(0, alpha)))
            self.canvas.blit(
                overlay,
                (
                    board_rect.centerx - width // 2,
                    board_rect.top + flash.row * cell,
                ),
            )

    def _draw_particles(self, board_rect: pygame.Rect) -> None:
        cell = self.layout.board_cell
        for particle in self.effects.particles:
            progress = particle.age / max(0.001, particle.lifetime)
            alpha = max(0, min(255, round(255 * (1.0 - progress))))
            radius = max(1, round(particle.radius * cell * (1.0 - 0.35 * progress)))
            point = (
                round(board_rect.left + particle.x * cell),
                round(board_rect.top + particle.y * cell),
            )
            color = _mix(particle.color, self.palette.board, 1.0 - alpha / 255.0)
            pygame.draw.circle(self.canvas, color, point, radius)

    def _draw_callouts(self) -> None:
        if not self.effects.callouts:
            return
        center_x = self.layout.board_rect.centerx
        base_y = self.layout.board_rect.top + round(self.layout.board_rect.height * 0.40)
        for index, callout in enumerate(reversed(self.effects.callouts)):
            progress = callout.age / max(0.001, callout.duration)
            alpha = 255 if progress < 0.72 else round(255 * (1.0 - progress) / 0.28)
            rise = round(self._scaled(18) * _ease_out_cubic(progress))
            text = self._text(callout.text, 38 if index == 0 else 25, callout.color, bold=True).copy()
            text.set_alpha(max(0, min(255, alpha)))
            shadow = self._text(callout.text, 38 if index == 0 else 25, (2, 4, 8), bold=True).copy()
            shadow.set_alpha(max(0, min(180, alpha)))
            target = text.get_rect(center=(center_x, base_y - rise - index * self._scaled(36)))
            self.canvas.blit(shadow, target.move(self._scaled(3), self._scaled(4)))
            self.canvas.blit(text, target)

    def _draw_overlay(self, title: str, detail: str, *, danger: bool = False) -> None:
        veil = pygame.Surface(self.logical_size, pygame.SRCALPHA)
        veil.fill((3, 6, 12, 178))
        self.canvas.blit(veil, (0, 0))
        width = min(self._scaled(460), self.logical_size[0] - self._scaled(32))
        height = self._scaled(190)
        card = pygame.Rect(0, 0, width, height)
        card.center = self.layout.board_rect.center
        pygame.draw.rect(self.canvas, (8, 13, 23), card.move(0, self._scaled(8)), border_radius=self._scaled(14))
        pygame.draw.rect(self.canvas, self.palette.panel, card, border_radius=self._scaled(14))
        pygame.draw.rect(
            self.canvas,
            self.palette.danger if danger else self.palette.accent,
            card,
            width=max(1, self._scaled(2)),
            border_radius=self._scaled(14),
        )
        title_surface = self._text(
            title,
            48,
            self.palette.danger if danger else self.palette.text,
            bold=True,
        )
        detail_surface = self._text(detail, 19, self.palette.muted_text, bold=True)
        self.canvas.blit(title_surface, title_surface.get_rect(center=(card.centerx, card.centery - self._scaled(24))))
        self.canvas.blit(detail_surface, detail_surface.get_rect(center=(card.centerx, card.centery + self._scaled(38))))

    def _draw_preview_piece(
        self,
        piece: int,
        rect: pygame.Rect,
        *,
        alpha: int,
        large: bool = False,
    ) -> None:
        cells = piece_cells(piece, 0)
        if not cells:
            return
        min_x = min(x for x, _ in cells)
        max_x = max(x for x, _ in cells)
        min_y = min(y for _, y in cells)
        max_y = max(y for _, y in cells)
        columns = max_x - min_x + 1
        rows = max_y - min_y + 1
        maximum = self._scaled(27 if large else 22)
        cell = max(4, min(maximum, rect.width // max(1, columns), rect.height // max(1, rows)))
        width, height = columns * cell, rows * cell
        origin_x = rect.centerx - width // 2
        origin_y = rect.centery - height // 2
        sprite = self._block_sprite(piece, cell, "preview")
        if alpha < 255:
            sprite = sprite.copy()
            sprite.set_alpha(max(0, min(255, alpha)))
        for x, y in cells:
            col = x - min_x
            row = max_y - y
            self.canvas.blit(sprite, (origin_x + col * cell, origin_y + row * cell))

    def _panel(self, rect: pygame.Rect) -> None:
        radius = max(3, self._scaled(10))
        pygame.draw.rect(
            self.canvas,
            (3, 7, 14),
            rect.move(0, self._scaled(5)),
            border_radius=radius,
        )
        pygame.draw.rect(self.canvas, self.palette.panel, rect, border_radius=radius)
        pygame.draw.rect(
            self.canvas,
            self.palette.panel_border,
            rect,
            width=max(1, self._scaled(1)),
            border_radius=radius,
        )

    def _panel_title(self, rect: pygame.Rect, label: str) -> None:
        surface = self._text(label, 16, self.palette.muted_text, bold=True)
        self.canvas.blit(surface, (rect.left + self._scaled(16), rect.top + self._scaled(14)))
        pygame.draw.line(
            self.canvas,
            self.palette.accent,
            (rect.left + self._scaled(16), rect.top + self._scaled(40)),
            (rect.left + self._scaled(54), rect.top + self._scaled(40)),
            max(1, self._scaled(2)),
        )

    def _block_sprite(self, piece: int, cell: int, style: str) -> pygame.Surface:
        key = (int(piece), int(cell), str(style))
        cached = self._sprite_cache.get(key)
        if cached is not None:
            return cached
        size = max(2, int(cell))
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        base = PIECE_COLORS.get(int(piece), PIECE_COLORS[GARBAGE_CELL_ID])
        margin = max(1, size // 22)
        radius = max(2, size // 7)
        rect = pygame.Rect(margin, margin, size - margin * 2, size - margin * 2)

        if style == "ghost":
            fill = (*_mix(base, self.palette.board, 0.45), 30)
            outline = (*_mix(base, (255, 255, 255), 0.18), 150)
            pygame.draw.rect(surface, fill, rect, border_radius=radius)
            pygame.draw.rect(
                surface,
                outline,
                rect.inflate(-max(1, size // 9), -max(1, size // 9)),
                width=max(1, size // 14),
                border_radius=max(1, radius - 1),
            )
        else:
            if style == "locked":
                base = _mix(base, self.palette.board, 0.16)
            elif style in {"active", "preview"}:
                base = _mix(base, (255, 255, 255), 0.06)
            shadow = _mix(base, (0, 0, 0), 0.55)
            pygame.draw.rect(surface, shadow, rect.move(0, max(1, size // 13)), border_radius=radius)
            pygame.draw.rect(surface, base, rect, border_radius=radius)
            inner = rect.inflate(-max(2, size // 6), -max(2, size // 6))
            pygame.draw.rect(surface, _mix(base, (255, 255, 255), 0.08), inner, border_radius=max(1, radius // 2))
            highlight = max(1, size // 12)
            pygame.draw.line(
                surface,
                _mix(base, (255, 255, 255), 0.42),
                (rect.left + radius, rect.top + highlight),
                (rect.right - radius, rect.top + highlight),
                highlight,
            )
            pygame.draw.line(
                surface,
                _mix(base, (0, 0, 0), 0.28),
                (rect.left + radius, rect.bottom - highlight),
                (rect.right - radius, rect.bottom - highlight),
                highlight,
            )
        self._sprite_cache[key] = surface
        return surface

    def _font(self, base_size: int, bold: bool) -> pygame.font.Font:
        size = max(8, self._scaled(base_size))
        key = (size, bool(bold))
        font = self._font_cache.get(key)
        if font is None:
            font = pygame.font.Font(None, size)
            font.set_bold(bool(bold))
            self._font_cache[key] = font
        return font

    def _text(
        self,
        text: str,
        base_size: int,
        color: tuple[int, int, int],
        *,
        bold: bool = False,
    ) -> pygame.Surface:
        key = (str(text), int(base_size), tuple(color), bool(bold))
        cached = self._text_cache.get(key)
        if cached is not None:
            self._text_cache.move_to_end(key)
            return cached
        surface = self._font(base_size, bold).render(str(text), True, color)
        self._text_cache[key] = surface
        if len(self._text_cache) > 384:
            self._text_cache.popitem(last=False)
        return surface

    def _scaled(self, value: float) -> int:
        return max(1, round(float(value) * self.layout.scale))

    def _make_background(self) -> pygame.Surface:
        width, height = self.logical_size
        background = pygame.Surface((width, height))
        for y in range(height):
            amount = y / max(1, height - 1)
            color = _mix(self.palette.background_top, self.palette.background_bottom, amount)
            pygame.draw.line(background, color, (0, y), (width, y))
        # Very subtle fixed glows add depth without requiring bitmap assets.
        glow = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.circle(
            glow,
            (*self.palette.accent, 13),
            (width // 2, round(height * 0.30)),
            max(1, round(min(width, height) * 0.34)),
        )
        pygame.draw.circle(
            glow,
            (176, 94, 255, 8),
            (round(width * 0.10), round(height * 0.83)),
            max(1, round(min(width, height) * 0.22)),
        )
        background.blit(glow, (0, 0))
        return background


def _normalize_board(board: Any) -> list[list[int]]:
    if hasattr(board, "tolist"):
        board = board.tolist()
    if board is None:
        board = ()
    try:
        values = list(board)
    except TypeError:
        values = []
    if len(values) == BOARD_ROWS * BOARD_COLS and (
        not values or not isinstance(values[0], (list, tuple))
    ):
        values = [values[row * BOARD_COLS : (row + 1) * BOARD_COLS] for row in range(BOARD_ROWS)]

    normalized = [[EMPTY_CELL_ID for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
    for row in range(min(BOARD_ROWS, len(values))):
        try:
            source = list(values[row])
        except TypeError:
            continue
        for col in range(min(BOARD_COLS, len(source))):
            try:
                normalized[row][col] = int(source[col])
            except (TypeError, ValueError):
                normalized[row][col] = EMPTY_CELL_ID
    return normalized


def _occupied(value: Any) -> bool:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return False
    return number != EMPTY_CELL_ID and number >= 0


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _coerce_pose(value: Any, *, fallback: PiecePose | None = None) -> PiecePose | None:
    if value is None:
        return None
    if isinstance(value, PiecePose):
        return value
    if isinstance(value, (int, float)) and fallback is not None:
        return PiecePose(fallback.piece, fallback.rotation, fallback.x, float(value))
    piece = _get(value, "piece", fallback.piece if fallback else None)
    if piece is None:
        return None
    try:
        return PiecePose(
            int(piece),
            int(_get(value, "rotation", fallback.rotation if fallback else 0)),
            float(_get(value, "x", fallback.x if fallback else 0.0)),
            float(_get(value, "y", fallback.y if fallback else 0.0)),
        )
    except (TypeError, ValueError):
        return None


def _action_name(action: int | str) -> str:
    if isinstance(action, str):
        normalized = action.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "rotate_cw": "cw",
            "rotate_ccw": "ccw",
            "rotate_180": "180",
            "harddrop": "hard_drop",
            "softdrop": "soft_drop",
        }
        return aliases.get(normalized, normalized)
    try:
        number = int(action)
    except (TypeError, ValueError):
        return "none"
    return {
        ACTION_NONE: "none",
        ACTION_LEFT: "left",
        ACTION_RIGHT: "right",
        ACTION_SOFT_DROP: "soft_drop",
        ACTION_HARD_DROP: "hard_drop",
        ACTION_CW: "cw",
        ACTION_CCW: "ccw",
        ACTION_180: "180",
        ACTION_HOLD: "hold",
    }.get(number, "none")


def _clear_label(
    lines: int,
    spin_clear: bool,
    spin_type: int,
    b2b_bonus: bool,
    combo: int,
) -> str:
    line_names = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 4: "TETRIS"}
    if spin_clear:
        spin = "T-SPIN MINI" if spin_type == 1 else "T-SPIN"
        main = f"{spin} {line_names.get(lines, '')}".strip()
    else:
        main = line_names.get(lines, f"{lines} LINES")
    if b2b_bonus:
        main = f"B2B · {main}"
    if combo > 0:
        main = f"{main} · {combo + 1} COMBO"
    return main


def _format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds) // 60
    remaining = int(seconds) % 60
    return f"{minutes:02d}:{remaining:02d}"


def _mix(
    first: tuple[int, int, int],
    second: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, float(amount)))
    return tuple(
        max(0, min(255, round(a + (b - a) * amount)))
        for a, b in zip(first, second, strict=True)
    )


def _ease_out_cubic(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return 1.0 - (1.0 - value) ** 3


__all__ = [
    "ACTION_180",
    "ACTION_CCW",
    "ACTION_CW",
    "ACTION_HARD_DROP",
    "ACTION_HOLD",
    "ACTION_LEFT",
    "ACTION_NONE",
    "ACTION_RIGHT",
    "ACTION_SOFT_DROP",
    "BOARD_COLS",
    "BOARD_ROWS",
    "DEFAULT_LOGICAL_SIZE",
    "EMPTY_CELL_ID",
    "FeedbackLayer",
    "GARBAGE_CELL_ID",
    "LogicalLayout",
    "PIECE_COLORS",
    "Palette",
    "PiecePose",
    "PoseTween",
    "Renderer",
    "SynthSoundBank",
    "landing_y",
    "piece_cells",
    "visible_cells",
]
