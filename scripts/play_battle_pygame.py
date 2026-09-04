"""Watch an RL placement policy play a live two-player Tetris battle.

The heavyweight training modules are deliberately imported only after argument
parsing.  This keeps ``--help`` and the renderer usable in lightweight Python
environments that do not have PyTorch or Gymnasium installed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame


BOARD_ROWS = 20
BOARD_COLS = 10
EMPTY_CELL_ID = 255
OWN_OBSERVATION_SIZE = 254
PLACEMENT_ACTION_DIM = 3200

BG = (8, 11, 18)
PANEL = (15, 21, 32)
PANEL_LIGHT = (21, 29, 43)
BOARD_BG = (7, 11, 18)
GRID = (27, 36, 51)
TEXT = (231, 239, 250)
MUTED = (133, 150, 174)
LEARNER = (61, 225, 211)
OPPONENT = (255, 102, 153)
WARNING = (255, 104, 91)
GARBAGE = (101, 112, 130)

PIECE_COLORS: dict[int, tuple[int, int, int]] = {
    0: (38, 218, 238),
    1: (250, 214, 66),
    2: (177, 91, 238),
    3: (255, 153, 64),
    4: (74, 117, 239),
    5: (83, 213, 112),
    6: (242, 79, 91),
    7: GARBAGE,
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch a TetrisV2 checkpoint battle Cold Clear in pygame."
    )
    parser.add_argument("checkpoint", type=Path, help="RL checkpoint (.pt) to watch")
    parser.add_argument(
        "--algo",
        choices=("flow_dqn", "battle_dqn", "dqn", "ppo"),
        default="flow_dqn",
        help="Checkpoint algorithm (default: flow_dqn)",
    )
    parser.add_argument(
        "--opponent",
        choices=("cold_clear", "random"),
        default="cold_clear",
        help="Opponent policy (default: cold_clear)",
    )
    parser.add_argument("--seed", type=int, default=123, help="Initial match seed")
    parser.add_argument("--fps", type=_positive_int, default=60, help="Render frame rate")
    parser.add_argument(
        "--step-ms",
        type=_nonnegative_int,
        default=110,
        help="Delay between atomic placement decisions (default: 110)",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        default=None,
        help="Override maximum joint placements (default: checkpoint rule or 500)",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device")
    parser.add_argument(
        "--lib", type=Path, default=None, help="Path to the native tetris_v2 C API library"
    )
    parser.add_argument(
        "--cold-clear-think-ms",
        type=_nonnegative_int,
        default=0,
        help="Cold Clear wall-clock budget; 0 uses deterministic fixed work",
    )
    parser.add_argument(
        "--auto-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start the next seed after showing the result (default: on)",
    )
    parser.add_argument("--cell", type=_positive_int, default=26, help="Board cell size")
    parser.add_argument(
        "--screenshot", type=Path, default=None, help="Save the last rendered frame"
    )
    parser.add_argument("--max-frames", type=_nonnegative_int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


class BattlePolicyLike(Protocol):
    identifier: str
    kind: str

    def reset(self, seed: int) -> None: ...

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        player: int,
        env: Any,
    ) -> int: ...


@dataclass
class SinglePlayerPolicyAdapter:
    """Expose a 254-observation placement policy as a battle policy."""

    policy: Any
    identifier: str
    kind: str = "single_player_checkpoint"

    def reset(self, seed: int) -> None:
        # Current single-player inference policies are stateless.  Seed numpy
        # anyway so a future stochastic adapter has a stable match boundary.
        self._seed = int(seed)

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        player: int,
        env: Any,
    ) -> int:
        del player, env
        own = np.asarray(observation, dtype=np.float32).reshape(-1)[:OWN_OBSERVATION_SIZE]
        if own.shape != (OWN_OBSERVATION_SIZE,):
            raise ValueError("single-player battle adapter requires at least 254 observations")
        return int(
            self.policy.act(
                own,
                deterministic=True,
                temperature=1.0,
                epsilon=0.0,
                action_mask=np.asarray(action_mask, dtype=np.float32),
            )
        )


@dataclass
class LoadedLearner:
    policy: BattlePolicyLike
    checkpoint_metadata: dict[str, object]


def load_learner(algo: str, checkpoint: Path, *, device: str) -> LoadedLearner:
    """Load only after CLI parsing so ``--help`` has no torch dependency."""

    if algo == "battle_dqn":
        from tetris_v2.rl.battle.cli import (  # heavyweight, intentionally local
            load_evaluation_policy,
            policy_checkpoint_metadata,
        )

        policy = load_evaluation_policy(
            checkpoint,
            device=device,
            identifier=checkpoint.stem,
            kind="learner",
        )
        return LoadedLearner(policy, policy_checkpoint_metadata(policy))

    from tetris_v2.rl.policy import load_policy  # heavyweight, intentionally local

    loaded = load_policy(algo, checkpoint, device=device)
    if int(loaded.obs_dim) != OWN_OBSERVATION_SIZE:
        raise ValueError(
            f"{algo} checkpoint observation size is {loaded.obs_dim}; expected "
            f"{OWN_OBSERVATION_SIZE} for battle adaptation"
        )
    if int(loaded.action_dim) != PLACEMENT_ACTION_DIM:
        raise ValueError(
            f"{algo} checkpoint action size is {loaded.action_dim}; expected "
            f"{PLACEMENT_ACTION_DIM}"
        )
    policy = SinglePlayerPolicyAdapter(loaded, checkpoint.stem)
    return LoadedLearner(policy, dict(getattr(loaded, "metadata", {})))


def load_opponent(mode: str, *, seed: int, cold_clear_think_ms: int) -> BattlePolicyLike:
    """Construct the selected native-compatible opponent lazily."""

    from tetris_v2.rl.battle.policies import (  # heavyweight, intentionally local
        ColdClearBattlePolicy,
        RandomBattlePolicy,
    )

    if mode == "cold_clear":
        return ColdClearBattlePolicy(think_ms=int(cold_clear_think_ms))
    return RandomBattlePolicy(seed=int(seed))


def make_battle_env(
    *,
    seed: int,
    lib_path: Path | None,
    max_steps: int | None,
    checkpoint_metadata: Mapping[str, object],
) -> Any:
    """Create a BattleEnv, inheriting Battle-DQN rules where available."""

    from tetris_v2.rl.battle.config import BattleRewardConfig, BattleRulesConfig
    from tetris_v2.rl.battle.env import BattleEnv

    default_rules = BattleRulesConfig().to_dict()
    stored_rules = checkpoint_metadata.get("rules", {})
    if not isinstance(stored_rules, Mapping):
        raise ValueError("battle checkpoint rules configuration is malformed")
    rule_values = {**default_rules, **dict(stored_rules)}
    if max_steps is not None:
        rule_values["max_steps"] = int(max_steps)

    default_rewards = BattleRewardConfig().to_dict()
    stored_rewards = checkpoint_metadata.get("rewards", {})
    if not isinstance(stored_rewards, Mapping):
        raise ValueError("battle checkpoint rewards configuration is malformed")

    return BattleEnv(
        seed=int(seed),
        lib_path=lib_path,
        rules=BattleRulesConfig(**rule_values),
        reward_config=BattleRewardConfig(**{**default_rewards, **dict(stored_rewards)}),
    )


class BattleSession:
    """Small UI-independent match controller around atomic joint steps."""

    def __init__(
        self,
        env: Any,
        learner: BattlePolicyLike,
        opponent: BattlePolicyLike,
        *,
        seed: int,
        learner_seat: int = 0,
    ):
        self.env = env
        self.learner = learner
        self.opponent = opponent
        self.seed = int(seed)
        self.learner_seat = int(learner_seat)
        if self.learner_seat not in (0, 1):
            raise ValueError("learner_seat must be 0 or 1")
        self.observations: tuple[np.ndarray, np.ndarray]
        self.masks: tuple[np.ndarray, np.ndarray]
        self.info: dict[str, Any]
        self.last_actions: tuple[int | None, int | None] = (None, None)
        self.last_rewards = (0.0, 0.0)
        self.done = False
        self.reset(self.seed)

    def policy_for_seat(self, seat: int) -> BattlePolicyLike:
        return self.learner if int(seat) == self.learner_seat else self.opponent

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        observations, masks, info = self.env.reset(seed=self.seed)
        self.observations = tuple(np.asarray(value, dtype=np.float32) for value in observations)  # type: ignore[assignment]
        self.masks = tuple(np.asarray(value, dtype=np.float32) for value in masks)  # type: ignore[assignment]
        self.info = dict(info)
        self.learner.reset(self.seed * 2 + 1)
        self.opponent.reset(self.seed * 2 + 2)
        self.last_actions = (None, None)
        self.last_rewards = (0.0, 0.0)
        self.done = False

    def next_seed(self) -> None:
        self.reset(self.seed + 1)

    def swap_seats(self) -> None:
        self.learner_seat = 1 - self.learner_seat
        self.reset(self.seed)

    def step(self) -> dict[str, Any]:
        if self.done:
            return self.info
        actions: list[int] = []
        for seat in (0, 1):
            action = int(
                self.policy_for_seat(seat).select_action(
                    self.observations[seat],
                    self.masks[seat],
                    player=seat,
                    env=self.env,
                )
            )
            actions.append(action)
        observations, rewards, terminated, truncated, info = self.env.step(
            (actions[0], actions[1])
        )
        self.observations = tuple(np.asarray(value, dtype=np.float32) for value in observations)  # type: ignore[assignment]
        self.info = dict(info)
        self.masks = tuple(
            np.asarray(value, dtype=np.float32)
            for value in self.info.get("action_masks", self.masks)
        )  # type: ignore[assignment]
        self.last_actions = (actions[0], actions[1])
        self.last_rewards = (float(rewards[0]), float(rewards[1]))
        self.done = bool(terminated or truncated)
        return self.info

    def board_piece_ids(self, seat: int) -> np.ndarray:
        return np.asarray(
            self.env.runtime.board_piece_ids(int(seat), include_active=True),
            dtype=np.uint8,
        ).reshape(BOARD_ROWS, BOARD_COLS)


def decode_action_name(action: int | None) -> str:
    if action is None:
        return "waiting"
    value = int(action)
    use_hold = value >= 1600
    pose = value - 1600 if use_hold else value
    rotation, remainder = divmod(pose, BOARD_ROWS * BOARD_COLS)
    y, x = divmod(remainder, BOARD_COLS)
    prefix = "HOLD + " if use_hold else ""
    return f"{prefix}x{x} y{y} r{rotation}"


def _lighten(color: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
    return tuple(min(255, channel + amount) for channel in color)


def draw_board(
    surface: pygame.Surface,
    board: np.ndarray,
    rect: pygame.Rect,
    *,
    cell: int,
    accent: tuple[int, int, int],
    incoming: int,
) -> None:
    pygame.draw.rect(surface, (3, 6, 11), rect.inflate(18, 18), border_radius=12)
    pygame.draw.rect(surface, accent, rect.inflate(8, 8), width=2, border_radius=8)
    pygame.draw.rect(surface, BOARD_BG, rect)
    cells = np.asarray(board, dtype=np.uint8).reshape(BOARD_ROWS, BOARD_COLS)
    for row in range(BOARD_ROWS):
        for column in range(BOARD_COLS):
            square = pygame.Rect(
                rect.x + column * cell,
                rect.y + row * cell,
                cell,
                cell,
            )
            piece = int(cells[row, column])
            if piece != EMPTY_CELL_ID:
                color = PIECE_COLORS.get(piece, GARBAGE)
                inner = square.inflate(-2, -2)
                pygame.draw.rect(surface, color, inner, border_radius=max(2, cell // 7))
                pygame.draw.line(
                    surface,
                    _lighten(color, 34),
                    (inner.left + 3, inner.top + 2),
                    (inner.right - 3, inner.top + 2),
                    width=max(1, cell // 10),
                )
                pygame.draw.rect(surface, (0, 0, 0), inner, width=1, border_radius=3)
                if piece == 7:
                    pygame.draw.line(
                        surface,
                        (137, 148, 166),
                        inner.bottomleft,
                        inner.topright,
                        width=1,
                    )
            pygame.draw.rect(surface, GRID, square, width=1)

    if incoming > 0:
        meter_h = max(5, min(rect.height, round(rect.height * min(incoming, 20) / 20)))
        meter = pygame.Rect(rect.right + 9, rect.bottom - meter_h, 7, meter_h)
        pygame.draw.rect(surface, (64, 30, 36), (meter.x, rect.y, meter.w, rect.h), border_radius=4)
        pygame.draw.rect(surface, WARNING, meter, border_radius=4)


@dataclass(frozen=True)
class SceneLayout:
    size: tuple[int, int]
    left_board: pygame.Rect
    right_board: pygame.Rect
    cell: int


def scene_layout(cell: int) -> SceneLayout:
    size = max(14, int(cell))
    board_w, board_h = BOARD_COLS * size, BOARD_ROWS * size
    margin = 64
    gap = 150
    board_y = 130
    left = pygame.Rect(margin, board_y, board_w, board_h)
    right = pygame.Rect(margin + board_w + gap, board_y, board_w, board_h)
    return SceneLayout(
        (right.right + margin, board_y + board_h + 160),
        left,
        right,
        size,
    )


@lru_cache(maxsize=32)
def _font(size: int, *, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont("Inter,Segoe UI,DejaVu Sans", int(size), bold=bold)


def _text(
    surface: pygame.Surface,
    value: str,
    position: tuple[int, int],
    *,
    size: int,
    color: tuple[int, int, int] = TEXT,
    bold: bool = False,
    center: bool = False,
) -> pygame.Rect:
    rendered = _font(size, bold=bold).render(str(value), True, color)
    rect = rendered.get_rect(center=position) if center else rendered.get_rect(topleft=position)
    surface.blit(rendered, rect)
    return rect


def _players(info: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    raw = info.get("players", ({}, {}))
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return {}, {}
    return raw[0], raw[1]


def _step_stats(info: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    raw = info.get("step_stats", ({}, {}))
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return {}, {}
    return raw[0], raw[1]


def _policy_title(policy: BattlePolicyLike, *, learner: bool) -> str:
    if learner:
        return str(getattr(policy, "identifier", "AI CHECKPOINT")).replace("_", " ").upper()
    value = str(getattr(policy, "identifier", "OPPONENT")).replace("_", " ").upper()
    return "COLD CLEAR" if value == "COLD CLEAR" else value


def draw_scene(
    session: BattleSession,
    *,
    cell: int,
    paused: bool,
    step_ms: int,
    status: str,
) -> pygame.Surface:
    layout = scene_layout(cell)
    surface = pygame.Surface(layout.size)
    surface.fill(BG)

    # Subtle header and footer panels keep the boards visually dominant.
    pygame.draw.rect(surface, PANEL, (20, 18, layout.size[0] - 40, 90), border_radius=18)
    _text(surface, "TETRISV2  //  BATTLE LAB", (42, 34), size=16, color=MUTED, bold=True)
    opponent_name = _policy_title(session.opponent, learner=False)
    _text(surface, f"AI CHECKPOINT  vs  {opponent_name}", (42, 57), size=28, bold=True)
    header_meta = _font(15, bold=True).render(
        f"SEED {session.seed}   •   JOINT STEP {int(session.info.get('step', 0))}",
        True,
        MUTED,
    )
    surface.blit(header_meta, header_meta.get_rect(topright=(layout.size[0] - 42, 43)))

    players = _players(session.info)
    boards = (session.board_piece_ids(0), session.board_piece_ids(1))
    board_rects = (layout.left_board, layout.right_board)
    accents = tuple(
        LEARNER if seat == session.learner_seat else OPPONENT for seat in (0, 1)
    )

    for seat in (0, 1):
        player = players[seat]
        policy = session.policy_for_seat(seat)
        learner = seat == session.learner_seat
        rect = board_rects[seat]
        accent = accents[seat]
        label = _policy_title(policy, learner=learner)
        _text(surface, f"P{seat + 1}  {label}", (rect.x, rect.y - 40), size=19, color=accent, bold=True)
        incoming = int(player.get("incoming_garbage", 0))
        draw_board(
            surface,
            boards[seat],
            rect,
            cell=layout.cell,
            accent=accent,
            incoming=incoming,
        )
        stats_y = rect.bottom + 17
        _text(
            surface,
            f"{int(player.get('placements', 0)):03d} PCS   "
            f"{int(player.get('lines_cleared', 0)):03d} LINES   "
            f"{int(player.get('attack_generated', 0)):03d} ATK",
            (rect.x, stats_y),
            size=14,
            color=TEXT,
            bold=True,
        )
        _text(
            surface,
            f"SENT {int(player.get('garbage_sent', 0)):03d}   "
            f"INCOMING {incoming:02d}",
            (rect.x, stats_y + 25),
            size=13,
            color=WARNING if incoming else MUTED,
            bold=True,
        )
        _text(
            surface,
            decode_action_name(session.last_actions[seat]),
            (rect.x, stats_y + 49),
            size=12,
            color=MUTED,
        )

    center_x = (layout.left_board.right + layout.right_board.left) // 2
    step_stats = _step_stats(session.info)
    left_sent = int(step_stats[0].get("garbage_sent", 0))
    right_sent = int(step_stats[1].get("garbage_sent", 0))
    pygame.draw.circle(surface, PANEL_LIGHT, (center_x, layout.left_board.centery), 52)
    _text(surface, "ATTACK", (center_x, layout.left_board.centery - 24), size=12, color=MUTED, bold=True, center=True)
    _text(
        surface,
        f"{left_sent}  < >  {right_sent}",
        (center_x, layout.left_board.centery + 2),
        size=25,
        color=TEXT,
        bold=True,
        center=True,
    )
    _text(
        surface,
        f"{step_ms} ms",
        (center_x, layout.left_board.centery + 31),
        size=12,
        color=MUTED,
        center=True,
    )

    controls = "SPACE pause   . single-step   R reset   N next seed   S swap seats   +/- speed   ESC quit"
    _text(
        surface,
        controls,
        (layout.size[0] // 2, layout.size[1] - 23),
        size=13,
        color=MUTED,
        center=True,
    )
    _text(
        surface,
        status,
        (layout.size[0] // 2, layout.size[1] - 52),
        size=13,
        color=(172, 206, 255),
        center=True,
    )

    if paused or session.done:
        shade = pygame.Surface(layout.size, pygame.SRCALPHA)
        shade.fill((2, 5, 10, 150))
        surface.blit(shade, (0, 0))
        card = pygame.Rect(0, 0, min(460, layout.size[0] - 60), 150)
        card.center = (layout.size[0] // 2, layout.left_board.centery)
        pygame.draw.rect(surface, PANEL_LIGHT, card, border_radius=18)
        pygame.draw.rect(surface, (64, 78, 99), card, width=2, border_radius=18)
        if session.done:
            winner = session.info.get("winner")
            if winner in (0, 1):
                winner_policy = session.policy_for_seat(int(winner))
                heading = f"{_policy_title(winner_policy, learner=int(winner) == session.learner_seat)} WINS"
                subheading = f"Player {int(winner) + 1} • seed {session.seed}"
            else:
                heading = "DRAW"
                subheading = f"Step limit reached • seed {session.seed}"
        else:
            heading = "PAUSED"
            subheading = "Press Space to resume or . to advance once"
        _text(surface, heading, (card.centerx, card.y + 49), size=32, bold=True, center=True)
        _text(surface, subheading, (card.centerx, card.y + 93), size=15, color=MUTED, center=True)

    return surface


def present(logical: pygame.Surface, screen: pygame.Surface) -> pygame.Rect:
    target = screen.get_rect()
    scale = min(target.width / logical.get_width(), target.height / logical.get_height())
    width = max(1, round(logical.get_width() * scale))
    height = max(1, round(logical.get_height() * scale))
    fitted = pygame.Rect(0, 0, width, height)
    fitted.center = target.center
    screen.fill((2, 4, 8))
    if logical.get_size() == fitted.size:
        screen.blit(logical, fitted)
    else:
        screen.blit(pygame.transform.smoothscale(logical, fitted.size), fitted)
    return fitted


def _open_window(size: tuple[int, int]) -> pygame.Surface:
    flags = pygame.RESIZABLE | pygame.DOUBLEBUF
    try:
        return pygame.display.set_mode(size, flags, vsync=1)
    except (TypeError, pygame.error):
        return pygame.display.set_mode(size, flags)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        learner = load_learner(args.algo, args.checkpoint, device=str(args.device))
        opponent = load_opponent(
            args.opponent,
            seed=int(args.seed) + 1,
            cold_clear_think_ms=int(args.cold_clear_think_ms),
        )
        env = make_battle_env(
            seed=int(args.seed),
            lib_path=args.lib,
            max_steps=None if args.max_steps is None else int(args.max_steps),
            checkpoint_metadata=(
                learner.checkpoint_metadata if args.algo == "battle_dqn" else {}
            ),
        )
    except ModuleNotFoundError as exc:
        dependency = exc.name or "an optional dependency"
        raise SystemExit(
            f"Battle playback needs {dependency!r}. Install the project dependencies "
            "in this interpreter with `python -m pip install -e .`."
        ) from None

    pygame_started = False
    last_frame: pygame.Surface | None = None
    try:
        session = BattleSession(env, learner.policy, opponent, seed=int(args.seed))
        pygame.init()
        pygame_started = True
        pygame.display.set_caption(
            f"TetrisV2 Battle — {learner.policy.identifier} vs {opponent.identifier}"
        )
        screen = _open_window(scene_layout(int(args.cell)).size)
        pygame.event.set_blocked(None)
        pygame.event.set_allowed((pygame.QUIT, pygame.KEYDOWN, pygame.VIDEORESIZE))
        clock = pygame.time.Clock()
        paused = False
        running = True
        frame_count = 0
        step_ms = int(args.step_ms)
        next_step_at = pygame.time.get_ticks() + step_ms
        done_at: int | None = None
        single_step = False
        status = f"Loaded {args.checkpoint.name} • learner in Player 1"

        while running:
            now = pygame.time.get_ticks()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    screen = _open_window((max(480, event.w), max(480, event.h)))
                elif event.type == pygame.KEYDOWN and not getattr(event, "repeat", False):
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        next_step_at = now + step_ms
                        status = "Paused" if paused else "Running"
                    elif event.key == pygame.K_PERIOD:
                        single_step = True
                        paused = True
                    elif event.key == pygame.K_r:
                        session.reset()
                        done_at = None
                        next_step_at = now + step_ms
                        status = f"Reset seed {session.seed}"
                    elif event.key == pygame.K_n:
                        session.next_seed()
                        done_at = None
                        next_step_at = now + step_ms
                        status = f"Next seed {session.seed}"
                    elif event.key == pygame.K_s:
                        session.swap_seats()
                        done_at = None
                        next_step_at = now + step_ms
                        status = f"Swapped seats • learner in Player {session.learner_seat + 1}"
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        step_ms = max(0, step_ms - 20)
                        next_step_at = now + step_ms
                        status = f"Decision delay {step_ms} ms"
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        step_ms = min(2000, step_ms + 20)
                        next_step_at = now + step_ms
                        status = f"Decision delay {step_ms} ms"

            should_step = (single_step or (not paused and now >= next_step_at)) and not session.done
            if should_step:
                session.step()
                single_step = False
                next_step_at = pygame.time.get_ticks() + step_ms
                status = (
                    f"Player 1 {decode_action_name(session.last_actions[0])}  •  "
                    f"Player 2 {decode_action_name(session.last_actions[1])}"
                )
                if session.done:
                    done_at = pygame.time.get_ticks()

            if (
                session.done
                and args.auto_reset
                and done_at is not None
                and now - done_at >= 1400
            ):
                session.next_seed()
                done_at = None
                next_step_at = now + step_ms
                status = f"Auto-reset • seed {session.seed}"

            last_frame = draw_scene(
                session,
                cell=int(args.cell),
                paused=paused,
                step_ms=step_ms,
                status=status,
            )
            present(last_frame, screen)
            pygame.display.flip()
            frame_count += 1
            if args.max_frames and frame_count >= int(args.max_frames):
                running = False
            clock.tick(int(args.fps))

        if args.screenshot is not None and last_frame is not None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(last_frame, str(args.screenshot))
    finally:
        env.close()
        if pygame_started:
            _font.cache_clear()
            pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
