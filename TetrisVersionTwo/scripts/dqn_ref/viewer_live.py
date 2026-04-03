from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pygame
except Exception:  # pragma: no cover - optional dependency at runtime
    pygame = None  # type: ignore[assignment]


BOARD_ROWS = 20
BOARD_COLS = 10

BG = (10, 14, 20)
PANEL = (18, 24, 34)
PANEL_ALT = (22, 30, 42)
TEXT = (230, 236, 245)
TEXT_DIM = (160, 172, 188)
ACCENT = (72, 132, 255)
ACTIVE = (86, 214, 146)
DONE = (90, 160, 255)
FAILED = (255, 115, 115)
EMPTY = (23, 30, 43)
FILLED = (130, 170, 255)
GRID = (36, 48, 66)
EMPTY_CELL_ID = 255
PIECE_COLORS = {
    0: (0, 230, 230),   # I
    1: (240, 220, 0),   # O
    2: (170, 70, 230),  # T
    3: (255, 150, 60),  # L
    4: (80, 110, 255),  # J
    5: (80, 220, 100),  # S
    6: (240, 90, 90),   # Z
}


@dataclass
class AgentViewState:
    agent_index: int
    generation: int = 0
    status: str = "idle"
    episode_index: int = 0
    games_per_agent: int = 0
    step_in_episode: int = 0
    lines_total: int = 0
    episode_return_running: float = 0.0
    epsilon: float = 0.0
    loss_last: float = 0.0
    fitness_provisional: float = 0.0
    fitness_final: Optional[float] = None
    board: np.ndarray = field(
        default_factory=lambda: np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.uint8)
    )
    board_piece_ids: np.ndarray = field(
        default_factory=lambda: np.full((BOARD_ROWS, BOARD_COLS), EMPTY_CELL_ID, dtype=np.uint8)
    )
    recent_episodes: List[Dict[str, float]] = field(default_factory=list)
    last_update_ts: float = 0.0


class LiveTrainingViewer:
    def __init__(
        self,
        *,
        total_agents: int,
        total_generations: int,
        games_per_agent: int,
        fullscreen: bool = True,
        fps: int = 20,
        grid_padding: int = 8,
        min_tile_px: int = 6,
        initial_selected_agent: int = 1,
        run_dir: str = "",
    ):
        self.total_agents = int(max(1, total_agents))
        self.total_generations = int(max(1, total_generations))
        self.games_per_agent = int(max(1, games_per_agent))
        self.fullscreen = bool(fullscreen)
        self.fps = int(max(5, fps))
        self.grid_padding = int(max(2, grid_padding))
        self.min_tile_px = int(max(2, min_tile_px))
        self.selected_agent = int(max(1, min(initial_selected_agent, self.total_agents)))
        self.run_dir = run_dir

        self.ready = False
        self.closed = False
        self._screen = None
        self._clock = None
        self._font = None
        self._small = None
        self._mono = None
        self._last_frame_ts = 0.0
        self._start_ts = time.time()
        self._agent_rects: Dict[int, Tuple[int, int, int, int]] = {}
        self._windowed_size = (1600, 900)
        self._global_generation = 0
        self._completed_agents = 0
        self._events_processed = 0

        self.agents: Dict[int, AgentViewState] = {
            i: AgentViewState(agent_index=i) for i in range(1, self.total_agents + 1)
        }

        self._init_pygame()

    def _init_pygame(self) -> None:
        if pygame is None:
            self.ready = False
            return
        try:
            pygame.init()
            flags = pygame.FULLSCREEN if self.fullscreen else 0
            if self.fullscreen:
                self._screen = pygame.display.set_mode((0, 0), flags)
            else:
                self._screen = pygame.display.set_mode(self._windowed_size)
            pygame.display.set_caption("DQNRef Live GA Training Viewer")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("Segoe UI", 20, bold=True)
            self._small = pygame.font.SysFont("Segoe UI", 16)
            self._mono = pygame.font.SysFont("Consolas", 16)
            self.ready = True
            self._last_frame_ts = time.time()
        except Exception:
            self.ready = False
            self.closed = True

    def close(self) -> None:
        if not self.ready:
            self.closed = True
            return
        self.closed = True
        try:
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass

    def _toggle_fullscreen(self) -> None:
        if not self.ready or pygame is None:
            return
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self._screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self._screen = pygame.display.set_mode(self._windowed_size)

    def _status_color(self, status: str) -> Tuple[int, int, int]:
        if status == "done":
            return DONE
        if status == "failed":
            return FAILED
        if status == "active":
            return ACTIVE
        return TEXT_DIM

    def _normalize_board(self, payload: object) -> np.ndarray:
        if isinstance(payload, np.ndarray):
            arr = payload.astype(np.uint8, copy=False)
            if arr.shape == (BOARD_ROWS, BOARD_COLS):
                return arr
            return arr.reshape(BOARD_ROWS, BOARD_COLS)
        arr = np.asarray(payload, dtype=np.uint8)
        if arr.size != BOARD_ROWS * BOARD_COLS:
            return np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.uint8)
        return arr.reshape(BOARD_ROWS, BOARD_COLS)

    def _normalize_piece_ids(self, payload: object) -> np.ndarray:
        if isinstance(payload, np.ndarray):
            arr = payload.astype(np.uint8, copy=False)
            if arr.shape == (BOARD_ROWS, BOARD_COLS):
                return arr
            return arr.reshape(BOARD_ROWS, BOARD_COLS)
        arr = np.asarray(payload, dtype=np.uint8)
        if arr.size != BOARD_ROWS * BOARD_COLS:
            return np.full((BOARD_ROWS, BOARD_COLS), EMPTY_CELL_ID, dtype=np.uint8)
        return arr.reshape(BOARD_ROWS, BOARD_COLS)

    def process_event(self, event: Dict[str, object]) -> None:
        event_type = str(event.get("type", ""))
        agent_index = int(event.get("agent_index", 0) or 0)
        generation = int(event.get("generation", 0) or 0)
        ts = float(event.get("timestamp", time.time()))

        if generation > 0:
            self._global_generation = max(self._global_generation, generation)

        if agent_index <= 0 or agent_index > self.total_agents:
            if event_type == "generation_done":
                self._completed_agents = 0
            self._events_processed += 1
            return

        state = self.agents[agent_index]
        state.generation = generation if generation > 0 else state.generation
        state.last_update_ts = ts
        state.games_per_agent = int(event.get("games_per_agent", state.games_per_agent or self.games_per_agent))

        if event_type == "agent_started":
            state.status = "active"
            state.episode_index = 0
            state.step_in_episode = 0
            state.fitness_final = None
            state.recent_episodes.clear()
            state.lines_total = 0
            state.episode_return_running = 0.0
            state.fitness_provisional = 0.0
            state.board_piece_ids.fill(EMPTY_CELL_ID)
        elif event_type == "step_snapshot":
            state.status = str(event.get("status", "active"))
            state.episode_index = int(event.get("episode_index", state.episode_index))
            state.step_in_episode = int(event.get("step_in_episode", state.step_in_episode))
            state.lines_total = int(event.get("lines_total", state.lines_total))
            state.episode_return_running = float(
                event.get("episode_return_running", state.episode_return_running)
            )
            state.epsilon = float(event.get("epsilon", state.epsilon))
            state.loss_last = float(event.get("loss_last", state.loss_last))
            state.fitness_provisional = float(
                event.get("fitness_provisional", state.fitness_provisional)
            )
            if "board" in event:
                state.board = self._normalize_board(event.get("board"))
            if "board_piece_ids" in event:
                state.board_piece_ids = self._normalize_piece_ids(event.get("board_piece_ids"))
        elif event_type == "episode_done":
            state.status = str(event.get("status", state.status or "active"))
            state.episode_index = int(event.get("episode_index", state.episode_index))
            state.step_in_episode = int(event.get("survival_length", state.step_in_episode))
            state.lines_total = int(event.get("lines_total", state.lines_total))
            state.episode_return_running = float(
                event.get("episode_return", state.episode_return_running)
            )
            state.epsilon = float(event.get("epsilon", state.epsilon))
            state.loss_last = float(event.get("loss", state.loss_last))
            state.fitness_provisional = float(
                event.get("fitness_provisional", state.fitness_provisional)
            )
            if "board" in event:
                state.board = self._normalize_board(event.get("board"))
            if "board_piece_ids" in event:
                state.board_piece_ids = self._normalize_piece_ids(event.get("board_piece_ids"))
            recent = {
                "episode_index": float(state.episode_index),
                "episode_return": float(event.get("episode_return", 0.0)),
                "lines": float(event.get("lines_cleared", 0.0)),
                "steps": float(event.get("survival_length", 0.0)),
            }
            state.recent_episodes.append(recent)
            if len(state.recent_episodes) > 8:
                state.recent_episodes = state.recent_episodes[-8:]
        elif event_type == "agent_done":
            state.status = "done"
            state.fitness_final = float(event.get("fitness", state.fitness_provisional))
            state.fitness_provisional = float(state.fitness_final)
            state.episode_index = int(event.get("episodes_completed", state.episode_index))
            self._completed_agents += 1

        self._events_processed += 1

    def _draw_board(
        self,
        x: int,
        y: int,
        tile: int,
        board: np.ndarray,
        board_piece_ids: Optional[np.ndarray],
        border_color: Tuple[int, int, int],
    ) -> None:
        if not self.ready:
            return
        screen = self._screen
        assert screen is not None
        w = BOARD_COLS * tile
        h = BOARD_ROWS * tile
        pygame.draw.rect(screen, (6, 9, 14), (x - 2, y - 2, w + 4, h + 4), border_radius=4)
        pygame.draw.rect(screen, border_color, (x - 2, y - 2, w + 4, h + 4), 2, border_radius=4)
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                v = int(board[r, c])
                color = FILLED if v else EMPTY
                if board_piece_ids is not None:
                    pid = int(board_piece_ids[r, c])
                    if pid in PIECE_COLORS:
                        color = PIECE_COLORS[pid]
                    elif pid == EMPTY_CELL_ID and v == 0:
                        color = EMPTY
                rx = x + c * tile
                ry = y + r * tile
                pygame.draw.rect(screen, color, (rx, ry, tile - 1, tile - 1))
                if tile >= 8:
                    pygame.draw.rect(screen, GRID, (rx, ry, tile - 1, tile - 1), 1)

    def _layout_grid(self, left_rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        _, _, width, height = left_rect
        pad = self.grid_padding
        n = self.total_agents
        best = (1, n, 1, 1)  # cols, rows, card_w, card_h
        best_tile = -1

        for cols in range(1, n + 1):
            rows = int(math.ceil(n / float(cols)))
            card_w = int((width - (cols + 1) * pad) / cols)
            card_h = int((height - (rows + 1) * pad) / rows)
            if card_w <= 28 or card_h <= 28:
                continue
            board_tile = min(
                int((card_w - 10) / BOARD_COLS),
                int((card_h - 44) / BOARD_ROWS),
            )
            if board_tile < self.min_tile_px:
                continue
            if board_tile > best_tile:
                best_tile = board_tile
                best = (cols, rows, card_w, card_h)
        return best

    def _handle_input(self) -> None:
        if not self.ready or pygame is None:
            return
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.close()
                return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_q:
                    self.close()
                    return
                if ev.key == pygame.K_F11:
                    self._toggle_fullscreen()
                elif ev.key == pygame.K_TAB:
                    self.selected_agent += 1
                    if self.selected_agent > self.total_agents:
                        self.selected_agent = 1
                elif ev.key == pygame.K_r:
                    self.selected_agent = 1
                elif ev.key == pygame.K_LEFT:
                    self.selected_agent = max(1, self.selected_agent - 1)
                elif ev.key == pygame.K_RIGHT:
                    self.selected_agent = min(self.total_agents, self.selected_agent + 1)
                elif ev.key == pygame.K_UP:
                    self.selected_agent = max(1, self.selected_agent - 5)
                elif ev.key == pygame.K_DOWN:
                    self.selected_agent = min(self.total_agents, self.selected_agent + 5)
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                for agent_idx, rect in self._agent_rects.items():
                    x, y, w, h = rect
                    if x <= mx <= x + w and y <= my <= y + h:
                        self.selected_agent = int(agent_idx)
                        break

    def tick(self) -> None:
        if not self.ready or self.closed:
            return
        now = time.time()
        min_dt = 1.0 / float(max(5, self.fps))
        if now - self._last_frame_ts < min_dt:
            return
        self._last_frame_ts = now
        self._handle_input()
        if self.closed:
            return

        screen = self._screen
        font = self._font
        small = self._small
        mono = self._mono
        assert screen is not None and font is not None and small is not None and mono is not None

        w, h = screen.get_size()
        top_h = 58
        left_w = int(w * 0.68)
        right_w = w - left_w
        pad = self.grid_padding

        screen.fill(BG)
        pygame.draw.rect(screen, PANEL_ALT, (0, 0, w, top_h))
        pygame.draw.line(screen, GRID, (0, top_h), (w, top_h), 2)

        elapsed = now - self._start_ts
        active_count = sum(1 for s in self.agents.values() if s.status == "active")
        done_count = sum(1 for s in self.agents.values() if s.status == "done")
        top_line = (
            f"DQNRef GA Live Viewer  |  gen {self._global_generation}/{self.total_generations}  |  "
            f"agents done {done_count}/{self.total_agents}  |  active {active_count}  |  "
            f"elapsed {elapsed:,.1f}s  |  events {self._events_processed}"
        )
        screen.blit(font.render(top_line, True, TEXT), (12, 16))
        if self.run_dir:
            screen.blit(small.render(f"run: {self.run_dir}", True, TEXT_DIM), (12, 38))

        left_rect = (0, top_h, left_w, h - top_h)
        right_rect = (left_w, top_h, right_w, h - top_h)
        pygame.draw.rect(screen, PANEL, left_rect)
        pygame.draw.rect(screen, PANEL_ALT, right_rect)
        pygame.draw.line(screen, GRID, (left_w, top_h), (left_w, h), 2)

        cols, rows, card_w, card_h = self._layout_grid(left_rect)
        self._agent_rects.clear()
        for agent_idx in range(1, self.total_agents + 1):
            row = (agent_idx - 1) // cols
            col = (agent_idx - 1) % cols
            x = pad + col * (card_w + pad)
            y = top_h + pad + row * (card_h + pad)
            rect = (x, y, card_w, card_h)
            self._agent_rects[agent_idx] = rect

            st = self.agents[agent_idx]
            card_bg = PANEL_ALT if agent_idx != self.selected_agent else (26, 40, 62)
            border = self._status_color(st.status)
            pygame.draw.rect(screen, card_bg, rect, border_radius=6)
            pygame.draw.rect(screen, border, rect, 2, border_radius=6)

            tile = max(
                self.min_tile_px,
                min(int((card_w - 10) / BOARD_COLS), int((card_h - 44) / BOARD_ROWS)),
            )
            bx = x + (card_w - BOARD_COLS * tile) // 2
            by = y + 6
            self._draw_board(bx, by, tile, st.board, st.board_piece_ids, border)

            label = f"A{agent_idx:02d}  {st.status}"
            ep = f"ep {st.episode_index}/{st.games_per_agent or self.games_per_agent}"
            fit = st.fitness_final if st.fitness_final is not None else st.fitness_provisional
            line2 = f"{ep}  L={st.lines_total}  F={fit:.2f}"
            ly = y + card_h - 32
            screen.blit(small.render(label, True, TEXT), (x + 6, ly - 16))
            screen.blit(small.render(line2, True, TEXT_DIM), (x + 6, ly))

        sel = self.agents[self.selected_agent]
        rx, ry, rw, rh = right_rect
        screen.blit(
            font.render(f"Selected Agent A{self.selected_agent:02d}", True, TEXT),
            (rx + 14, ry + 10),
        )
        screen.blit(
            small.render(
                "Controls: Click/Arrows/Tab select | F11 fullscreen | R reset focus | Q close viewer",
                True,
                TEXT_DIM,
            ),
            (rx + 14, ry + 36),
        )

        board_area_h = int(rh * 0.54)
        board_pad = 14
        max_tile_w = int((rw - 2 * board_pad - 8) / BOARD_COLS)
        max_tile_h = int((board_area_h - 72) / BOARD_ROWS)
        tile_big = max(self.min_tile_px + 2, min(max_tile_w, max_tile_h))
        big_bw = BOARD_COLS * tile_big
        bx = rx + (rw - big_bw) // 2
        by = ry + 66
        self._draw_board(bx, by, tile_big, sel.board, sel.board_piece_ids, self._status_color(sel.status))

        metrics_y = by + BOARD_ROWS * tile_big + 14
        metrics = [
            f"status: {sel.status}",
            f"generation: {sel.generation}",
            f"episode: {sel.episode_index}/{sel.games_per_agent or self.games_per_agent}",
            f"step: {sel.step_in_episode}",
            f"lines total: {sel.lines_total}",
            f"running return: {sel.episode_return_running:.2f}",
            f"epsilon: {sel.epsilon:.5f}",
            f"last loss: {sel.loss_last:.4f}",
            (
                f"fitness: {sel.fitness_final:.4f}"
                if sel.fitness_final is not None
                else f"fitness provisional: {sel.fitness_provisional:.4f}"
            ),
        ]
        for i, line in enumerate(metrics):
            screen.blit(mono.render(line, True, TEXT), (rx + 16, metrics_y + i * 20))

        recent_y = metrics_y + len(metrics) * 20 + 10
        screen.blit(small.render("Recent Episodes", True, TEXT), (rx + 16, recent_y))
        if not sel.recent_episodes:
            screen.blit(small.render("No completed episodes yet.", True, TEXT_DIM), (rx + 16, recent_y + 22))
        else:
            recent = sel.recent_episodes[-6:][::-1]
            for i, ep in enumerate(recent):
                txt = (
                    f"ep {int(ep['episode_index']):>3}  "
                    f"ret={ep['episode_return']:>9.2f}  "
                    f"lines={int(ep['lines']):>3}  "
                    f"steps={int(ep['steps']):>4}"
                )
                screen.blit(mono.render(txt, True, TEXT_DIM), (rx + 16, recent_y + 24 + i * 18))

        pygame.display.flip()
        self._clock.tick(self.fps)
