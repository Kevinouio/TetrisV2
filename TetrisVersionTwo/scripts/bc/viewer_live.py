from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
ACTIVE = (86, 214, 146)
DONE = (90, 160, 255)
FAILED = (255, 115, 115)
IDLE = (116, 132, 154)
EMPTY = (23, 30, 43)
FILLED = (130, 170, 255)
GRID = (36, 48, 66)
EMPTY_CELL_ID = 255

PIECE_COLORS = {
    0: (0, 230, 230),  # I
    1: (240, 220, 0),  # O
    2: (170, 70, 230),  # T
    3: (255, 150, 60),  # L
    4: (80, 110, 255),  # J
    5: (80, 220, 100),  # S
    6: (240, 90, 90),  # Z
}


def _event_type_name(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("type", "")).strip()


def queue_put_best_effort(event_queue: Any, event: Dict[str, object]) -> bool:
    if event_queue is None:
        return False
    try:
        event_queue.put_nowait(event)
        return True
    except Exception:
        pass

    event_type = _event_type_name(event)
    if event_type == "step_snapshot":
        return False

    # If queue is full, preferentially evict stale step snapshots so terminal
    # and run-level events can still get through.
    drained: List[object] = []
    removed_step = False
    for _ in range(64):
        try:
            existing = event_queue.get_nowait()
        except Exception:
            break
        existing_type = _event_type_name(existing)
        if (not removed_step) and existing_type == "step_snapshot":
            removed_step = True
            continue
        drained.append(existing)

    if not removed_step:
        for existing in drained:
            try:
                event_queue.put_nowait(existing)
            except Exception:
                break
        return False

    pushed = False
    try:
        event_queue.put_nowait(event)
        pushed = True
    except Exception:
        pushed = False

    for existing in drained:
        try:
            event_queue.put_nowait(existing)
        except Exception:
            break
    return pushed


def board_for_event(board: object) -> List[int]:
    arr = np.asarray(board, dtype=np.uint8)
    if arr.size != BOARD_ROWS * BOARD_COLS:
        return [0] * (BOARD_ROWS * BOARD_COLS)
    return arr.reshape(-1).astype(np.uint8, copy=False).tolist()


def piece_ids_for_event(piece_ids: object) -> List[int]:
    arr = np.asarray(piece_ids, dtype=np.uint8)
    if arr.size != BOARD_ROWS * BOARD_COLS:
        return [255] * (BOARD_ROWS * BOARD_COLS)
    return arr.reshape(-1).astype(np.uint8, copy=False).tolist()


@dataclass
class WorkerViewState:
    worker_slot: int
    worker_key: str = ""
    label: str = ""
    worker_pid: int = -1
    status: str = "idle"
    episode_id: int = -1
    episode_steps: int = 0
    episodes_completed: int = 0
    lines_total: int = 0
    transitions_total: int = 0
    transitions_last_episode: int = 0
    expert_steps: int = 0
    learner_steps: int = 0
    invalid_learner_raw_argmax: int = 0
    unseen_learner_fallback: int = 0
    board: np.ndarray = field(
        default_factory=lambda: np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.uint8)
    )
    board_piece_ids: np.ndarray = field(
        default_factory=lambda: np.full((BOARD_ROWS, BOARD_COLS), EMPTY_CELL_ID, dtype=np.uint8)
    )
    recent_episodes: List[Dict[str, float]] = field(default_factory=list)
    last_update_ts: float = 0.0


class LiveCollectionViewer:
    def __init__(
        self,
        *,
        mode: str,
        total_workers: int,
        total_episodes: int,
        fullscreen: bool = True,
        fps: int = 20,
        grid_padding: int = 8,
        min_tile_px: int = 6,
        initial_selected_worker: int = 1,
        run_dir: str = "",
        round_id: int = 0,
        beta: float = 0.0,
    ):
        self.mode = str(mode)
        self.total_workers = int(max(1, total_workers))
        self.total_episodes = int(max(1, total_episodes))
        self.fullscreen = bool(fullscreen)
        self.fps = int(max(5, fps))
        self.grid_padding = int(max(2, grid_padding))
        self.min_tile_px = int(max(2, min_tile_px))
        self.selected_worker_slot = int(max(1, initial_selected_worker))
        self.run_dir = str(run_dir)

        self.round_id = int(max(0, round_id))
        self.beta = float(beta)

        self.ready = False
        self.closed = False
        self._screen = None
        self._clock = None
        self._font = None
        self._small = None
        self._mono = None
        self._last_frame_ts = 0.0
        self._start_ts = time.time()
        self._windowed_size = (1600, 900)
        self._worker_rects: Dict[int, Tuple[int, int, int, int]] = {}
        self._events_processed = 0
        self._page_index = 0
        self._stale_timeout_sec = 8.0

        self.run_status = "running"
        self.episodes_completed = 0
        self.episodes_with_data = 0
        self.transitions_collected = 0
        self.episodes_per_sec = 0.0
        self.eta_seconds: Optional[float] = None
        self.skipped_no_legal = 0
        self.skipped_invalid_expert = 0
        self.skipped_missing_tuple = 0
        self.failed_steps = 0
        self.invalid_learner_raw_argmax = 0
        self.unseen_learner_fallback = 0
        self.expert_steps = 0
        self.learner_steps = 0

        self.workers: Dict[int, WorkerViewState] = {}
        self._key_to_slot: Dict[str, int] = {}
        self._next_dynamic_slot = 1

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
            pygame.display.set_caption("BC/DAgger Live Collection Viewer")
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
        if status in ("active", "running", "stopping"):
            return ACTIVE
        return IDLE

    def _normalize_board(self, payload: object) -> np.ndarray:
        arr = np.asarray(payload, dtype=np.uint8)
        if arr.size != BOARD_ROWS * BOARD_COLS:
            return np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.uint8)
        return arr.reshape(BOARD_ROWS, BOARD_COLS)

    def _normalize_piece_ids(self, payload: object) -> np.ndarray:
        arr = np.asarray(payload, dtype=np.uint8)
        if arr.size != BOARD_ROWS * BOARD_COLS:
            return np.full((BOARD_ROWS, BOARD_COLS), EMPTY_CELL_ID, dtype=np.uint8)
        return arr.reshape(BOARD_ROWS, BOARD_COLS)

    def _parse_worker_pid(self, key: str) -> int:
        if not key.startswith("pid:"):
            return -1
        try:
            return int(key.split(":", 1)[1])
        except Exception:
            return -1

    def _allocate_slot(self, preferred_slot: int) -> int:
        preferred = int(preferred_slot)
        if preferred > 0 and preferred not in self.workers:
            return preferred
        slot = int(max(1, self._next_dynamic_slot))
        while slot in self.workers:
            slot += 1
        self._next_dynamic_slot = slot + 1
        return slot

    def _resolve_worker_slot(self, event: Dict[str, object]) -> int:
        slot_raw = event.get("worker_slot", 0)
        try:
            slot_hint = int(slot_raw)
        except Exception:
            slot_hint = 0

        key_raw = event.get("worker_key")
        key = str(key_raw).strip() if key_raw is not None else ""
        pid = self._parse_worker_pid(key) if key else -1

        if key and key in self._key_to_slot:
            slot = self._key_to_slot[key]
        else:
            slot = self._allocate_slot(slot_hint)
            if key:
                self._key_to_slot[key] = slot

        if slot not in self.workers:
            label = f"W{slot:02d}"
            if pid > 0:
                label = f"PID {pid}"
            self.workers[slot] = WorkerViewState(
                worker_slot=slot,
                worker_key=key,
                worker_pid=pid,
                label=label,
                status="active",
                last_update_ts=time.time(),
            )

        state = self.workers[slot]
        if key:
            state.worker_key = key
        if pid > 0:
            state.worker_pid = pid
            if not state.label or state.label.startswith("W"):
                state.label = f"PID {pid}"

        label_raw = event.get("worker_label")
        if label_raw is not None:
            label = str(label_raw).strip()
            if label:
                state.label = label
        elif state.worker_pid > 0:
            state.label = f"PID {state.worker_pid}"
        return slot

    def _set_selected_slot(self, slot: int, *, snap_page: bool) -> None:
        if slot not in self.workers:
            return
        self.selected_worker_slot = int(slot)
        if not snap_page:
            return
        slots = sorted(self.workers.keys())
        if not slots:
            self._page_index = 0
            return
        capacity = getattr(self, "_last_page_capacity", 1)
        page_capacity = max(1, int(capacity))
        idx = slots.index(int(slot))
        self._page_index = int(idx // page_capacity)

    def _navigate_selection(self, delta: int) -> None:
        slots = sorted(self.workers.keys())
        if not slots:
            return
        if self.selected_worker_slot not in slots:
            self._set_selected_slot(slots[0], snap_page=True)
            return
        idx = slots.index(self.selected_worker_slot)
        new_idx = max(0, min(len(slots) - 1, idx + int(delta)))
        self._set_selected_slot(slots[new_idx], snap_page=True)

    def _switch_page(self, delta: int) -> None:
        slots = sorted(self.workers.keys())
        if not slots:
            return
        page_capacity = max(1, int(getattr(self, "_last_page_capacity", 1)))
        page_count = int(max(1, math.ceil(len(slots) / float(page_capacity))))
        if page_count <= 1:
            self._page_index = 0
            return
        self._page_index = (int(self._page_index) + int(delta)) % page_count
        start = self._page_index * page_capacity
        end = min(len(slots), start + page_capacity)
        visible = slots[start:end]
        if visible:
            self._set_selected_slot(visible[0], snap_page=False)

    def _visible_slots(self, slots: List[int], page_capacity: int) -> Tuple[List[int], int]:
        total = len(slots)
        if total <= 0:
            return [], 1
        cap = max(1, int(page_capacity))
        page_count = int(max(1, math.ceil(total / float(cap))))
        self._page_index = max(0, min(int(self._page_index), page_count - 1))
        start = self._page_index * cap
        end = min(total, start + cap)
        return slots[start:end], page_count

    def _prune_stale_workers(self, now: float) -> None:
        stale_slots: List[int] = []
        for slot, state in self.workers.items():
            if state.last_update_ts <= 0:
                continue
            age = float(now - state.last_update_ts)
            if age <= float(self._stale_timeout_sec):
                continue
            if state.status in ("done", "failed", "idle") or age > float(self._stale_timeout_sec * 4.0):
                stale_slots.append(int(slot))
        if not stale_slots:
            return
        stale_set = set(stale_slots)
        for slot in stale_slots:
            self.workers.pop(slot, None)
        if self.selected_worker_slot in stale_set:
            if self.workers:
                self.selected_worker_slot = sorted(self.workers.keys())[0]
            else:
                self.selected_worker_slot = 1
                self._page_index = 0
        stale_keys = [key for key, slot in self._key_to_slot.items() if slot in stale_set]
        for key in stale_keys:
            self._key_to_slot.pop(key, None)

    def process_event(self, event: Dict[str, object]) -> None:
        event_type = str(event.get("type", "")).strip()
        ts = float(event.get("timestamp", time.time()))

        if event_type in ("run_started", "run_progress", "run_done"):
            self.run_status = str(event.get("status", self.run_status))
            self.episodes_completed = int(event.get("episodes_completed", self.episodes_completed))
            self.episodes_with_data = int(event.get("episodes_with_data", self.episodes_with_data))
            self.transitions_collected = int(event.get("transitions_collected", self.transitions_collected))
            self.episodes_per_sec = float(event.get("episodes_per_sec", self.episodes_per_sec))
            eta = event.get("eta_seconds")
            self.eta_seconds = None if eta is None else float(eta)
            self.skipped_no_legal = int(event.get("skipped_no_legal", self.skipped_no_legal))
            self.skipped_invalid_expert = int(event.get("skipped_invalid_expert", self.skipped_invalid_expert))
            self.skipped_missing_tuple = int(event.get("skipped_missing_tuple", self.skipped_missing_tuple))
            self.failed_steps = int(event.get("failed_steps", self.failed_steps))
            self.invalid_learner_raw_argmax = int(
                event.get("invalid_learner_raw_argmax", self.invalid_learner_raw_argmax)
            )
            self.unseen_learner_fallback = int(
                event.get("unseen_learner_fallback", self.unseen_learner_fallback)
            )
            self.expert_steps = int(event.get("expert_steps", self.expert_steps))
            self.learner_steps = int(event.get("learner_steps", self.learner_steps))
            self.round_id = int(event.get("round_id", self.round_id))
            self.beta = float(event.get("beta", self.beta))
            self.total_episodes = int(event.get("episodes_total", self.total_episodes))
            self.total_workers = int(max(1, int(event.get("collect_workers", self.total_workers))))
            run_dir_val = event.get("run_dir")
            if run_dir_val is not None:
                self.run_dir = str(run_dir_val)
            mode_val = event.get("mode")
            if mode_val is not None:
                self.mode = str(mode_val)

        if event_type in ("worker_started", "step_snapshot", "episode_done", "worker_done"):
            slot = self._resolve_worker_slot(event)
            state = self.workers[slot]
            state.last_update_ts = ts

            if event_type == "worker_started":
                state.status = str(event.get("status", "active"))
                state.episode_steps = 0
                state.transitions_last_episode = 0
            elif event_type == "step_snapshot":
                state.status = str(event.get("status", "active"))
                state.episode_id = int(event.get("episode_id", state.episode_id))
                state.episode_steps = int(event.get("step_in_episode", state.episode_steps))
                state.lines_total = int(event.get("lines_total", state.lines_total))
                state.transitions_total = int(event.get("transitions_collected", state.transitions_total))
                state.expert_steps = int(event.get("expert_steps", state.expert_steps))
                state.learner_steps = int(event.get("learner_steps", state.learner_steps))
                state.invalid_learner_raw_argmax = int(
                    event.get("invalid_learner_raw_argmax", state.invalid_learner_raw_argmax)
                )
                state.unseen_learner_fallback = int(
                    event.get("unseen_learner_fallback", state.unseen_learner_fallback)
                )
                if "board" in event:
                    state.board = self._normalize_board(event.get("board"))
                if "board_piece_ids" in event:
                    state.board_piece_ids = self._normalize_piece_ids(event.get("board_piece_ids"))
            elif event_type == "episode_done":
                state.status = str(event.get("status", state.status or "active"))
                state.episode_id = int(event.get("episode_id", state.episode_id))
                state.episode_steps = int(event.get("survival_length", state.episode_steps))
                state.lines_total = int(event.get("lines_total", state.lines_total))
                state.episodes_completed = int(event.get("episodes_completed", state.episodes_completed))
                state.transitions_total = int(event.get("transitions_collected", state.transitions_total))
                state.transitions_last_episode = int(
                    event.get("episode_transitions", state.transitions_last_episode)
                )
                state.expert_steps = int(event.get("expert_steps", state.expert_steps))
                state.learner_steps = int(event.get("learner_steps", state.learner_steps))
                state.invalid_learner_raw_argmax = int(
                    event.get("invalid_learner_raw_argmax", state.invalid_learner_raw_argmax)
                )
                state.unseen_learner_fallback = int(
                    event.get("unseen_learner_fallback", state.unseen_learner_fallback)
                )
                if "board" in event:
                    state.board = self._normalize_board(event.get("board"))
                if "board_piece_ids" in event:
                    state.board_piece_ids = self._normalize_piece_ids(event.get("board_piece_ids"))
                recent = {
                    "episode_id": float(state.episode_id),
                    "steps": float(state.episode_steps),
                    "lines": float(state.lines_total),
                    "transitions": float(state.transitions_last_episode),
                }
                state.recent_episodes.append(recent)
                if len(state.recent_episodes) > 8:
                    state.recent_episodes = state.recent_episodes[-8:]
            elif event_type == "worker_done":
                state.status = str(event.get("status", "done"))

        self._events_processed += 1
        if self.selected_worker_slot not in self.workers and self.workers:
            self.selected_worker_slot = sorted(self.workers.keys())[0]

    def _draw_board(
        self,
        x: int,
        y: int,
        tile: int,
        board: np.ndarray,
        board_piece_ids: np.ndarray,
        border_color: Tuple[int, int, int],
    ) -> None:
        if not self.ready:
            return
        screen = self._screen
        assert screen is not None
        width = BOARD_COLS * tile
        height = BOARD_ROWS * tile
        pygame.draw.rect(screen, (6, 9, 14), (x - 2, y - 2, width + 4, height + 4), border_radius=4)
        pygame.draw.rect(screen, border_color, (x - 2, y - 2, width + 4, height + 4), 2, border_radius=4)
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                occupied = int(board[r, c])
                color = FILLED if occupied else EMPTY
                pid = int(board_piece_ids[r, c])
                if pid in PIECE_COLORS:
                    color = PIECE_COLORS[pid]
                elif pid == EMPTY_CELL_ID and occupied == 0:
                    color = EMPTY
                rx = x + c * tile
                ry = y + r * tile
                pygame.draw.rect(screen, color, (rx, ry, tile - 1, tile - 1))
                if tile >= 8:
                    pygame.draw.rect(screen, GRID, (rx, ry, tile - 1, tile - 1), 1)

    def _layout_grid(self, rect: Tuple[int, int, int, int], count: int) -> Tuple[int, int, int, int]:
        _, _, width, height = rect
        pad = self.grid_padding
        n = int(max(1, count))
        best = (1, n, 1, 1)
        best_tile = -1
        for cols in range(1, n + 1):
            rows = int(math.ceil(n / float(cols)))
            card_w = int((width - (cols + 1) * pad) / cols)
            card_h = int((height - (rows + 1) * pad) / rows)
            if card_w <= 40 or card_h <= 40:
                continue
            board_tile = min(
                int((card_w - 12) / BOARD_COLS),
                int((card_h - 56) / BOARD_ROWS),
            )
            if board_tile < self.min_tile_px:
                continue
            if board_tile > best_tile:
                best_tile = board_tile
                best = (cols, rows, card_w, card_h)
        return best

    def _grid_capacity(self, rect: Tuple[int, int, int, int]) -> int:
        _, _, width, height = rect
        pad = self.grid_padding
        card_min_w = BOARD_COLS * self.min_tile_px + 24
        card_min_h = BOARD_ROWS * self.min_tile_px + 68
        cols = max(1, int((width - pad) // max(1, card_min_w + pad)))
        rows = max(1, int((height - pad) // max(1, card_min_h + pad)))
        return int(max(1, cols * rows))

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
                    self._navigate_selection(+1)
                elif ev.key == pygame.K_r:
                    slots = sorted(self.workers.keys())
                    if slots:
                        self._set_selected_slot(slots[0], snap_page=True)
                elif ev.key == pygame.K_LEFT:
                    self._navigate_selection(-1)
                elif ev.key == pygame.K_RIGHT:
                    self._navigate_selection(+1)
                elif ev.key == pygame.K_UP:
                    self._navigate_selection(-5)
                elif ev.key == pygame.K_DOWN:
                    self._navigate_selection(+5)
                elif ev.key in (pygame.K_PAGEUP, pygame.K_LEFTBRACKET):
                    self._switch_page(-1)
                elif ev.key in (pygame.K_PAGEDOWN, pygame.K_RIGHTBRACKET):
                    self._switch_page(+1)
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                for slot, rect in self._worker_rects.items():
                    x, y, w, h = rect
                    if x <= mx <= x + w and y <= my <= y + h:
                        self._set_selected_slot(int(slot), snap_page=True)
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

        self._prune_stale_workers(now)

        screen = self._screen
        font = self._font
        small = self._small
        mono = self._mono
        assert screen is not None and font is not None and small is not None and mono is not None

        slots = sorted(self.workers.keys())
        if slots and self.selected_worker_slot not in self.workers:
            self.selected_worker_slot = slots[0]

        width, height = screen.get_size()
        top_h = 74
        bottom_h = 188
        grid_h = max(120, height - top_h - bottom_h)
        grid_rect = (0, top_h, width, grid_h)
        detail_rect = (0, top_h + grid_h, width, height - (top_h + grid_h))
        self._last_page_capacity = max(1, self._grid_capacity(grid_rect))

        visible_slots, page_count = self._visible_slots(slots, self._last_page_capacity)
        if slots and self.selected_worker_slot in slots:
            selected_idx = slots.index(self.selected_worker_slot)
            selected_page = selected_idx // max(1, int(self._last_page_capacity))
            if selected_page != self._page_index:
                self._page_index = selected_page
                visible_slots, page_count = self._visible_slots(slots, self._last_page_capacity)

        screen.fill(BG)
        pygame.draw.rect(screen, PANEL_ALT, (0, 0, width, top_h))
        pygame.draw.line(screen, GRID, (0, top_h), (width, top_h), 2)

        elapsed = now - self._start_ts
        active_count = sum(1 for s in self.workers.values() if s.status in ("active", "running", "stopping"))
        done_count = sum(1 for s in self.workers.values() if s.status == "done")
        eta_text = "unknown" if self.eta_seconds is None else f"{float(self.eta_seconds):.1f}s"
        top_line = (
            f"{self.mode.upper()} Live Worker Wall  |  status={self.run_status}  |  "
            f"episodes {self.episodes_completed}/{self.total_episodes}  |  with_data {self.episodes_with_data}  |  "
            f"transitions {self.transitions_collected}  |  eps/s {self.episodes_per_sec:.2f}  |  eta {eta_text}"
        )
        screen.blit(font.render(top_line, True, TEXT), (12, 14))
        line2 = (
            f"workers active {active_count}  done {done_count}  visible {len(visible_slots)}  "
            f"all_seen {len(self.workers)}  page {self._page_index + 1}/{page_count}  "
            f"elapsed {elapsed:,.1f}s  events {self._events_processed}"
        )
        screen.blit(small.render(line2, True, TEXT_DIM), (12, 42))

        pygame.draw.rect(screen, PANEL, grid_rect)
        pygame.draw.line(
            screen,
            GRID,
            (0, detail_rect[1]),
            (width, detail_rect[1]),
            2,
        )
        pygame.draw.rect(screen, PANEL_ALT, detail_rect)

        self._worker_rects.clear()
        if visible_slots:
            cols, rows, card_w, card_h = self._layout_grid(grid_rect, len(visible_slots))
            gx, gy, _, _ = grid_rect
            pad = self.grid_padding
            for order, slot in enumerate(visible_slots):
                row = order // cols
                col = order % cols
                x = gx + pad + col * (card_w + pad)
                y = gy + pad + row * (card_h + pad)
                rect = (x, y, card_w, card_h)
                self._worker_rects[slot] = rect

                state = self.workers[slot]
                selected = slot == self.selected_worker_slot
                card_bg = PANEL_ALT if not selected else (26, 40, 62)
                border = self._status_color(state.status)
                pygame.draw.rect(screen, card_bg, rect, border_radius=6)
                pygame.draw.rect(screen, border, rect, 2, border_radius=6)

                tile = max(
                    self.min_tile_px,
                    min(int((card_w - 12) / BOARD_COLS), int((card_h - 58) / BOARD_ROWS)),
                )
                bx = x + (card_w - BOARD_COLS * tile) // 2
                by = y + 6
                self._draw_board(bx, by, tile, state.board, state.board_piece_ids, border)

                label = state.label or (f"PID {state.worker_pid}" if state.worker_pid > 0 else f"W{slot:02d}")
                age = max(0.0, now - state.last_update_ts) if state.last_update_ts > 0 else 0.0
                line1 = f"{label}  {state.status}"
                line2_card = (
                    f"ep={state.episode_id}  step={state.episode_steps}  "
                    f"tr={state.transitions_total}  age={age:.1f}s"
                )
                ly = y + card_h - 34
                screen.blit(small.render(line1, True, TEXT), (x + 6, ly - 16))
                screen.blit(small.render(line2_card, True, TEXT_DIM), (x + 6, ly))
        else:
            msg = "Waiting for worker telemetry..."
            screen.blit(font.render(msg, True, TEXT_DIM), (24, top_h + 24))

        dx, dy, dw, dh = detail_rect
        screen.blit(
            small.render(
                "Controls: Click/Arrows/Tab select | PageUp/PageDown or [/] page | "
                "F11 fullscreen | R reset | Q close",
                True,
                TEXT_DIM,
            ),
            (dx + 12, dy + 10),
        )

        selected_state: Optional[WorkerViewState] = None
        if self.selected_worker_slot in self.workers:
            selected_state = self.workers[self.selected_worker_slot]
        elif slots:
            selected_state = self.workers[slots[0]]

        left_x = dx + 12
        metrics_y = dy + 36

        if selected_state is not None:
            label = selected_state.label or (
                f"PID {selected_state.worker_pid}" if selected_state.worker_pid > 0 else f"W{selected_state.worker_slot:02d}"
            )
            age = max(0.0, now - selected_state.last_update_ts) if selected_state.last_update_ts > 0 else 0.0
            screen.blit(font.render(f"Selected: {label}", True, TEXT), (left_x, metrics_y))
            metrics_y += 28
            selected_metrics = [
                f"status={selected_state.status}  episode={selected_state.episode_id}  step={selected_state.episode_steps}",
                f"lines={selected_state.lines_total}  worker_transitions={selected_state.transitions_total}  "
                f"episodes_done={selected_state.episodes_completed}  age={age:.1f}s",
            ]
            for line in selected_metrics:
                screen.blit(mono.render(line, True, TEXT), (left_x, metrics_y))
                metrics_y += 20
        else:
            screen.blit(font.render("Selected: none", True, TEXT), (left_x, metrics_y))
            metrics_y += 28

        expert_total = int(self.expert_steps)
        learner_total = int(self.learner_steps)
        expert_rate = 0.0
        if (expert_total + learner_total) > 0:
            expert_rate = float(expert_total) / float(expert_total + learner_total)
        run_metrics = [
            f"round_id={self.round_id}  beta={self.beta:.4f}  mode={self.mode}",
            f"expert/learner={expert_total}/{learner_total}  expert_rate={expert_rate:.3f}",
            f"invalid_learner_argmax={self.invalid_learner_raw_argmax}  "
            f"unseen_fallback={self.unseen_learner_fallback}  failed_steps={self.failed_steps}",
            f"skips no_legal/invalid/missing={self.skipped_no_legal}/"
            f"{self.skipped_invalid_expert}/{self.skipped_missing_tuple}",
        ]
        for line in run_metrics:
            screen.blit(mono.render(line, True, TEXT_DIM), (left_x, metrics_y))
            metrics_y += 20

        if self.run_dir:
            screen.blit(small.render(f"run: {self.run_dir}", True, TEXT_DIM), (left_x, dy + dh - 24))

        pygame.display.flip()
        self._clock.tick(self.fps)
