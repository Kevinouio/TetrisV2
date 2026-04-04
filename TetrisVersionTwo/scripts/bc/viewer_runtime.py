from __future__ import annotations

import multiprocessing as mp
import queue as queue_mod
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from .viewer_telemetry import queue_put_best_effort


class LiveViewerRuntime:
    """Shared runtime wrapper for BC/DAgger live pygame viewer lifecycle."""

    def __init__(
        self,
        *,
        log_prefix: str,
        enabled: bool,
        mode: str,
        total_workers: int,
        total_episodes: int,
        fullscreen: bool,
        fps: int,
        grid_padding: int,
        min_tile_px: int,
        initial_selected_worker: int,
        run_dir: str,
        viewer_max_queue: int,
        reopen_file: Path,
        round_id: int = 0,
        beta: float = 0.0,
    ) -> None:
        self.log_prefix = str(log_prefix).strip() or "viewer"
        self.mode = str(mode).strip() or "collect_data"
        self.total_workers = int(max(1, total_workers))
        self.total_episodes = int(max(1, total_episodes))
        self.fullscreen = bool(fullscreen)
        self.fps = int(max(5, fps))
        self.grid_padding = int(max(2, grid_padding))
        self.min_tile_px = int(max(2, min_tile_px))
        self.initial_selected_worker = int(max(1, initial_selected_worker))
        self.run_dir = str(run_dir)
        self.reopen_file = Path(reopen_file)
        self.round_id = int(max(0, round_id))
        self.beta = float(beta)
        self.enabled = bool(enabled)

        self._viewer_cls: Any = None
        self._viewer: Any = None
        self._queue: Any = None
        self._manager: Any = None
        self._max_queue = int(max(1, viewer_max_queue))
        self._event_buffer: Deque[Dict[str, object]] = deque(maxlen=self._max_queue)

        self._events_processed = 0
        self._frames_rendered = 0
        self._last_seen_frame_counter = 0
        self._restart_count = 0

        self._stall_force_after_sec = 1.5
        self._stall_restart_after_sec = 4.0
        self._last_force_draw_attempt_ts = 0.0
        self._last_restart_attempt_ts = 0.0

        if not self.enabled:
            return

        try:
            from .viewer_live import LiveCollectionViewer

            self._viewer_cls = LiveCollectionViewer
            self._viewer = self._create_viewer_instance()
            if self._viewer is None:
                print(
                    f"[{self.log_prefix}] warning: viewer init failed; starting headless. "
                    f"Create '{self.reopen_file}' to retry open."
                )
            if self.total_workers > 1:
                self._manager = mp.Manager()
                self._queue = self._manager.Queue(max(1, int(self._max_queue)))
            self.pump(force_draw=True)
        except Exception as exc:
            print(f"[{self.log_prefix}] warning: viewer unavailable ({exc}); continuing headless.")
            self.enabled = False
            self._viewer_cls = None
            self._viewer = None
            self._queue = None
            self._manager = None

    @property
    def worker_queue(self) -> Any:
        if not self.enabled:
            return None
        return self._queue

    def _create_viewer_instance(self) -> Any:
        if self._viewer_cls is None:
            return None
        try:
            viewer = self._viewer_cls(
                mode=self.mode,
                total_workers=self.total_workers,
                total_episodes=self.total_episodes,
                fullscreen=self.fullscreen,
                fps=self.fps,
                grid_padding=self.grid_padding,
                min_tile_px=self.min_tile_px,
                initial_selected_worker=self.initial_selected_worker,
                run_dir=self.run_dir,
                round_id=self.round_id,
                beta=self.beta,
            )
            if not bool(getattr(viewer, "ready", False)):
                return None
            self._last_seen_frame_counter = 0
            return viewer
        except Exception:
            return None

    def _sync_frame_counters(self) -> None:
        viewer = self._viewer
        if viewer is None:
            return
        frame_counter = int(getattr(viewer, "rendered_frames", 0))
        if frame_counter > self._last_seen_frame_counter:
            self._frames_rendered += int(frame_counter - self._last_seen_frame_counter)
            self._last_seen_frame_counter = int(frame_counter)

    def _viewer_last_render_ts(self) -> float:
        viewer = self._viewer
        if viewer is None:
            return 0.0
        return float(getattr(viewer, "last_render_timestamp", 0.0))

    def _deliver_event_to_viewer(self, event_payload: Dict[str, object]) -> None:
        viewer = self._viewer
        if viewer is None or bool(getattr(viewer, "closed", False)):
            return
        try:
            viewer.process_event(event_payload)
            self._events_processed += 1
        except Exception as exc:
            print(
                f"[{self.log_prefix}] warning: viewer event processing failed ({exc}); "
                "attempting viewer restart."
            )
            self._restart_viewer(reason="event processing failure")

    def emit(self, event: Dict[str, object]) -> None:
        if not self.enabled:
            return
        payload = dict(event)
        payload.setdefault("mode", self.mode)
        payload.setdefault("timestamp", float(time.time()))
        if self._queue is not None:
            if not queue_put_best_effort(self._queue, payload):
                self._event_buffer.append(payload)
            return
        self._event_buffer.append(payload)
        self._deliver_event_to_viewer(payload)

    def _tick_viewer(self, *, force_draw: bool) -> None:
        viewer = self._viewer
        if viewer is None or bool(getattr(viewer, "closed", False)):
            return
        try:
            viewer.tick(force=bool(force_draw))
        except Exception as exc:
            print(f"[{self.log_prefix}] warning: viewer tick failed ({exc}); attempting restart.")
            self._restart_viewer(reason="tick failure")
            return
        self._sync_frame_counters()

    def _restart_viewer(self, *, reason: str) -> None:
        now = time.time()
        if self._viewer_cls is None:
            return
        if now - self._last_restart_attempt_ts < 0.5:
            return
        if self._restart_count >= 1:
            return
        self._last_restart_attempt_ts = now

        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
        self._viewer = self._create_viewer_instance()
        if self._viewer is None:
            print(f"[{self.log_prefix}] warning: viewer restart failed ({reason}).")
            return
        self._restart_count += 1
        print(f"[{self.log_prefix}] warning: viewer restarted ({reason}).")
        for ev in self._event_buffer:
            self._deliver_event_to_viewer(ev)
        self._tick_viewer(force_draw=True)

    def _watchdog(self) -> None:
        if not self.enabled:
            return
        viewer = self._viewer
        if viewer is None or bool(getattr(viewer, "closed", False)):
            return
        now = time.time()
        last_render = self._viewer_last_render_ts()
        if last_render <= 0.0:
            if now - self._last_force_draw_attempt_ts >= self._stall_force_after_sec:
                self._last_force_draw_attempt_ts = now
                self._tick_viewer(force_draw=True)
            return

        age = float(max(0.0, now - last_render))
        if age >= self._stall_force_after_sec and (
            now - self._last_force_draw_attempt_ts
        ) >= self._stall_force_after_sec:
            self._last_force_draw_attempt_ts = now
            self._tick_viewer(force_draw=True)
            last_render = self._viewer_last_render_ts()
            age = float(max(0.0, now - last_render)) if last_render > 0.0 else age

        if age >= self._stall_restart_after_sec:
            self._restart_viewer(reason=f"stalled frames ({age:.1f}s)")

    def pump(self, *, force_draw: bool = False) -> None:
        if not self.enabled:
            return

        if self._queue is not None:
            for _ in range(2048):
                try:
                    event_payload = self._queue.get_nowait()
                except queue_mod.Empty:
                    break
                except Exception:
                    break
                if not isinstance(event_payload, dict):
                    continue
                self._event_buffer.append(event_payload)
                self._deliver_event_to_viewer(event_payload)

        if self._viewer is not None and bool(getattr(self._viewer, "closed", False)):
            self._viewer = None

        if self._viewer is None and self._viewer_cls is not None and self.reopen_file.exists():
            try:
                self.reopen_file.unlink(missing_ok=True)
            except Exception:
                pass
            self._viewer = self._create_viewer_instance()
            if self._viewer is not None:
                for ev in self._event_buffer:
                    self._deliver_event_to_viewer(ev)
                print(f"[{self.log_prefix}] viewer reopened via trigger: {self.reopen_file}")
                self._tick_viewer(force_draw=True)
            else:
                print(
                    f"[{self.log_prefix}] warning: viewer reopen failed. "
                    f"Create '{self.reopen_file}' again to retry."
                )

        self._tick_viewer(force_draw=bool(force_draw))
        self._watchdog()

    def emit_starting_workers(self, *, extra_fields: Optional[Dict[str, object]] = None) -> None:
        if not self.enabled:
            return
        base = dict(extra_fields or {})
        for slot in range(1, self.total_workers + 1):
            event_payload = {
                "type": "worker_started",
                "status": "starting",
                "worker_slot": int(slot),
                "worker_key": f"slot:{slot}",
                "worker_label": f"W{slot:02d}",
                **base,
            }
            self.emit(event_payload)
        self.pump(force_draw=True)

    def health_snapshot(self) -> Dict[str, object]:
        if not self.enabled:
            return {}
        self._sync_frame_counters()
        payload: Dict[str, object] = {
            "viewer_events_processed": int(self._events_processed),
            "viewer_frames_rendered": int(self._frames_rendered),
            "viewer_restart_count": int(self._restart_count),
        }
        last_render = self._viewer_last_render_ts()
        if last_render > 0.0:
            payload["viewer_last_frame_age_sec"] = float(max(0.0, time.time() - last_render))
        return payload

    def close(self) -> None:
        viewer = self._viewer
        self._viewer = None
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception:
                pass
            self._manager = None
        self._queue = None
