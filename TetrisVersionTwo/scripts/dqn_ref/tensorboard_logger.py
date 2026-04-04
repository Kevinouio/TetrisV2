from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class DQNTensorBoardLogger:
    """Best-effort TensorBoard scalar logger for DQN-ref training."""

    _EPISODE_TAGS = {
        "episode_return": "episode/return",
        "lines_cleared": "episode/lines_cleared",
        "survival_length": "episode/survival_length",
        "epsilon": "episode/epsilon",
        "learning_rate": "episode/learning_rate",
        "loss": "episode/loss",
    }

    _EPISODE_REWARD_TERM_TAGS = {
        "game_over_term": "episode/reward_terms/game_over_term",
        "survival_term": "episode/reward_terms/survival_term",
        "y_pos_term": "episode/reward_terms/y_pos_term",
        "total_height_term": "episode/reward_terms/total_height_term",
        "lines_term": "episode/reward_terms/lines_term",
        "holes_term": "episode/reward_terms/holes_term",
        "bumpiness_term": "episode/reward_terms/bumpiness_term",
        "pillar_term": "episode/reward_terms/pillar_term",
        "high_placement_penalty_term": "episode/reward_terms/high_placement_penalty_term",
    }

    _GENERATION_TAGS = {
        "best_fitness": "generation/best_fitness",
        "mean_fitness": "generation/mean_fitness",
        "std_fitness": "generation/std_fitness",
        "diversity": "generation/diversity",
        "uniform_rate": "generation/uniform_rate",
        "alpha_rate": "generation/alpha_rate",
        "mutate_rate": "generation/mutate_rate",
        "checkpoint_saved": "generation/checkpoint_saved",
    }

    def __init__(
        self,
        *,
        enabled: bool,
        log_root_dir: Path,
        backfill_enabled: bool = True,
        flush_every_events: int = 64,
    ) -> None:
        self.enabled = bool(enabled)
        self.log_root_dir = Path(log_root_dir)
        self.backfill_enabled = bool(backfill_enabled)
        self.flush_every_events = max(1, int(flush_every_events))

        self.session_dir: Optional[Path] = None
        self._writer: Optional[Any] = None
        self._writer_unavailable = False
        self._writer_unavailable_reason = ""

        self.global_episode_step = 0
        self.global_generation_step = 0
        self._pending_events_since_flush = 0

    @property
    def ready(self) -> bool:
        return self._writer is not None

    @property
    def unavailable_reason(self) -> str:
        return str(self._writer_unavailable_reason)

    def _ensure_writer(self) -> bool:
        if not self.enabled:
            return False
        if self._writer is not None:
            return True
        if self._writer_unavailable:
            return False

        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except Exception as exc:
            self._writer_unavailable = True
            self._writer_unavailable_reason = f"{type(exc).__name__}: {exc}"
            return False

        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self.session_dir = self.log_root_dir / timestamp
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.session_dir))
        return True

    @staticmethod
    def _to_float(payload: Dict[str, Any], key: str, default: float = 0.0) -> float:
        raw = payload.get(key, default)
        if raw is None:
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    def _emit_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is None:
            return
        self._writer.add_scalar(tag, float(value), int(step))
        self._pending_events_since_flush += 1
        if self._pending_events_since_flush >= self.flush_every_events:
            self.flush()

    def log_episode_row(self, row: Dict[str, Any]) -> None:
        if not self._ensure_writer():
            return
        step = int(self.global_episode_step)
        for key, tag in self._EPISODE_TAGS.items():
            self._emit_scalar(tag, self._to_float(row, key, 0.0), step)
        for key, tag in self._EPISODE_REWARD_TERM_TAGS.items():
            self._emit_scalar(tag, self._to_float(row, key, 0.0), step)
        self.global_episode_step += 1

    def log_generation_row(self, row: Dict[str, Any]) -> None:
        if not self._ensure_writer():
            return
        step = int(self.global_generation_step)
        for key, tag in self._GENERATION_TAGS.items():
            self._emit_scalar(tag, self._to_float(row, key, 0.0), step)
        self.global_generation_step += 1

    def backfill_from_csv(
        self,
        *,
        episode_csv_path: Path,
        generation_csv_path: Path,
    ) -> Tuple[int, int]:
        if not self.enabled or not self.backfill_enabled:
            return 0, 0
        if not self._ensure_writer():
            return 0, 0

        episode_rows = 0
        generation_rows = 0

        if episode_csv_path.exists():
            with episode_csv_path.open("r", newline="", encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    if not row:
                        continue
                    self.log_episode_row(dict(row))
                    episode_rows += 1

        if generation_csv_path.exists():
            with generation_csv_path.open("r", newline="", encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    if not row:
                        continue
                    self.log_generation_row(dict(row))
                    generation_rows += 1

        self.flush()
        return episode_rows, generation_rows

    def flush(self) -> None:
        if self._writer is None:
            return
        self._writer.flush()
        self._pending_events_since_flush = 0

    def close(self) -> None:
        writer = self._writer
        self._writer = None
        if writer is None:
            return
        try:
            writer.flush()
        except Exception:
            pass
        try:
            writer.close()
        except Exception:
            pass
        self._pending_events_since_flush = 0
