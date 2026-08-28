"""Reusable episode metrics for RL checkpoint evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class EpisodeMetrics:
    episode: int
    seed: int
    placements: int
    lines: int
    episode_return: float
    topout: bool
    truncated: bool
    illegal_actions: int = 0

    def to_dict(self) -> dict[str, int | float | bool]:
        return {
            "episode": self.episode,
            "seed": self.seed,
            "placements": self.placements,
            "lines": self.lines,
            "return": self.episode_return,
            "topout": self.topout,
            "truncated": self.truncated,
            "illegal_actions": self.illegal_actions,
        }


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "min": int(np.min(data)),
        "p5": float(np.percentile(data, 5)),
        "median": float(np.median(data)),
        "mean": float(np.mean(data)),
    }


def summarize_episodes(episodes: Sequence[EpisodeMetrics]) -> dict[str, object]:
    if not episodes:
        raise ValueError("At least one episode is required for evaluation metrics.")

    return {
        "episode_count": len(episodes),
        "placements": _distribution([episode.placements for episode in episodes]),
        "lines": _distribution([episode.lines for episode in episodes]),
        "mean_return": float(np.mean([episode.episode_return for episode in episodes])),
        "topout_rate": float(np.mean([episode.topout for episode in episodes])),
        "truncation_rate": float(np.mean([episode.truncated for episode in episodes])),
        "illegal_actions": int(sum(episode.illegal_actions for episode in episodes)),
    }


def evaluate_gate(
    episodes: Sequence[EpisodeMetrics],
    *,
    min_placements: Optional[int],
    min_lines: Optional[int],
) -> dict[str, object]:
    failed_episodes = [
        episode.episode
        for episode in episodes
        if (min_placements is not None and episode.placements < min_placements)
        or (min_lines is not None and episode.lines < min_lines)
        or episode.illegal_actions > 0
    ]
    return {
        "enabled": min_placements is not None or min_lines is not None or bool(failed_episodes),
        "passed": not failed_episodes,
        "min_placements": min_placements,
        "min_lines": min_lines,
        "max_illegal_actions": 0,
        "failed_episodes": failed_episodes,
    }


__all__ = ["EpisodeMetrics", "evaluate_gate", "summarize_episodes"]
