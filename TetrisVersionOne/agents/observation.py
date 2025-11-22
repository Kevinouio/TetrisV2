"""Flattening and normalisation utilities for Tetris observations."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces


DEFAULT_NORMALISATION_FACTORS: Dict[str, np.ndarray | float] = {
    "current": np.array([6.0, 3.0, 20.0, 10.0], dtype=np.float32),
    "next": 6.0,
    "queue": 6.0,
    "hold": 6.0,
    "combo": 10.0,
    "back_to_back": 1.0,
    "level": 30.0,
    "lines": 200.0,
    "score": 100_000.0,
    "pending_garbage": 20.0,
    "opponent_pending": 20.0,
    "opponent_combo": 10.0,
    "opponent_back_to_back": 1.0,
}


class ObservationProcessor:
    """Flattens and normalises dict observations for the agent."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        normalization_overrides: Optional[Dict[str, np.ndarray | float]] = None,
        clip_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("ObservationProcessor requires a Dict observation space.")
        self.observation_space = observation_space
        self.keys = tuple(observation_space.spaces.keys())
        self._sizes: Dict[str, int] = {}
        flat_dim = 0
        self.slices: Dict[str, Tuple[int, int]] = {}
        self.board_shape: Optional[Tuple[int, int]] = None
        self.board_dim: int = 0
        for key in self.keys:
            space = observation_space.spaces[key]
            if isinstance(space, spaces.Discrete):
                size = 1
            else:
                shape = getattr(space, "shape", ()) or ()
                size = int(np.prod(shape, dtype=int)) if shape else 1
            self._sizes[key] = size
            flat_dim += size
            self.slices[key] = (flat_dim - size, flat_dim)
            if key == "board" and getattr(space, "shape", None):
                self.board_shape = tuple(int(dim) for dim in space.shape)  # type: ignore[arg-type]
                self.board_dim = size
        self.flat_dim = flat_dim
        self._clip_range = clip_range
        self._normalisers: Dict[str, np.ndarray] = {}
        overrides = normalization_overrides or {}
        for key in self.keys:
            factor = overrides.get(key, DEFAULT_NORMALISATION_FACTORS.get(key))
            if factor is None:
                continue
            arr = np.asarray(factor, dtype=np.float32)
            arr = np.where(arr == 0.0, 1.0, np.abs(arr))
            self._normalisers[key] = arr

    def flatten(self, observation: Dict[str, Any]) -> np.ndarray:
        """Convert a dict observation into a flat, scaled float32 vector."""
        flat = np.zeros(self.flat_dim, dtype=np.float32)
        offset = 0
        low, high = self._clip_range
        for key in self.keys:
            value = observation[key]
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            norm = self._normalisers.get(key)
            if norm is not None:
                arr = arr / norm
            arr = np.clip(arr, low, high)
            target = self._sizes[key]
            if arr.size < target:
                padded = np.zeros(target, dtype=np.float32)
                padded[: arr.size] = arr
                arr = padded
            elif arr.size > target:
                arr = arr[:target]
            flat[offset : offset + target] = arr
            offset += target
        return flat
