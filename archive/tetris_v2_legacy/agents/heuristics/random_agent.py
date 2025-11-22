"""Simple policies for smoke-testing environments."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from gymnasium import spaces


class RandomTetrisAgent:
    """Chooses random discrete actions – useful as a sparring partner."""

    def __init__(self, action_space: spaces.Discrete, seed: Optional[int] = None) -> None:
        self.action_space = action_space
        self._rng = np.random.default_rng(seed)

    def act(self, observation: Any | None = None) -> int:
        if hasattr(self.action_space, "sample"):
            return int(self.action_space.sample())
        return int(self._rng.integers(0, 8))
