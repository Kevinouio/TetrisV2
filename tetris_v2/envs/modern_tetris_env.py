"""Modern Tetris Gymnasium environment backed by the shared ruleset."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from tetris_v2.rendering import PygameBoardRenderer
from . import utils
from .modern_ruleset import ModernRuleset

Action = int


class ModernTetrisEnv(gym.Env):
    """Wrapper around ModernRuleset that exposes a Gymnasium API."""

    metadata = ModernRuleset.metadata

    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        queue_size: int = 5,
        lock_delay_frames: int = 30,
        line_clear_delay_frames: int = 20,
        reward_mode: str = "score",
        time_limit_seconds: Optional[float] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        if time_limit_seconds is None and reward_mode == "score":
            time_limit_seconds = 180.0
        self._frame_limit = (
            None if time_limit_seconds is None else int(time_limit_seconds * self.metadata["render_fps"])
        )
        self._frames = 0
        self._rules = ModernRuleset(
            queue_size=queue_size,
            lock_delay_frames=lock_delay_frames,
            line_clear_delay_frames=line_clear_delay_frames,
            max_steps=max_steps,
        )
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=8, shape=(20, 10), dtype=np.int8),
                "current": spaces.Box(low=-10_000, high=10_000, shape=(4,), dtype=np.int16),
                "queue": spaces.Box(low=0, high=7, shape=(queue_size,), dtype=np.int8),
                "hold": spaces.Box(low=-1, high=7, shape=(), dtype=np.int8),
                "combo": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int16),
                "back_to_back": spaces.Discrete(2),
                "level": spaces.Box(low=0, high=1000, shape=(), dtype=np.int16),
                "lines": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int16),
                "score": spaces.Box(low=0, high=2**31 - 1, shape=(), dtype=np.int32),
                "pending_garbage": spaces.Box(low=0, high=200, shape=(), dtype=np.int16),
            }
        )
        self.action_space = spaces.Discrete(8)
        self._renderer: Optional[PygameBoardRenderer] = None

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if options and "seed" in options:
            seed = options["seed"]
        obs = self._rules.reset(seed=seed)
        self._frames = 0
        return obs, {}

    def step(self, action: Action):
        result = self._rules.step(action)
        reward = self._compute_reward(result)
        info = dict(result.info)
        self._frames += 1
        truncated = False
        if self._frame_limit is not None and self._frames >= self._frame_limit:
            truncated = True
            info["time_limit_reached"] = True
        if self._frame_limit:
            remaining = max(self._frame_limit - self._frames, 0)
            info["time_remaining_frames"] = remaining
        return result.observation, reward, result.terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "rgb_array":
            board = self._rules.board_matrix()
            return utils.render_board_rgb(board)
        if self.render_mode == "human":
            rgb = utils.render_board_rgb(self._rules.board_matrix())
            observation = self._rules.snapshot()
            hud = {
                "Score": int(observation["score"]),
                "Lines": int(observation["lines"]),
                "Level": int(observation["level"]),
                "Combo": int(observation["combo"]),
                "Pending": int(observation["pending_garbage"]),
            }
            if self._frame_limit:
                remaining = max(self._frame_limit - self._frames, 0)
                seconds = remaining / self.metadata["render_fps"]
                hud["Time"] = f"{int(seconds // 60)}:{int(seconds % 60):02d}"
            hold_id = int(observation["hold"])
            hold_image = utils.render_piece_preview(hold_id) if hold_id >= 0 else None
            queue_ids = list(self._rules._queue)[:3]
            queue_images = [utils.render_piece_preview(pid) for pid in queue_ids]
            if self._renderer is None:
                self._renderer = PygameBoardRenderer(
                    title="Modern Tetris",
                    board_shape=rgb.shape[:2],
                )
            self._renderer.draw(rgb, hud, hold_image=hold_image, queue_images=queue_images)
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    def _compute_reward(self, result) -> float:
        if self.reward_mode == "attack":
            return float(result.attack)
        if self.reward_mode == "lines":
            return float(result.lines_cleared)
        return float(result.score_delta) / 100.0
