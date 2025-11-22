"""Modern Tetris Gymnasium environment backed by the shared ruleset."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..rendering import PygameBoardRenderer
from . import utils
from .modern_ruleset import (
    ModernRuleset,
    MOVE_NONE,
    MOVE_LEFT,
    MOVE_RIGHT,
    ROTATE_CW,
    ROTATE_CCW,
    SOFT_DROP,
    HARD_DROP,
    HOLD,
    ROTATE_180,
)

Action = int


class ModernTetrisEnv(gym.Env):
    """Wrapper around ModernRuleset that exposes a Gymnasium API."""

    metadata = ModernRuleset.metadata

    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        preview_pieces: int = 5,
        lock_delay_frames: int = 30,
        line_clear_delay_frames: int = 0,
        reward_mode: str = "score",
        time_limit_seconds: Optional[float] = None,
        max_steps: Optional[int] = None,
        soft_drop_factor: float = 6.0,
        das_frames: int = 10,
        arr_frames: int = 2,
        allowed_pieces: Optional[Sequence[str | int]] = None,
        include_hold: bool = True,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        self.preview_pieces = max(1, int(preview_pieces))
        self._frame_limit = (
            None if time_limit_seconds is None else int(time_limit_seconds * self.metadata["render_fps"])
        )
        self._frames = 0
        self._rules = ModernRuleset(
            queue_size=self.preview_pieces,
            lock_delay_frames=lock_delay_frames,
            line_clear_delay_frames=line_clear_delay_frames,
            soft_drop_factor=soft_drop_factor,
            max_steps=max_steps,
            das_frames=das_frames,
            arr_frames=arr_frames,
            allowed_pieces=allowed_pieces,
        )
        # Mapping from discrete index -> low-level ruleset action.
        self._action_map = [
            MOVE_NONE,
            MOVE_LEFT,
            MOVE_RIGHT,
            ROTATE_CW,
            ROTATE_CCW,
            SOFT_DROP,
            HARD_DROP,
            ROTATE_180,
        ]
        if include_hold:
            self._action_map.append(HOLD)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=8, shape=(20, 10), dtype=np.int8),
                "current": spaces.Box(low=-10_000, high=10_000, shape=(4,), dtype=np.int16),
                "queue": spaces.Box(low=0, high=7, shape=(self.preview_pieces,), dtype=np.int8),
                "hold": spaces.Box(low=-1, high=7, shape=(), dtype=np.int8),
                "combo": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int16),
                "back_to_back": spaces.Discrete(2),
                "level": spaces.Box(low=0, high=1000, shape=(), dtype=np.int16),
                "lines": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int16),
                "score": spaces.Box(low=0, high=2**31 - 1, shape=(), dtype=np.int32),
                "pending_garbage": spaces.Box(low=0, high=200, shape=(), dtype=np.int16),
            }
        )
        self.action_space = spaces.Discrete(len(self._action_map))
        self._renderer: Optional[PygameBoardRenderer] = None
        self._skip_frame_advance = False

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if options and "seed" in options:
            seed = options["seed"]
        obs = self._rules.reset(seed=seed)
        self._frames = 0
        return obs, {}

    def step(self, action: Action):
        try:
            mapped = self._action_map[int(action)]
        except (IndexError, TypeError, ValueError):
            raise AssertionError(f"Invalid action index {action}") from None
        result = self._rules.step(mapped)
        reward = self._compute_reward(result)
        info = dict(result.info)
        if getattr(self, "_skip_frame_advance", False):
            pass
        else:
            self._frames += 1
        truncated = False
        if info.get("max_steps_reached"):
            truncated = True
            info["time_limit_reached"] = True
        if self._frame_limit is not None and self._frames >= self._frame_limit:
            truncated = True
            info["time_limit_reached"] = True
        if self._frame_limit:
            remaining = max(self._frame_limit - self._frames, 0)
            info["time_remaining_frames"] = remaining
        self._skip_frame_advance = False
        return result.observation, reward, result.terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        current_piece = self._rules.current_piece()
        ghost_piece = self._rules.ghost_piece()
        base_board = self._rules.board_matrix(include_current=False)
        if self.render_mode == "rgb_array":
            return utils.render_board_rgb(base_board, current=current_piece, ghost=ghost_piece)
        if self.render_mode == "human":
            rgb = utils.render_board_rgb(base_board, current=current_piece, ghost=ghost_piece)
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
            queue_ids = list(self._rules._queue)[: self.preview_pieces]
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
