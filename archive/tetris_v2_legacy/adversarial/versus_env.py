"""Two-player adversarial environment built on the modern ruleset."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from tetris_v2.rendering import PygameBoardRenderer
from tetris_v2.agents.heuristics.random_agent import RandomTetrisAgent
from tetris_v2.envs import utils
from tetris_v2.envs.modern_ruleset import ModernRuleset


class VersusEnv(gym.Env):
    """Player vs. AI opponent using modern rules, garbage + 1v1 scoring."""

    metadata = ModernRuleset.metadata

    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        opponent_policy: Optional[RandomTetrisAgent] = None,
        reward_mode: str = "attack",
        time_limit_seconds: Optional[float] = 120.0,
        soft_drop_factor: float = 6.0,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        self._frame_limit = (
            None if time_limit_seconds is None else int(time_limit_seconds * self.metadata["render_fps"])
        )
        self._frames = 0
        self._player = ModernRuleset(soft_drop_factor=soft_drop_factor)
        self._opponent = ModernRuleset(soft_drop_factor=soft_drop_factor)
        self._latest_opponent_obs: Optional[Dict[str, np.ndarray]] = None
        self._opponent_policy = opponent_policy or RandomTetrisAgent(spaces.Discrete(8))
        base_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=8, shape=(20, 10), dtype=np.int8),
                "current": spaces.Box(low=-10_000, high=10_000, shape=(4,), dtype=np.int16),
                "queue": spaces.Box(low=0, high=7, shape=(5,), dtype=np.int8),
                "hold": spaces.Box(low=-1, high=7, shape=(), dtype=np.int8),
                "combo": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int16),
                "back_to_back": spaces.Discrete(2),
                "level": spaces.Box(low=0, high=1000, shape=(), dtype=np.int16),
                "lines": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int16),
                "score": spaces.Box(low=0, high=2**31 - 1, shape=(), dtype=np.int32),
                "pending_garbage": spaces.Box(low=0, high=200, shape=(), dtype=np.int16),
            }
        )
        self.observation_space = spaces.Dict(
            {
                **base_space.spaces,
                "opponent_board": spaces.Box(low=0, high=8, shape=(20, 10), dtype=np.int8),
                "opponent_pending": spaces.Box(low=0, high=200, shape=(), dtype=np.int16),
                "opponent_combo": spaces.Box(low=0, high=10_000, shape=(), dtype=np.int16),
                "opponent_back_to_back": spaces.Discrete(2),
            }
        )
        self.action_space = spaces.Discrete(9)
        self._renderer: Optional[PygameBoardRenderer] = None
        self._skip_frame_advance = False

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        player_obs = self._player.reset(seed=seed)
        opponent_seed = None
        if seed is not None:
            opponent_seed = seed + 1
        opponent_obs = self._opponent.reset(seed=opponent_seed)
        self._latest_opponent_obs = opponent_obs
        self._frames = 0
        return self._build_observation(player_obs, opponent_obs), {}

    def step(self, action: int):
        if self._latest_opponent_obs is None:
            self._latest_opponent_obs = self._opponent.snapshot()
        opponent_action = self._opponent_policy.act(self._latest_opponent_obs)
        player_result = self._player.step(action)
        opponent_result = self._opponent.step(opponent_action)
        self._latest_opponent_obs = opponent_result.observation

        cancel = min(player_result.attack, opponent_result.attack)
        player_attack = max(0, player_result.attack - cancel)
        opponent_attack = max(0, opponent_result.attack - cancel)
        if player_attack:
            self._opponent.queue_garbage(player_attack)
        if opponent_attack:
            self._player.queue_garbage(opponent_attack)

        reward = self._compute_reward(player_result, player_attack, opponent_attack)
        info = dict(player_result.info)
        info["opponent_attack"] = opponent_attack
        info["player_attack"] = player_attack
        info["opponent_top_out"] = opponent_result.terminated
        terminated = player_result.terminated or opponent_result.terminated

        if not getattr(self, "_skip_frame_advance", False):
            self._frames += 1
        self._skip_frame_advance = False
        truncated = False
        if self._frame_limit is not None and self._frames >= self._frame_limit:
            truncated = True
            info["time_limit_reached"] = True
        info["time_remaining_frames"] = (
            0 if self._frame_limit is None else max(self._frame_limit - self._frames, 0)
        )
        obs = self._build_observation(player_result.observation, opponent_result.observation)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return None
        frame = self._compose_frame()
        if self.render_mode == "rgb_array":
            return frame
        if self.render_mode == "human" and frame is not None:
            player_snapshot = self._player.snapshot()
            opponent_snapshot = self._opponent.snapshot()
            hud = {
                "Score": int(player_snapshot["score"]),
                "Lines": int(player_snapshot["lines"]),
                "Pending": int(player_snapshot["pending_garbage"]),
                "Opp Pending": int(opponent_snapshot["pending_garbage"]),
            }
            if self._frame_limit:
                remaining = max(self._frame_limit - self._frames, 0)
                seconds = remaining / self.metadata["render_fps"]
                hud["Time"] = f"{int(seconds // 60)}:{int(seconds % 60):02d}"
            hold_id = int(player_snapshot["hold"])
            hold_image = utils.render_piece_preview(hold_id) if hold_id >= 0 else None
            queue_images = [
                utils.render_piece_preview(pid)
                for pid in list(self._player._queue)[: self._player.queue_size]
            ]
            if self._renderer is None:
                self._renderer = PygameBoardRenderer(title="Tetris Versus", board_shape=frame.shape[:2])
            self._renderer.draw(frame, hud, hold_image=hold_image, queue_images=queue_images)
        return None

    def _compose_frame(self) -> Optional[np.ndarray]:
        player = utils.render_board_rgb(
            self._player.board_matrix(include_current=False),
            current=self._player.current_piece(),
            ghost=self._player.ghost_piece(),
        )
        opponent = utils.render_board_rgb(
            self._opponent.board_matrix(include_current=False),
            current=self._opponent.current_piece(),
            ghost=self._opponent.ghost_piece(),
        )
        gap = np.zeros((player.shape[0], 20, 3), dtype=np.uint8)
        return np.concatenate([player, gap, opponent], axis=1)

    def close(self):
        if self._renderer:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    def _build_observation(self, player_obs, opponent_obs):
        return {
            **player_obs,
            "opponent_board": opponent_obs["board"],
            "opponent_pending": opponent_obs["pending_garbage"],
            "opponent_combo": opponent_obs["combo"],
            "opponent_back_to_back": opponent_obs["back_to_back"],
        }

    def _compute_reward(self, player_result, player_attack: int, opponent_attack: int) -> float:
        if self.reward_mode == "score":
            return player_result.score_delta / 100.0
        if self.reward_mode == "lines":
            return float(player_result.lines_cleared)
        return float(player_attack - opponent_attack)
