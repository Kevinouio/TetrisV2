"""Common wrappers for Tetris environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from gymnasium import ObservationWrapper, RewardWrapper, Wrapper
from gymnasium import spaces


class IdentityWrapper(Wrapper):
    """No-op wrapper retained for compatibility."""


class FloatBoardWrapper(ObservationWrapper):
    """Normalise the board channel to floats in [0, 1]."""

    def __init__(self, env, *, board_scale: float = 7.0):
        super().__init__(env)
        self.board_scale = board_scale
        board_space = self.observation_space["board"]
        self.observation_space = spaces.Dict(
            {
                **self.observation_space.spaces,
                "board": spaces.Box(
                    low=board_space.low.min() / board_scale,
                    high=board_space.high.max() / board_scale,
                    shape=board_space.shape,
                    dtype=np.float32,
                ),
            }
        )

    def observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        obs = dict(observation)
        obs["board"] = observation["board"].astype(np.float32) / self.board_scale
        return obs


class RewardScaleWrapper(RewardWrapper):
    """Scale rewards for stabilising value targets."""

    def __init__(self, env, scale: float = 1.0):
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, reward: float) -> float:
        return reward * self.scale


@dataclass
class AgentRewardConfig:
    """Common activity-focused reward weights shared across environments."""

    base_reward_scale: float = 1.0
    line_clear_bonus: float = 0.5
    survival_reward: float = 0.02
    top_out_penalty: float = 2.0
    hole_penalty: float = 0.75
    hole_clear_reward: float = 0.5
    height_penalty: float = 0.01
    height_drop_reward: float = 0.005
    bumpiness_penalty: float = 0.02
    bumpiness_drop_reward: float = 0.01
    idle_penalty: float = 0.05
    board_change_bonus: float = 0.1
    excess_rotation_penalty: float = 0.0
    time_decay_penalty: float = 0.0


@dataclass
class EnvironmentRewardConfig:
    """Environment-specific shaping weights."""

    combo_reward: float = 0.05
    attack_reward: float = 0.02
    perfect_clear_reward: float = 2.0
    tspin_reward: float = 0.5
    pending_garbage_penalty: float = 0.01
    hard_drop_reward: float = 0.02
    drop_bonus_scale: float = 0.005
    level_reward: float = 0.0


@dataclass
class _BoardMetrics:
    holes: int
    aggregate_height: int
    bumpiness: int


def _compute_board_metrics(board: np.ndarray) -> _BoardMetrics:
    """Return coarse hand-crafted features for reward shaping."""
    grid = np.asarray(board)
    if grid.ndim != 2:
        grid = grid.reshape(grid.shape[-2], grid.shape[-1])
    height, width = grid.shape
    heights: list[int] = []
    holes = 0
    for col in range(width):
        column = grid[:, col]
        filled = np.where(column > 0)[0]
        if filled.size == 0:
            heights.append(0)
            continue
        top_index = filled[0]
        heights.append(int(height - top_index))
        holes += int(np.count_nonzero(column[top_index:] == 0))
    bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))
    aggregate_height = int(sum(heights))
    return _BoardMetrics(holes=holes, aggregate_height=aggregate_height, bumpiness=int(bumpiness))


class UniversalRewardWrapper(RewardWrapper):
    """Combines agent-level activity shaping with environment-aware bonuses."""

    def __init__(
        self,
        env,
        *,
        env_kind: str = "generic",
        agent_config: Optional[AgentRewardConfig] = None,
        env_config: Optional[EnvironmentRewardConfig] = None,
        board_key: str = "board",
    ):
        super().__init__(env)
        self.agent_config = agent_config or AgentRewardConfig()
        self.env_config = env_config or EnvironmentRewardConfig()
        self.env_kind = (env_kind or "generic").lower()
        self.board_key = board_key
        self._previous_metrics: Optional[_BoardMetrics] = None

    # Gymnasium wrappers override reset to keep local state.
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._previous_metrics = self._extract_metrics(obs)
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = self._shape_reward(obs, float(base_reward), terminated, truncated, info)
        return obs, shaped_reward, terminated, truncated, info

    def _extract_metrics(self, observation: Dict[str, Any]) -> _BoardMetrics:
        board = np.asarray(observation[self.board_key])
        # Boards wrapped with FloatBoardWrapper are scaled to [0, 1]; this still works
        # because any non-zero entry indicates the presence of a block.
        binary_board = np.where(board > 0, 1, 0)
        return _compute_board_metrics(binary_board)

    def _shape_reward(
        self,
        observation: Dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> float:
        metrics = self._extract_metrics(observation)
        prev = self._previous_metrics or metrics
        total = reward * self.agent_config.base_reward_scale
        total += self._agent_activity_reward(prev, metrics, terminated, truncated, info)
        total += self._environment_reward(info)
        self._previous_metrics = metrics
        return total

    def _agent_activity_reward(
        self,
        prev: _BoardMetrics,
        current: _BoardMetrics,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> float:
        cfg = self.agent_config
        total = 0.0

        hole_delta = current.holes - prev.holes
        if hole_delta > 0:
            total -= cfg.hole_penalty * hole_delta
        elif hole_delta < 0:
            total += cfg.hole_clear_reward * (-hole_delta)

        height_delta = current.aggregate_height - prev.aggregate_height
        if height_delta > 0:
            total -= cfg.height_penalty * height_delta
        elif height_delta < 0:
            total += cfg.height_drop_reward * (-height_delta)

        bump_delta = current.bumpiness - prev.bumpiness
        if bump_delta > 0:
            total -= cfg.bumpiness_penalty * bump_delta
        elif bump_delta < 0:
            total += cfg.bumpiness_drop_reward * (-bump_delta)

        board_changed = (
            hole_delta != 0 or height_delta != 0 or bump_delta != 0 or int(info.get("lines_cleared", 0)) > 0
        )
        if board_changed:
            total += cfg.board_change_bonus
        elif self._previous_metrics is not None and not terminated and not truncated:
            total -= cfg.idle_penalty

        total += cfg.line_clear_bonus * float(info.get("lines_cleared", 0))
        if not terminated and not truncated:
            total += cfg.survival_reward
            if cfg.time_decay_penalty:
                total -= cfg.time_decay_penalty
        if terminated:
            total -= cfg.top_out_penalty

        excess_rot = float(info.get("excess_rotations", 0) or 0.0)
        if excess_rot and cfg.excess_rotation_penalty:
            total -= cfg.excess_rotation_penalty * excess_rot
        return total

    def _environment_reward(self, info: Dict[str, Any]) -> float:
        cfg = self.env_config
        reward = 0.0
        combo = info.get("combo")
        if combo:
            reward += cfg.combo_reward * float(combo)
        attack = info.get("attack")
        if attack:
            reward += cfg.attack_reward * float(attack)
        if info.get("perfect_clear"):
            reward += cfg.perfect_clear_reward
        if info.get("t_spin"):
            reward += cfg.tspin_reward
        pending = info.get("pending_garbage")
        if pending:
            reward -= cfg.pending_garbage_penalty * float(pending)
        hard_drop = info.get("hard_drop_distance")
        if hard_drop:
            reward += cfg.hard_drop_reward * float(hard_drop)

        if "nes" in self.env_kind:
            drop_points = info.get("drop_points")
            if drop_points:
                reward += cfg.drop_bonus_scale * float(drop_points) / 100.0
            level = info.get("level")
            if level:
                reward += cfg.level_reward * float(level)

        return reward


# Backwards compatibility aliases.
AdvancedRewardConfig = AgentRewardConfig
AdvancedRewardWrapper = UniversalRewardWrapper
