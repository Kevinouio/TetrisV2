"""Common wrappers for Tetris environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import ObservationWrapper, RewardWrapper, Wrapper
from gymnasium import spaces

from tetris_v2.envs import utils
from tetris_v2.envs.modern_ruleset import (
    HARD_DROP as MODERN_HARD_DROP,
    HOLD as MODERN_HOLD,
    MOVE_LEFT as MODERN_MOVE_LEFT,
    MOVE_NONE as MODERN_MOVE_NONE,
    MOVE_RIGHT as MODERN_MOVE_RIGHT,
    SOFT_DROP as MODERN_SOFT_DROP,
    ROTATE_CCW as MODERN_ROTATE_CCW,
    ROTATE_CW as MODERN_ROTATE_CW,
    ROTATE_180 as MODERN_ROTATE_180,
)
from tetris_v2.envs.modern_tetris_env import ModernTetrisEnv
from tetris_v2.envs.nes_tetris_env import NesTetrisEnv
from tetris_v2.envs.placement_planner import (
    ActionCodes,
    HoldContext,
    PlacementActionSpec,
    PlacementPlan,
    build_action_specs,
    compute_action_plans,
)


class IdentityWrapper(Wrapper):
    """No-op wrapper retained for compatibility."""

class _ProxyWrapperMixin:
    """Allows forwarding of private attributes for downstream wrappers."""

    def __getattr__(self, name: str):
        if name == "_np_random":
            raise AttributeError(
                "Can't access `_np_random` of a wrapper, use `self.unwrapped._np_random` or `self.np_random`."
            )
        return getattr(self.env, name)


class FloatBoardWrapper(_ProxyWrapperMixin, ObservationWrapper):
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


class RewardScaleWrapper(_ProxyWrapperMixin, RewardWrapper):
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
    hard_drop_reward: float = 0.0
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


class UniversalRewardWrapper(_ProxyWrapperMixin, RewardWrapper):
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
        total += self.agent_activity_reward(prev, metrics, terminated, truncated, info)
        total += self.environment_reward(info)
        self._previous_metrics = metrics
        return total

    def agent_activity_reward(
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

    def environment_reward(self, info: Dict[str, Any]) -> float:
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


class PlacementActionWrapper(_ProxyWrapperMixin, Wrapper):
    """Collapse low-level inputs into discrete placement actions."""

    def __init__(self, env, *, allow_hold: Optional[bool] = None):
        super().__init__(env)
        base = env.unwrapped
        self._is_modern = isinstance(base, ModernTetrisEnv)
        self._is_nes = isinstance(base, NesTetrisEnv)
        if not (self._is_modern or self._is_nes):
            raise TypeError("PlacementActionWrapper only supports the bundled Tetris envs.")
        supports_hold = self._is_modern
        if allow_hold is False:
            supports_hold = False
        self._supports_hold = supports_hold
        if allow_hold is True and not self._is_modern:
            self._supports_hold = False
        if self._is_modern:
            self._noop_action = MODERN_MOVE_NONE
            self._hard_drop_action = MODERN_HARD_DROP
            self._left_action = MODERN_MOVE_LEFT
            self._right_action = MODERN_MOVE_RIGHT
            self._rotate_cw = MODERN_ROTATE_CW
            self._rotate_ccw = MODERN_ROTATE_CCW
            self._rotate_180 = MODERN_ROTATE_180
            self._hold_action = MODERN_HOLD
            self._soft_drop_action = MODERN_SOFT_DROP
        else:
            self._noop_action = NesTetrisEnv.MOVE_NONE
            self._hard_drop_action = NesTetrisEnv.HARD_DROP
            self._left_action = NesTetrisEnv.MOVE_LEFT
            self._right_action = NesTetrisEnv.MOVE_RIGHT
            self._rotate_cw = NesTetrisEnv.ROTATE_CW
            self._rotate_ccw = None
            self._rotate_180 = None
            self._hold_action = None
            self._soft_drop_action = NesTetrisEnv.SOFT_DROP

        self._action_codes = ActionCodes(
            move_left=self._left_action,
            move_right=self._right_action,
            soft_drop=self._soft_drop_action,
            hard_drop=self._hard_drop_action,
            rotate_cw=self._rotate_cw,
            rotate_ccw=self._rotate_ccw,
            rotate_180=self._rotate_180,
            hold=self._hold_action,
            neutral=self._noop_action,
        )
        self._specs = build_action_specs(allow_hold=self._supports_hold)
        self.action_space = spaces.Discrete(len(self._specs))
        self.observation_space = env.observation_space
        self._last_obs: Optional[Dict[str, Any]] = None
        self._last_mask = np.ones(len(self._specs), dtype=np.int8)
        self._done = False
        self._action_plans: List[Optional[PlacementPlan]] = [None] * len(self._specs)

    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._done = False
        self._last_obs = obs
        mask, plans = self._refresh_action_library(obs)
        self._last_mask = mask
        self._action_plans = plans
        return obs, self._attach_mask(info, mask)

    # ------------------------------------------------------------------
    def step(self, action: int):
        if self._done:
            raise RuntimeError("step() called after the episode terminated; call reset() first.")
        if not self.action_space.contains(action):
            raise AssertionError(f"Invalid placement action {action}")
        if self._last_obs is None:
            raise RuntimeError("Environment reset() must be called before step().")

        placement = self._specs[action]
        total_reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        success = True
        obs = self._last_obs

        plan = self._action_plans[action]
        last_info: Dict[str, Any] = {}
        actions_run = 0
        if plan is None:
            success = False
        else:
            for code in plan.actions:
                obs, reward, terminated, truncated, step_info = self.env.step(code)
                total_reward += reward
                last_info = step_info
                actions_run += 1
                if terminated or truncated:
                    break
            success = success and actions_run == len(plan.actions)
        info = last_info

        if not (terminated or truncated):
            obs, wait_reward, terminated, truncated, wait_info = self._advance_until_piece(obs, terminated, truncated)
            total_reward += wait_reward
            if wait_info:
                info = wait_info

        self._finalise(obs, terminated, truncated)
        info = self._final_info(info, action, placement, success)
        info = self._attach_mask(info, self._last_mask)
        return obs, total_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _finalise(self, obs, terminated, truncated) -> None:
        self._last_obs = obs
        if terminated or truncated:
            self._done = True
            self._last_mask = np.zeros(len(self._specs), dtype=np.int8)
            self._action_plans = [None] * len(self._specs)
        else:
            mask, plans = self._refresh_action_library(obs)
            self._last_mask = mask
            self._action_plans = plans

    def _attach_mask(self, info: Optional[Dict[str, Any]], mask: np.ndarray) -> Dict[str, Any]:
        data = dict(info or {})
        data["action_mask"] = mask.astype(np.int8)
        return data

    def _final_info(
        self,
        info: Dict[str, Any],
        action_idx: int,
        spec: PlacementActionSpec,
        success: bool,
    ) -> Dict[str, Any]:
        data = dict(info or {})
        data["placement_action_index"] = action_idx
        data["placement_target_rotation"] = spec.rotation
        data["placement_target_column"] = spec.column
        data["placement_used_hold"] = spec.use_hold
        data["placement_success"] = bool(success)
        plan = self._action_plans[action_idx]
        if plan is not None:
            data["placement_landing_row"] = plan.landing_row
        return data

    def _refresh_action_library(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, List[Optional[PlacementPlan]]]:
        mask = np.zeros(len(self._specs), dtype=np.int8)
        plans: List[Optional[PlacementPlan]] = [None] * len(self._specs)
        board = self._board_matrix()
        current_piece = self._current_piece_state(obs)
        hold_ctx = self._build_hold_context()
        plan_map = compute_action_plans(
            board=board,
            specs=self._specs,
            current_piece=current_piece,
            action_codes=self._action_codes,
            hold_context=hold_ctx,
            is_modern=self._is_modern,
        )
        for idx, spec in enumerate(self._specs):
            plan = plan_map.get(spec)
            if plan is None:
                continue
            mask[idx] = 1
            plans[idx] = plan
        return mask, plans

    def _build_hold_context(self) -> HoldContext:
        if not self._supports_hold:
            return HoldContext(False, False, None, None)
        rules = getattr(self.env.unwrapped, "_rules", None)
        if rules is None:
            return HoldContext(True, False, None, None)
        hold_available = bool(getattr(rules, "_hold_available", False))
        hold_piece_id = getattr(rules, "_hold_piece", None)
        queue_ids = list(getattr(rules, "_queue", []))
        queue_head = queue_ids[0] if queue_ids else None
        return HoldContext(True, hold_available, hold_piece_id, queue_head)

    def available_action_descriptions(self) -> List[Dict[str, Any]]:
        """Return metadata about currently valid placement actions."""
        descriptions: List[Dict[str, Any]] = []
        for idx, plan in enumerate(self._action_plans):
            if plan is None:
                continue
            descriptions.append(
                {
                    "index": idx,
                    "use_hold": plan.spec.use_hold,
                    "rotation": plan.spec.rotation,
                    "column": plan.spec.column,
                    "landing_row": plan.landing_row,
                    "sequence_length": len(plan.actions),
                }
            )
        return descriptions

    def _advance_until_piece(self, obs, terminated: bool, truncated: bool):
        total_reward = 0.0
        info: Dict[str, Any] = {}
        current_obs = obs
        done = terminated or truncated
        while not done:
            if self._current_piece_state(current_obs) is not None:
                break
            current_obs, reward, terminated, truncated, step_info = self.env.step(self._noop_action)
            total_reward += reward
            info = step_info
            done = terminated or truncated
        return current_obs, total_reward, terminated, truncated, info

    def _current_piece_state(self, obs: Optional[Dict[str, Any]]):
        if obs is None or "current" not in obs:
            return None
        cur = obs["current"]
        piece_id = int(cur[0])
        if piece_id < 0:
            return None
        rotation = int(cur[1]) % 4
        row = int(cur[2]) + utils.HIDDEN_ROWS
        col = int(cur[3])
        return utils.PieceState(piece_id=piece_id, rotation=rotation, row=row, col=col)

    def _board_matrix(self) -> np.ndarray:
        base = self.env.unwrapped
        board = getattr(base, "_board", None)
        if board is not None:
            return board
        rules = getattr(base, "_rules", None)
        if rules is not None:
            return rules._board
        raise RuntimeError("Unsupported environment structure for placement wrapper.")

# Backwards compatibility aliases.
AdvancedRewardConfig = AgentRewardConfig
AdvancedRewardWrapper = UniversalRewardWrapper
