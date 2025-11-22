"""Curriculum helpers for staged Tetris training."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import numpy as np
from gymnasium import Wrapper

from . import utils
from .state_presets import BoardPreset, BoardPresetLibrary, apply_board_preset
from .wrappers import _ProxyWrapperMixin


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum phase."""

    name: str
    max_global_steps: Optional[int]
    description: str = ""
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    agent_reward_overrides: Dict[str, float] = field(default_factory=dict)
    env_reward_overrides: Dict[str, float] = field(default_factory=dict)
    max_moves: Optional[int] = None
    end_on_line_clear: bool = False
    board_setup: Optional[str] = None
    max_height: Optional[int] = None


class CurriculumManager:
    """Keeps track of which stage should be active for a given global step."""

    def __init__(self, stages: List[CurriculumStage]):
        if not stages:
            raise ValueError("Curriculum requires at least one stage.")
        self._stages = stages

    @property
    def stages(self) -> List[CurriculumStage]:
        return list(self._stages)

    def stage_for_step(self, global_step: int) -> CurriculumStage:
        for stage in self._stages:
            if stage.max_global_steps is None or global_step < stage.max_global_steps:
                return stage
        return self._stages[-1]


def build_default_curriculum(total_timesteps: int) -> CurriculumManager:
    """Return a CurriculumManager implementing the staged plan described in docs."""

    # Helper to clamp stage end so we don't create empty windows when total steps are small.
    def _cap(value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        return min(value, total_timesteps)

    stages = [
        CurriculumStage(
            name="line_discovery",
            max_global_steps=_cap(1_000_000),
            description="Short horizons to learn clearing lines without creating holes.",
            agent_reward_overrides={
                "line_clear_bonus": 2.0,
                "hole_penalty": 0.1,
                "hole_clear_reward": 0.0,
                "height_penalty": 0.0,
                "height_drop_reward": 0.0,
                "bumpiness_penalty": 0.0,
                "bumpiness_drop_reward": 0.0,
                "survival_reward": 0.0,
                "idle_penalty": 0.0,
                "time_decay_penalty": 0.01,  # small step cost
                "top_out_penalty": 1.0,
                "board_change_bonus": 0.0,
            },
            env_reward_overrides={
                "combo_reward": 0.0,
                "attack_reward": 0.0,
                "pending_garbage_penalty": 0.0,
                "hard_drop_reward": 0.0,
            },
            env_kwargs={"max_steps": 100},
            max_moves=100,
            board_setup="nes_easy_line_clear",
        ),
        CurriculumStage(
            name="stack_health",
            max_global_steps=_cap(5_000_000),
            description="Longer horizons rewarding stable stacks and penalising messy growth.",
            agent_reward_overrides={
                "line_clear_bonus": 1.5,
                "hole_penalty": 0.3,
                "hole_clear_reward": 0.0,
                "height_penalty": 0.02,
                "height_drop_reward": 0.0,
                "bumpiness_penalty": 0.02,
                "bumpiness_drop_reward": 0.0,
                "survival_reward": 0.0,
                "idle_penalty": 0.0,
                "time_decay_penalty": 0.0,
                "top_out_penalty": 0.5,
                "board_change_bonus": 0.0,
            },
            env_reward_overrides={
                "combo_reward": 0.0,
                "attack_reward": 0.0,
                "pending_garbage_penalty": 0.0,
                "hard_drop_reward": 0.0,
            },
            env_kwargs={"max_steps": 300},
            max_moves=300,
        ),
        CurriculumStage(
            name="full_nes_play",
            max_global_steps=None,
            description="Near-final reward shaping; push toward normal NES play.",
            agent_reward_overrides={
                "line_clear_bonus": 1.0,
                "hole_penalty": 0.1,
                "hole_clear_reward": 0.0,
                "height_penalty": 0.01,
                "height_drop_reward": 0.0,
                "bumpiness_penalty": 0.01,
                "bumpiness_drop_reward": 0.0,
                "survival_reward": 0.0,
                "idle_penalty": 0.0,
                "time_decay_penalty": 0.0,
                "top_out_penalty": 1.0,
                "board_change_bonus": 0.0,
            },
            env_reward_overrides={
                "combo_reward": 0.0,
                "attack_reward": 0.0,
                "pending_garbage_penalty": 0.0,
                "hard_drop_reward": 0.0,
            },
            env_kwargs={"max_steps": 500},
            max_moves=500,
        ),
    ]
    return CurriculumManager(stages)


def build_modern_placement_curriculum(total_timesteps: int) -> CurriculumManager:
    """Curriculum tuned for placement-action agents in the modern environment."""

    def _cap(value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        return min(value, total_timesteps)

    stages: List[CurriculumStage] = []

    stage_defs = [
        (
            "guided_line_clears",
            2_000_000,
            CurriculumStage(
                name="guided_line_clears",
                max_global_steps=None,
                description="Start near line clears with slow pieces and limited bag to learn precise placements.",
                env_kwargs={
                    "max_steps": 60,
                    "allowed_pieces": ("I", "J", "L", "T"),
                    "das_frames": 14,
                    "arr_frames": 6,
                    "soft_drop_factor": 4.0,
                },
                agent_reward_overrides={
                    "line_clear_bonus": 4.0,
                    "idle_penalty": 0.0,
                    "hole_penalty": 0.4,
                    "survival_reward": 0.05,
                },
                env_reward_overrides={
                    "combo_reward": 0.0,
                    "attack_reward": 0.0,
                    "hard_drop_reward": 0.01,
                },
                board_setup="modern_easy_line_clear",
                max_moves=60,
                end_on_line_clear=True,
                max_height=8,
            ),
        ),
        (
            "restricted_stack_building",
            3_500_000,
            CurriculumStage(
                name="restricted_stack_building",
                max_global_steps=None,
                description="Focus on flat stacking with a limited height cap and reduced piece variety.",
                env_kwargs={
                    "max_steps": 140,
                    "allowed_pieces": ("I", "J", "L", "O", "T"),
                    "das_frames": 12,
                    "arr_frames": 4,
                },
                agent_reward_overrides={
                    "line_clear_bonus": 2.5,
                    "hole_penalty": 0.8,
                    "bumpiness_penalty": 0.05,
                    "height_penalty": 0.04,
                },
                env_reward_overrides={
                    "combo_reward": 0.02,
                    "attack_reward": 0.01,
                },
                board_setup="modern_stack_intro",
                max_moves=140,
                max_height=12,
            ),
        ),
        (
            "height_constrained_survival",
            5_000_000,
            CurriculumStage(
                name="height_constrained_survival",
                max_global_steps=None,
                description="Increase variety while keeping a reasonable height ceiling to encourage stability.",
                env_kwargs={
                    "max_steps": 220,
                    "allowed_pieces": None,
                    "das_frames": 10,
                    "arr_frames": 3,
                },
                agent_reward_overrides={
                    "line_clear_bonus": 1.5,
                    "hole_penalty": 1.0,
                    "height_penalty": 0.06,
                    "bumpiness_penalty": 0.05,
                },
                env_reward_overrides={
                    "combo_reward": 0.04,
                    "attack_reward": 0.02,
                },
                board_setup="modern_column_focus",
                max_moves=220,
                max_height=16,
            ),
        ),
    ]

    previous_cap = 0
    for _, target, template in stage_defs:
        cap = _cap(target)
        if cap is not None and cap <= previous_cap:
            continue
        stage = replace(template, max_global_steps=cap)
        stages.append(stage)
        if cap is not None:
            previous_cap = cap

    stages.append(
        CurriculumStage(
            name="full_gameplay",
            max_global_steps=None,
            description="Full-speed modern Tetris with the complete bag and standard speeds.",
            env_kwargs={},
        )
    )
    return CurriculumManager(stages)


class CurriculumEpisodeWrapper(_ProxyWrapperMixin, Wrapper):
    """Limits episode length / termination and injects curriculum board setups."""

    def __init__(self, env, stage: CurriculumStage, presets: Optional[BoardPresetLibrary] = None):
        super().__init__(env)
        self.stage = stage
        self._moves = 0
        self._preset_library = presets

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._moves = 0
        if self.stage.board_setup:
            custom_obs = _apply_board_setup(self.env, self.stage.board_setup, self._preset_library)
            if custom_obs is not None:
                obs = custom_obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._moves += 1
        info = dict(info)
        if self.stage.end_on_line_clear and int(info.get("lines_cleared", 0)) > 0:
            truncated = True
            info["curriculum_line_cleared"] = True
        if self.stage.max_moves is not None and self._moves >= self.stage.max_moves:
            truncated = True
            info["curriculum_max_moves"] = True
        if self.stage.max_height is not None:
            board = np.asarray(obs.get("board"))
            if board is not None and _max_column_height(board) >= self.stage.max_height:
                truncated = True
                info["curriculum_max_height"] = True
        return obs, reward, terminated, truncated, info


def apply_overrides(config, overrides: Dict[str, float]):
    """Return a shallow copy of `config` with dataclass attributes overridden."""
    new_cfg = replace(config)
    for key, value in overrides.items():
        if not hasattr(new_cfg, key):
            raise ValueError(f"Unknown reward config field '{key}'.")
        setattr(new_cfg, key, float(value))
    return new_cfg


def _apply_board_setup(env, preset: str, preset_library: Optional[BoardPresetLibrary]):
    preset_lower = preset.lower()
    if preset_lower == "nes_easy_line_clear":
        return _nes_easy_line_clear(env)
    if preset_lower == "modern_easy_line_clear":
        return _modern_easy_line_clear(env)
    if preset_lower == "modern_stack_intro":
        return _modern_stack_intro(env)
    if preset_lower == "modern_column_focus":
        return _modern_column_focus(env)
    if preset_library:
        key = preset
        if preset_lower.startswith("preset:"):
            key = preset.split(":", 1)[1]
        preset_obj = _lookup_preset(preset_library, key)
        if preset_obj is not None:
            return apply_board_preset(env, preset_obj)
    return None


def _max_column_height(board: np.ndarray) -> int:
    if board is None:
        return 0
    binary = np.where(board > 0, 1, 0)
    rows, cols = binary.shape
    max_height = 0
    for col in range(cols):
        filled = np.where(binary[:, col] > 0)[0]
        if filled.size:
            height = rows - filled[0]
            if height > max_height:
                max_height = height
    return max_height


def _lookup_preset(presets: BoardPresetLibrary, key: str) -> Optional[BoardPreset]:
    if key in presets:
        return presets[key]
    lowered = key.lower()
    for name, preset in presets.items():
        if name.lower() == lowered:
            return preset
    return None


def _nes_easy_line_clear(env):
    from .nes_tetris_env import NesTetrisEnv

    base = env.unwrapped
    if not isinstance(base, NesTetrisEnv):
        return None
    board = utils.create_board()
    target_row = utils.TOTAL_ROWS - 1
    board[target_row] = np.ones_like(board[target_row])
    gap_start = 3
    gap_width = 4
    board[target_row, gap_start : gap_start + gap_width] = 0

    piece = utils.PieceState(
        piece_id=utils.NAME_TO_ID["I"],
        rotation=0,
        row=utils.NES_SPAWN_ROW,
        col=gap_start,
    )
    base._board = board
    base._current = piece
    base._gravity_timer = 0
    base._steps = 0
    base._consecutive_rotations = 0
    base._top_out = False
    return base._observe()


def _modern_easy_line_clear(env):
    from .modern_tetris_env import ModernTetrisEnv

    base = env.unwrapped
    if not isinstance(base, ModernTetrisEnv):
        return None
    rules = base._rules
    board = utils.create_board()
    target_row = utils.TOTAL_ROWS - 1
    board[target_row] = np.ones_like(board[target_row])
    board[target_row, 3:7] = 0
    piece = utils.spawn_piece(utils.NAME_TO_ID["I"])
    piece = piece.moved(d_col=1)
    _reset_modern_state(rules, board, piece)
    return rules._observe()


def _modern_stack_intro(env):
    from .modern_tetris_env import ModernTetrisEnv

    base = env.unwrapped
    if not isinstance(base, ModernTetrisEnv):
        return None
    rules = base._rules
    board = utils.create_board()
    for row in range(utils.TOTAL_ROWS - 4, utils.TOTAL_ROWS - 1):
        board[row, :3] = 2
    piece = utils.spawn_piece(utils.NAME_TO_ID["L"])
    _reset_modern_state(rules, board, piece)
    return rules._observe()


def _modern_column_focus(env):
    from .modern_tetris_env import ModernTetrisEnv

    base = env.unwrapped
    if not isinstance(base, ModernTetrisEnv):
        return None
    rules = base._rules
    board = utils.create_board()
    for row in range(utils.TOTAL_ROWS - 8, utils.TOTAL_ROWS - 2):
        board[row, 4:6] = 3
    piece = utils.spawn_piece(utils.NAME_TO_ID["T"])
    _reset_modern_state(rules, board, piece)
    return rules._observe()


def _reset_modern_state(rules, board: np.ndarray, piece: utils.PieceState) -> None:
    rules._board = board
    rules._current = piece
    rules._pending_garbage.clear()
    rules._queue.clear()
    rules._fill_queue()
    rules._hold_piece = None
    rules._hold_available = True
    rules._combo = 0
    rules._back_to_back = False
    rules._score = 0
    rules._steps = 0
    rules._lines = 0
    rules._level = 1
    rules._line_clear_timer = 0
    rules._ground_frames = 0
    rules._manipulation_count = 0
    rules._rotation_streak = 0
    rules._top_out = False
