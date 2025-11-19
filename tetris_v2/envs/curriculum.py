"""Curriculum helpers for staged Tetris training."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import numpy as np
from gymnasium import Wrapper

from . import utils
from .state_presets import BoardPreset, BoardPresetLibrary, apply_board_preset


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
            name="line_clear_bootstrap",
            max_global_steps=_cap(1_000_000),
            description="Expose one-move line clears and terminate after success/failure.",
            agent_reward_overrides={
                "line_clear_bonus": 5.0,
                "idle_penalty": 0.2,
                "survival_reward": 0.01,
                "board_change_bonus": 0.5,
                "hole_penalty": 0.2,
                "bumpiness_penalty": 0.0,
                "excess_rotation_penalty": 0.5,
                "time_decay_penalty": 0.05,
            },
            env_reward_overrides={
                "combo_reward": 0.0,
                "attack_reward": 0.0,
                "pending_garbage_penalty": 0.0,
                "hard_drop_reward": 0.0,
            },
            env_kwargs={"max_steps": 50},
            max_moves=50,
            end_on_line_clear=True,
            board_setup="nes_easy_line_clear",
        ),
        CurriculumStage(
            name="short_survival",
            max_global_steps=_cap(1_400_000),
            description="Learn to clear at least one line before topping out.",
            agent_reward_overrides={
                "line_clear_bonus": 2.5,
                "idle_penalty": 0.1,
                "hole_penalty": 0.5,
                "height_penalty": 0.02,
                "bumpiness_penalty": 0.02,
                "survival_reward": 0.02,
            },
            env_reward_overrides={
                "combo_reward": 0.02,
                "attack_reward": 0.01,
                "pending_garbage_penalty": 0.005,
                "hard_drop_reward": 0.02,
            },
            env_kwargs={"max_steps": 120},
            max_moves=120,
        ),
        CurriculumStage(
            name="stability_building",
            max_global_steps=_cap(1_900_000),
            description="Encourage stable stacks and mid-length survival.",
            agent_reward_overrides={
                "line_clear_bonus": 1.5,
                "hole_penalty": 0.9,
                "height_penalty": 0.05,
                "bumpiness_penalty": 0.04,
                "height_drop_reward": 0.01,
                "bumpiness_drop_reward": 0.02,
            },
            env_reward_overrides={
                "combo_reward": 0.05,
                "attack_reward": 0.02,
                "pending_garbage_penalty": 0.015,
                "hard_drop_reward": 0.02,
            },
            env_kwargs={"max_steps": 250},
            max_moves=250,
        ),
        CurriculumStage(
            name="full_gameplay",
            max_global_steps=None,
            description="Run with near-final rules and minimal shaping.",
            agent_reward_overrides={
                "line_clear_bonus": 1.0,
                "idle_penalty": 0.05,
                "hole_penalty": 1.0,
                "height_penalty": 0.08,
                "bumpiness_penalty": 0.05,
                "survival_reward": 0.02,
            },
            env_reward_overrides={
                "combo_reward": 0.05,
                "attack_reward": 0.03,
                "pending_garbage_penalty": 0.02,
                "hard_drop_reward": 0.03,
            },
            env_kwargs={},
            max_moves=None,
        ),
    ]
    return CurriculumManager(stages)


class CurriculumEpisodeWrapper(Wrapper):
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
    if preset_library:
        key = preset
        if preset_lower.startswith("preset:"):
            key = preset.split(":", 1)[1]
        preset_obj = _lookup_preset(preset_library, key)
        if preset_obj is not None:
            return apply_board_preset(env, preset_obj)
    return None


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
