"""Utilities shared across single-agent trainers."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from tetris_v2.envs.wrappers import AgentRewardConfig, EnvironmentRewardConfig


def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
    """Linearly interpolate between start/end over `duration` steps."""
    if duration <= 0:
        return end
    clamped = min(max(step, 0), duration)
    mix = clamped / float(duration)
    return start + mix * (end - start)


def parse_key_value_overrides(pairs: Optional[Iterable[str]]) -> Dict[str, float]:
    """Parse strings of the form key=value into a float dictionary."""
    overrides: Dict[str, float] = {}
    if not pairs:
        return overrides
    for raw in pairs:
        if "=" not in raw:
            raise ValueError(f"Expected KEY=VALUE format, got '{raw}'.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{raw}': key cannot be empty.")
        try:
            overrides[key] = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value in override '{raw}'.") from exc
    return overrides


def build_agent_reward_config(
    overrides: Optional[Iterable[str]] = None,
) -> AgentRewardConfig:
    """Return an AgentRewardConfig with optional overrides."""
    config = AgentRewardConfig()
    mapping = parse_key_value_overrides(overrides)
    for key, value in mapping.items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown AgentRewardConfig field '{key}'.")
        setattr(config, key, float(value))
    return config


def build_environment_reward_config(
    overrides: Optional[Iterable[str]] = None,
) -> EnvironmentRewardConfig:
    """Return an EnvironmentRewardConfig with optional overrides."""
    config = EnvironmentRewardConfig()
    mapping = parse_key_value_overrides(overrides)
    for key, value in mapping.items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown EnvironmentRewardConfig field '{key}'.")
        setattr(config, key, float(value))
    return config


def build_advanced_reward_config(
    overrides: Optional[Iterable[str]] = None,
) -> AgentRewardConfig:
    """Backward compatible alias for build_agent_reward_config."""
    return build_agent_reward_config(overrides)
