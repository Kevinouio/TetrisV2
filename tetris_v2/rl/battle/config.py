"""Configuration values for deterministic two-player battle matches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_ATTACK_TABLE = (0, 0, 1, 2, 4)


@dataclass(frozen=True)
class BattleRulesConfig:
    """Rules that affect native battle resolution.

    Advanced attack bonuses stay explicit and disabled until the native engine
    has a tested rule for each of them.
    """

    attack_table: tuple[int, ...] = DEFAULT_ATTACK_TABLE
    garbage_delay: int = 1
    max_steps: int = 500
    mirrored_piece_seeds: bool = True
    garbage_holes_per_row: int = 1
    combo_attack_enabled: bool = False
    back_to_back_attack_enabled: bool = False
    spin_attack_enabled: bool = False
    perfect_clear_attack_enabled: bool = False

    def __post_init__(self) -> None:
        table = tuple(int(value) for value in self.attack_table)
        if len(table) != 5 or any(value < 0 for value in table):
            raise ValueError("attack_table must contain exactly five non-negative entries")
        if self.garbage_delay < 0:
            raise ValueError("garbage_delay must be non-negative")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.garbage_holes_per_row != 1:
            raise ValueError("battle mode currently supports exactly one hole per garbage row")
        if any(
            (
                self.combo_attack_enabled,
                self.back_to_back_attack_enabled,
                self.spin_attack_enabled,
                self.perfect_clear_attack_enabled,
            )
        ):
            raise ValueError("advanced battle attack bonuses are not implemented")
        object.__setattr__(self, "attack_table", table)

    def attack_for_lines(self, lines_cleared: int) -> int:
        lines = max(0, min(int(lines_cleared), len(self.attack_table) - 1))
        return int(self.attack_table[lines])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BattleRewardConfig:
    """Weights for antisymmetric, terminal-heavy battle rewards."""

    terminal: float = 20.0
    attack: float = 0.05
    cancellation: float = 0.03
    line_clear: float = 0.01
    board_quality: float = 0.02
    height: float = 0.02
    holes: float = 0.03
    garbage: float = 0.04

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


__all__ = [
    "BattleRewardConfig",
    "BattleRulesConfig",
    "DEFAULT_ATTACK_TABLE",
]
