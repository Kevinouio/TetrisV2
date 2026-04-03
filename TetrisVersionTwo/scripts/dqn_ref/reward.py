from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

from .features import as_feature_array


@dataclass(frozen=True)
class RewardTerms:
    total: float
    game_over_term: float
    survival_term: float
    y_pos_term: float
    total_height_term: float
    lines_term: float
    holes_term: float
    bumpiness_term: float
    pillar_term: float
    high_placement_penalty_term: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "total": float(self.total),
            "game_over_term": float(self.game_over_term),
            "survival_term": float(self.survival_term),
            "y_pos_term": float(self.y_pos_term),
            "total_height_term": float(self.total_height_term),
            "lines_term": float(self.lines_term),
            "holes_term": float(self.holes_term),
            "bumpiness_term": float(self.bumpiness_term),
            "pillar_term": float(self.pillar_term),
            "high_placement_penalty_term": float(self.high_placement_penalty_term),
        }


class ReferenceReward:
    """Faithful port of Tetris-A.I Version2 Agent.calculate_rewards."""

    def __init__(self, weights: Mapping[str, float]):
        required = (
            "game_over",
            "survival_instinct",
            "total_height",
            "lines_removed",
            "holes",
            "bumpiness",
            "pillar",
            "y_pos_reward",
            "y_pos_punish",
        )
        missing = [k for k in required if k not in weights]
        if missing:
            raise ValueError(f"Missing reward weights: {missing}")
        self.weights = {k: float(v) for k, v in weights.items()}

    def compute(self, features: Sequence[float], finished: bool) -> RewardTerms:
        total_heights, bumpiness, lines_removed, holes, y_pos, _pillar = as_feature_array(features).tolist()
        calc_reward = 0.0

        board_half_full = (total_heights >= 110.0) or (total_heights >= 90.0 and bumpiness >= 10.0)

        if total_heights >= 140.0 or (total_heights >= 110.0 and bumpiness >= 12.0):
            hole_penalty = -2.743561101942274
        elif total_heights >= 90.0 or (total_heights >= 70.0 and bumpiness >= 9.0):
            hole_penalty = -4.743561101942274
        else:
            hole_penalty = self.weights["holes"]

        pillar_penalty = self.weights["pillar"] if (holes > 0.0 or board_half_full) else 0.0

        if total_heights <= 40.0:
            high_placement_penalty = (10.0 - y_pos) * 2.0
        elif total_heights <= 100.0:
            high_placement_penalty = 10.0 - y_pos
        else:
            high_placement_penalty = 0.0

        high_placement_penalty_term = -high_placement_penalty if y_pos >= 12.0 else 0.0
        calc_reward += high_placement_penalty_term

        game_over_term = -self.weights["game_over"] if finished else 0.0
        calc_reward += game_over_term

        survival_term = self.weights["survival_instinct"]
        calc_reward += survival_term

        if y_pos >= 9.0:
            y_pos_term = self.weights["y_pos_reward"]
        else:
            y_pos_term = -((10.0 - y_pos) * 0.2 - self.weights["y_pos_punish"])
        calc_reward += y_pos_term

        total_height_term = self.weights["total_height"] * total_heights
        calc_reward += total_height_term

        lines_term = (2.0 ** float(lines_removed)) * self.weights["lines_removed"]
        if int(lines_removed) == 4:
            lines_term += 5000.0
        calc_reward += lines_term

        holes_term = hole_penalty * holes
        calc_reward += holes_term

        bumpiness_term = self.weights["bumpiness"] * bumpiness
        calc_reward += bumpiness_term

        pillar_term = pillar_penalty
        calc_reward += pillar_term

        return RewardTerms(
            total=float(calc_reward),
            game_over_term=float(game_over_term),
            survival_term=float(survival_term),
            y_pos_term=float(y_pos_term),
            total_height_term=float(total_height_term),
            lines_term=float(lines_term),
            holes_term=float(holes_term),
            bumpiness_term=float(bumpiness_term),
            pillar_term=float(pillar_term),
            high_placement_penalty_term=float(high_placement_penalty_term),
        )

