"""Fixed-evaluation curriculum state for Battle-DQN self-play."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PromotionRequirement:
    opponent: str
    min_win_rate: float
    min_matches: int = 100
    max_illegal_actions: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_win_rate <= 1.0:
            raise ValueError("min_win_rate must be in [0, 1]")
        if self.min_matches <= 0:
            raise ValueError("min_matches must be positive")


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    opponent_mix: Mapping[str, float]
    promotion: tuple[PromotionRequirement, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name cannot be empty")
        if not self.opponent_mix:
            raise ValueError("opponent_mix cannot be empty")
        if any(weight < 0.0 for weight in self.opponent_mix.values()):
            raise ValueError("opponent weights cannot be negative")
        total = sum(self.opponent_mix.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"opponent_mix must sum to 1.0, got {total}")


def default_curriculum() -> tuple[CurriculumStage, ...]:
    """Four stages with conservative development gates and a 20/30/50 mix."""

    return (
        CurriculumStage(
            name="random",
            opponent_mix={"random": 1.0},
            promotion=(PromotionRequirement("random", 0.90),),
        ),
        CurriculumStage(
            name="heuristic",
            opponent_mix={"random": 0.30, "cold_clear": 0.70},
            promotion=(
                PromotionRequirement("random", 0.95),
                PromotionRequirement("cold_clear", 0.55),
            ),
        ),
        CurriculumStage(
            name="frozen_self_play",
            opponent_mix={
                "random": 0.10,
                "cold_clear": 0.20,
                "frozen": 0.60,
                "current": 0.10,
            },
            promotion=(
                PromotionRequirement("random", 0.95),
                PromotionRequirement("cold_clear", 0.60),
                PromotionRequirement("frozen", 0.52),
            ),
        ),
        CurriculumStage(
            name="mixed",
            opponent_mix={
                "random": 0.20,
                "cold_clear": 0.30,
                "frozen": 0.50,
                "current": 0.0,
            },
        ),
    )


@dataclass
class CurriculumState:
    stages: tuple[CurriculumStage, ...] = field(default_factory=default_curriculum)
    stage_index: int = 0
    entered_at_step: int = 0
    promotion_history: list[dict[str, object]] = field(default_factory=list)

    @property
    def current(self) -> CurriculumStage:
        return self.stages[self.stage_index]

    @property
    def complete(self) -> bool:
        return self.stage_index == len(self.stages) - 1

    def evaluate(
        self,
        reports: Mapping[str, Mapping[str, float | int]],
    ) -> tuple[bool, list[str]]:
        """Check the current gate using fixed-seed evaluator summaries."""

        failures: list[str] = []
        for requirement in self.current.promotion:
            report = reports.get(requirement.opponent)
            if report is None:
                failures.append(f"missing report for {requirement.opponent}")
                continue
            matches = int(report.get("match_count", 0))
            win_rate = float(report.get("win_rate", 0.0))
            illegal = report.get("illegal_action_count", 0)
            if isinstance(illegal, Sequence) and not isinstance(illegal, (str, bytes)):
                illegal_count = sum(int(value) for value in illegal)
            else:
                illegal_count = int(illegal)
            if matches < requirement.min_matches:
                failures.append(
                    f"{requirement.opponent}: matches {matches} < {requirement.min_matches}"
                )
            if win_rate < requirement.min_win_rate:
                failures.append(
                    f"{requirement.opponent}: win_rate {win_rate:.3f} < "
                    f"{requirement.min_win_rate:.3f}"
                )
            if illegal_count > requirement.max_illegal_actions:
                failures.append(
                    f"{requirement.opponent}: illegal_actions {illegal_count} > "
                    f"{requirement.max_illegal_actions}"
                )
        return not failures, failures

    def maybe_promote(
        self,
        reports: Mapping[str, Mapping[str, float | int]],
        *,
        global_step: int,
    ) -> bool:
        if self.complete:
            return False
        passed, failures = self.evaluate(reports)
        self.promotion_history.append(
            {
                "stage": self.current.name,
                "global_step": int(global_step),
                "passed": passed,
                "failures": failures,
                "reports": {name: dict(report) for name, report in reports.items()},
            }
        )
        if passed:
            self.stage_index += 1
            self.entered_at_step = int(global_step)
        return passed

    def state_dict(self) -> dict[str, object]:
        return {
            "stages": [
                {
                    "name": stage.name,
                    "opponent_mix": dict(stage.opponent_mix),
                    "promotion": [asdict(requirement) for requirement in stage.promotion],
                }
                for stage in self.stages
            ],
            "stage_index": self.stage_index,
            "entered_at_step": self.entered_at_step,
            "promotion_history": list(self.promotion_history),
        }

    @classmethod
    def from_state_dict(cls, value: Mapping[str, object]) -> "CurriculumState":
        raw_stages = value.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("Curriculum checkpoint has no stages")
        stages = []
        for raw_stage in raw_stages:
            if not isinstance(raw_stage, Mapping):
                raise ValueError("Invalid curriculum stage")
            promotion = tuple(
                PromotionRequirement(**dict(item))
                for item in raw_stage.get("promotion", [])  # type: ignore[arg-type]
            )
            stages.append(
                CurriculumStage(
                    name=str(raw_stage["name"]),
                    opponent_mix={
                        str(name): float(weight)
                        for name, weight in dict(raw_stage["opponent_mix"]).items()  # type: ignore[arg-type]
                    },
                    promotion=promotion,
                )
            )
        state = cls(
            stages=tuple(stages),
            stage_index=int(value.get("stage_index", 0)),
            entered_at_step=int(value.get("entered_at_step", 0)),
            promotion_history=[dict(item) for item in value.get("promotion_history", [])],  # type: ignore[arg-type]
        )
        if not 0 <= state.stage_index < len(state.stages):
            raise ValueError("Curriculum stage index is out of range")
        return state


__all__ = [
    "CurriculumStage",
    "CurriculumState",
    "PromotionRequirement",
    "default_curriculum",
]
