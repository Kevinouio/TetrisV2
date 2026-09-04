"""Evaluation scheduling and reproducibility helpers for battle matches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from time import perf_counter
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from tetris_v2.rl.battle.metrics import BattleMatchMetrics, summarize_battle_matches
from tetris_v2.rl.battle.policies import BattlePolicy


@dataclass(frozen=True)
class ScheduledBattle:
    match: int
    seed: int
    learner_seat: int


@dataclass(frozen=True)
class ScriptedGarbagePressure:
    rows: int = 0
    interval: int = 0
    hole_column: int = 4
    delay: int = 0

    def __post_init__(self) -> None:
        if self.rows < 0 or self.interval < 0 or self.delay < 0:
            raise ValueError("Scripted garbage values cannot be negative")
        if not 0 <= self.hole_column < 10:
            raise ValueError("hole_column must be in [0, 9]")


def paired_seat_schedule(*, matches: int, seed: int) -> list[ScheduledBattle]:
    """Create fixed-seed pairs where the learner plays each seed from both seats."""

    if matches <= 0:
        raise ValueError("matches must be positive")
    if matches % 2:
        raise ValueError("paired seat evaluation requires an even match count")
    schedule = []
    for pair in range(matches // 2):
        match_seed = int(seed) + pair
        schedule.append(ScheduledBattle(2 * pair + 1, match_seed, 0))
        schedule.append(ScheduledBattle(2 * pair + 2, match_seed, 1))
    return schedule


def deterministic_match_fingerprint(match: BattleMatchMetrics) -> dict[str, object]:
    """Return fields that fixed seeds must reproduce, excluding wall-clock timing."""

    value = match.to_dict()
    value.pop("inference_ms", None)
    return value


def compare_repeated_matches(
    first: Sequence[BattleMatchMetrics],
    second: Sequence[BattleMatchMetrics],
) -> dict[str, object]:
    """Compare two evaluation passes without accepting timing as game state."""

    if len(first) != len(second):
        return {
            "passed": False,
            "mismatches": [f"match counts differ: {len(first)} != {len(second)}"],
        }
    mismatches = []
    for index, (left, right) in enumerate(zip(first, second, strict=True), start=1):
        if deterministic_match_fingerprint(left) != deterministic_match_fingerprint(right):
            mismatches.append(f"match {index} differs")
    return {"passed": not mismatches, "mismatches": mismatches}


def reward_component_totals(
    component_rows: Iterable[Mapping[str, float]],
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in component_rows:
        for name, value in row.items():
            totals[name] = totals.get(name, 0.0) + float(value)
    return totals


def _update_trace(
    digest: "hashlib._Hash",
    observations: Sequence[np.ndarray],
    actions: Sequence[int] | None,
    rewards: Sequence[float] | None,
    info: Mapping[str, object],
) -> None:
    for observation in observations:
        digest.update(np.ascontiguousarray(observation, dtype=np.float32).tobytes())
    if actions is not None:
        digest.update(np.asarray(actions, dtype=np.int64).tobytes())
    if rewards is not None:
        digest.update(np.asarray(rewards, dtype=np.float64).tobytes())
    deterministic_info = {
        name: info.get(name)
        for name in ("step", "winner", "result", "step_stats", "board_stats")
    }
    digest.update(
        json.dumps(deterministic_info, sort_keys=True, allow_nan=False).encode("utf-8")
    )


def run_battle_match(
    env: object,
    policies: tuple[BattlePolicy, BattlePolicy],
    *,
    scheduled: ScheduledBattle,
    opponent_type: str,
    opponent_checkpoint: str | None = None,
    pressure: ScriptedGarbagePressure | None = None,
) -> BattleMatchMetrics:
    """Run one exploration-free joint match and retain a deterministic trace hash."""

    observations, masks, info = env.reset(seed=scheduled.seed)  # type: ignore[attr-defined]
    observations = tuple(np.asarray(value, dtype=np.float32) for value in observations)
    masks = tuple(np.asarray(value, dtype=np.float32) for value in masks)
    for seat, policy in enumerate(policies):
        # Paired evaluation must keep each logical policy's RNG stream fixed
        # while swapping physical seats. Otherwise a random opponent receives
        # a different trajectory in the second half of a seed pair and seat
        # fairness is confounded with opponent sampling noise.
        logical_role = 0 if seat == scheduled.learner_seat else 1
        policy.reset(scheduled.seed * 4 + logical_role + 1)

    pressure_config = pressure or ScriptedGarbagePressure()
    if pressure_config.rows > 0 and hasattr(env, "enqueue_garbage"):
        observations, masks, info = env.enqueue_garbage(  # type: ignore[attr-defined]
            scheduled.learner_seat,
            [pressure_config.hole_column] * pressure_config.rows,
            delay=pressure_config.delay,
        )

    returns = [0.0, 0.0]
    illegal = [0, 0]
    inference_totals = [0.0, 0.0]
    inference_counts = [0, 0]
    height_totals = [0.0, 0.0]
    hole_totals = [0.0, 0.0]
    max_height_seen = [
        int(info["board_stats"][seat]["max_height"]) for seat in (0, 1)
    ]
    board_samples = 0
    component_rows: list[Mapping[str, float]] = []
    trace = hashlib.sha256()
    _update_trace(trace, observations, None, None, info)
    terminated = False
    truncated = False
    local_winner: int | None = None

    while not (terminated or truncated):
        actions = []
        for seat, policy in enumerate(policies):
            started = perf_counter()
            action = int(
                policy.select_action(
                    observations[seat],
                    masks[seat],
                    player=seat,
                    env=env,
                )
            )
            inference_totals[seat] += (perf_counter() - started) * 1000.0
            inference_counts[seat] += 1
            if not 0 <= action < masks[seat].size or masks[seat][action] <= 0.5:
                illegal[seat] += 1
            actions.append(action)

        if any(illegal):
            local_winner = (
                None if illegal[0] and illegal[1] else (1 if illegal[0] else 0)
            )
            terminated = True
            break

        observations, rewards, terminated, truncated, info = env.step(tuple(actions))  # type: ignore[attr-defined]
        observations = tuple(np.asarray(value, dtype=np.float32) for value in observations)
        masks = tuple(
            np.asarray(value, dtype=np.float32) for value in info["action_masks"]
        )
        for seat in (0, 1):
            returns[seat] += float(rewards[seat])
        learner_components = info.get("reward_components", ({}, {}))[
            scheduled.learner_seat
        ]
        component_rows.append(learner_components)
        for seat, board in enumerate(info["board_stats"]):
            height_totals[seat] += float(board["max_height"])
            hole_totals[seat] += float(board["holes"])
            max_height_seen[seat] = max(
                max_height_seen[seat], int(board["max_height"])
            )
        board_samples += 1
        _update_trace(trace, observations, actions, rewards, info)

        step = int(info.get("step", 0))
        if (
            pressure_config.rows > 0
            and pressure_config.interval > 0
            and step > 0
            and step % pressure_config.interval == 0
            and not (terminated or truncated)
        ):
            observations, masks, info = env.enqueue_garbage(  # type: ignore[attr-defined]
                scheduled.learner_seat,
                [pressure_config.hole_column] * pressure_config.rows,
                delay=pressure_config.delay,
            )

    winner = local_winner if any(illegal) else info.get("winner")
    winner = None if winner is None else int(winner)
    learner_result = (
        "draw"
        if winner is None
        else ("win" if winner == scheduled.learner_seat else "loss")
    )
    players = info["players"]
    sample_count = max(1, board_samples)
    reward_totals = reward_component_totals(component_rows)
    return BattleMatchMetrics(
        match=scheduled.match,
        seed=scheduled.seed,
        learner_seat=scheduled.learner_seat,
        opponent_type=opponent_type,
        opponent_checkpoint=opponent_checkpoint,
        result=learner_result,
        winner=winner,
        match_steps=int(info.get("step", 0)),
        placements=tuple(int(players[seat]["placements"]) for seat in (0, 1)),
        lines_cleared=tuple(int(players[seat]["lines_cleared"]) for seat in (0, 1)),
        attack_generated=tuple(
            int(players[seat]["attack_generated"]) for seat in (0, 1)
        ),
        attack_sent=tuple(int(players[seat]["garbage_sent"]) for seat in (0, 1)),
        garbage_received=tuple(
            int(players[seat]["garbage_received"]) for seat in (0, 1)
        ),
        garbage_cancelled=tuple(
            int(players[seat]["garbage_cancelled"]) for seat in (0, 1)
        ),
        top_out=tuple(bool(players[seat]["top_out"]) for seat in (0, 1)),
        average_height=tuple(value / sample_count for value in height_totals),
        maximum_height=(max_height_seen[0], max_height_seen[1]),
        average_holes=tuple(value / sample_count for value in hole_totals),
        illegal_actions=(illegal[0], illegal[1]),
        inference_ms=tuple(
            inference_totals[seat] / max(1, inference_counts[seat]) for seat in (0, 1)
        ),
        inference_decisions=(inference_counts[0], inference_counts[1]),
        board_samples=(board_samples, board_samples),
        returns=(returns[0], returns[1]),
        reward_components=reward_totals,
        trace_hash=trace.hexdigest(),
        garbage_applied=tuple(
            int(players[seat].get("garbage_applied", 0)) for seat in (0, 1)
        ),
    )


def evaluate_paired_battles(
    learner: BattlePolicy,
    opponent: BattlePolicy,
    *,
    env_factory: Callable[[], object],
    matches: int,
    seed: int,
    pressure: ScriptedGarbagePressure | None = None,
) -> tuple[list[BattleMatchMetrics], dict[str, object]]:
    """Evaluate both physical seats for every seed with no exploration changes."""

    schedule = paired_seat_schedule(matches=matches, seed=seed)
    env = env_factory()
    results = []
    try:
        for item in schedule:
            policies = (
                (learner, opponent) if item.learner_seat == 0 else (opponent, learner)
            )
            results.append(
                run_battle_match(
                    env,
                    policies,
                    scheduled=item,
                    opponent_type=opponent.kind,
                    opponent_checkpoint=getattr(opponent, "checkpoint", None),
                    pressure=pressure,
                )
            )
    finally:
        env.close()  # type: ignore[attr-defined]
    return results, summarize_battle_matches(results)


__all__ = [
    "ScheduledBattle",
    "ScriptedGarbagePressure",
    "compare_repeated_matches",
    "deterministic_match_fingerprint",
    "evaluate_paired_battles",
    "paired_seat_schedule",
    "reward_component_totals",
    "run_battle_match",
]
