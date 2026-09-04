"""Configurable zero-sum rewards for atomic battle steps."""

from __future__ import annotations

from dataclasses import dataclass

from tetris_v2.rl.battle.config import BattleRewardConfig
from tetris_v2.rl.battle.stats import BoardStats, PlayerStepStats, board_quality


REWARD_COMPONENT_NAMES = (
    "terminal_reward",
    "attack_reward",
    "cancellation_reward",
    "line_clear_reward",
    "board_quality_reward",
    "height_penalty",
    "hole_penalty",
    "garbage_penalty",
    "total_reward",
)


@dataclass(frozen=True)
class BattleReward:
    rewards: tuple[float, float]
    components: tuple[dict[str, float], dict[str, float]]


def compute_battle_rewards(
    previous_boards: tuple[BoardStats, BoardStats],
    current_boards: tuple[BoardStats, BoardStats],
    events: tuple[PlayerStepStats, PlayerStepStats],
    *,
    winner: int | None = None,
    terminated: bool = False,
    config: BattleRewardConfig | None = None,
) -> BattleReward:
    """Return exactly antisymmetric rewards and component logs.

    Board terms use changes from the previous state, so a policy cannot collect
    a persistent dense reward merely by maintaining the same board.
    """

    weights = config or BattleRewardConfig()
    event0, event1 = events
    old0, old1 = previous_boards
    new0, new1 = current_boards

    terminal_signal = 0.0
    if terminated and winner in (0, 1):
        terminal_signal = 1.0 if winner == 0 else -1.0

    quality_change0 = board_quality(new0) - board_quality(old0)
    quality_change1 = board_quality(new1) - board_quality(old1)
    height_change0 = (new0.max_height - old0.max_height) / 20.0
    height_change1 = (new1.max_height - old1.max_height) / 20.0
    hole_change0 = (new0.holes - old0.holes) / 200.0
    hole_change1 = (new1.holes - old1.holes) / 200.0

    player0 = {
        "terminal_reward": float(weights.terminal * terminal_signal),
        "attack_reward": float(weights.attack * (event0.garbage_sent - event1.garbage_sent)),
        "cancellation_reward": float(
            weights.cancellation * (event0.garbage_cancelled - event1.garbage_cancelled)
        ),
        "line_clear_reward": float(
            weights.line_clear * (event0.lines_cleared - event1.lines_cleared)
        ),
        "board_quality_reward": float(
            weights.board_quality * (quality_change0 - quality_change1)
        ),
        "height_penalty": float(weights.height * (height_change1 - height_change0)),
        "hole_penalty": float(weights.holes * (hole_change1 - hole_change0)),
        "garbage_penalty": float(
            weights.garbage * (event1.garbage_applied - event0.garbage_applied)
        ),
    }
    total = float(sum(player0.values()))
    player0["total_reward"] = total
    player1 = {name: -value for name, value in player0.items()}
    return BattleReward(rewards=(total, -total), components=(player0, player1))


__all__ = [
    "BattleReward",
    "BattleRewardConfig",
    "REWARD_COMPONENT_NAMES",
    "compute_battle_rewards",
]
