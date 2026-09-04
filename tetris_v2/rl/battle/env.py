"""Canonical player-perspective joint environment for native Tetris battles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
from gymnasium import spaces

from tetris_v2.rl.actions import PLACEMENT_ACTION_DIM
from tetris_v2.rl.battle.config import BattleRewardConfig, BattleRulesConfig
from tetris_v2.rl.battle.reward import REWARD_COMPONENT_NAMES, compute_battle_rewards
from tetris_v2.rl.battle.runtime import BattleRuntime
from tetris_v2.rl.battle.stats import (
    BattleStats,
    PlayerBattleStats,
    PlayerStepStats,
    compute_board_stats,
)


OWN_OBS_SLICE = slice(0, 254)
OPP_BOARD_SLICE = slice(254, 454)
BATTLE_FEATURE_SLICE = slice(454, 470)
BATTLE_OBSERVATION_DIM = BATTLE_FEATURE_SLICE.stop
BATTLE_OBSERVATION_SCHEMA = "tetris_v2_battle_470_v1"

BATTLE_FEATURE_NAMES = (
    "own_incoming_garbage",
    "own_next_garbage_delay",
    "opponent_incoming_garbage",
    "opponent_next_garbage_delay",
    "own_aggregate_height",
    "own_max_height",
    "own_holes",
    "own_bumpiness",
    "own_wells",
    "opponent_aggregate_height",
    "opponent_max_height",
    "opponent_holes",
    "opponent_bumpiness",
    "opponent_wells",
    "height_advantage",
    "hole_advantage",
)


class BattleRuntimeLike(Protocol):
    def action_dim(self) -> int: ...
    def reset(self, seed: int) -> None: ...
    def observation(self, player: int) -> np.ndarray: ...
    def decision_mask(self, player: int) -> np.ndarray: ...
    def board(self, player: int) -> np.ndarray: ...
    def meta(self) -> dict[str, Any]: ...
    def step(self, actions: tuple[int, int]) -> dict[str, Any]: ...
    def enqueue_garbage(self, player: int, holes: list[int], delay: int = 0) -> bool: ...
    def bot_choose(self, player: int, think_ms: int = 0) -> dict[str, Any]: ...
    def close(self) -> None: ...


def _public_features(
    players: tuple[PlayerBattleStats, PlayerBattleStats],
    perspective: int,
    rules: BattleRulesConfig,
) -> np.ndarray:
    own = players[perspective]
    opponent = players[1 - perspective]
    own_board = own.board
    opponent_board = opponent.board
    delay_scale = max(1, rules.garbage_delay)
    values = np.asarray(
        [
            own.incoming_garbage / 40.0,
            max(0, own.next_garbage_delay) / delay_scale,
            opponent.incoming_garbage / 40.0,
            max(0, opponent.next_garbage_delay) / delay_scale,
            own_board.aggregate_height / 200.0,
            own_board.max_height / 20.0,
            own_board.holes / 200.0,
            own_board.bumpiness / 180.0,
            own_board.wells / 420.0,
            opponent_board.aggregate_height / 200.0,
            opponent_board.max_height / 20.0,
            opponent_board.holes / 200.0,
            opponent_board.bumpiness / 180.0,
            opponent_board.wells / 420.0,
            (opponent_board.max_height - own_board.max_height + 20.0) / 40.0,
            (opponent_board.holes - own_board.holes + 200.0) / 400.0,
        ],
        dtype=np.float32,
    )
    return np.clip(values, 0.0, 1.0)


class BattleEnv:
    """Two-player joint API over an atomic native battle coordinator.

    ``reset`` returns ``(observations, masks, info)``. ``step`` accepts both
    pre-state decisions and returns ``(observations, rewards, terminated,
    truncated, info)``. Each pair is ordered by physical seat, while every
    observation is canonical from that seat's own perspective.
    """

    def __init__(
        self,
        *,
        seed: int = 1,
        lib_path: Optional[Path] = None,
        rules: BattleRulesConfig | None = None,
        reward_config: BattleRewardConfig | None = None,
        runtime: BattleRuntimeLike | None = None,
    ):
        self.rules = rules or BattleRulesConfig()
        self.reward_config = reward_config or BattleRewardConfig()
        self.seed_value = int(seed)
        self._seed_rng = np.random.default_rng(self.seed_value)
        self.runtime: BattleRuntimeLike = runtime or BattleRuntime(
            lib_path=lib_path,
            seed=self.seed_value,
            rules=self.rules,
        )
        action_dim = int(self.runtime.action_dim())
        if action_dim != PLACEMENT_ACTION_DIM:
            self.runtime.close()
            raise RuntimeError(
                f"battle action schema mismatch: Python={PLACEMENT_ACTION_DIM}, native={action_dim}"
            )
        self.action_space = spaces.Tuple((spaces.Discrete(action_dim), spaces.Discrete(action_dim)))
        player_observation = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(BATTLE_OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Tuple((player_observation, player_observation))
        self.stats = BattleStats()
        self._masks = (
            np.zeros(action_dim, dtype=np.float32),
            np.zeros(action_dim, dtype=np.float32),
        )
        self._done = False

    def _boards(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.runtime.board(0), dtype=np.uint8).reshape(20, 10),
            np.asarray(self.runtime.board(1), dtype=np.uint8).reshape(20, 10),
        )

    def _refresh_public_state(self, meta: dict[str, Any]) -> None:
        boards = self._boards()
        player_meta = meta["players"]
        for player in (0, 1):
            stats = self.stats.players[player]
            native = player_meta[player]
            stats.board = compute_board_stats(boards[player])
            for name in (
                "placements",
                "lines_cleared",
                "attack_generated",
                "garbage_cancelled",
                "garbage_sent",
                "garbage_received",
                "garbage_applied",
            ):
                if name in native:
                    setattr(stats, name, int(native[name]))
            if "score" in native:
                stats.score = float(native["score"])
            stats.incoming_garbage = int(native["incoming_garbage"])
            stats.next_garbage_delay = int(native["next_garbage_delay"])
            stats.top_out = bool(native["top_out"])

    def _observations(self) -> tuple[np.ndarray, np.ndarray]:
        observations = []
        for player in (0, 1):
            native = np.asarray(self.runtime.observation(player), dtype=np.float32)
            if native.ndim != 1 or native.size != BATTLE_OBSERVATION_DIM:
                raise RuntimeError(
                    f"native battle observation must expose exactly {BATTLE_OBSERVATION_DIM} fields"
                )
            observations.append(native.copy())
        return observations[0], observations[1]

    def _action_masks(self) -> tuple[np.ndarray, np.ndarray]:
        masks = tuple(
            np.asarray(self.runtime.decision_mask(player), dtype=np.float32).copy()
            for player in (0, 1)
        )
        expected = (PLACEMENT_ACTION_DIM,)
        if masks[0].shape != expected or masks[1].shape != expected:
            raise RuntimeError("native battle action mask has the wrong shape")
        return masks[0], masks[1]

    def _info(
        self,
        reward_components: tuple[dict[str, float], dict[str, float]],
    ) -> dict[str, Any]:
        return {
            "seed": int(self.seed_value),
            "observation_schema": BATTLE_OBSERVATION_SCHEMA,
            "step": int(self.stats.steps),
            "winner": self.stats.winner,
            "result": self.stats.result,
            "terminated": bool(self.stats.terminated),
            "truncated": bool(self.stats.truncated),
            "action_masks": self._masks,
            "legal_action_counts": tuple(int(np.count_nonzero(mask > 0.5)) for mask in self._masks),
            "players": tuple(player.to_dict() for player in self.stats.players),
            "step_stats": tuple(player.last_step.to_dict() for player in self.stats.players),
            "board_stats": tuple(player.board.to_dict() for player in self.stats.players),
            "reward_components": reward_components,
        }

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self.seed_value = int(seed)
            self._seed_rng = np.random.default_rng(self.seed_value)
        else:
            self.seed_value = int(self._seed_rng.integers(0, 2**32, dtype=np.uint32))
        self.runtime.reset(self.seed_value)
        self.stats = BattleStats()
        self._refresh_public_state(self.runtime.meta())
        self._masks = self._action_masks()
        self._done = False
        observations = self._observations()
        empty_components = tuple(
            {name: 0.0 for name in REWARD_COMPONENT_NAMES} for _ in (0, 1)
        )
        return observations, self._masks, self._info(empty_components)  # type: ignore[arg-type]

    def step(self, actions: tuple[int, int]):
        if self._done:
            raise RuntimeError("battle match is finished; call reset before stepping again")
        if len(actions) != 2:
            raise ValueError("battle step requires one action for each player")
        selected = (int(actions[0]), int(actions[1]))
        for player, action in enumerate(selected):
            if action < 0 or action >= PLACEMENT_ACTION_DIM or self._masks[player][action] <= 0.5:
                raise ValueError(f"illegal battle action for player {player}: {action}")

        previous_boards = (self.stats.players[0].board, self.stats.players[1].board)
        outcome = self.runtime.step(selected)
        if not outcome["success"]:
            raise RuntimeError("native battle runtime rejected a prevalidated joint step")

        player_outcomes = outcome["players"]
        events = tuple(
            PlayerStepStats(
                raw_reward=float(player_outcomes[player].get("raw_reward", 0.0)),
                lines_cleared=int(player_outcomes[player]["lines_cleared"]),
                attack_generated=int(player_outcomes[player]["attack_generated"]),
                garbage_cancelled=int(player_outcomes[player]["garbage_cancelled"]),
                garbage_sent=int(player_outcomes[player]["garbage_sent"]),
                garbage_received=int(player_outcomes[player]["garbage_received"]),
                garbage_applied=int(player_outcomes[player]["garbage_applied"]),
                incoming_garbage=int(player_outcomes[player]["incoming_garbage"]),
                next_garbage_delay=int(player_outcomes[player]["next_garbage_delay"]),
                top_out=bool(player_outcomes[player]["top_out"]),
            )
            for player in (0, 1)
        )
        boards = self._boards()
        board_stats = (compute_board_stats(boards[0]), compute_board_stats(boards[1]))
        for player in (0, 1):
            self.stats.players[player].record(events[player], board_stats[player])

        self.stats.steps += 1
        self.stats.winner = outcome.get("winner")
        native_terminated = bool(outcome.get("terminated", False))
        top_out_terminal = any(event.top_out for event in events)
        reached_limit = self.stats.steps >= self.rules.max_steps and not top_out_terminal
        self.stats.terminated = top_out_terminal or (native_terminated and not reached_limit)
        self.stats.truncated = reached_limit
        self._done = self.stats.terminated or self.stats.truncated

        reward = compute_battle_rewards(
            previous_boards,
            board_stats,
            events,
            winner=self.stats.winner,
            terminated=self.stats.terminated,
            config=self.reward_config,
        )
        self._masks = (
            (
                np.zeros(PLACEMENT_ACTION_DIM, dtype=np.float32),
                np.zeros(PLACEMENT_ACTION_DIM, dtype=np.float32),
            )
            if self.stats.terminated
            else self._action_masks()
        )
        observations = self._observations()
        return (
            observations,
            reward.rewards,
            self.stats.terminated,
            self.stats.truncated,
            self._info(reward.components),
        )

    def enqueue_garbage(
        self,
        player: int,
        holes: list[int] | tuple[int, ...],
        delay: int = 0,
    ):
        """Queue deterministic scripted pressure and refresh every public view."""

        if self._done:
            raise RuntimeError("cannot enqueue garbage after the battle has finished")
        seat = int(player)
        columns = [int(hole) for hole in holes]
        if seat not in (0, 1):
            raise ValueError("battle player must be 0 or 1")
        if not self.runtime.enqueue_garbage(seat, columns, int(delay)):
            raise ValueError("native battle runtime rejected scripted garbage")
        self._refresh_public_state(self.runtime.meta())
        self._masks = self._action_masks()
        observations = self._observations()
        empty_components = tuple(
            {name: 0.0 for name in REWARD_COMPONENT_NAMES} for _ in (0, 1)
        )
        return observations, self._masks, self._info(empty_components)  # type: ignore[arg-type]

    def close(self) -> None:
        self.runtime.close()


__all__ = [
    "BATTLE_FEATURE_NAMES",
    "BATTLE_FEATURE_SLICE",
    "BATTLE_OBSERVATION_DIM",
    "BATTLE_OBSERVATION_SCHEMA",
    "BattleEnv",
    "BattleRuntimeLike",
    "OPP_BOARD_SLICE",
    "OWN_OBS_SLICE",
]
