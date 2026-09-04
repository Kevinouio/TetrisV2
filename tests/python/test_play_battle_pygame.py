from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from scripts import play_battle_pygame as battle_view


class FakePolicy:
    def __init__(self, identifier: str, action: int):
        self.identifier = identifier
        self.kind = "fake"
        self.action = int(action)
        self.reset_seeds: list[int] = []
        self.calls: list[tuple[int, float]] = []

    def reset(self, seed: int) -> None:
        self.reset_seeds.append(int(seed))

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        player: int,
        env: Any,
    ) -> int:
        del env
        self.calls.append((int(player), float(observation[0])))
        assert action_mask[self.action] > 0.5
        return self.action


class FakeRuntime:
    def __init__(self) -> None:
        self.boards = np.full((2, 20, 10), 255, dtype=np.uint8)

    def board_piece_ids(self, player: int, *, include_active: bool) -> np.ndarray:
        board = self.boards[int(player)].copy()
        if include_active:
            board[0, 4] = int(player)
        return board


class FakeBattleEnv:
    def __init__(self) -> None:
        self.runtime = FakeRuntime()
        self.closed = False
        self.step_calls: list[tuple[int, int]] = []
        self.step_number = 0

    @staticmethod
    def _masks() -> tuple[np.ndarray, np.ndarray]:
        mask = np.zeros(3200, dtype=np.float32)
        mask[3] = 1.0
        mask[7] = 1.0
        return mask.copy(), mask.copy()

    def _info(self, *, done: bool = False) -> dict[str, Any]:
        return {
            "step": self.step_number,
            "winner": 0 if done else None,
            "action_masks": self._masks(),
            "players": (
                {
                    "placements": self.step_number,
                    "lines_cleared": 2,
                    "attack_generated": 1,
                    "garbage_sent": 1,
                    "incoming_garbage": 0,
                },
                {
                    "placements": self.step_number,
                    "lines_cleared": 1,
                    "attack_generated": 0,
                    "garbage_sent": 0,
                    "incoming_garbage": 2,
                },
            ),
            "step_stats": ({"garbage_sent": 1}, {"garbage_sent": 0}),
        }

    def reset(self, *, seed: int):
        self.seed = int(seed)
        self.step_number = 0
        observations = (
            np.full(470, 10.0, dtype=np.float32),
            np.full(470, 20.0, dtype=np.float32),
        )
        return observations, self._masks(), self._info()

    def step(self, actions: tuple[int, int]):
        self.step_calls.append(tuple(map(int, actions)))
        self.step_number += 1
        done = self.step_number >= 2
        observations = (
            np.full(470, 11.0, dtype=np.float32),
            np.full(470, 21.0, dtype=np.float32),
        )
        return observations, (0.5, -0.5), done, False, self._info(done=done)

    def close(self) -> None:
        self.closed = True


class FakeLoadedPolicy:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def act(self, observation: np.ndarray, **kwargs: Any) -> int:
        self.calls.append({"observation": observation.copy(), **kwargs})
        return 7


@pytest.fixture(scope="module", autouse=True)
def pygame_runtime():
    pygame.init()
    yield
    pygame.quit()


def test_parse_args_defaults_and_validation() -> None:
    args = battle_view.parse_args(["flow.pt"])
    assert args.checkpoint == Path("flow.pt")
    assert args.algo == "flow_dqn"
    assert args.opponent == "cold_clear"
    assert args.step_ms == 110
    assert args.max_steps is None
    assert args.auto_reset

    args = battle_view.parse_args(
        ["battle.pt", "--algo", "battle_dqn", "--no-auto-reset", "--step-ms", "0"]
    )
    assert args.algo == "battle_dqn"
    assert not args.auto_reset
    assert args.step_ms == 0

    with pytest.raises(SystemExit):
        battle_view.parse_args(["flow.pt", "--fps", "0"])


def test_single_player_adapter_uses_only_own_254_features_and_battle_mask() -> None:
    loaded = FakeLoadedPolicy()
    adapter = battle_view.SinglePlayerPolicyAdapter(loaded, "flow")
    observation = np.arange(470, dtype=np.float32)
    mask = np.zeros(3200, dtype=np.float32)
    mask[7] = 1.0
    action = adapter.select_action(observation, mask, player=1, env=object())
    assert action == 7
    call = loaded.calls[0]
    np.testing.assert_array_equal(call["observation"], observation[:254])
    np.testing.assert_array_equal(call["action_mask"], mask)
    assert call["deterministic"] is True


def test_session_steps_atomically_and_swaps_policy_seats() -> None:
    env = FakeBattleEnv()
    learner = FakePolicy("flow", 3)
    opponent = FakePolicy("cold_clear", 7)
    session = battle_view.BattleSession(env, learner, opponent, seed=9)

    session.step()
    assert env.step_calls == [(3, 7)]
    assert learner.calls[-1] == (0, 10.0)
    assert opponent.calls[-1] == (1, 20.0)
    assert session.last_rewards == (0.5, -0.5)

    session.swap_seats()
    assert session.learner_seat == 1
    session.step()
    assert env.step_calls[-1] == (7, 3)
    assert opponent.calls[-1] == (0, 10.0)
    assert learner.calls[-1] == (1, 20.0)

    session.next_seed()
    assert session.seed == 10
    assert learner.reset_seeds[-1] == 21
    assert opponent.reset_seeds[-1] == 22


def test_scene_draws_two_piece_id_boards_and_winner_overlay() -> None:
    env = FakeBattleEnv()
    session = battle_view.BattleSession(
        env,
        FakePolicy("flow_dqn", 3),
        FakePolicy("cold_clear", 7),
        seed=4,
    )
    scene = battle_view.draw_scene(
        session,
        cell=18,
        paused=False,
        step_ms=90,
        status="Running",
    )
    layout = battle_view.scene_layout(18)
    assert scene.get_size() == layout.size
    assert scene.get_at(layout.left_board.center) != pygame.Color(*battle_view.BG, 255)
    assert scene.get_at(layout.right_board.center) != pygame.Color(*battle_view.BG, 255)

    session.step()
    session.step()
    assert session.done
    finished = battle_view.draw_scene(
        session,
        cell=18,
        paused=False,
        step_ms=90,
        status="Finished",
    )
    assert finished.get_at((10, 200)) != scene.get_at((10, 200))


def test_action_names_include_hold_and_scene_present_letterboxes() -> None:
    assert battle_view.decode_action_name(None) == "waiting"
    assert battle_view.decode_action_name(1600).startswith("HOLD +")
    logical = pygame.Surface((800, 600))
    screen = pygame.Surface((1000, 500))
    fitted = battle_view.present(logical, screen)
    assert fitted.size == (667, 500)
    assert fitted.center == screen.get_rect().center


def test_headless_main_saves_frame_and_closes_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = FakeBattleEnv()
    learner = FakePolicy("flow_dqn", 3)
    opponent = FakePolicy("cold_clear", 7)
    monkeypatch.setattr(
        battle_view,
        "load_learner",
        lambda *args, **kwargs: battle_view.LoadedLearner(learner, {}),
    )
    monkeypatch.setattr(
        battle_view,
        "load_opponent",
        lambda *args, **kwargs: opponent,
    )
    monkeypatch.setattr(
        battle_view,
        "make_battle_env",
        lambda *args, **kwargs: env,
    )
    screenshot = tmp_path / "battle.png"
    result = battle_view.main(
        [
            "flow.pt",
            "--max-frames",
            "2",
            "--step-ms",
            "0",
            "--cell",
            "14",
            "--screenshot",
            str(screenshot),
        ]
    )
    assert result == 0
    assert env.closed
    assert env.step_calls
    assert screenshot.is_file()
    assert screenshot.stat().st_size > 0
