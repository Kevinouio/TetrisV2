from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import eval_rl
from tetris_v2.rl.evaluation import EpisodeMetrics, evaluate_gate, summarize_episodes


def test_episode_summary_and_strict_gate() -> None:
    episodes = [
        EpisodeMetrics(1, 10, 10, 1, 100.0, True, False),
        EpisodeMetrics(2, 11, 20, 2, 200.0, False, True),
        EpisodeMetrics(3, 12, 30, 3, 300.0, True, False),
        EpisodeMetrics(4, 13, 40, 4, 400.0, False, False),
    ]

    summary = summarize_episodes(episodes)
    assert summary["placements"] == {"min": 10, "p5": 11.5, "median": 25.0, "mean": 25.0}
    assert summary["lines"] == {"min": 1, "p5": 1.15, "median": 2.5, "mean": 2.5}
    assert summary["mean_return"] == 250.0
    assert summary["topout_rate"] == 0.5
    assert summary["truncation_rate"] == 0.25
    assert summary["illegal_actions"] == 0

    passing = evaluate_gate(episodes, min_placements=10, min_lines=1)
    assert passing["passed"]
    assert passing["failed_episodes"] == []

    failing = evaluate_gate(episodes, min_placements=20, min_lines=2)
    assert not failing["passed"]
    assert failing["failed_episodes"] == [1]

    illegal = evaluate_gate(
        [EpisodeMetrics(1, 20, 20, 2, 200.0, False, True, illegal_actions=1)],
        min_placements=20,
        min_lines=2,
    )
    assert not illegal["passed"]
    assert illegal["failed_episodes"] == [1]


class FakePolicy:
    obs_dim = 2
    action_dim = 3

    def act(self, *_: object, **__: object) -> int:
        return 0


class FakeIllegalPolicy(FakePolicy):
    def act(self, *_: object, **__: object) -> int:
        return self.action_dim


class FakeEnv:
    action_space = SimpleNamespace(n=3)
    observation_space = SimpleNamespace(shape=(2,))

    def __init__(self, **_: object) -> None:
        self.closed = False
        self.steps = 0
        self.seed = 0

    def reset(self, *, seed: int):
        self.seed = seed
        self.steps = 0
        return np.zeros(2, dtype=np.float32), {
            "action_mask": np.ones(3, dtype=np.float32),
            "lines": 0,
            "placements": 0,
        }

    def step(self, _: int):
        self.steps += 1
        info = {
            "action_mask": np.ones(3, dtype=np.float32),
            "success": True,
            "selected_is_hold": False,
            "lines": 0,
            "placements": self.steps,
            "top_out": False,
        }
        if self.seed == 10:
            if self.steps == 2:
                info["selected_is_hold"] = True
            terminated = self.steps == 3
            if terminated:
                info["lines"] = 1
                info["top_out"] = True
            return np.zeros(2, dtype=np.float32), 50.0, terminated, False, info
        return np.zeros(2, dtype=np.float32), 0.0, False, True, info

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("algo", ["ppo", "dqn", "flow_dqn"])
def test_eval_json_report_and_failure_exit(
    algo: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loaded_algorithms = []
    monkeypatch.setattr(
        eval_rl,
        "load_policy",
        lambda requested, *_args, **_kwargs: loaded_algorithms.append(requested) or FakePolicy(),
    )
    monkeypatch.setattr(eval_rl, "CCTetrisEnv", FakeEnv)

    status = eval_rl.main(
        [
            "checkpoint.pt",
            "--algo",
            algo,
            "--episodes",
            "2",
            "--seed",
            "10",
            "--min-placements",
            "3",
            "--min-lines",
            "1",
            "--json",
        ]
    )

    assert status == 1
    assert loaded_algorithms == [algo]
    report = json.loads(capsys.readouterr().out)
    assert report["algorithm"] == algo
    assert report["episodes"] == [
        {
            "episode": 1,
            "seed": 10,
            "placements": 3,
            "lines": 1,
            "return": 150.0,
            "topout": True,
            "truncated": False,
            "illegal_actions": 0,
        },
        {
            "episode": 2,
            "seed": 11,
            "placements": 1,
            "lines": 0,
            "return": 0.0,
            "topout": False,
            "truncated": True,
            "illegal_actions": 0,
        },
    ]
    assert report["summary"]["placements"]["min"] == 1
    assert report["summary"]["topout_rate"] == 0.5
    assert report["summary"]["illegal_actions"] == 0
    assert report["gate"]["passed"] is False
    assert report["gate"]["failed_episodes"] == [2]


def test_eval_writes_json_file_and_passes_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(eval_rl, "load_policy", lambda *_args, **_kwargs: FakePolicy())
    monkeypatch.setattr(eval_rl, "CCTetrisEnv", FakeEnv)
    output = tmp_path / "evaluation.json"

    status = eval_rl.main(
        [
            "checkpoint.pt",
            "--algo",
            "dqn",
            "--episodes",
            "1",
            "--seed",
            "10",
            "--min-placements",
            "3",
            "--min-lines",
            "1",
            "--json-output",
            str(output),
        ]
    )

    assert status == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["gate"]["passed"] is True
    assert report["summary"]["lines"]["min"] == 1


def test_eval_records_illegal_action_and_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(eval_rl, "load_policy", lambda *_args, **_kwargs: FakeIllegalPolicy())
    monkeypatch.setattr(eval_rl, "CCTetrisEnv", FakeEnv)

    status = eval_rl.main(
        ["checkpoint.pt", "--algo", "flow_dqn", "--episodes", "1", "--json"]
    )

    assert status == 1
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == 2
    assert report["episodes"][0]["illegal_actions"] == 1
    assert report["summary"]["illegal_actions"] == 1
    assert report["gate"]["failed_episodes"] == [1]
