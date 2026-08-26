from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import eval_rl, generate_expert_dataset, pretrain_rl_dqn_expert, train_rl_dqn_hybrid
from tetris_v2.rl.actions import PLACEMENT_ACTION_DIM
from tetris_v2.rl.dqn import train as dqn_train
from tetris_v2.rl.policy import load_policy
from tetris_v2.rl.ppo import train as ppo_train


def assert_checkpoint(path: Path, algo: str) -> None:
    assert path.is_file()
    policy = load_policy(algo, path)
    assert policy.obs_dim > 0
    assert policy.action_dim == PLACEMENT_ACTION_DIM
    for key in ("policy_loss", "value_loss", "entropy", "avg_loss"):
        if key in policy.metadata:
            assert math.isfinite(float(policy.metadata[key]))


def test_ppo_train_and_evaluate_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "ppo"
    assert (
        ppo_train.main(
            [
                "--total-timesteps",
                "32",
                "--num-envs",
                "2",
                "--n-steps",
                "16",
                "--minibatch-size",
                "16",
                "--update-epochs",
                "1",
                "--hidden-sizes",
                "32",
                "--max-steps",
                "4",
                "--seed",
                "13",
                "--log-interval",
                "1000000",
                "--eval-frequency",
                "1000000",
                "--checkpoint-frequency",
                "1000000",
                "--log-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    checkpoint = output_dir / "ppo_final.pt"
    assert_checkpoint(checkpoint, "ppo")
    assert (
        eval_rl.main(
            [
                str(checkpoint),
                "--algo",
                "ppo",
                "--episodes",
                "1",
                "--max-steps",
                "10",
                "--seed",
                "401",
            ]
        )
        == 0
    )


def test_dqn_expert_and_hybrid_training_smoke(tmp_path: Path) -> None:
    dqn_dir = tmp_path / "dqn"
    assert (
        dqn_train.main(
            [
                "--total-timesteps",
                "24",
                "--buffer-size",
                "64",
                "--warmup-steps",
                "4",
                "--batch-size",
                "4",
                "--hidden-sizes",
                "32",
                "--max-steps",
                "5",
                "--seed",
                "17",
                "--log-interval",
                "1000000",
                "--eval-frequency",
                "1000000",
                "--checkpoint-frequency",
                "1000000",
                "--log-dir",
                str(dqn_dir),
            ]
        )
        == 0
    )
    dqn_checkpoint = dqn_dir / "dqn_final.pt"
    assert_checkpoint(dqn_checkpoint, "dqn")

    expert_dir = tmp_path / "expert"
    assert (
        generate_expert_dataset.main(
            [
                "--output-dir",
                str(expert_dir),
                "--episodes",
                "1",
                "--max-steps",
                "3",
                "--think-ms",
                "1",
                "--random-action-prob",
                "0.25",
                "--shard-size",
                "4",
                "--seed",
                "29",
            ]
        )
        == 0
    )

    pretrain_dir = tmp_path / "pretrain"
    assert (
        pretrain_rl_dqn_expert.main(
            [
                "--dataset-dir",
                str(expert_dir),
                "--updates",
                "2",
                "--batch-size",
                "2",
                "--hidden-sizes",
                "32",
                "--log-interval",
                "1000000",
                "--checkpoint-frequency",
                "1000000",
                "--seed",
                "31",
                "--log-dir",
                str(pretrain_dir),
            ]
        )
        == 0
    )
    pretrain_checkpoint = pretrain_dir / "dqn_expert_pretrain.pt"
    assert_checkpoint(pretrain_checkpoint, "dqn")

    hybrid_dir = tmp_path / "hybrid"
    assert (
        train_rl_dqn_hybrid.main(
            [
                "--offline-dataset-dir",
                str(expert_dir),
                "--total-timesteps",
                "8",
                "--buffer-size",
                "32",
                "--warmup-steps",
                "4",
                "--batch-size",
                "4",
                "--learning-rate",
                "0.0002",
                "--gamma",
                "0.95",
                "--target-sync-interval",
                "7",
                "--hidden-sizes",
                "32",
                "--max-steps",
                "4",
                "--log-interval",
                "1000000",
                "--eval-frequency",
                "1000000",
                "--checkpoint-frequency",
                "1000000",
                "--online-expert-interval",
                "2",
                "--expert-think-ms",
                "1",
                "--seed",
                "37",
                "--log-dir",
                str(hybrid_dir),
                "--init-checkpoint",
                str(pretrain_checkpoint),
            ]
        )
        == 0
    )
    hybrid_checkpoint = hybrid_dir / "dqn_hybrid_final.pt"
    assert_checkpoint(hybrid_checkpoint, "dqn")
    hybrid_policy = load_policy("dqn", hybrid_checkpoint)
    assert hybrid_policy.dqn_agent is not None
    assert hybrid_policy.dqn_agent.config.learning_rate == pytest.approx(0.0002)
    assert hybrid_policy.dqn_agent.config.gamma == pytest.approx(0.95)
    assert hybrid_policy.dqn_agent.config.target_sync_interval == 7


class LoadOnlyAgent:
    def __init__(self, *_: object, **__: object):
        raise AssertionError("A supplied checkpoint must be loaded, not replaced by a fresh agent.")

    @classmethod
    def load(cls, path: str, **_: object):
        raise FileNotFoundError(path)


def test_pretrain_rejects_missing_init_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SimpleNamespace(
        obs=np.zeros((1, 3), dtype=np.float32),
        action_mask=np.ones((1, 4), dtype=np.uint8),
    )
    monkeypatch.setattr(pretrain_rl_dqn_expert, "DQNAgent", LoadOnlyAgent)
    monkeypatch.setattr(pretrain_rl_dqn_expert, "load_dataset_directory", lambda _: dataset)
    missing = tmp_path / "missing-pretrain.pt"

    with pytest.raises(FileNotFoundError, match="missing-pretrain"):
        pretrain_rl_dqn_expert.main(
            [
                "--dataset-dir",
                str(tmp_path),
                "--updates",
                "1",
                "--init-checkpoint",
                str(missing),
            ]
        )


def test_hybrid_rejects_missing_init_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SimpleNamespace(
        obs=np.zeros((1, 3), dtype=np.float32),
        action_mask=np.ones((1, PLACEMENT_ACTION_DIM), dtype=np.uint8),
    )

    class FakeRuntime:
        pass

    class FakeEnv:
        def __init__(self, **_: object):
            self.runtime = FakeRuntime()
            self.action_space = SimpleNamespace(n=PLACEMENT_ACTION_DIM)

        def reset(self, **_: object):
            return np.zeros(3, dtype=np.float32), {
                "action_mask": np.ones(PLACEMENT_ACTION_DIM, dtype=np.float32)
            }

        def close(self) -> None:
            pass

    monkeypatch.setattr(train_rl_dqn_hybrid, "DQNAgent", LoadOnlyAgent)
    monkeypatch.setattr(train_rl_dqn_hybrid, "CCTetrisEnv", FakeEnv)
    monkeypatch.setattr(train_rl_dqn_hybrid, "load_dataset_directory", lambda _: dataset)
    missing = tmp_path / "missing-hybrid.pt"

    with pytest.raises(FileNotFoundError, match="missing-hybrid"):
        train_rl_dqn_hybrid.main(
            [
                "--offline-dataset-dir",
                str(tmp_path),
                "--total-timesteps",
                "1",
                "--init-checkpoint",
                str(missing),
            ]
        )
