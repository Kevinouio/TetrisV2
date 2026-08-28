from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tetris_v2.rl.actions import PLACEMENT_ACTION_DIM, encode_action
from tetris_v2.rl.ppo import pretrain, train
from tetris_v2.rl.ppo.core import (
    ACTION_ORDER,
    ActorCritic,
    PPOAgent,
    PPOConfig,
    RolloutBuffer,
    StructuredActorCritic,
    clipped_value_loss,
)


def test_structured_ppo_has_independent_actor_and_critic_and_exact_action_ids() -> None:
    agent = PPOAgent(
        PPOConfig(obs_dim=254, action_dim=PLACEMENT_ACTION_DIM, device="cpu")
    )
    assert agent.network_type == "placement_conv"
    assert isinstance(agent.policy, StructuredActorCritic)
    actor_params = {id(parameter) for parameter in agent.policy.actor.parameters()}
    critic_params = {id(parameter) for parameter in agent.policy.critic.parameters()}
    assert actor_params.isdisjoint(critic_params)

    target = encode_action(use_hold=True, rotation=3, y=39, x=9)
    mask = np.zeros(PLACEMENT_ACTION_DIM, dtype=np.float32)
    mask[target] = 1.0
    action, _, _ = agent.act(
        np.zeros(254, dtype=np.float32),
        deterministic=True,
        action_mask=mask,
    )
    assert action == target == PLACEMENT_ACTION_DIM - 1


def test_structured_behavior_cloning_skips_critic_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = PPOAgent(
        PPOConfig(obs_dim=254, action_dim=PLACEMENT_ACTION_DIM, device="cpu")
    )
    assert isinstance(agent.policy, StructuredActorCritic)

    def fail_critic(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("BC must not evaluate the value network")

    monkeypatch.setattr(agent.policy.critic, "forward", fail_critic)
    masks = np.zeros((2, PLACEMENT_ACTION_DIM), dtype=np.float32)
    masks[:, :2] = 1.0
    loss, agreement = agent.behavior_cloning_loss(
        np.zeros((2, 254), dtype=np.float32),
        np.asarray([0, 1], dtype=np.int64),
        masks,
    )
    assert torch.isfinite(loss)
    assert 0.0 <= float(agreement) <= 1.0


def test_clipped_value_loss_uses_larger_error() -> None:
    values = torch.tensor([2.0])
    old_values = torch.tensor([0.0])
    returns = torch.tensor([2.0])
    loss = clipped_value_loss(values, old_values, returns, 0.2)
    assert float(loss) == pytest.approx(0.5 * (2.0 - 0.2) ** 2)


def test_ppo_kl_stop_and_stability_metrics_are_finite() -> None:
    agent = PPOAgent(
        PPOConfig(
            obs_dim=3,
            action_dim=2,
            hidden_sizes=(8,),
            target_kl=1e-6,
            device="cpu",
        )
    )
    buffer = RolloutBuffer(n_steps=4, num_envs=1, obs_dim=3)
    for _ in range(4):
        buffer.add(
            obs=np.zeros((1, 3), dtype=np.float32),
            actions=np.zeros(1, dtype=np.int64),
            rewards=np.ones(1, dtype=np.float32),
            dones=np.zeros(1, dtype=np.float32),
            values=np.zeros(1, dtype=np.float32),
            log_probs=np.asarray([-10.0], dtype=np.float32),
            action_masks=np.ones((1, 2), dtype=np.float32),
        )
    buffer.advantages[: buffer.step] = 1.0
    buffer.returns[: buffer.step] = 1.0
    before = [parameter.detach().clone() for parameter in agent.policy.parameters()]
    metrics = agent.update(buffer, batch_size=4, epochs=4)

    assert metrics["early_stopped"] == 1.0
    assert agent.update_steps == 0
    for expected, actual in zip(before, agent.policy.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    for name in (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
    ):
        assert np.isfinite(metrics[name])


def test_ppo_update_can_mix_in_expert_behavior_cloning() -> None:
    class ExpertBatch:
        @staticmethod
        def sample(batch_size: int, rng: np.random.Generator):
            del rng
            return {
                "obs": np.zeros((batch_size, 3), dtype=np.float32),
                "action_mask": np.ones((batch_size, 2), dtype=np.uint8),
                "teacher_best_action": np.ones(batch_size, dtype=np.int64),
            }

    agent = PPOAgent(
        PPOConfig(
            obs_dim=3,
            action_dim=2,
            hidden_sizes=(8,),
            target_kl=None,
            device="cpu",
        )
    )
    observations = np.zeros((2, 3), dtype=np.float32)
    masks = np.ones((2, 2), dtype=np.float32)
    actions, log_probs, values = agent.act_batch(observations, action_mask=masks)
    buffer = RolloutBuffer(n_steps=1, num_envs=2, obs_dim=3)
    buffer.add(
        observations,
        actions,
        np.ones(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        values,
        log_probs,
        masks,
    )
    buffer.advantages[0] = 1.0
    buffer.returns[0] = 1.0

    metrics = agent.update(
        buffer,
        batch_size=2,
        epochs=1,
        expert_dataset=ExpertBatch(),
        expert_batch_size=2,
        bc_coef=0.5,
        expert_rng=np.random.default_rng(5),
    )
    assert metrics["bc_loss"] > 0.0
    assert 0.0 <= metrics["teacher_top1_agreement"] <= 1.0


def test_legacy_ppo_checkpoint_without_network_type_still_loads(tmp_path: Path) -> None:
    config = PPOConfig(
        obs_dim=3,
        action_dim=4,
        hidden_sizes=(8,),
        network_type="mlp",
        device="cpu",
    )
    original = PPOAgent(config)
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "algo": "ppo",
            "config": {
                "obs_dim": 3,
                "action_dim": 4,
                "hidden_sizes": (8,),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "entropy_coef": 0.01,
                "value_coef": 0.5,
                "max_grad_norm": 0.5,
            },
            "state_dict": original.policy.state_dict(),
            "optimizer_state_dict": original.optimizer.state_dict(),
            "metadata": {"global_step": 7},
        },
        path,
    )

    loaded, metadata = PPOAgent.load(str(path), device="cpu")
    assert loaded.network_type == "mlp"
    assert isinstance(loaded.policy, ActorCritic)
    assert metadata["global_step"] == 7


def test_structured_checkpoint_records_schema_and_round_trips(tmp_path: Path) -> None:
    agent = PPOAgent(
        PPOConfig(obs_dim=254, action_dim=PLACEMENT_ACTION_DIM, device="cpu")
    )
    agent.update_steps = 9
    path = tmp_path / "structured.pt"
    agent.save(str(path), metadata={"global_step": 12})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["checkpoint_version"] == 2
    assert payload["action_order"] == ACTION_ORDER
    assert payload["config"]["network_type"] == "placement_conv"

    restored, metadata = PPOAgent.load(str(path), device="cpu")
    assert restored.network_type == "placement_conv"
    assert restored.update_steps == 9
    assert metadata["global_step"] == 12
    initialized, _ = PPOAgent.load(str(path), device="cpu", restore_optimizer=False)
    assert initialized.update_steps == 0
    assert initialized.optimizer.state_dict()["state"] == {}


class TinyExpertDataset:
    def __init__(self):
        self.obs = np.zeros((2, 254), dtype=np.float32)
        self.action_mask = np.zeros((2, PLACEMENT_ACTION_DIM), dtype=np.uint8)
        self.action_mask[:, :2] = 1
        self.teacher_best_action = np.asarray([0, 1], dtype=np.int64)

    def sample(self, batch_size: int, rng: np.random.Generator):
        indices = rng.integers(0, 2, size=batch_size)
        return {
            "obs": self.obs[indices],
            "action_mask": self.action_mask[indices],
            "teacher_best_action": self.teacher_best_action[indices],
        }


def test_ppo_expert_pretrain_cli_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pretrain, "_load_expert_data", lambda *_: TinyExpertDataset())
    output = tmp_path / "pretrain"
    assert (
        pretrain.main(
            [
                "--dataset-dir",
                str(tmp_path),
                "--updates",
                "2",
                "--batch-size",
                "2",
                "--checkpoint-frequency",
                "0",
                "--log-interval",
                "0",
                "--device",
                "cpu",
                "--log-dir",
                str(output),
            ]
        )
        == 0
    )
    checkpoint = output / "ppo_expert_pretrain.pt"
    agent, metadata = PPOAgent.load(str(checkpoint), device="cpu")
    assert agent.network_type == "placement_conv"
    assert metadata["global_step"] == 2
    assert np.isfinite(metadata["bc_loss"])


def test_ppo_train_resume_and_init_have_distinct_counters(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    common = [
        "--num-envs",
        "1",
        "--n-steps",
        "2",
        "--minibatch-size",
        "2",
        "--update-epochs",
        "1",
        "--hidden-sizes",
        "8",
        "--network-type",
        "mlp",
        "--max-steps",
        "2",
        "--log-interval",
        "0",
        "--eval-frequency",
        "0",
        "--checkpoint-frequency",
        "0",
        "--device",
        "cpu",
    ]
    assert train.main(["--total-timesteps", "4", "--log-dir", str(first_dir), *common]) == 0
    first_checkpoint = first_dir / "ppo_final.pt"
    first, first_metadata = PPOAgent.load(str(first_checkpoint), device="cpu")
    assert first_metadata["global_step"] == 4
    assert first.update_steps == 2
    first_optimizer_step = int(next(iter(first.optimizer.state.values()))["step"])

    resume_dir = tmp_path / "resume"
    assert (
        train.main(
            [
                "--total-timesteps",
                "6",
                "--resume-checkpoint",
                str(first_checkpoint),
                "--log-dir",
                str(resume_dir),
                *common,
            ]
        )
        == 0
    )
    resumed, resumed_metadata = PPOAgent.load(
        str(resume_dir / "ppo_final.pt"), device="cpu"
    )
    assert resumed_metadata["global_step"] == 6
    assert resumed.update_steps == 3
    assert int(next(iter(resumed.optimizer.state.values()))["step"]) == first_optimizer_step + 1
    assert resumed_metadata["learning_rate"] <= first_metadata["learning_rate"]

    init_dir = tmp_path / "init"
    assert (
        train.main(
            [
                "--total-timesteps",
                "2",
                "--init-checkpoint",
                str(first_checkpoint),
                "--log-dir",
                str(init_dir),
                *common,
            ]
        )
        == 0
    )
    initialized, initialized_metadata = PPOAgent.load(
        str(init_dir / "ppo_final.pt"), device="cpu"
    )
    assert initialized_metadata["global_step"] == 2
    assert initialized.update_steps == 1
