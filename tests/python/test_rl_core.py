from __future__ import annotations

import numpy as np
import pytest
import torch

from tetris_v2.rl.dqn.core import DQNAgent, DQNConfig
from tetris_v2.rl.dqn.expert_losses import pairwise_ranking_loss
from tetris_v2.rl.ppo.core import PPOAgent, PPOConfig, RolloutBuffer
from tetris_v2.rl.ppo.train import _bootstrap_truncated_rewards


def test_ppo_gae_stops_at_terminal_boundary() -> None:
    buffer = RolloutBuffer(n_steps=3, num_envs=1, obs_dim=1)
    for reward, done, value in zip(
        (1.0, 2.0, 3.0),
        (0.0, 1.0, 0.0),
        (10.0, 20.0, 30.0),
        strict=True,
    ):
        buffer.add(
            obs=np.zeros((1, 1), dtype=np.float32),
            actions=np.zeros(1, dtype=np.int64),
            rewards=np.asarray([reward], dtype=np.float32),
            dones=np.asarray([done], dtype=np.float32),
            values=np.asarray([value], dtype=np.float32),
            log_probs=np.zeros(1, dtype=np.float32),
            action_masks=np.ones((1, 2), dtype=np.float32),
        )

    buffer.compute_returns_and_advantages(
        last_values=np.asarray([40.0], dtype=np.float32),
        last_dones=np.asarray([0.0], dtype=np.float32),
        gamma=1.0,
        gae_lambda=1.0,
    )

    np.testing.assert_allclose(buffer.returns[:, 0], [3.0, 2.0, 43.0])


def test_ppo_bootstraps_time_limits_without_changing_episode_rewards() -> None:
    class FixedValueAgent:
        @staticmethod
        def value_batch(observations: np.ndarray) -> np.ndarray:
            return observations[:, 0]

    final_observations = np.empty(3, dtype=object)
    final_observations[:] = [None, np.asarray([4.0]), np.asarray([9.0])]
    rewards = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    training_rewards = _bootstrap_truncated_rewards(
        FixedValueAgent(),
        rewards,
        terminated=np.asarray([False, False, True]),
        truncated=np.asarray([False, True, True]),
        infos={"final_obs": final_observations},
        gamma=0.5,
    )

    np.testing.assert_allclose(training_rewards, [1.0, 4.0, 3.0])
    np.testing.assert_allclose(rewards, [1.0, 2.0, 3.0])


@pytest.mark.parametrize("epsilon", [0.0, 1.0])
def test_dqn_rejects_empty_action_mask(epsilon: float) -> None:
    agent = DQNAgent(DQNConfig(obs_dim=3, action_dim=4, hidden_sizes=(8,), device="cpu"))

    with pytest.raises(ValueError, match="no legal actions"):
        agent.select_action(
            np.zeros(3, dtype=np.float32),
            epsilon=epsilon,
            action_mask=np.zeros(4, dtype=np.float32),
        )


def test_ppo_rejects_empty_action_mask_row() -> None:
    agent = PPOAgent(PPOConfig(obs_dim=3, action_dim=4, hidden_sizes=(8,), device="cpu"))
    observations = np.zeros((2, 3), dtype=np.float32)
    masks = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="no legal actions"):
        agent.act_batch(observations, deterministic=True, action_mask=masks)


def test_pairwise_expert_loss_is_vectorized_and_orders_actions() -> None:
    q_values = torch.tensor([[0.0, 1.0]], requires_grad=True)
    teacher_best = torch.tensor([0])
    action_mask = torch.ones((1, 2))

    wrong_loss = pairwise_ranking_loss(
        q_values,
        teacher_best,
        action_mask,
        rng=np.random.default_rng(7),
        pairs_per_sample=8,
    )
    correct_loss = pairwise_ranking_loss(
        torch.tensor([[2.0, 0.0]]),
        teacher_best,
        action_mask,
        rng=np.random.default_rng(7),
        pairs_per_sample=8,
    )

    assert wrong_loss > correct_loss
    wrong_loss.backward()
    assert q_values.grad is not None
    assert q_values.grad[0, 0] < 0
    assert q_values.grad[0, 1] > 0
