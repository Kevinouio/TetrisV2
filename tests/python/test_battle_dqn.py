from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tetris_v2.rl.actions import PLACEMENT_ACTION_DIM
from tetris_v2.rl.battle.checkpoint import (
    load_battle_training_checkpoint,
    save_battle_training_checkpoint,
)
from tetris_v2.rl.battle.dqn import (
    BATTLE_FEATURE_SLICE,
    BATTLE_OBSERVATION_DIM,
    BATTLE_OBSERVATION_SCHEMA,
    BattleDQNAgent,
    BattleDQNConfig,
    LinearSchedule,
)
from tetris_v2.rl.battle.opponents import OpponentDescriptor, OpponentPool
from tetris_v2.rl.battle.replay import (
    PACKED_ACTION_MASK_BYTES,
    PackedBattleReplayBuffer,
)
from tetris_v2.rl.dqn.core import DQNAgent, DQNConfig


def _mask(*actions: int) -> np.ndarray:
    value = np.zeros(PLACEMENT_ACTION_DIM, dtype=np.uint8)
    value[list(actions)] = 1
    return value


def _terminal_batch(batch_size: int = 2) -> dict[str, np.ndarray]:
    observations = np.zeros((batch_size, BATTLE_OBSERVATION_DIM), dtype=np.float32)
    masks = np.zeros((batch_size, PLACEMENT_ACTION_DIM), dtype=np.uint8)
    masks[:, 0] = 1
    return {
        "obs": observations,
        "action_masks": masks,
        "actions": np.zeros(batch_size, dtype=np.int64),
        "rewards": np.ones(batch_size, dtype=np.float32),
        "next_obs": observations.copy(),
        "next_action_masks": np.zeros_like(masks),
        "terminated": np.ones(batch_size, dtype=np.float32),
        "truncated": np.zeros(batch_size, dtype=np.float32),
    }


def test_battle_network_warm_start_preserves_q_map_and_resets_optimizer() -> None:
    torch.manual_seed(9)
    source = DQNAgent(
        DQNConfig(
            obs_dim=254,
            action_dim=PLACEMENT_ACTION_DIM,
            device="cpu",
        )
    )
    agent = BattleDQNAgent(BattleDQNConfig(seed=31, device="cpu"))
    agent.update(_terminal_batch())
    assert agent.update_steps == 1
    assert agent.optimizer.state

    agent.warm_start_from_dqn(source)
    assert agent.update_steps == 0
    assert not agent.optimizer.state
    observations = np.random.default_rng(7).random(
        (3, BATTLE_OBSERVATION_DIM), dtype=np.float32
    )
    assert observations[:, BATTLE_FEATURE_SLICE].shape == (3, 16)
    with torch.no_grad():
        source_q = source.online(torch.from_numpy(observations[:, :254]))
        battle_q = agent.online(torch.from_numpy(observations))
        target_q = agent.target(torch.from_numpy(observations))
    torch.testing.assert_close(battle_q, source_q, rtol=0, atol=0)
    torch.testing.assert_close(target_q, battle_q, rtol=0, atol=0)

    only_last = _mask(PLACEMENT_ACTION_DIM - 1)
    assert agent.select_action(
        observations[0],
        action_mask=only_last,
        deterministic=True,
    ) == PLACEMENT_ACTION_DIM - 1
    with pytest.raises(ValueError, match="no legal actions"):
        agent.select_action(
            observations[0],
            action_mask=np.zeros(PLACEMENT_ACTION_DIM, dtype=np.uint8),
        )

    incompatible = DQNAgent(
        DQNConfig(
            obs_dim=254,
            action_dim=PLACEMENT_ACTION_DIM,
            hidden_sizes=(8,),
            network_type="mlp",
            device="cpu",
        )
    )
    with pytest.raises(ValueError, match="placement-convolution"):
        agent.warm_start_from_dqn(incompatible)


def test_frozen_battle_checkpoint_is_compact_and_loadable(tmp_path: Path) -> None:
    agent = BattleDQNAgent(BattleDQNConfig(seed=3, device="cpu"))
    observation = np.random.default_rng(8).random(BATTLE_OBSERVATION_DIM, dtype=np.float32)
    expected = agent.q_values(observation)
    path = tmp_path / "frozen.pt"
    agent.save_frozen(path, metadata={"generation": 4})

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["algo"] == "battle_dqn"
    assert payload["checkpoint_type"] == "frozen_policy"
    assert payload["observation_schema"]["name"] == BATTLE_OBSERVATION_SCHEMA
    assert "target_state_dict" not in payload
    assert "optimizer_state_dict" not in payload

    loaded, metadata = BattleDQNAgent.load_frozen(path, device="cpu")
    np.testing.assert_array_equal(loaded.q_values(observation), expected)
    assert metadata == {"generation": 4}
    assert loaded.update_steps == 0
    assert not loaded.optimizer.state


class _LookupQ(nn.Module):
    def __init__(self, current: torch.Tensor, following: torch.Tensor):
        super().__init__()
        self.register_buffer("current", current)
        self.register_buffer("following", following)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        is_following = observations[:, :1] > 0.5
        return torch.where(is_following, self.following, self.current).expand(
            observations.shape[0], -1
        )


def test_double_dqn_uses_legal_online_action_and_bootstraps_truncation() -> None:
    agent = BattleDQNAgent(
        BattleDQNConfig(gamma=0.5, double_dqn=True, device="cpu")
    )
    current = torch.zeros((1, PLACEMENT_ACTION_DIM))
    online_next = torch.zeros_like(current)
    online_next[0, 1] = 100.0  # Illegal.
    online_next[0, 2] = 5.0
    target_next = torch.zeros_like(current)
    target_next[0, 0] = 20.0
    target_next[0, 2] = 4.0
    agent.online = _LookupQ(current, online_next)
    agent.target = _LookupQ(current, target_next)

    batch = {
        "obs": np.zeros((1, BATTLE_OBSERVATION_DIM), dtype=np.float32),
        "actions": np.asarray([0], dtype=np.int64),
        "rewards": np.asarray([2.0], dtype=np.float32),
        "next_obs": np.ones((1, BATTLE_OBSERVATION_DIM), dtype=np.float32),
        "next_action_masks": _mask(0, 2)[None, :],
        "terminated": np.asarray([0.0], dtype=np.float32),
        "truncated": np.asarray([1.0], dtype=np.float32),
    }
    # Online selects legal action 2, target evaluates it at 4: target=2+0.5*4=4.
    assert agent.compute_td_loss(batch).item() == pytest.approx(3.5)

    batch["terminated"] = np.asarray([1.0], dtype=np.float32)
    batch["next_action_masks"] = np.zeros((1, PLACEMENT_ACTION_DIM), dtype=np.uint8)
    # True terminals do not inspect the empty next mask: target=2.
    assert agent.compute_td_loss(batch).item() == pytest.approx(1.5)


def test_packed_replay_round_trip_preserves_ring_masks_boundaries_and_rng(
    tmp_path: Path,
) -> None:
    replay = PackedBattleReplayBuffer(capacity=3, seed=17)
    observations = [
        np.full(BATTLE_OBSERVATION_DIM, index, dtype=np.float32)
        for index in range(4)
    ]
    for index in range(4):
        current = _mask(index)
        terminal = index == 3
        following = np.zeros(PLACEMENT_ACTION_DIM, dtype=np.uint8) if terminal else _mask(index + 1)
        replay.add(
            observations[index],
            current,
            index,
            index + 0.5,
            observations[index],
            following,
            terminal,
            index == 2,
        )
    assert len(replay) == 3
    assert replay.action_masks.shape == (3, PACKED_ACTION_MASK_BYTES)
    assert [replay.transition(index)["action"] for index in range(3)] == [1, 2, 3]
    assert replay.transition(1)["truncated"] is True
    assert replay.transition(2)["terminated"] is True
    assert not np.any(replay.transition(2)["next_action_mask"])

    path = tmp_path / "replay.npz"
    replay.save(path)
    expected = replay.sample(8)
    restored = PackedBattleReplayBuffer.load(path)
    actual = restored.sample(8)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_opponent_pool_keeps_initial_newest_and_seeded_recent_older_state() -> None:
    pool = OpponentPool(
        max_frozen=4,
        seed=22,
        recent_window=1,
        recent_probability=0.5,
    )
    for generation in range(10):
        pool.add(
            OpponentDescriptor(
                identifier=f"checkpoint-{generation}",
                kind="checkpoint",
                checkpoint=f"checkpoint-{generation}.pt",
                generation=generation,
                created_step=generation * 100,
            )
        )
    generations = [descriptor.generation for descriptor in pool.frozen]
    assert len(generations) == 4
    assert generations[0] == 0
    assert generations[-1] == 9
    assert len(set(generations)) == 4

    buckets = {
        pool.sample_selection({"frozen": 1.0}).frozen_bucket
        for _ in range(30)
    }
    assert buckets == {"recent", "older"}
    state = pool.state_dict()
    expected = [
        pool.sample_selection({"random": 0.2, "cold_clear": 0.3, "frozen": 0.5}).to_dict()
        for _ in range(20)
    ]
    restored = OpponentPool.from_state_dict(state)
    actual = [
        restored.sample_selection(
            {"random": 0.2, "cold_clear": 0.3, "frozen": 0.5}
        ).to_dict()
        for _ in range(20)
    ]
    assert actual == expected


def test_training_checkpoint_restores_all_rng_streams_and_refuses_mid_episode(
    tmp_path: Path,
) -> None:
    agent = BattleDQNAgent(BattleDQNConfig(seed=101, device="cpu"))
    replay = PackedBattleReplayBuffer(capacity=4, seed=202)
    observation = np.zeros(BATTLE_OBSERVATION_DIM, dtype=np.float32)
    action_mask = _mask(0, 1)
    replay.add(
        observation,
        action_mask,
        0,
        1.0,
        observation,
        action_mask,
        False,
        True,
    )
    agent.update(replay.sample(1))
    pool = OpponentPool(max_frozen=4, seed=303, recent_window=1)
    initial_policy = tmp_path / "initial.pt"
    agent.save_frozen(initial_policy, metadata={"identifier": "initial"})
    pool.add(
        OpponentDescriptor(
            "initial",
            "checkpoint",
            checkpoint=str(initial_policy),
            generation=0,
        )
    )
    epsilon = LinearSchedule(1.0, 0.05, 1_000)
    learning_rate = LinearSchedule(1e-3, 1e-4, 1_000)
    checkpoint = tmp_path / "battle_training.pt"

    with pytest.raises(ValueError, match="episode boundaries"):
        save_battle_training_checkpoint(
            checkpoint,
            agent=agent,
            replay=replay,
            opponent_pool=pool,
            global_step=17,
            episode_index=3,
            epsilon_schedule=epsilon,
            learning_rate_schedule=learning_rate,
            training_config={"seed": 44},
            at_episode_boundary=False,
        )

    random.seed(404)
    np.random.seed(505)
    torch.manual_seed(606)
    paths = save_battle_training_checkpoint(
        checkpoint,
        agent=agent,
        replay=replay,
        opponent_pool=pool,
        global_step=17,
        episode_index=3,
        epsilon_schedule=epsilon,
        learning_rate_schedule=learning_rate,
        training_config={"seed": 44, "opponent_mix": {"random": 0.2, "frozen": 0.8}},
        at_episode_boundary=True,
        extra={"wall_seconds": 12.5},
    )
    assert paths.replay_sidecar.is_file()
    initial_policy.unlink()
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(4)
    expected_replay = replay.sample(6)
    expected_pool = [
        pool.sample({"random": 0.5, "frozen": 0.5}).identifier for _ in range(8)
    ]
    expected_action = agent.select_action(
        observation,
        action_mask=action_mask,
        epsilon=1.0,
    )

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    restored = load_battle_training_checkpoint(checkpoint, device="cpu")
    assert restored.global_step == 17
    assert restored.episode_index == 3
    assert restored.epsilon_schedule == epsilon
    assert restored.learning_rate_schedule == learning_rate
    assert restored.training_config["seed"] == 44
    assert restored.extra == {"wall_seconds": 12.5}
    embedded = restored.opponent_pool.embedded_checkpoint("initial")
    assert embedded is not None
    embedded_agent, embedded_metadata = BattleDQNAgent.from_frozen_payload(
        embedded,
        device="cpu",
    )
    assert embedded_metadata["identifier"] == "initial"
    np.testing.assert_array_equal(
        embedded_agent.q_values(observation),
        agent.q_values(observation),
    )
    assert restored.agent.update_steps == agent.update_steps
    for expected, actual in zip(
        agent.online.parameters(), restored.agent.online.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)

    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    torch.testing.assert_close(torch.rand(4), expected_torch)
    actual_replay = restored.replay.sample(6)
    for name in expected_replay:
        np.testing.assert_array_equal(actual_replay[name], expected_replay[name])
    assert [
        restored.opponent_pool.sample({"random": 0.5, "frozen": 0.5}).identifier
        for _ in range(8)
    ] == expected_pool
    assert restored.agent.select_action(
        observation,
        action_mask=action_mask,
        epsilon=1.0,
    ) == expected_action


def test_linear_schedule_hooks() -> None:
    schedule = LinearSchedule(start=1.0, end=0.0, duration=100, start_step=10)
    assert schedule.value(0) == 1.0
    assert schedule.value(60) == pytest.approx(0.5)
    assert schedule.value(200) == 0.0
    agent = BattleDQNAgent(BattleDQNConfig(device="cpu"))
    assert agent.epsilon_at(60, schedule) == pytest.approx(0.5)
    assert agent.apply_learning_rate_schedule(60, schedule) == pytest.approx(0.5)
    assert agent.current_learning_rate == pytest.approx(0.5)
