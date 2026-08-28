from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from scripts import eval_rl
from tetris_v2.rl.actions import PLACEMENT_ACTION_DIM, encode_action
from tetris_v2.rl.flow_dqn.core import (
    ACTION_MAP_SHAPE,
    SOURCE_NOISE_STD,
    FlowDQNAgent,
    FlowDQNConfig,
    FlowReplayBuffer,
    StructuredQNetwork,
    action_map_to_flat,
    flat_to_action_map,
)
from tetris_v2.rl.flow_dqn.train import _mixed_online_batch, main as train_main, parse_args


def make_agent(**overrides) -> FlowDQNAgent:
    values = {
        "obs_dim": 254,
        "action_dim": PLACEMENT_ACTION_DIM,
        "channels": 2,
        "flow_steps": 2,
        "device": "cpu",
    }
    values.update(overrides)
    return FlowDQNAgent(FlowDQNConfig(**values))


def test_action_map_preserves_stable_hold_rotation_y_x_order() -> None:
    flat = torch.arange(PLACEMENT_ACTION_DIM).reshape(1, -1)
    placement_map = flat_to_action_map(flat)

    action = encode_action(use_hold=True, rotation=2, y=17, x=6)
    assert placement_map.shape == (1, *ACTION_MAP_SHAPE)
    assert placement_map[0, 6, 17, 6].item() == action
    torch.testing.assert_close(action_map_to_flat(placement_map), flat)


def test_structured_critic_outputs_full_placement_map() -> None:
    network = StructuredQNetwork(obs_dim=254, channels=2)
    assert network(torch.zeros((2, 254))).shape == (2, 8, 40, 10)


class RecordingActor(nn.Module):
    def __init__(self, preferred_action: int, preference: float = 10.0):
        super().__init__()
        self.preferred_action = preferred_action
        self.preference = preference
        self.sources: list[torch.Tensor] = []

    def forward(self, obs: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        self.sources.append(source.detach().clone())
        logits = torch.zeros((obs.shape[0], PLACEMENT_ACTION_DIM), device=obs.device)
        logits[:, self.preferred_action] = self.preference
        return flat_to_action_map(logits)


def test_action_selection_uses_zero_noise_deterministically_and_gaussian_stochastically() -> None:
    agent = make_agent()
    actor = RecordingActor(preferred_action=99)
    agent.actor = actor
    mask = np.zeros(PLACEMENT_ACTION_DIM, dtype=np.float32)
    mask[[42, 99]] = 1.0
    obs = np.zeros(254, dtype=np.float32)

    assert agent.select_action(obs, deterministic=True, action_mask=mask) == 99
    assert torch.count_nonzero(actor.sources[-1]) == 0

    assert agent.select_action(obs, deterministic=False, temperature=0.25, action_mask=mask) == 99
    assert torch.count_nonzero(actor.sources[-1]) > 0
    assert actor.sources[-1].std().item() == pytest.approx(0.25 * SOURCE_NOISE_STD, rel=0.1)

    # Stochasticity comes only from the Gaussian latent; there is no second
    # categorical draw that can ignore the actor's masked argmax.
    agent.actor = RecordingActor(preferred_action=99, preference=1e-6)
    assert {
        agent.select_action(obs, deterministic=False, action_mask=mask)
        for _ in range(20)
    } == {99}


class ConstantFlow(nn.Module):
    def __init__(self, velocity: torch.Tensor):
        super().__init__()
        self.velocity = velocity
        self.inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.encode_calls = 0

    def forward(
        self,
        obs: torch.Tensor,
        action_map: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_encoded(self.encode_state(obs), action_map, time)

    def encode_state(self, obs: torch.Tensor) -> torch.Tensor:
        self.encode_calls += 1
        return obs

    def forward_encoded(
        self,
        state_features: torch.Tensor,
        action_map: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        del state_features
        self.inputs.append((action_map.detach().clone(), time.detach().clone()))
        return self.velocity.expand_as(action_map)


def test_euler_flow_integration_and_flow_matching_endpoint() -> None:
    agent = make_agent(flow_steps=10)
    obs = torch.zeros((1, 254))
    source = torch.zeros((1, *ACTION_MAP_SHAPE))
    agent.flow = ConstantFlow(torch.ones((1, *ACTION_MAP_SHAPE)))

    integrated = agent.integrate_flow(obs, source)
    torch.testing.assert_close(integrated, torch.ones_like(integrated))
    assert agent.flow.encode_calls == 1
    assert len(agent.flow.inputs) == 10
    torch.testing.assert_close(agent.flow.inputs[-1][1], torch.tensor([[0.9]]))

    endpoint_velocity = torch.zeros((1, PLACEMENT_ACTION_DIM))
    endpoint_velocity[:, 7] = 1.0
    agent.flow = ConstantFlow(flat_to_action_map(endpoint_velocity))
    loss = agent.flow_matching_loss(
        obs,
        torch.tensor([7]),
        source_noise=source,
        times=torch.tensor([[0.25]]),
    )
    assert loss.item() == pytest.approx(0.0)
    interpolated, time = agent.flow.inputs[-1]
    assert action_map_to_flat(interpolated)[0, 7].item() == pytest.approx(0.25)
    assert time.item() == pytest.approx(0.25)


def test_cached_euler_state_matches_reencoding_each_step() -> None:
    torch.manual_seed(9)
    agent = make_agent(flow_steps=3)
    obs = torch.rand((2, 254))
    source = agent.sample_source_noise(2)

    cached = agent.integrate_flow(obs, source)
    manual = source
    for step in range(3):
        time = torch.full((2, 1), step / 3.0)
        manual = manual + agent.flow(obs, manual, time) / 3.0

    torch.testing.assert_close(cached, manual)


def test_flow_loss_keeps_selected_one_hot_endpoint_at_unit_scale() -> None:
    agent = make_agent()
    agent.flow = ConstantFlow(torch.zeros((1, *ACTION_MAP_SHAPE)))
    loss = agent.flow_matching_loss(
        torch.zeros((1, 254)),
        torch.tensor([123]),
        source_noise=torch.zeros((1, *ACTION_MAP_SHAPE)),
        times=torch.tensor([[0.5]]),
    )
    # Per-map reduction keeps the selected coordinate meaningful instead of 1 / 3200.
    assert loss.item() == pytest.approx(1.0)


def test_flow_t0_auxiliary_conditions_first_euler_step_on_executed_action() -> None:
    agent = make_agent(flow_t0_ce_coef=1.0, action_logit_scale=10.0)
    obs = torch.zeros((1, 254))
    source = torch.zeros((1, *ACTION_MAP_SHAPE))
    actions = torch.tensor([9])
    mask = torch.zeros((1, PLACEMENT_ACTION_DIM))
    mask[0, [7, 9]] = 1.0

    correct_velocity = torch.zeros((1, PLACEMENT_ACTION_DIM))
    correct_velocity[0, 9] = 1.0
    agent.flow = ConstantFlow(flat_to_action_map(correct_velocity))
    total, matching, t0_ce, agreement = agent.flow_losses(
        obs,
        actions,
        action_masks=mask,
        source_noise=source,
        times=torch.tensor([[0.0]]),
    )
    assert matching.item() == pytest.approx(0.0)
    assert t0_ce.item() < 1e-3
    assert agreement.item() == pytest.approx(1.0)
    torch.testing.assert_close(total, matching + t0_ce)

    agent.flow = ConstantFlow(torch.zeros((1, *ACTION_MAP_SHAPE)))
    _, _, unconditioned_ce, unconditioned_agreement = agent.flow_losses(
        obs,
        actions,
        action_masks=mask,
        source_noise=source,
        times=torch.tensor([[0.0]]),
    )
    assert unconditioned_ce.item() == pytest.approx(np.log(2.0))
    assert unconditioned_agreement.item() == pytest.approx(0.0)


class FixedMap(nn.Module):
    def __init__(self, values: torch.Tensor):
        super().__init__()
        self.register_buffer("values", values)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.values.expand(obs.shape[0], -1, -1, -1)


class LearnableMapActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros((1, *ACTION_MAP_SHAPE)))

    def forward(self, obs: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        del source
        return self.logits.expand(obs.shape[0], -1, -1, -1)


def test_bellman_target_stops_only_at_true_terminal() -> None:
    agent = make_agent(gamma=0.5)
    actor = RecordingActor(preferred_action=7)
    agent.actor = actor
    agent.target_critic1 = FixedMap(torch.full((1, *ACTION_MAP_SHAPE), 4.0))
    agent.target_critic2 = FixedMap(torch.full((1, *ACTION_MAP_SHAPE), 6.0))

    next_masks = torch.zeros((2, PLACEMENT_ACTION_DIM))
    # A terminal row may have no next legal action.  A truncated row must bootstrap.
    next_masks[1, 7] = 1.0
    targets = agent.bellman_targets(
        rewards=torch.tensor([1.0, 2.0]),
        next_obs=torch.zeros((2, 254)),
        terminated=torch.tensor([1.0, 0.0]),
        next_action_masks=next_masks,
    )

    torch.testing.assert_close(targets, torch.tensor([1.0, 4.0]))


def test_polyak_update_moves_both_target_critics() -> None:
    agent = make_agent(tau=0.25)
    with torch.no_grad():
        for critic in (agent.critic1, agent.critic2):
            for parameter in critic.parameters():
                parameter.fill_(1.0)
        for target in (agent.target_critic1, agent.target_critic2):
            for parameter in target.parameters():
                parameter.zero_()

    agent.polyak_update()
    for target in (agent.target_critic1, agent.target_critic2):
        for parameter in target.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 0.25))


def test_normalized_q_guidance_is_scale_invariant() -> None:
    agent = make_agent(normalized_q=True)
    q_values = torch.tensor([1.0, 3.0], requires_grad=True)
    base = agent.q_guidance_loss(q_values)
    scaled = agent.q_guidance_loss(q_values * 7.0)

    torch.testing.assert_close(base, scaled)
    base.backward()
    assert torch.all(q_values.grad < 0)


def test_masked_q_guidance_favors_the_best_legal_action_only() -> None:
    agent = make_agent(distill_coef=0.0, normalized_q=True)
    actor = LearnableMapActor()
    agent.actor = actor
    agent.flow = ConstantFlow(torch.zeros((1, *ACTION_MAP_SHAPE)))
    q = torch.zeros((1, PLACEMENT_ACTION_DIM))
    q[0, 1] = 2.0
    q[0, 2] = 100.0  # Illegal and therefore excluded from the policy expectation.
    agent.critic1 = FixedMap(flat_to_action_map(q))
    agent.critic2 = FixedMap(flat_to_action_map(q))
    mask = torch.zeros((1, PLACEMENT_ACTION_DIM))
    mask[0, :2] = 1.0

    total, *_ = agent.actor_losses(
        torch.zeros((1, 254)),
        mask,
        source_noise=torch.zeros((1, *ACTION_MAP_SHAPE)),
    )
    total.backward()
    gradient = action_map_to_flat(actor.logits.grad)
    assert gradient[0, 1] < 0
    assert gradient[0, 0] > 0
    assert gradient[0, 2] == 0


def test_direct_actor_behavior_ce_favors_executed_legal_action() -> None:
    agent = make_agent(
        distill_coef=0.0,
        q_guidance_coef=0.0,
        actor_bc_coef=1.0,
        action_logit_scale=10.0,
    )
    actor = LearnableMapActor()
    agent.actor = actor
    agent.flow = ConstantFlow(torch.zeros((1, *ACTION_MAP_SHAPE)))
    q = flat_to_action_map(torch.zeros((1, PLACEMENT_ACTION_DIM)))
    agent.critic1 = FixedMap(q)
    agent.critic2 = FixedMap(q)
    mask = torch.zeros((1, PLACEMENT_ACTION_DIM))
    mask[0, [1, 2]] = 1.0

    total, _, _, behavior_ce, agreement, entropy = agent.actor_losses(
        torch.zeros((1, 254)),
        mask,
        actions=torch.tensor([2]),
        source_noise=torch.zeros((1, *ACTION_MAP_SHAPE)),
    )
    total.backward()
    gradient = action_map_to_flat(actor.logits.grad)
    assert gradient[0, 2] < 0
    assert gradient[0, 1] > 0
    assert gradient[0, 3] == 0
    assert behavior_ce.item() == pytest.approx(np.log(2.0))
    assert agreement.item() == pytest.approx(0.0)
    assert entropy.item() == pytest.approx(np.log(2.0))


def test_replay_round_trips_bit_packed_masks_and_boundaries() -> None:
    replay = FlowReplayBuffer(capacity=2, obs_dim=3, action_dim=PLACEMENT_ACTION_DIM)
    current = np.zeros(PLACEMENT_ACTION_DIM, dtype=np.uint8)
    following = np.zeros(PLACEMENT_ACTION_DIM, dtype=np.uint8)
    current[[0, 1703, 3199]] = 1
    following[[8, 91]] = 1
    replay.add(
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        current,
        1703,
        2.5,
        np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
        following,
        False,
        True,
    )

    batch = replay.sample(1, np.random.default_rng(5))
    np.testing.assert_array_equal(batch["action_masks"][0], current)
    np.testing.assert_array_equal(batch["next_action_masks"][0], following)
    assert batch["actions"][0] == 1703
    assert batch["terminated"][0] == 0.0
    assert batch["truncated"][0] == 1.0


def test_update_has_finite_flow_distillation_q_and_critic_losses() -> None:
    torch.manual_seed(4)
    agent = make_agent(flow_steps=1)
    mask = np.zeros((1, PLACEMENT_ACTION_DIM), dtype=np.uint8)
    mask[0, :4] = 1
    batch = {
        "obs": np.zeros((1, 254), dtype=np.float32),
        "action_masks": mask,
        "actions": np.asarray([2], dtype=np.int64),
        "rewards": np.asarray([1.0], dtype=np.float32),
        "next_obs": np.zeros((1, 254), dtype=np.float32),
        "next_action_masks": np.zeros_like(mask),
        "terminated": np.asarray([1.0], dtype=np.float32),
        "truncated": np.asarray([0.0], dtype=np.float32),
    }

    metrics = agent.update(batch, source="offline")
    assert all(np.isfinite(value) for value in metrics.values())
    assert agent.update_steps == 1
    assert agent.offline_update_steps == 1
    assert agent.online_update_steps == 0


def test_checkpoint_round_trip_restores_all_models_optimizers_schema_and_counters(
    tmp_path: Path,
) -> None:
    agent = make_agent()
    agent.update_steps = 11
    agent.offline_update_steps = 7
    agent.online_update_steps = 4
    agent.environment_steps = 91
    path = tmp_path / "flow.pt"
    agent.save(path, metadata={"tag": "round-trip"})

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["algo"] == "flow_dqn"
    assert payload["observation_schema"] == {
        "name": "tetris_v2_254",
        "obs_dim": 254,
    }
    assert payload["action_schema"] == {
        "shape": [8, 40, 10],
        "order": "hold-major,rotation-major,y-major,x-major",
        "action_dim": PLACEMENT_ACTION_DIM,
    }
    for key in (
        "actor_state_dict",
        "flow_state_dict",
        "critic1_state_dict",
        "critic2_state_dict",
        "target_critic1_state_dict",
        "target_critic2_state_dict",
        "actor_optimizer_state_dict",
        "flow_optimizer_state_dict",
        "critic_optimizer_state_dict",
    ):
        assert key in payload

    loaded, metadata = FlowDQNAgent.load(path, device="cpu")
    assert metadata == {"tag": "round-trip"}
    assert loaded.update_steps == 11
    assert loaded.offline_update_steps == 7
    assert loaded.online_update_steps == 4
    assert loaded.environment_steps == 91
    for expected, actual in zip(agent.actor.parameters(), loaded.actor.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)

    for key in ("flow_t0_ce_coef", "actor_bc_coef", "action_logit_scale"):
        payload["config"].pop(key)
    torch.save(payload, path)
    legacy_loaded, _ = FlowDQNAgent.load(path, device="cpu")
    assert legacy_loaded.config.flow_t0_ce_coef == pytest.approx(1.0)
    assert legacy_loaded.config.actor_bc_coef == pytest.approx(1.0)
    assert legacy_loaded.config.action_logit_scale == pytest.approx(10.0)

    payload["observation_schema"]["obs_dim"] = 451
    torch.save(payload, path)
    with pytest.raises(ValueError, match="observation schema"):
        FlowDQNAgent.load(path, device="cpu")

    payload["observation_schema"]["obs_dim"] = 254
    payload["action_schema"]["action_dim"] = 97
    torch.save(payload, path)
    with pytest.raises(ValueError, match="action layout"):
        FlowDQNAgent.load(path, device="cpu")


def test_training_cli_exposes_offline_online_and_q_sweep_flags() -> None:
    args = parse_args(
        [
            "--offline-dataset-dir",
            "data/v3",
            "--offline-updates",
            "200",
            "--online-timesteps",
            "500",
            "--distill-q-coef",
            "0.3",
            "--online-offline-fraction",
            "0.5",
            "--flow-t0-ce-coef",
            "0.75",
            "--actor-bc-coef",
            "0.5",
            "--action-logit-scale",
            "12",
        ]
    )
    assert args.offline_dataset_dir == Path("data/v3")
    assert args.offline_updates == 200
    assert args.online_timesteps == 500
    assert args.distill_q_coef == pytest.approx(0.3)
    assert args.online_offline_fraction == pytest.approx(0.5)
    assert args.flow_t0_ce_coef == pytest.approx(0.75)
    assert args.actor_bc_coef == pytest.approx(0.5)
    assert args.action_logit_scale == pytest.approx(12.0)


def test_online_batches_mix_packed_replay_with_expert_transitions() -> None:
    replay = FlowReplayBuffer(capacity=2, obs_dim=3, action_dim=PLACEMENT_ACTION_DIM)
    mask = np.ones(PLACEMENT_ACTION_DIM, dtype=np.uint8)
    replay.add(
        np.zeros(3, dtype=np.float32),
        mask,
        1,
        0.0,
        np.zeros(3, dtype=np.float32),
        mask,
        False,
        False,
    )

    class Dataset:
        @staticmethod
        def sample(batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
            del rng
            return {
                "obs": np.ones((batch_size, 3), dtype=np.float32),
                "action_mask": np.ones((batch_size, PLACEMENT_ACTION_DIM), dtype=np.uint8),
                "executed_action": np.full(batch_size, 2, dtype=np.int64),
                "reward": np.ones(batch_size, dtype=np.float32),
                "next_obs": np.ones((batch_size, 3), dtype=np.float32),
                "next_action_mask": np.ones(
                    (batch_size, PLACEMENT_ACTION_DIM), dtype=np.uint8
                ),
                "terminated": np.zeros(batch_size, dtype=np.uint8),
                "truncated": np.zeros(batch_size, dtype=np.uint8),
            }

    batch = _mixed_online_batch(
        replay,
        Dataset(),
        batch_size=4,
        offline_fraction=0.5,
        rng=np.random.default_rng(3),
    )
    assert sorted(batch["actions"].tolist()) == [1, 1, 2, 2]


def test_trainer_rejects_replay_that_can_never_reach_warmup() -> None:
    with pytest.raises(ValueError, match="online updates can never begin"):
        train_main(
            [
                "--online-timesteps",
                "1",
                "--buffer-size",
                "2",
                "--warmup-steps",
                "3",
                "--batch-size",
                "1",
            ]
        )


def test_flow_trainer_checkpoint_and_shared_evaluation_smoke(tmp_path: Path) -> None:
    output = tmp_path / "flow_online"
    assert train_main(
        [
            "--online-timesteps",
            "2",
            "--offline-updates",
            "0",
            "--buffer-size",
            "4",
            "--warmup-steps",
            "1",
            "--batch-size",
            "1",
            "--channels",
            "2",
            "--flow-steps",
            "1",
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
            "--log-dir",
            str(output),
        ]
    ) == 0

    checkpoint = output / "flow_dqn_final.pt"
    agent, _ = FlowDQNAgent.load(checkpoint, device="cpu")
    assert agent.environment_steps == 2
    assert agent.online_update_steps >= 1
    assert eval_rl.main(
        [
            str(checkpoint),
            "--algo",
            "flow_dqn",
            "--episodes",
            "1",
            "--max-steps",
            "2",
            "--device",
            "cpu",
        ]
    ) == 0
