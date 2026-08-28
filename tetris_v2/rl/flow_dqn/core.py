"""PyTorch-native discrete Flow-DQN for TetrisV2.

This is a discrete, masked-action adaptation of Flow Q-Learning.  The original
algorithm targets continuous actions; here every learned action tensor follows
TetrisV2's stable ``(hold, rotation, y, x)`` placement layout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tetris_v2.rl.actions import (
    BOARD_ROWS,
    BOARD_WIDTH,
    PLACEMENT_ACTION_DIM,
    PLACEMENT_MAP_SHAPE,
)


ACTION_MAP_SHAPE = PLACEMENT_MAP_SHAPE
ACTION_ORDER = "hold-major,rotation-major,y-major,x-major"
CURRENT_OBSERVATION_DIM = 254
PACKED_MASK_BYTES = (PLACEMENT_ACTION_DIM + 7) // 8
SOURCE_NOISE_STD = PLACEMENT_ACTION_DIM**-0.5


def flat_to_action_map(values: Tensor) -> Tensor:
    """View a final 3,200-value dimension as ``[8, 40, 10]``."""

    if values.shape[-1] != PLACEMENT_ACTION_DIM:
        raise ValueError(
            f"Expected a final action dimension of {PLACEMENT_ACTION_DIM}, got {values.shape[-1]}"
        )
    return values.reshape(*values.shape[:-1], *ACTION_MAP_SHAPE)


def action_map_to_flat(values: Tensor) -> Tensor:
    """Flatten one or more placement maps without changing action-ID order."""

    if tuple(values.shape[-3:]) != ACTION_MAP_SHAPE:
        raise ValueError(f"Expected action map shape {ACTION_MAP_SHAPE}, got {values.shape[-3:]}")
    return values.reshape(*values.shape[:-3], PLACEMENT_ACTION_DIM)


class PlacementStateEncoder(nn.Module):
    """Turn the 254-value observation into a 40-by-10 spatial feature map."""

    def __init__(self, obs_dim: int, channels: int):
        super().__init__()
        if obs_dim < 200:
            raise ValueError("Placement-map networks require the 200 board observation values.")
        self.obs_dim = int(obs_dim)
        self.board_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.board_global = nn.Linear(200, channels)
        self.context = nn.Linear(obs_dim - 200, channels) if obs_dim > 200 else None
        self.refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, channels)

        y = torch.linspace(0.0, 1.0, BOARD_ROWS).view(1, 1, BOARD_ROWS, 1)
        x = torch.linspace(0.0, 1.0, BOARD_WIDTH).view(1, 1, 1, BOARD_WIDTH)
        self.register_buffer("y_coord", y.expand(1, 1, BOARD_ROWS, BOARD_WIDTH).clone())
        self.register_buffer("x_coord", x.expand(1, 1, BOARD_ROWS, BOARD_WIDTH).clone())

    def forward(self, obs: Tensor) -> Tensor:  # type: ignore[override]
        batch = obs.shape[0]
        board_flat = obs[:, :200]
        board = board_flat.reshape(batch, 1, 20, BOARD_WIDTH)
        board = F.pad(board, (0, 0, 0, BOARD_ROWS - 20))
        spatial = torch.cat(
            (
                board,
                self.y_coord.expand(batch, -1, -1, -1),
                self.x_coord.expand(batch, -1, -1, -1),
            ),
            dim=1,
        )
        conditioning = self.board_global(board_flat)
        if self.context is not None:
            conditioning = conditioning + self.context(obs[:, 200:])
        features = F.silu(self.board_conv(spatial) + conditioning[:, :, None, None])
        return F.silu(self.norm(features + self.refine(features)))


class StructuredQNetwork(nn.Module):
    """A scalar Q value at every stable Tetris placement ID."""

    def __init__(self, obs_dim: int, channels: int = 32):
        super().__init__()
        self.state = PlacementStateEncoder(obs_dim, channels)
        self.refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, channels)
        self.output = nn.Conv2d(channels, ACTION_MAP_SHAPE[0], kernel_size=1)
        _initialize(self)

    def forward(self, obs: Tensor) -> Tensor:  # type: ignore[override]
        features = self.state(obs)
        features = F.silu(self.norm(features + self.refine(features)))
        return self.output(features)


class FlowVectorField(nn.Module):
    """Time-conditioned velocity field over the full placement map."""

    def __init__(self, obs_dim: int, channels: int = 32):
        super().__init__()
        self.state = PlacementStateEncoder(obs_dim, channels)
        self.action = nn.Conv2d(ACTION_MAP_SHAPE[0], channels, kernel_size=3, padding=1)
        self.time = nn.Sequential(
            nn.Linear(1, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
        )
        self.output = nn.Conv2d(channels, ACTION_MAP_SHAPE[0], kernel_size=1)
        _initialize(self, small_output=True)

    def encode_state(self, obs: Tensor) -> Tensor:
        """Encode state once for repeated vector-field evaluations."""

        return self.state(obs)

    def forward_encoded(
        self,
        state_features: Tensor,
        action_map: Tensor,
        time: Tensor,
    ) -> Tensor:
        """Evaluate the field from cached state features."""

        time = time.reshape(state_features.shape[0], 1).to(dtype=state_features.dtype)
        features = state_features + self.action(action_map) + self.time(time)[:, :, None, None]
        return self.output(self.refine(F.silu(features)))

    def forward(self, obs: Tensor, action_map: Tensor, time: Tensor) -> Tensor:  # type: ignore[override]
        return self.forward_encoded(self.encode_state(obs), action_map, time)


class OneStepPlacementActor(nn.Module):
    """Distilled one-pass policy from Gaussian noise to placement logits."""

    def __init__(self, obs_dim: int, channels: int = 32):
        super().__init__()
        self.state = PlacementStateEncoder(obs_dim, channels)
        self.source = nn.Conv2d(ACTION_MAP_SHAPE[0], channels, kernel_size=3, padding=1)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
        )
        self.output = nn.Conv2d(channels, ACTION_MAP_SHAPE[0], kernel_size=1)
        _initialize(self, small_output=True)

    def forward(self, obs: Tensor, source_noise: Tensor) -> Tensor:  # type: ignore[override]
        features = F.silu(self.state(obs) + self.source(source_noise))
        return self.output(self.refine(features))


def _initialize(module: nn.Module, *, small_output: bool = False) -> None:
    for child in module.modules():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(child.weight, nonlinearity="relu")
            if child.bias is not None:
                nn.init.zeros_(child.bias)
    output = getattr(module, "output", None)
    if small_output and isinstance(output, nn.Conv2d):
        nn.init.uniform_(output.weight, -1e-3, 1e-3)


@dataclass
class FlowDQNConfig:
    obs_dim: int
    action_dim: int
    channels: int = 32
    critic_learning_rate: float = 3e-4
    flow_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    flow_steps: int = 10
    distill_coef: float = 1.0
    q_guidance_coef: float = 1.0
    flow_t0_ce_coef: float = 1.0
    actor_bc_coef: float = 1.0
    action_logit_scale: float = 10.0
    normalized_q: bool = True
    policy_temperature: float = 1.0
    source_noise_std: float = SOURCE_NOISE_STD
    max_grad_norm: float = 10.0
    device: Optional[str] = None


class FlowReplayBuffer:
    """Ring replay with bit-packed current and next legal-action masks."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        if action_dim != PLACEMENT_ACTION_DIM:
            raise ValueError(f"Flow-DQN requires action_dim={PLACEMENT_ACTION_DIM}")
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        self.obs = np.empty((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.empty((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty(self.capacity, dtype=np.int64)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.terminated = np.empty(self.capacity, dtype=np.uint8)
        self.truncated = np.empty(self.capacity, dtype=np.uint8)
        self.action_masks = np.empty((self.capacity, PACKED_MASK_BYTES), dtype=np.uint8)
        self.next_action_masks = np.empty((self.capacity, PACKED_MASK_BYTES), dtype=np.uint8)
        self.size = 0
        self.pos = 0

    def __len__(self) -> int:
        return self.size

    @staticmethod
    def _pack(mask: np.ndarray) -> np.ndarray:
        return np.packbits(np.asarray(mask) > 0.5, bitorder="little")

    @staticmethod
    def _unpack(mask: np.ndarray) -> np.ndarray:
        return np.unpackbits(mask, axis=-1, count=PLACEMENT_ACTION_DIM, bitorder="little")

    def add(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        next_action_mask: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        index = self.pos
        self.obs[index] = obs
        self.action_masks[index] = self._pack(action_mask)
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.next_obs[index] = next_obs
        self.next_action_masks[index] = self._pack(next_action_mask)
        self.terminated[index] = bool(terminated)
        self.truncated[index] = bool(truncated)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        generator = rng or np.random.default_rng()
        indices = generator.integers(0, self.size, size=int(batch_size))
        return {
            "obs": self.obs[indices],
            "action_masks": self._unpack(self.action_masks[indices]),
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "next_action_masks": self._unpack(self.next_action_masks[indices]),
            "terminated": self.terminated[indices].astype(np.float32),
            "truncated": self.truncated[indices].astype(np.float32),
        }


class FlowDQNAgent:
    """Twin-Q discrete flow agent with one-step placement inference."""

    def __init__(self, config: FlowDQNConfig):
        if config.action_dim != PLACEMENT_ACTION_DIM:
            raise ValueError(f"Flow-DQN requires action_dim={PLACEMENT_ACTION_DIM}")
        if config.flow_steps <= 0:
            raise ValueError("flow_steps must be positive.")
        self.config = config
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.actor = OneStepPlacementActor(config.obs_dim, config.channels).to(self.device)
        self.flow = FlowVectorField(config.obs_dim, config.channels).to(self.device)
        self.critic1 = StructuredQNetwork(config.obs_dim, config.channels).to(self.device)
        self.critic2 = StructuredQNetwork(config.obs_dim, config.channels).to(self.device)
        self.target_critic1 = StructuredQNetwork(config.obs_dim, config.channels).to(self.device)
        self.target_critic2 = StructuredQNetwork(config.obs_dim, config.channels).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_critic1.requires_grad_(False)
        self.target_critic2.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_learning_rate)
        self.flow_optimizer = torch.optim.Adam(self.flow.parameters(), lr=config.flow_learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            tuple(self.critic1.parameters()) + tuple(self.critic2.parameters()),
            lr=config.critic_learning_rate,
        )

        self.update_steps = 0
        self.offline_update_steps = 0
        self.online_update_steps = 0
        self.environment_steps = 0

    def _obs_tensor(self, obs: np.ndarray | Tensor) -> Tensor:
        if isinstance(obs, Tensor):
            return obs.to(self.device, dtype=torch.float32)
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def _flat_mask(self, mask: np.ndarray | Tensor, batch_size: int) -> Tensor:
        value = torch.as_tensor(mask, device=self.device) if not isinstance(mask, Tensor) else mask.to(self.device)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        value = value > 0.5
        expected = (batch_size, PLACEMENT_ACTION_DIM)
        if tuple(value.shape) != expected:
            raise ValueError(f"Action mask shape mismatch: expected {expected}, got {tuple(value.shape)}")
        if torch.any(~value.any(dim=-1)):
            raise ValueError("Action mask contains no legal actions.")
        return value

    def _masked_logits(self, logits: Tensor, mask: np.ndarray | Tensor) -> Tuple[Tensor, Tensor]:
        flat = action_map_to_flat(logits)
        legal = self._flat_mask(mask, flat.shape[0])
        return flat.masked_fill(~legal, torch.finfo(flat.dtype).min), legal

    def policy_logits(self, obs: Tensor, source_noise: Optional[Tensor] = None) -> Tensor:
        if source_noise is None:
            source_noise = torch.zeros(
                (obs.shape[0], *ACTION_MAP_SHAPE), dtype=obs.dtype, device=obs.device
            )
        return self.actor(obs, source_noise)

    def sample_source_noise(self, batch_size: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
        """Draw a Gaussian whose expected full-map norm is approximately one."""

        return torch.randn(
            (batch_size, *ACTION_MAP_SHAPE), dtype=dtype, device=self.device
        ) * float(self.config.source_noise_std)

    def select_action(
        self,
        obs: np.ndarray,
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        obs_tensor = self._obs_tensor(obs).reshape(1, -1)
        with torch.no_grad():
            if deterministic:
                source = torch.zeros((1, *ACTION_MAP_SHAPE), device=self.device)
            else:
                source = self.sample_source_noise(1) * max(float(temperature), 0.0)
            logits = self.actor(obs_tensor, source)
            flat = action_map_to_flat(logits)
            if action_mask is not None:
                flat, _ = self._masked_logits(logits, action_mask)
            return int(flat.argmax(dim=-1).item())

    def integrate_flow(
        self,
        obs: Tensor,
        source_noise: Tensor,
        *,
        steps: Optional[int] = None,
    ) -> Tensor:
        """Euler-integrate the learned vector field from time zero to one."""

        count = int(steps or self.config.flow_steps)
        if count <= 0:
            raise ValueError("Euler integration requires at least one step.")
        value = source_noise
        state_features = self.flow.encode_state(obs)
        dt = 1.0 / float(count)
        for step in range(count):
            time = torch.full((obs.shape[0], 1), step * dt, dtype=obs.dtype, device=obs.device)
            value = value + dt * self.flow.forward_encoded(state_features, value, time)
        return value

    def flow_matching_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        *,
        action_masks: Optional[Tensor] = None,
        source_noise: Optional[Tensor] = None,
        times: Optional[Tensor] = None,
    ) -> Tensor:
        return self.flow_losses(
            obs,
            actions,
            action_masks=action_masks,
            source_noise=source_noise,
            times=times,
        )[0]

    def flow_losses(
        self,
        obs: Tensor,
        actions: Tensor,
        *,
        action_masks: Optional[Tensor],
        source_noise: Optional[Tensor] = None,
        times: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return total, uniform flow-matching, t=0 CE, and t=0 agreement."""

        batch_size = obs.shape[0]
        source = source_noise
        if source is None:
            source = self.sample_source_noise(batch_size, dtype=obs.dtype)
        endpoint = F.one_hot(actions.long(), num_classes=PLACEMENT_ACTION_DIM).to(obs.dtype)
        endpoint = flat_to_action_map(endpoint)
        if times is None:
            times = torch.rand((batch_size, 1), device=obs.device, dtype=obs.dtype)
        interpolation = times.reshape(batch_size, 1, 1, 1)
        interpolated = (1.0 - interpolation) * source + interpolation * endpoint
        target_velocity = endpoint - source
        state_features = self.flow.encode_state(obs)
        predicted_velocity = self.flow.forward_encoded(state_features, interpolated, times)
        # A mean over 3,200 cells would dilute the one selected endpoint by
        # 1/3,200.  Sum within each map and average only across the batch.
        matching = (predicted_velocity - target_velocity).square().flatten(1).sum(dim=1).mean()

        if action_masks is None:
            zero = matching.new_zeros(())
            return matching, matching, zero, zero

        zero_time = torch.zeros((batch_size, 1), dtype=obs.dtype, device=obs.device)
        initial_velocity = self.flow.forward_encoded(state_features, source, zero_time)
        initial_endpoint = source + initial_velocity
        initial_logits, _ = self._masked_logits(
            initial_endpoint * self.config.action_logit_scale,
            action_masks,
        )
        t0_ce = F.cross_entropy(initial_logits, actions.long())
        t0_agreement = (initial_logits.argmax(dim=-1) == actions.long()).float().mean()
        total = matching + self.config.flow_t0_ce_coef * t0_ce
        return total, matching, t0_ce, t0_agreement

    def actor_losses(
        self,
        obs: Tensor,
        action_masks: Tensor,
        *,
        actions: Optional[Tensor] = None,
        source_noise: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size = obs.shape[0]
        source = source_noise
        if source is None:
            source = self.sample_source_noise(batch_size, dtype=obs.dtype)
        with torch.no_grad():
            flow_target = self.integrate_flow(obs, source).detach()
            q_values = torch.minimum(self.critic1(obs), self.critic2(obs)).detach()
        prediction = self.actor(obs, source)
        distillation = (prediction - flow_target).square().flatten(1).sum(dim=1).mean()

        scaled_prediction = prediction * self.config.action_logit_scale
        masked_logits, _ = self._masked_logits(scaled_prediction, action_masks)
        probabilities = torch.softmax(masked_logits / max(self.config.policy_temperature, 1e-6), dim=-1)
        flat_q = action_map_to_flat(q_values)
        expected_q = (probabilities * flat_q).sum(dim=-1)
        q_guidance = self.q_guidance_loss(expected_q)
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1).mean()

        behavior_ce = prediction.new_zeros(())
        agreement = prediction.new_zeros(())
        if actions is not None:
            behavior_ce = F.cross_entropy(masked_logits, actions.long())
            agreement = (masked_logits.argmax(dim=-1) == actions.long()).float().mean()

        total = (
            self.config.distill_coef * distillation
            + self.config.q_guidance_coef * q_guidance
            + self.config.actor_bc_coef * behavior_ce
        )
        return total, distillation, q_guidance, behavior_ce, agreement, entropy

    def q_guidance_loss(self, expected_q: Tensor) -> Tensor:
        """Maximize actor Q with the stop-gradient normalization used by FQL."""

        scale = expected_q.new_tensor(1.0)
        if self.config.normalized_q:
            scale = expected_q.detach().abs().mean().clamp(min=1e-6)
        return -expected_q.mean() / scale

    def _tensor_batch(self, batch: Mapping[str, np.ndarray | Tensor]) -> Dict[str, Tensor]:
        float_fields = ("obs", "rewards", "next_obs", "terminated", "truncated")
        tensors = {
            key: self._obs_tensor(batch[key]) if key in float_fields else torch.as_tensor(batch[key], device=self.device)
            for key in (
                "obs",
                "action_masks",
                "actions",
                "rewards",
                "next_obs",
                "next_action_masks",
                "terminated",
                "truncated",
            )
        }
        tensors["actions"] = tensors["actions"].long()
        return tensors

    def bellman_targets(
        self,
        rewards: Tensor,
        next_obs: Tensor,
        terminated: Tensor,
        next_action_masks: Tensor,
    ) -> Tensor:
        """Build targets; true terminals stop bootstrap while time limits do not."""

        targets = rewards.clone()
        active = terminated < 0.5
        if not torch.any(active):
            return targets
        active_obs = next_obs[active]
        active_masks = next_action_masks[active]
        with torch.no_grad():
            source = torch.zeros((active_obs.shape[0], *ACTION_MAP_SHAPE), device=self.device)
            actor_logits = self.actor(active_obs, source)
            masked_logits, _ = self._masked_logits(actor_logits, active_masks)
            next_actions = masked_logits.argmax(dim=-1)
            q1 = action_map_to_flat(self.target_critic1(active_obs))
            q2 = action_map_to_flat(self.target_critic2(active_obs))
            next_q = torch.minimum(q1, q2).gather(1, next_actions[:, None]).squeeze(1)
            targets[active] = targets[active] + self.config.gamma * next_q
        return targets

    def _critic_loss_tensors(self, tensors: Mapping[str, Tensor]) -> Tuple[Tensor, Tensor]:
        targets = self.bellman_targets(
            tensors["rewards"],
            tensors["next_obs"],
            tensors["terminated"],
            tensors["next_action_masks"],
        )
        actions = tensors["actions"].unsqueeze(1)
        q1 = action_map_to_flat(self.critic1(tensors["obs"])).gather(1, actions).squeeze(1)
        q2 = action_map_to_flat(self.critic2(tensors["obs"])).gather(1, actions).squeeze(1)
        loss = F.smooth_l1_loss(q1, targets) + F.smooth_l1_loss(q2, targets)
        return loss, targets

    def critic_loss(self, batch: Mapping[str, np.ndarray | Tensor]) -> Tuple[Tensor, Tensor]:
        return self._critic_loss_tensors(self._tensor_batch(batch))

    def _step_optimizer(self, optimizer: torch.optim.Optimizer, loss: Tensor, parameters) -> None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm)
        optimizer.step()

    def polyak_update(self) -> None:
        with torch.no_grad():
            torch._foreach_lerp_(
                tuple(self.target_critic1.parameters()),
                tuple(self.critic1.parameters()),
                self.config.tau,
            )
            torch._foreach_lerp_(
                tuple(self.target_critic2.parameters()),
                tuple(self.critic2.parameters()),
                self.config.tau,
            )

    def update(
        self,
        batch: Mapping[str, np.ndarray | Tensor],
        *,
        source: str = "online",
    ) -> Dict[str, float]:
        tensors = self._tensor_batch(batch)

        critic_loss, targets = self._critic_loss_tensors(tensors)
        critic_parameters = tuple(self.critic1.parameters()) + tuple(self.critic2.parameters())
        self._step_optimizer(self.critic_optimizer, critic_loss, critic_parameters)

        flow_loss, flow_matching, flow_t0_ce, flow_t0_agreement = self.flow_losses(
            tensors["obs"],
            tensors["actions"],
            action_masks=tensors["action_masks"],
        )
        self._step_optimizer(self.flow_optimizer, flow_loss, self.flow.parameters())

        actor_total, distillation, q_guidance, actor_bc, actor_agreement, actor_entropy = (
            self.actor_losses(
                tensors["obs"],
                tensors["action_masks"],
                actions=tensors["actions"],
            )
        )
        self._step_optimizer(self.actor_optimizer, actor_total, self.actor.parameters())
        self.polyak_update()

        self.update_steps += 1
        if source == "offline":
            self.offline_update_steps += 1
        else:
            self.online_update_steps += 1
        return {
            "critic_loss": float(critic_loss.detach().item()),
            "flow_loss": float(flow_loss.detach().item()),
            "flow_matching_loss": float(flow_matching.detach().item()),
            "flow_t0_ce_loss": float(flow_t0_ce.detach().item()),
            "flow_t0_agreement": float(flow_t0_agreement.detach().item()),
            "actor_loss": float(actor_total.detach().item()),
            "distillation_loss": float(distillation.detach().item()),
            "q_guidance_loss": float(q_guidance.detach().item()),
            "actor_bc_loss": float(actor_bc.detach().item()),
            "actor_executed_agreement": float(actor_agreement.detach().item()),
            "actor_masked_entropy": float(actor_entropy.detach().item()),
            "target_mean": float(targets.detach().mean().item()),
        }

    def save(self, path: str | Path, *, metadata: Optional[Dict[str, object]] = None) -> None:
        payload = {
            "checkpoint_version": 1,
            "algo": "flow_dqn",
            "config": asdict(self.config),
            "observation_schema": {
                "name": (
                    "tetris_v2_254"
                    if self.config.obs_dim == CURRENT_OBSERVATION_DIM
                    else "flat"
                ),
                "obs_dim": self.config.obs_dim,
            },
            "action_schema": {
                "shape": list(ACTION_MAP_SHAPE),
                "order": ACTION_ORDER,
                "action_dim": PLACEMENT_ACTION_DIM,
            },
            "actor_state_dict": self.actor.state_dict(),
            "flow_state_dict": self.flow.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "target_critic1_state_dict": self.target_critic1.state_dict(),
            "target_critic2_state_dict": self.target_critic2.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "flow_optimizer_state_dict": self.flow_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "training_counters": {
                "update_steps": self.update_steps,
                "offline_update_steps": self.offline_update_steps,
                "online_update_steps": self.online_update_steps,
                "environment_steps": self.environment_steps,
            },
            "metadata": metadata or {},
        }
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: Optional[str] = None,
    ) -> Tuple["FlowDQNAgent", Dict[str, object]]:
        payload = torch.load(path, map_location=device or "cpu", weights_only=False)
        if payload.get("algo") != "flow_dqn":
            raise ValueError("Checkpoint is not a Flow-DQN checkpoint.")
        schema = payload.get("action_schema", {})
        if (
            tuple(schema.get("shape", ())) != ACTION_MAP_SHAPE
            or schema.get("order") != ACTION_ORDER
            or int(schema.get("action_dim", -1)) != PLACEMENT_ACTION_DIM
        ):
            raise ValueError("Flow-DQN checkpoint uses an incompatible action layout.")

        config_values = dict(payload["config"])
        observation_schema = payload.get("observation_schema")
        if observation_schema is not None:
            expected_name = (
                "tetris_v2_254"
                if int(config_values["obs_dim"]) == CURRENT_OBSERVATION_DIM
                else "flat"
            )
            if (
                observation_schema.get("name") != expected_name
                or int(observation_schema.get("obs_dim", -1)) != int(config_values["obs_dim"])
            ):
                raise ValueError("Flow-DQN checkpoint uses an incompatible observation schema.")
        config_values["device"] = device
        agent = cls(FlowDQNConfig(**config_values))
        agent.actor.load_state_dict(payload["actor_state_dict"])
        agent.flow.load_state_dict(payload["flow_state_dict"])
        agent.critic1.load_state_dict(payload["critic1_state_dict"])
        agent.critic2.load_state_dict(payload["critic2_state_dict"])
        agent.target_critic1.load_state_dict(payload["target_critic1_state_dict"])
        agent.target_critic2.load_state_dict(payload["target_critic2_state_dict"])
        agent.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        agent.flow_optimizer.load_state_dict(payload["flow_optimizer_state_dict"])
        agent.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
        counters = payload.get("training_counters", {})
        agent.update_steps = int(counters.get("update_steps", 0))
        agent.offline_update_steps = int(counters.get("offline_update_steps", 0))
        agent.online_update_steps = int(counters.get("online_update_steps", 0))
        agent.environment_steps = int(counters.get("environment_steps", 0))
        return agent, dict(payload.get("metadata", {}))


__all__ = [
    "ACTION_MAP_SHAPE",
    "ACTION_ORDER",
    "SOURCE_NOISE_STD",
    "FlowDQNAgent",
    "FlowDQNConfig",
    "FlowReplayBuffer",
    "FlowVectorField",
    "OneStepPlacementActor",
    "StructuredQNetwork",
    "action_map_to_flat",
    "flat_to_action_map",
]
