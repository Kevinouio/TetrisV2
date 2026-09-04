"""CPU-friendly Double-DQN components for player-perspective battle Tetris."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tetris_v2.rl.actions import (
    BOARD_ROWS,
    BOARD_WIDTH,
    PLACEMENT_ACTION_DIM,
    PLACEMENT_CHANNELS,
    PLACEMENT_MAP_SHAPE,
)
from tetris_v2.rl.dqn.core import PlacementConvNet

if TYPE_CHECKING:
    from tetris_v2.rl.dqn.core import DQNAgent


OWN_OBSERVATION_DIM = 254
OPPONENT_BOARD_CELLS = 200
BATTLE_FEATURE_DIM = 16
BATTLE_OBSERVATION_DIM = OWN_OBSERVATION_DIM + OPPONENT_BOARD_CELLS + BATTLE_FEATURE_DIM
BATTLE_OBSERVATION_SCHEMA = "tetris_v2_battle_470_v1"
BATTLE_ACTION_ORDER = "hold,rotation,y,x"

OWN_OBSERVATION_SLICE = slice(0, OWN_OBSERVATION_DIM)
OPPONENT_BOARD_SLICE = slice(
    OWN_OBSERVATION_DIM,
    OWN_OBSERVATION_DIM + OPPONENT_BOARD_CELLS,
)
BATTLE_FEATURE_SLICE = slice(
    OWN_OBSERVATION_DIM + OPPONENT_BOARD_CELLS,
    BATTLE_OBSERVATION_DIM,
)
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


@dataclass(frozen=True)
class LinearSchedule:
    """A serializable linear schedule with an explicit starting step."""

    start: float
    end: float
    duration: int
    start_step: int = 0

    def __post_init__(self) -> None:
        if self.duration < 0:
            raise ValueError("Schedule duration cannot be negative.")

    def value(self, step: int) -> float:
        if self.duration == 0:
            return float(self.end)
        progress = (int(step) - int(self.start_step)) / float(self.duration)
        progress = min(1.0, max(0.0, progress))
        return float(self.start + progress * (self.end - self.start))

    def state_dict(self) -> Dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "LinearSchedule":
        return cls(
            start=float(state["start"]),
            end=float(state["end"]),
            duration=int(state["duration"]),
            start_step=int(state.get("start_step", 0)),
        )


class BattlePlacementQNet(nn.Module):
    """Structured Q map with a checkpoint-compatible single-player branch.

    The first 254 inputs are passed through the existing ``PlacementConvNet``
    unchanged. Public opponent and garbage/board-summary features contribute a
    residual Q map. The residual projection starts at zero, so importing a
    single-player network preserves its Q values exactly at initialization.
    """

    def __init__(self, obs_dim: int = BATTLE_OBSERVATION_DIM, channels: int = 32):
        super().__init__()
        if int(obs_dim) != BATTLE_OBSERVATION_DIM:
            raise ValueError(
                f"BattlePlacementQNet requires obs_dim={BATTLE_OBSERVATION_DIM}, got {obs_dim}."
            )
        self.obs_dim = int(obs_dim)
        self.channels = int(channels)
        if self.channels <= 0:
            raise ValueError("channels must be positive.")

        self.own = PlacementConvNet()
        self.opponent_board_conv = nn.Conv2d(3, self.channels, kernel_size=3, padding=1)
        self.opponent_global = nn.Linear(OPPONENT_BOARD_CELLS, self.channels)
        self.battle_context = nn.Linear(BATTLE_FEATURE_DIM, self.channels)
        self.battle_refine = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1)
        self.battle_output = nn.Conv2d(
            self.channels,
            PLACEMENT_CHANNELS,
            kernel_size=1,
        )
        self.reset_battle_branch()

    def reset_battle_branch(self) -> None:
        """Reset new battle features while keeping their initial contribution zero."""

        for module in (
            self.opponent_board_conv,
            self.opponent_global,
            self.battle_context,
            self.battle_refine,
        ):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.battle_output.weight)
        nn.init.zeros_(self.battle_output.bias)

    def forward(self, observation: Tensor) -> Tensor:  # type: ignore[override]
        if observation.ndim != 2 or observation.shape[1] != self.obs_dim:
            raise ValueError(
                f"Expected battle observations shaped [batch,{self.obs_dim}], "
                f"got {tuple(observation.shape)}."
            )
        batch = observation.shape[0]
        own_q = self.own(observation[:, OWN_OBSERVATION_SLICE])

        opponent_flat = observation[:, OPPONENT_BOARD_SLICE]
        opponent_board = opponent_flat.reshape(batch, 1, 20, BOARD_WIDTH)
        opponent_board = F.pad(opponent_board, (0, 0, 0, BOARD_ROWS - 20))
        spatial = torch.cat(
            (
                opponent_board,
                self.own.y_coord.expand(batch, -1, -1, -1),
                self.own.x_coord.expand(batch, -1, -1, -1),
            ),
            dim=1,
        )
        conditioning = self.opponent_global(opponent_flat) + self.battle_context(
            observation[:, BATTLE_FEATURE_SLICE]
        )
        features = F.relu(
            self.opponent_board_conv(spatial) + conditioning[:, :, None, None]
        )
        features = F.relu(features + self.battle_refine(features))
        residual = self.battle_output(features).reshape(batch, PLACEMENT_ACTION_DIM)
        return own_q + residual


@dataclass
class BattleDQNConfig:
    obs_dim: int = BATTLE_OBSERVATION_DIM
    action_dim: int = PLACEMENT_ACTION_DIM
    channels: int = 32
    learning_rate: float = 1e-3
    gamma: float = 0.99
    target_sync_interval: int = 1_000
    max_grad_norm: float = 10.0
    double_dqn: bool = True
    seed: int = 0
    device: Optional[str] = None


class BattleDQNAgent:
    """Masked Double DQN for a shared battle policy."""

    def __init__(self, config: BattleDQNConfig):
        if int(config.obs_dim) != BATTLE_OBSERVATION_DIM:
            raise ValueError(
                f"Battle DQN requires obs_dim={BATTLE_OBSERVATION_DIM}, got {config.obs_dim}."
            )
        if int(config.action_dim) != PLACEMENT_ACTION_DIM:
            raise ValueError(
                f"Battle DQN requires action_dim={PLACEMENT_ACTION_DIM}, got {config.action_dim}."
            )
        self.config = config
        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.online = BattlePlacementQNet(config.obs_dim, config.channels).to(self.device)
        self.target = BattlePlacementQNet(config.obs_dim, config.channels).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=float(config.learning_rate),
        )
        self.update_steps = 0
        self.action_rng = np.random.default_rng(config.seed)

    @property
    def current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def set_learning_rate(self, learning_rate: float) -> float:
        value = float(learning_rate)
        for group in self.optimizer.param_groups:
            group["lr"] = value
        return value

    def apply_learning_rate_schedule(self, step: int, schedule: LinearSchedule) -> float:
        return self.set_learning_rate(schedule.value(step))

    @staticmethod
    def epsilon_at(step: int, schedule: LinearSchedule) -> float:
        return schedule.value(step)

    def _obs_tensor(self, value: np.ndarray | Tensor) -> Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _masked_q(self, q_values: Tensor, action_mask: np.ndarray | Tensor) -> Tensor:
        mask = torch.as_tensor(action_mask, device=self.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        legal = mask > 0.5
        if tuple(legal.shape) != tuple(q_values.shape):
            raise ValueError(
                f"Action mask shape mismatch: mask={tuple(legal.shape)}, "
                f"q={tuple(q_values.shape)}."
            )
        if torch.any(~legal.any(dim=-1)):
            raise ValueError("Action mask contains no legal actions.")
        return q_values.masked_fill(~legal, torch.finfo(q_values.dtype).min)

    def select_actions(
        self,
        observations: np.ndarray,
        *,
        action_masks: np.ndarray,
        epsilon: float = 0.0,
        deterministic: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float32)
        masks = np.asarray(action_masks)
        if obs.ndim == 1:
            obs = obs[None, :]
        if masks.ndim == 1:
            masks = masks[None, :]
        if obs.shape != (masks.shape[0], self.config.obs_dim):
            raise ValueError("Observation batch does not match the action-mask batch.")

        with torch.no_grad():
            q_values = self.online(self._obs_tensor(obs))
            masked = self._masked_q(q_values, masks)
            actions = masked.argmax(dim=-1).cpu().numpy().astype(np.int64)

        explore_rate = 0.0 if deterministic else min(1.0, max(0.0, float(epsilon)))
        if explore_rate == 0.0:
            return actions
        generator = rng or self.action_rng
        explore = generator.random(obs.shape[0]) < explore_rate
        for index in np.flatnonzero(explore):
            legal = np.flatnonzero(masks[index] > 0.5)
            if legal.size == 0:
                raise ValueError("Action mask contains no legal actions.")
            actions[index] = int(generator.choice(legal))
        return actions

    def select_action(
        self,
        observation: np.ndarray,
        *,
        action_mask: np.ndarray,
        epsilon: float = 0.0,
        deterministic: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        actions = self.select_actions(
            observation,
            action_masks=action_mask,
            epsilon=epsilon,
            deterministic=deterministic,
            rng=rng,
        )
        return int(actions[0])

    def q_values(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        with torch.no_grad():
            values = self.online(self._obs_tensor(obs))
        return values.cpu().numpy()

    def compute_td_loss(self, batch: Mapping[str, np.ndarray | Tensor]) -> Tensor:
        obs = self._obs_tensor(batch["obs"])
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = self._obs_tensor(batch["rewards"])
        next_obs = self._obs_tensor(batch["next_obs"])
        terminated = self._obs_tensor(batch["terminated"])
        # Truncations intentionally bootstrap. Reading the field makes the
        # transition contract explicit and catches incomplete replay batches.
        self._obs_tensor(batch["truncated"])
        next_action_masks = torch.as_tensor(batch["next_action_masks"], device=self.device)

        predicted = self.online(obs).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_values = torch.zeros_like(rewards)
            bootstrap = terminated < 0.5
            if torch.any(bootstrap):
                active_obs = next_obs[bootstrap]
                active_masks = next_action_masks[bootstrap]
                if self.config.double_dqn:
                    online_next = self._masked_q(self.online(active_obs), active_masks)
                    next_actions = online_next.argmax(dim=-1, keepdim=True)
                    target_next = self.target(active_obs).gather(1, next_actions).squeeze(1)
                else:
                    target_next = self._masked_q(
                        self.target(active_obs),
                        active_masks,
                    ).max(dim=-1).values
                next_values[bootstrap] = target_next
            targets = rewards + float(self.config.gamma) * next_values
        return F.smooth_l1_loss(predicted, targets)

    def update(self, batch: Mapping[str, np.ndarray | Tensor]) -> Dict[str, float]:
        loss = self.compute_td_loss(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(),
            float(self.config.max_grad_norm),
        ) if self.config.max_grad_norm > 0 else torch.zeros((), device=self.device)
        self.optimizer.step()
        self.update_steps += 1
        if (
            self.config.target_sync_interval > 0
            and self.update_steps % int(self.config.target_sync_interval) == 0
        ):
            self.sync_target()
        return {
            "td_loss": float(loss.detach().item()),
            "grad_norm": float(grad_norm.detach().item()),
            "learning_rate": self.current_learning_rate,
            "update_steps": float(self.update_steps),
        }

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def reset_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=float(self.config.learning_rate),
        )
        self.update_steps = 0

    def warm_start_from_dqn(self, source: "DQNAgent") -> None:
        """Import a 254/3,200 placement-convolution DQN as fresh battle weights."""

        if int(source.config.obs_dim) != OWN_OBSERVATION_DIM:
            raise ValueError("Warm-start DQN must use the 254-value observation schema.")
        if int(source.config.action_dim) != PLACEMENT_ACTION_DIM:
            raise ValueError("Warm-start DQN must use the 3,200-action placement schema.")
        if source.network_type != "placement_conv" or not isinstance(
            source.online.net,
            PlacementConvNet,
        ):
            raise ValueError("Warm-start DQN must use the placement-convolution network.")

        self.online.reset_battle_branch()
        self.online.own.load_state_dict(source.online.net.state_dict())
        self.target.load_state_dict(self.online.state_dict())
        self.reset_optimizer()
        self.action_rng = np.random.default_rng(self.config.seed)

    def warm_start_from_checkpoint(
        self,
        checkpoint: str | Path,
    ) -> Dict[str, Any]:
        from tetris_v2.rl.dqn.core import DQNAgent

        source, metadata = DQNAgent.load(str(checkpoint), device=str(self.device))
        self.warm_start_from_dqn(source)
        return dict(metadata)

    def save_frozen(
        self,
        path: str | Path,
        *,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Path:
        """Save a compact inference snapshot without optimizer, target, or replay."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algo": "battle_dqn",
            "checkpoint_version": 1,
            "checkpoint_type": "frozen_policy",
            "observation_schema": {
                "name": BATTLE_OBSERVATION_SCHEMA,
                "obs_dim": BATTLE_OBSERVATION_DIM,
            },
            "action_schema": {
                "shape": list(PLACEMENT_MAP_SHAPE),
                "order": BATTLE_ACTION_ORDER,
                "action_dim": PLACEMENT_ACTION_DIM,
            },
            "config": asdict(self.config),
            "model_state_dict": self.online.state_dict(),
            "metadata": dict(metadata or {}),
        }
        temporary = destination.parent / f".{destination.name}.tmp"
        torch.save(payload, temporary)
        temporary.replace(destination)
        return destination

    @classmethod
    def load_frozen(
        cls,
        path: str | Path,
        *,
        device: Optional[str] = "cpu",
    ) -> tuple["BattleDQNAgent", Dict[str, object]]:
        payload = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("Frozen Battle DQN checkpoint payload is malformed.")
        return cls.from_frozen_payload(payload, device=device)

    @classmethod
    def from_frozen_payload(
        cls,
        payload: Mapping[str, object],
        *,
        device: Optional[str] = "cpu",
    ) -> tuple["BattleDQNAgent", Dict[str, object]]:
        """Load a compact policy stored directly inside a trainer checkpoint."""

        cpu_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() and torch.cuda.is_initialized()
            else None
        )
        try:
            cls._validate_state_schema(payload)
            if payload.get("checkpoint_type") != "frozen_policy":
                raise ValueError("Battle DQN checkpoint is not a frozen policy snapshot.")
            raw_config = payload.get("config")
            if not isinstance(raw_config, Mapping):
                raise ValueError("Frozen Battle DQN config is malformed.")
            config_values = dict(raw_config)
            config_values["device"] = device
            agent = cls(BattleDQNConfig(**config_values))
            agent.online.load_state_dict(payload["model_state_dict"])
            agent.sync_target()
            agent.reset_optimizer()
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise ValueError("Frozen Battle DQN metadata is malformed.")
            return agent, dict(metadata)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    def state_dict(self) -> Dict[str, object]:
        return {
            "algo": "battle_dqn",
            "checkpoint_version": 1,
            "observation_schema": {
                "name": BATTLE_OBSERVATION_SCHEMA,
                "obs_dim": BATTLE_OBSERVATION_DIM,
            },
            "action_schema": {
                "shape": list(PLACEMENT_MAP_SHAPE),
                "order": BATTLE_ACTION_ORDER,
                "action_dim": PLACEMENT_ACTION_DIM,
            },
            "config": asdict(self.config),
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_steps": int(self.update_steps),
            "action_rng_state": self.action_rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self._validate_state_schema(state)
        config = state["config"]
        if not isinstance(config, Mapping):
            raise ValueError("Battle DQN checkpoint config is malformed.")
        if int(config["obs_dim"]) != self.config.obs_dim or int(
            config["action_dim"]
        ) != self.config.action_dim:
            raise ValueError("Battle DQN checkpoint dimensions do not match this agent.")
        self.online.load_state_dict(state["online_state_dict"])  # type: ignore[arg-type]
        self.target.load_state_dict(state["target_state_dict"])  # type: ignore[arg-type]
        self.optimizer.load_state_dict(state["optimizer_state_dict"])  # type: ignore[arg-type]
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, Tensor):
                    optimizer_state[key] = value.to(self.device)
        self.update_steps = int(state.get("update_steps", 0))
        self.action_rng.bit_generator.state = state["action_rng_state"]  # type: ignore[assignment]

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
        *,
        device: Optional[str] = None,
    ) -> "BattleDQNAgent":
        cls._validate_state_schema(state)
        raw_config = state["config"]
        if not isinstance(raw_config, Mapping):
            raise ValueError("Battle DQN checkpoint config is malformed.")
        config_values = dict(raw_config)
        config_values["device"] = device
        agent = cls(BattleDQNConfig(**config_values))
        agent.load_state_dict(state)
        return agent

    @staticmethod
    def _validate_state_schema(state: Mapping[str, object]) -> None:
        if state.get("algo") != "battle_dqn":
            raise ValueError("Checkpoint is not a Battle DQN checkpoint.")
        observation = state.get("observation_schema")
        action = state.get("action_schema")
        if not isinstance(observation, Mapping) or (
            observation.get("name") != BATTLE_OBSERVATION_SCHEMA
            or int(observation.get("obs_dim", -1)) != BATTLE_OBSERVATION_DIM
        ):
            raise ValueError("Battle DQN checkpoint observation schema is incompatible.")
        if not isinstance(action, Mapping) or (
            tuple(action.get("shape", ())) != PLACEMENT_MAP_SHAPE
            or action.get("order") != BATTLE_ACTION_ORDER
            or int(action.get("action_dim", -1)) != PLACEMENT_ACTION_DIM
        ):
            raise ValueError("Battle DQN checkpoint action schema is incompatible.")


__all__ = [
    "BATTLE_ACTION_ORDER",
    "BATTLE_FEATURE_DIM",
    "BATTLE_FEATURE_NAMES",
    "BATTLE_FEATURE_SLICE",
    "BATTLE_OBSERVATION_DIM",
    "BATTLE_OBSERVATION_SCHEMA",
    "BattleDQNAgent",
    "BattleDQNConfig",
    "BattlePlacementQNet",
    "LinearSchedule",
    "OPPONENT_BOARD_SLICE",
    "OWN_OBSERVATION_SLICE",
]
