"""Native PyTorch DQN utilities (observation encoder, replay buffer, agent)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F


DEFAULT_NORMALISATION_FACTORS: Dict[str, np.ndarray | float] = {
    "current": np.array([6.0, 3.0, 20.0, 10.0], dtype=np.float32),
    "next": 6.0,
    "queue": 6.0,
    "hold": 6.0,
    "combo": 10.0,
    "back_to_back": 1.0,
    "level": 30.0,
    "lines": 200.0,
    "score": 100_000.0,
    "pending_garbage": 20.0,
    "opponent_pending": 20.0,
    "opponent_combo": 10.0,
    "opponent_back_to_back": 1.0,
}


class ObservationProcessor:
    """Flattens and normalises dict observations for the agent."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        normalization_overrides: Optional[Dict[str, np.ndarray | float]] = None,
        clip_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("ObservationProcessor requires a Dict observation space.")
        self.observation_space = observation_space
        self.keys = tuple(observation_space.spaces.keys())
        self._sizes: Dict[str, int] = {}
        flat_dim = 0
        for key in self.keys:
            space = observation_space.spaces[key]
            if isinstance(space, spaces.Discrete):
                size = 1
            else:
                shape = getattr(space, "shape", ()) or ()
                size = int(np.prod(shape, dtype=int)) if shape else 1
            self._sizes[key] = size
            flat_dim += size
        self.flat_dim = flat_dim
        self._clip_range = clip_range
        self._normalisers: Dict[str, np.ndarray] = {}
        overrides = normalization_overrides or {}
        for key in self.keys:
            factor = overrides.get(key, DEFAULT_NORMALISATION_FACTORS.get(key))
            if factor is None:
                continue
            arr = np.asarray(factor, dtype=np.float32)
            arr = np.where(arr == 0.0, 1.0, np.abs(arr))
            self._normalisers[key] = arr

    def flatten(self, observation: Dict[str, Any]) -> np.ndarray:
        """Convert a dict observation into a flat, scaled float32 vector."""
        flat = np.zeros(self.flat_dim, dtype=np.float32)
        offset = 0
        low, high = self._clip_range
        for key in self.keys:
            value = observation[key]
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            norm = self._normalisers.get(key)
            if norm is not None:
                arr = arr / norm
            arr = np.clip(arr, low, high)
            target = self._sizes[key]
            if arr.size < target:
                padded = np.zeros(target, dtype=np.float32)
                padded[: arr.size] = arr
                arr = padded
            elif arr.size > target:
                arr = arr[:target]
            flat[offset : offset + target] = arr
            offset += target
        return flat


class ReplayBuffer:
    """Simple uniform replay buffer."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.size = 0
        self.pos = 0

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        idx = self.pos
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        batch_indices = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "obs": self.obs[batch_indices],
            "next_obs": self.next_obs[batch_indices],
            "actions": self.actions[batch_indices],
            "rewards": self.rewards[batch_indices],
            "dones": self.dones[batch_indices],
        }


class QNetwork(nn.Module):
    """Feed-forward Q-network with optional dueling heads."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Sequence[int],
        *,
        dueling: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        dims = [input_dim] + list(hidden_sizes)
        for i in range(len(hidden_sizes)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()
        last_dim = dims[-1] if hidden_sizes else input_dim
        self.dueling = dueling
        if dueling:
            self.value_head = nn.Linear(last_dim, 1)
            self.advantage_head = nn.Linear(last_dim, output_dim)
        else:
            self.output_head = nn.Linear(last_dim, output_dim)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.feature_extractor(x)
        if self.dueling:
            advantage = self.advantage_head(features)
            value = self.value_head(features)
            advantage = advantage - advantage.mean(dim=1, keepdim=True)
            return value + advantage
        return self.output_head(features)


@dataclass
class AgentConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: Tuple[int, ...] = (512, 256)
    gamma: float = 0.99
    learning_rate: float = 2.5e-4
    target_sync_interval: int = 1_000
    max_grad_norm: float = 5.0
    device: Optional[str] = None
    use_dueling: bool = True
    use_double_q: bool = True


class DQNAgent:
    """Minimal DQN agent with hard target updates."""

    def __init__(self, config: AgentConfig):
        self.config = config
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.obs_dim = int(config.obs_dim)
        self.action_dim = int(config.action_dim)
        self.hidden_sizes = tuple(config.hidden_sizes)
        self.gamma = float(config.gamma)
        self.learning_rate = float(config.learning_rate)
        self.target_sync_interval = int(config.target_sync_interval)
        self.max_grad_norm = float(config.max_grad_norm)
        self.use_dueling = bool(config.use_dueling)
        self.use_double_q = bool(config.use_double_q)

        self.q_network = QNetwork(
            self.obs_dim,
            self.action_dim,
            self.hidden_sizes,
            dueling=self.use_dueling,
        ).to(self.device)
        self.target_network = QNetwork(
            self.obs_dim,
            self.action_dim,
            self.hidden_sizes,
            dueling=self.use_dueling,
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.train_steps = 0

    def act(
        self,
        obs: np.ndarray,
        *,
        epsilon: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() < epsilon:
            return int(rng.integers(0, self.action_dim))
        obs_tensor = torch.from_numpy(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self, batch: Dict[str, np.ndarray]) -> float:
        obs = torch.from_numpy(batch["obs"]).to(self.device)
        next_obs = torch.from_numpy(batch["next_obs"]).to(self.device)
        actions = torch.from_numpy(batch["actions"]).long().to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        dones = torch.from_numpy(batch["dones"]).to(self.device)

        q_values = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.use_double_q:
                next_actions = self.q_network(next_obs).argmax(dim=1, keepdim=True)
                target_q = self.target_network(next_obs)
                next_q = target_q.gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_network(next_obs).max(1).values
            targets = rewards + (1.0 - dones) * self.gamma * next_q
        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.train_steps += 1
        if self.train_steps % self.target_sync_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return float(loss.item())

    def save(self, path: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "config": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "hidden_sizes": self.hidden_sizes,
                "gamma": self.gamma,
                "learning_rate": self.learning_rate,
                "target_sync_interval": self.target_sync_interval,
                "max_grad_norm": self.max_grad_norm,
            },
            "state_dict": self.q_network.state_dict(),
            "target_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
            "metadata": metadata or {},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, *, device: Optional[str] = None) -> Tuple["DQNAgent", Dict[str, Any]]:
        payload = torch.load(path, map_location=device or "cpu")
        config_dict = payload["config"]
        config = AgentConfig(
            obs_dim=config_dict["obs_dim"],
            action_dim=config_dict["action_dim"],
            hidden_sizes=tuple(config_dict["hidden_sizes"]),
            gamma=config_dict["gamma"],
            learning_rate=config_dict["learning_rate"],
            target_sync_interval=config_dict["target_sync_interval"],
            max_grad_norm=config_dict["max_grad_norm"],
            device=device,
        )
        agent = cls(config)
        agent.q_network.load_state_dict(payload["state_dict"])
        agent.target_network.load_state_dict(payload["target_state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        agent.train_steps = payload.get("train_steps", 0)
        metadata = payload.get("metadata", {})
        return agent, metadata
