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
        self.slices: Dict[str, Tuple[int, int]] = {}
        self.board_shape: Optional[Tuple[int, int]] = None
        self.board_dim: int = 0
        for key in self.keys:
            space = observation_space.spaces[key]
            if isinstance(space, spaces.Discrete):
                size = 1
            else:
                shape = getattr(space, "shape", ()) or ()
                size = int(np.prod(shape, dtype=int)) if shape else 1
            self._sizes[key] = size
            flat_dim += size
            self.slices[key] = (flat_dim - size, flat_dim)
            if key == "board" and getattr(space, "shape", None):
                self.board_shape = tuple(int(dim) for dim in space.shape)  # type: ignore[arg-type]
                self.board_dim = size
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

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        return np.random.randint(0, self.size, size=int(batch_size))

    def _gather_batch(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "obs": self.obs[indices],
            "next_obs": self.next_obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
            "indices": indices,
        }

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        indices = self._sample_indices(batch_size)
        return self._gather_batch(indices)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        # Uniform buffer ignores priority updates.
        return


class PrioritizedReplayBuffer(ReplayBuffer):
    """Simple proportional prioritized replay buffer."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-6,
    ):
        super().__init__(capacity, obs_dim)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.beta_increment = float(beta_increment)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.epsilon = 1e-6

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        prev_pos = self.pos
        super().add(obs, action, reward, next_obs, done)
        idx = (prev_pos) % self.capacity
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[idx] = max(max_prio, self.epsilon)

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        scaled = self.priorities[: self.size] ** self.alpha
        probs = scaled / np.sum(scaled)
        return np.random.default_rng().choice(self.size, size=int(batch_size), p=probs)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        indices = self._sample_indices(batch_size)
        scaled = self.priorities[: self.size] ** self.alpha
        probs = scaled / np.sum(scaled)
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max() if weights.max() > 0 else 1.0
        self.beta = min(1.0, self.beta + self.beta_increment)
        batch = self._gather_batch(indices)
        batch["weights"] = weights.astype(np.float32)
        return batch

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.abs(priorities) + self.epsilon
        self.priorities[indices] = priorities


class QNetwork(nn.Module):
    """Feed-forward Q-network with optional dueling heads."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Sequence[int],
        *,
        dueling: bool = True,
        board_dim: int = 0,
        board_shape: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.board_dim = int(board_dim)
        self.board_shape = board_shape
        vector_dim = input_dim - self.board_dim
        if self.board_dim and self.board_shape is not None:
            channels = 32
            self.board_extractor = nn.Sequential(
                nn.Conv2d(1, channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            conv_out_dim = (channels * 2) * self.board_shape[0] * self.board_shape[1]
        else:
            self.board_extractor = None
            conv_out_dim = 0

        layers: list[nn.Module] = []
        mlp_input = conv_out_dim + max(vector_dim, 0)
        dims = [mlp_input] + list(hidden_sizes)
        for i in range(len(hidden_sizes)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()
        last_dim = dims[-1] if hidden_sizes else mlp_input
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
        if self.board_extractor is not None and self.board_dim > 0:
            board = x[:, : self.board_dim].view(-1, 1, *self.board_shape)  # type: ignore[arg-type]
            rest = x[:, self.board_dim :]
            board_features = self.board_extractor(board)
            if rest.shape[1] > 0:
                features_input = torch.cat([board_features, rest], dim=1)
            else:
                features_input = board_features
        else:
            features_input = x
        features = self.feature_extractor(features_input)
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
    board_dim: int = 0
    board_shape: Optional[Tuple[int, int]] = None


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
        self.board_dim = int(config.board_dim)
        self.board_shape = config.board_shape

        self.q_network = QNetwork(
            self.obs_dim,
            self.action_dim,
            self.hidden_sizes,
            dueling=self.use_dueling,
            board_dim=self.board_dim,
            board_shape=self.board_shape,
        ).to(self.device)
        self.target_network = QNetwork(
            self.obs_dim,
            self.action_dim,
            self.hidden_sizes,
            dueling=self.use_dueling,
            board_dim=self.board_dim,
            board_shape=self.board_shape,
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.train_steps = 0

    def act(
        self,
        obs: np.ndarray,
        *,
        epsilon: float = 0.0,
        temperature: float = 1.0,
        strategy: str = "epsilon",
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        if rng is None:
            rng = np.random.default_rng()
        obs_tensor = torch.from_numpy(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor).squeeze(0)
        strategy = strategy.lower()
        if strategy == "epsilon":
            if rng.random() < epsilon:
                return int(rng.integers(0, self.action_dim))
            return int(torch.argmax(q_values, dim=0).item())
        if strategy == "boltzmann":
            scale = max(float(temperature), 1e-6)
            logits = q_values / scale
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            if epsilon > 0:
                probs = (1.0 - epsilon) * probs + epsilon / self.action_dim
            probs = probs / probs.sum()
            return int(rng.choice(self.action_dim, p=probs))
        raise ValueError(f"Unsupported exploration strategy '{strategy}'.")

    def act_batch(
        self,
        obs_batch: np.ndarray,
        *,
        epsilon: float = 0.0,
        temperature: float = 1.0,
        strategy: str = "epsilon",
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if obs_batch.ndim != 2:
            raise ValueError("obs_batch must be 2D (batch, features).")
        obs_tensor = torch.from_numpy(obs_batch).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        batch_size = q_values.shape[0]
        strategy = strategy.lower()
        if strategy == "epsilon":
            greedy = torch.argmax(q_values, dim=1).cpu().numpy()
            if epsilon <= 0.0:
                return greedy.astype(np.int64)
            random_actions = rng.integers(0, self.action_dim, size=batch_size)
            masks = rng.random(batch_size) < epsilon
            actions = np.where(masks, random_actions, greedy)
            return actions.astype(np.int64)
        if strategy == "boltzmann":
            scale = max(float(temperature), 1e-6)
            logits = q_values / scale
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            if epsilon > 0:
                probs = (1.0 - epsilon) * probs + epsilon / self.action_dim
            actions = np.empty(batch_size, dtype=np.int64)
            for idx in range(batch_size):
                actions[idx] = int(rng.choice(self.action_dim, p=probs[idx] / probs[idx].sum()))
            return actions
        raise ValueError(f"Unsupported exploration strategy '{strategy}'.")

    def update(self, batch: Dict[str, np.ndarray]) -> Tuple[float, np.ndarray]:
        obs = torch.from_numpy(batch["obs"]).to(self.device)
        next_obs = torch.from_numpy(batch["next_obs"]).to(self.device)
        actions = torch.from_numpy(batch["actions"]).long().to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        dones = torch.from_numpy(batch["dones"]).to(self.device)
        weights_np = batch.get("weights")
        if weights_np is not None:
            weights = torch.from_numpy(weights_np).to(self.device)
        else:
            weights = torch.ones(len(actions), device=self.device)

        q_values = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.use_double_q:
                next_actions = self.q_network(next_obs).argmax(dim=1, keepdim=True)
                target_q = self.target_network(next_obs)
                next_q = target_q.gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_network(next_obs).max(1).values
            targets = rewards + (1.0 - dones) * self.gamma * next_q
        elementwise_loss = F.smooth_l1_loss(q_values, targets, reduction="none")
        loss = (weights * elementwise_loss).mean()
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.train_steps += 1
        if self.train_steps % self.target_sync_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        td_errors = (targets - q_values).detach().cpu().numpy()
        return float(loss.item()), td_errors

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
                "board_dim": self.board_dim,
                "board_shape": self.board_shape,
                "use_double_q": self.use_double_q,
                "use_dueling": self.use_dueling,
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
        payload = torch.load(path, map_location=device or "cpu", weights_only=False)
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
            board_dim=config_dict.get("board_dim", 0),
            board_shape=tuple(config_dict["board_shape"]) if config_dict.get("board_shape") else None,
            use_double_q=config_dict.get("use_double_q", True),
            use_dueling=config_dict.get("use_dueling", True),
        )
        agent = cls(config)
        agent.q_network.load_state_dict(payload["state_dict"])
        agent.target_network.load_state_dict(payload["target_state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        agent.train_steps = payload.get("train_steps", 0)
        metadata = payload.get("metadata", {})
        return agent, metadata
