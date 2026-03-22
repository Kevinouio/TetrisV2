"""Custom PyTorch DQN core for VersionTwo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class QNetwork(nn.Module):
    """Simple MLP Q-network."""

    def __init__(self, input_dim: int, action_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        dims = [input_dim] + list(hidden_sizes)
        for idx in range(len(hidden_sizes)):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.ReLU())
        last_dim = dims[-1] if hidden_sizes else input_dim
        layers.append(nn.Linear(last_dim, action_dim))
        self.net = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.net(x)


@dataclass
class DQNConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: Tuple[int, ...] = (512, 256)
    learning_rate: float = 1e-3
    gamma: float = 0.99
    target_sync_interval: int = 1_000
    max_grad_norm: float = 10.0
    double_dqn: bool = True
    device: Optional[str] = None


class ReplayBuffer:
    """Ring replay buffer storing transitions as contiguous numpy arrays."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be >= 1")
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.size = 0
        self.pos = 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        idx = self.pos
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.dones[idx] = 1.0 if done else 0.0
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("Cannot sample from empty replay buffer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }


class DQNAgent:
    """DQN agent with online/target networks and epsilon-greedy action selection."""

    def __init__(self, config: DQNConfig):
        self.config = config
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.online = QNetwork(config.obs_dim, config.action_dim, config.hidden_sizes).to(self.device)
        self.target = QNetwork(config.obs_dim, config.action_dim, config.hidden_sizes).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.update_steps = 0

    def select_action(
        self,
        obs: np.ndarray,
        *,
        epsilon: float = 0.0,
        deterministic: bool = False,
    ) -> int:
        eps = 0.0 if deterministic else max(0.0, float(epsilon))
        if not deterministic and np.random.random() < eps:
            return int(np.random.randint(0, self.config.action_dim))
        obs_tensor = torch.from_numpy(obs.astype(np.float32, copy=False)).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online(obs_tensor)
        return int(torch.argmax(q_values, dim=-1).item())

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs.astype(np.float32, copy=False)).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.online(obs_tensor)
        return q.squeeze(0).cpu().numpy()

    def update(self, batch: Dict[str, np.ndarray]) -> float:
        obs = torch.from_numpy(batch["obs"]).to(self.device)
        actions = torch.from_numpy(batch["actions"]).to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        next_obs = torch.from_numpy(batch["next_obs"]).to(self.device)
        dones = torch.from_numpy(batch["dones"]).to(self.device)

        q_pred = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.config.double_dqn:
                next_actions = torch.argmax(self.online(next_obs), dim=1, keepdim=True)
                next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
            else:
                next_q = torch.max(self.target(next_obs), dim=1).values
            q_target = rewards + (1.0 - dones) * self.config.gamma * next_q

        loss = F.smooth_l1_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.update_steps += 1
        if self.config.target_sync_interval > 0 and (self.update_steps % self.config.target_sync_interval) == 0:
            self.sync_target()

        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str, *, metadata: Optional[Dict[str, float]] = None) -> None:
        payload = {
            "algo": "dqn",
            "config": {
                "obs_dim": self.config.obs_dim,
                "action_dim": self.config.action_dim,
                "hidden_sizes": self.config.hidden_sizes,
                "learning_rate": self.config.learning_rate,
                "gamma": self.config.gamma,
                "target_sync_interval": self.config.target_sync_interval,
                "max_grad_norm": self.config.max_grad_norm,
                "double_dqn": self.config.double_dqn,
            },
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_steps": int(self.update_steps),
            "metadata": metadata or {},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, *, device: Optional[str] = None) -> Tuple["DQNAgent", Dict[str, float]]:
        payload = torch.load(path, map_location=device or "cpu", weights_only=False)
        if payload.get("algo") != "dqn":
            raise ValueError("Checkpoint is not a DQN checkpoint.")
        cfg = payload["config"]
        config = DQNConfig(
            obs_dim=int(cfg["obs_dim"]),
            action_dim=int(cfg["action_dim"]),
            hidden_sizes=tuple(int(v) for v in cfg["hidden_sizes"]),
            learning_rate=float(cfg["learning_rate"]),
            gamma=float(cfg["gamma"]),
            target_sync_interval=int(cfg["target_sync_interval"]),
            max_grad_norm=float(cfg["max_grad_norm"]),
            double_dqn=bool(cfg.get("double_dqn", True)),
            device=device,
        )
        agent = cls(config)
        agent.online.load_state_dict(payload["online_state_dict"])
        agent.target.load_state_dict(payload["target_state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        agent.update_steps = int(payload.get("update_steps", 0))
        metadata = payload.get("metadata", {})
        return agent, metadata


__all__ = ["DQNConfig", "ReplayBuffer", "DQNAgent"]
