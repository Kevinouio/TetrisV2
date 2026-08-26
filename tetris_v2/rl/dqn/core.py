"""Custom PyTorch DQN core for TetrisV2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PlacementConvNet(nn.Module):
    """Board-aware Q-map over (hold, rotation, landing row, column)."""

    def __init__(self):
        super().__init__()
        channels = 32
        self.board_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.board_global = nn.Linear(200, channels)
        self.context = nn.Linear(54, channels)
        self.refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.output = nn.Conv2d(channels, 8, kernel_size=1)

        y = torch.linspace(0.0, 1.0, 40).view(1, 1, 40, 1).expand(1, 1, 40, 10)
        x = torch.linspace(0.0, 1.0, 10).view(1, 1, 1, 10).expand(1, 1, 40, 10)
        self.register_buffer("y_coord", y)
        self.register_buffer("x_coord", x)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        batch = x.shape[0]
        board_flat = x[:, :200]
        board = board_flat.reshape(batch, 1, 20, 10)
        board = F.pad(board, (0, 0, 0, 20))
        spatial = torch.cat(
            [board, self.y_coord.expand(batch, -1, -1, -1), self.x_coord.expand(batch, -1, -1, -1)],
            dim=1,
        )
        conditioning = self.board_global(board_flat) + self.context(x[:, 200:254])
        features = F.relu(self.board_conv(spatial) + conditioning[:, :, None, None])
        features = F.relu(features + self.refine(features))
        return self.output(features).reshape(batch, 3200)


class QNetwork(nn.Module):
    """Q-network with a structured placement map or a legacy MLP."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        network_type: str = "mlp",
    ):
        super().__init__()
        if network_type == "placement_conv":
            self.net = PlacementConvNet()
            return
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
    network_type: str = "auto"
    device: Optional[str] = None


class ReplayBuffer:
    """Ring replay buffer storing transitions as contiguous numpy arrays."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be >= 1")
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.next_action_masks = np.zeros((self.capacity, action_dim), dtype=np.uint8)
        self.size = 0
        self.pos = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
    ) -> None:
        idx = self.pos
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.dones[idx] = 1.0 if done else 0.0
        self.next_action_masks[idx] = next_action_mask > 0.5
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
            "next_action_masks": self.next_action_masks[idx],
        }


class DQNAgent:
    """DQN agent with online/target networks and epsilon-greedy action selection."""

    def __init__(self, config: DQNConfig):
        self.config = config
        self.network_type = (
            "placement_conv"
            if config.network_type == "auto" and config.obs_dim == 254 and config.action_dim == 3200
            else ("mlp" if config.network_type == "auto" else config.network_type)
        )
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.online = QNetwork(
            config.obs_dim, config.action_dim, config.hidden_sizes, self.network_type
        ).to(self.device)
        self.target = QNetwork(
            config.obs_dim, config.action_dim, config.hidden_sizes, self.network_type
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.update_steps = 0

    def _masked_q(self, q_values: Tensor, action_mask: Optional[np.ndarray | Tensor]) -> Tensor:
        if action_mask is None:
            return q_values
        if isinstance(action_mask, np.ndarray):
            mask = torch.from_numpy(action_mask).to(self.device)
        else:
            mask = action_mask.to(self.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        mask = mask > 0.5
        if mask.shape != q_values.shape:
            raise ValueError(f"Action mask shape mismatch: mask={mask.shape}, q={q_values.shape}")
        if torch.any(~mask.any(dim=-1)):
            raise ValueError("Action mask contains no legal actions.")
        return q_values.masked_fill(~mask, torch.finfo(q_values.dtype).min)

    def select_action(
        self,
        obs: np.ndarray,
        *,
        epsilon: float = 0.0,
        deterministic: bool = False,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        eps = 0.0 if deterministic else max(0.0, float(epsilon))
        legal_indices = None
        if action_mask is not None:
            legal_indices = np.flatnonzero(np.asarray(action_mask, dtype=np.float32) > 0.5).astype(np.int64)
            if legal_indices.size == 0:
                raise ValueError("Action mask contains no legal actions.")
        if not deterministic and np.random.random() < eps:
            if legal_indices is None:
                return int(np.random.randint(0, self.config.action_dim))
            return int(np.random.choice(legal_indices))
        obs_tensor = torch.from_numpy(obs.astype(np.float32, copy=False)).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online(obs_tensor)
            q_values = self._masked_q(q_values, action_mask)
        return int(torch.argmax(q_values, dim=-1).item())

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs.astype(np.float32, copy=False)).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.online(obs_tensor)
        return q.squeeze(0).cpu().numpy()

    def compute_td_loss(self, batch: Dict[str, np.ndarray]) -> Tensor:
        obs = torch.from_numpy(batch["obs"]).to(self.device)
        actions = torch.from_numpy(batch["actions"]).to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        next_obs = torch.from_numpy(batch["next_obs"]).to(self.device)
        dones = torch.from_numpy(batch["dones"]).to(self.device)
        next_action_masks = torch.from_numpy(batch["next_action_masks"]).to(self.device)

        q_pred = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = torch.zeros_like(rewards)
            nonterminal = dones < 0.5
            if torch.any(nonterminal):
                active_next_obs = next_obs[nonterminal]
                active_next_masks = next_action_masks[nonterminal]
                if self.config.double_dqn:
                    online_next = self._masked_q(self.online(active_next_obs), active_next_masks)
                    next_actions = torch.argmax(online_next, dim=1, keepdim=True)
                    next_q[nonterminal] = self.target(active_next_obs).gather(1, next_actions).squeeze(1)
                else:
                    target_next = self._masked_q(self.target(active_next_obs), active_next_masks)
                    next_q[nonterminal] = torch.max(target_next, dim=1).values
            q_target = rewards + self.config.gamma * next_q

        return F.smooth_l1_loss(q_pred, q_target)

    def _apply_update(self, loss: Tensor) -> float:
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.update_steps += 1
        if self.config.target_sync_interval > 0 and (self.update_steps % self.config.target_sync_interval) == 0:
            self.sync_target()

        return float(loss.item())

    def update(self, batch: Dict[str, np.ndarray]) -> float:
        loss = self.compute_td_loss(batch)
        return self._apply_update(loss)

    def update_combined(
        self,
        batch: Dict[str, np.ndarray],
        *,
        bc_loss: Optional[Tensor] = None,
        pair_loss: Optional[Tensor] = None,
        lambda_bc: float = 0.0,
        lambda_pair: float = 0.0,
    ) -> Dict[str, float]:
        td_loss = self.compute_td_loss(batch)
        total = td_loss
        bc_value = 0.0
        pair_value = 0.0
        if bc_loss is not None and float(lambda_bc) != 0.0:
            total = total + float(lambda_bc) * bc_loss
            bc_value = float(bc_loss.detach().item())
        if pair_loss is not None and float(lambda_pair) != 0.0:
            total = total + float(lambda_pair) * pair_loss
            pair_value = float(pair_loss.detach().item())
        total_value = self._apply_update(total)
        return {
            "td_loss": float(td_loss.detach().item()),
            "bc_loss": bc_value,
            "pair_loss": pair_value,
            "total_loss": float(total_value),
        }

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
                "network_type": self.network_type,
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
            network_type=str(cfg.get("network_type", "mlp")),
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
