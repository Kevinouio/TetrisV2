"""Actor-critic networks, rollout storage, and PPO agent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F


class ActorCritic(nn.Module):
    """Shared network used by PPO for policy and value estimation."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        *,
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
            conv_out = (channels * 2) * self.board_shape[0] * self.board_shape[1]
        else:
            self.board_extractor = None
            conv_out = 0

        mlp_input = conv_out + max(vector_dim, 0)
        layers: list[nn.Module] = []
        dims = [mlp_input] + list(hidden_sizes)
        for idx in range(len(hidden_sizes)):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.ReLU())
        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()
        last_dim = dims[-1] if hidden_sizes else mlp_input
        self.policy_head = nn.Linear(last_dim, action_dim)
        self.value_head = nn.Linear(last_dim, 1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore[override]
        if self.board_extractor is not None and self.board_dim > 0:
            board = x[:, : self.board_dim].view(-1, 1, *self.board_shape)  # type: ignore[arg-type]
            rest = x[:, self.board_dim :]
            board_features = self.board_extractor(board)
            if rest.shape[1] > 0:
                feat_input = torch.cat([board_features, rest], dim=1)
            else:
                feat_input = board_features
        else:
            feat_input = x
        features = self.feature_extractor(feat_input)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values


@dataclass
class PPOConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: Tuple[int, ...] = (512, 256)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: Optional[str] = None
    board_dim: int = 0
    board_shape: Optional[Tuple[int, int]] = None


class RolloutBuffer:
    """Fixed-size rollout storage for on-policy updates supporting vector envs."""

    def __init__(self, n_steps: int, num_envs: int, obs_dim: int):
        self.n_steps = int(n_steps)
        self.num_envs = int(num_envs)
        self.obs = np.zeros((self.n_steps, self.num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.num_envs), dtype=np.int64)
        self.log_probs = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        self.advantages = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        self.returns = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        self.step = 0

    def reset(self) -> None:
        self.step = 0

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
    ) -> None:
        if self.step >= self.n_steps:
            raise ValueError("RolloutBuffer overflow. Did you forget to reset after an update?")
        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.log_probs[self.step] = log_probs
        self.step += 1

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        last_dones: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        if self.step == 0:
            return
        last_gae = np.zeros(self.num_envs, dtype=np.float32)
        for step in reversed(range(self.step)):
            if step == self.step - 1:
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
                next_value = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae
        returns = self.advantages[: self.step] + self.values[: self.step]
        flat_adv = self.advantages[: self.step].reshape(-1)
        norm_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        self.advantages[: self.step] = norm_adv.reshape(self.step, self.num_envs)
        self.returns[: self.step] = returns

    def num_samples(self) -> int:
        return self.step * self.num_envs

    def iter_minibatches(self, batch_size: int) -> Iterable[Dict[str, np.ndarray]]:
        total = self.num_samples()
        if total == 0:
            return
        obs = self.obs[: self.step].reshape(total, -1)
        actions = self.actions[: self.step].reshape(total)
        log_probs = self.log_probs[: self.step].reshape(total)
        advantages = self.advantages[: self.step].reshape(total)
        returns = self.returns[: self.step].reshape(total)
        values = self.values[: self.step].reshape(total)
        indices = np.arange(total)
        np.random.shuffle(indices)
        for start in range(0, total, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield {
                "obs": obs[batch_idx],
                "actions": actions[batch_idx],
                "log_probs": log_probs[batch_idx],
                "advantages": advantages[batch_idx],
                "returns": returns[batch_idx],
                "values": values[batch_idx],
            }


class PPOAgent:
    """Minimal PPO agent with clipped surrogate objectives."""

    def __init__(self, config: PPOConfig):
        self.config = config
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.policy = ActorCritic(
            config.obs_dim,
            config.action_dim,
            config.hidden_sizes,
            board_dim=config.board_dim,
            board_shape=config.board_shape,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)

    def act(
        self,
        obs: np.ndarray,
        *,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        deterministic: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[int, float, float]:
        if obs.ndim == 1:
            batch = obs[np.newaxis, :]
        else:
            batch = obs
        actions, log_probs, values = self.act_batch(
            batch,
            temperature=temperature,
            epsilon=epsilon,
            deterministic=deterministic,
            rng=rng,
        )
        return int(actions[0]), float(log_probs[0]), float(values[0])

    def act_batch(
        self,
        obs_batch: np.ndarray,
        *,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        deterministic: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()
        obs_tensor = torch.from_numpy(obs_batch).to(self.device)
        with torch.no_grad():
            logits, values = self.policy(obs_tensor)
        scale = max(float(temperature), 1e-6)
        scaled_logits = logits / scale
        probs = torch.softmax(scaled_logits, dim=-1)
        if epsilon > 0:
            uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = (1.0 - epsilon) * probs + epsilon * uniform
        if deterministic:
            action = torch.argmax(probs, dim=-1, keepdim=True)
            selected = probs.gather(-1, action).clamp(min=1e-8)
            log_prob = selected.log().squeeze(-1)
            actions_np = action.squeeze(-1).cpu().numpy()
            log_probs_np = log_prob.cpu().numpy()
        else:
            dist = Categorical(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            actions_np = action.cpu().numpy()
            log_probs_np = log_prob.cpu().numpy()
        values_np = values.squeeze(-1).cpu().numpy()
        actions_np = np.asarray(actions_np).reshape(-1).astype(np.int64)
        log_probs_np = np.asarray(log_probs_np).reshape(-1).astype(np.float32)
        values_np = np.asarray(values_np).reshape(-1).astype(np.float32)
        return actions_np, log_probs_np, values_np

    def value(self, obs: np.ndarray) -> float:
        return float(self.value_batch(obs[np.newaxis, :])[0])

    def value_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs_batch).to(self.device)
        with torch.no_grad():
            _, values = self.policy(obs_tensor)
        return values.squeeze(-1).cpu().numpy()

    def update(self, buffer: RolloutBuffer, batch_size: int, epochs: int) -> Dict[str, float]:
        if buffer.num_samples() == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        updates = 0
        clip_range = self.config.clip_range

        for _ in range(epochs):
            for batch in buffer.iter_minibatches(batch_size):
                obs = torch.from_numpy(batch["obs"]).to(self.device)
                actions = torch.from_numpy(batch["actions"]).to(self.device)
                old_log_probs = torch.from_numpy(batch["log_probs"]).to(self.device)
                advantages = torch.from_numpy(batch["advantages"]).to(self.device)
                returns = torch.from_numpy(batch["returns"]).to(self.device)

                logits, values = self.policy(obs)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                ratios = torch.exp(log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, returns)
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                total_policy += float(policy_loss.item())
                total_value += float(value_loss.item())
                total_entropy += float(entropy.item())
                updates += 1

        if updates == 0:
            updates = 1
        return {
            "policy_loss": total_policy / updates,
            "value_loss": total_value / updates,
            "entropy": total_entropy / updates,
        }

    def save(self, path: str, *, metadata: Optional[Dict[str, float]] = None) -> None:
        payload = {
            "config": {
                "obs_dim": self.config.obs_dim,
                "action_dim": self.config.action_dim,
                "hidden_sizes": self.config.hidden_sizes,
                "learning_rate": self.config.learning_rate,
                "gamma": self.config.gamma,
                "gae_lambda": self.config.gae_lambda,
                "clip_range": self.config.clip_range,
                "entropy_coef": self.config.entropy_coef,
                "value_coef": self.config.value_coef,
                "max_grad_norm": self.config.max_grad_norm,
                "board_dim": self.config.board_dim,
                "board_shape": self.config.board_shape,
            },
            "state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, *, device: Optional[str] = None) -> Tuple["PPOAgent", Dict[str, float]]:
        payload = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg = payload["config"]
        config = PPOConfig(
            obs_dim=cfg["obs_dim"],
            action_dim=cfg["action_dim"],
            hidden_sizes=tuple(cfg["hidden_sizes"]),
            learning_rate=cfg["learning_rate"],
            gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"],
            clip_range=cfg["clip_range"],
            entropy_coef=cfg["entropy_coef"],
            value_coef=cfg["value_coef"],
            max_grad_norm=cfg["max_grad_norm"],
            device=device,
            board_dim=cfg.get("board_dim", 0),
            board_shape=tuple(cfg["board_shape"]) if cfg.get("board_shape") else None,
        )
        agent = cls(config)
        agent.policy.load_state_dict(payload["state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        metadata = payload.get("metadata", {})
        return agent, metadata
