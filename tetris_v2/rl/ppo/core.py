"""Custom PyTorch PPO core for TetrisV2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F

from tetris_v2.rl.actions import (
    BOARD_ROWS,
    BOARD_WIDTH,
    PLACEMENT_ACTION_DIM,
    PLACEMENT_CHANNELS,
)


OBSERVATION_DIM = 254
VISIBLE_BOARD_CELLS = 200
ACTION_ORDER = "hold,rotation,y,x"


class ExpertDatasetLike(Protocol):
    """The small dataset surface PPO needs for auxiliary imitation updates."""

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]: ...


class ActorCritic(nn.Module):
    """Legacy shared MLP retained for old PPO checkpoint compatibility."""

    def __init__(self, input_dim: int, action_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        dims = [input_dim] + list(hidden_sizes)
        for idx in range(len(hidden_sizes)):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.ReLU())
        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()
        last_dim = dims[-1] if hidden_sizes else input_dim
        self.policy_head = nn.Linear(last_dim, action_dim)
        self.value_head = nn.Linear(last_dim, 1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore[override]
        features = self.feature_extractor(x)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values


class PlacementEncoder(nn.Module):
    """Board-aware spatial encoder over the stable 40-by-10 placement grid."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.board_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.board_global = nn.Linear(VISIBLE_BOARD_CELLS, channels)
        self.context = nn.Linear(OBSERVATION_DIM - VISIBLE_BOARD_CELLS, channels)
        self.refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        y = torch.linspace(0.0, 1.0, BOARD_ROWS).view(1, 1, BOARD_ROWS, 1)
        x = torch.linspace(0.0, 1.0, BOARD_WIDTH).view(1, 1, 1, BOARD_WIDTH)
        self.register_buffer(
            "y_coord", y.expand(1, 1, BOARD_ROWS, BOARD_WIDTH).clone()
        )
        self.register_buffer(
            "x_coord", x.expand(1, 1, BOARD_ROWS, BOARD_WIDTH).clone()
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, observation: Tensor) -> Tensor:  # type: ignore[override]
        batch = observation.shape[0]
        board_flat = observation[:, :VISIBLE_BOARD_CELLS]
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
        conditioning = self.board_global(board_flat) + self.context(
            observation[:, VISIBLE_BOARD_CELLS:OBSERVATION_DIM]
        )
        features = F.relu(self.board_conv(spatial) + conditioning[:, :, None, None])
        return F.relu(features + self.refine(features))


class PlacementPolicyNetwork(nn.Module):
    """Policy logits arranged as ``(hold, rotation, y, x)``."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.encoder = PlacementEncoder(channels)
        self.output = nn.Conv2d(channels, PLACEMENT_CHANNELS, kernel_size=1)
        nn.init.orthogonal_(self.output.weight, gain=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, observation: Tensor) -> Tensor:  # type: ignore[override]
        logits = self.output(self.encoder(observation))
        return logits.reshape(observation.shape[0], PLACEMENT_ACTION_DIM)


class PlacementValueNetwork(nn.Module):
    """Value estimator with parameters independent from the placement policy."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.encoder = PlacementEncoder(channels)
        self.value_head = nn.Linear(channels, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, observation: Tensor) -> Tensor:  # type: ignore[override]
        features = self.encoder(observation).mean(dim=(2, 3))
        return self.value_head(features).squeeze(-1)


class StructuredActorCritic(nn.Module):
    """Independent structured actor and critic exposed through one PPO module."""

    def __init__(self):
        super().__init__()
        self.actor = PlacementPolicyNetwork()
        self.critic = PlacementValueNetwork()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore[override]
        return self.actor(x), self.critic(x)


def clipped_value_loss(
    values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    clip_range: Optional[float],
) -> Tensor:
    """PPO value loss with the same trust region used by the policy update."""

    squared_error = (values - returns).square()
    if clip_range is None or clip_range <= 0:
        return 0.5 * squared_error.mean()
    clipped = old_values + torch.clamp(values - old_values, -clip_range, clip_range)
    clipped_error = (clipped - returns).square()
    return 0.5 * torch.maximum(squared_error, clipped_error).mean()


def explained_variance(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return the fraction of target variance explained by value predictions."""

    target_variance = float(np.var(targets))
    if target_variance <= 1e-12:
        return 0.0
    return float(1.0 - np.var(targets - predictions) / target_variance)


@dataclass
class PPOConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: Tuple[int, ...] = (512, 256)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: Optional[float] = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.03
    network_type: str = "auto"
    device: Optional[str] = None


class RolloutBuffer:
    """Fixed-size rollout storage for on-policy PPO updates."""

    def __init__(self, n_steps: int, num_envs: int, obs_dim: int):
        self.n_steps = int(n_steps)
        self.num_envs = int(num_envs)
        self.obs = np.zeros((self.n_steps, self.num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.num_envs), dtype=np.int64)
        self.log_probs = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        self.action_masks: Optional[np.ndarray] = None
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
        action_masks: np.ndarray,
    ) -> None:
        if self.step >= self.n_steps:
            raise ValueError("RolloutBuffer overflow.")
        if self.action_masks is None:
            self.action_masks = np.zeros(
                (self.n_steps, self.num_envs, action_masks.shape[-1]),
                dtype=np.uint8,
            )
        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.log_probs[self.step] = log_probs
        self.action_masks[self.step] = action_masks > 0.5
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
                next_non_terminal = 1.0 - self.dones[step]
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
        if self.action_masks is None:
            raise ValueError("RolloutBuffer action masks are not initialized.")
        action_masks = self.action_masks[: self.step].reshape(total, -1)
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
                "action_masks": action_masks[batch_idx],
            }


class PPOAgent:
    """Masked-action PPO with structured Tetris and legacy MLP networks."""

    def __init__(self, config: PPOConfig):
        self.config = config
        self.network_type = self._resolve_network_type(config)
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        if self.network_type == "placement_conv":
            self.policy: nn.Module = StructuredActorCritic().to(self.device)
        else:
            self.policy = ActorCritic(
                config.obs_dim,
                config.action_dim,
                config.hidden_sizes,
            ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.update_steps = 0

    @staticmethod
    def _resolve_network_type(config: PPOConfig) -> str:
        network_type = str(config.network_type)
        if network_type == "auto":
            if config.obs_dim == OBSERVATION_DIM and config.action_dim == PLACEMENT_ACTION_DIM:
                return "placement_conv"
            return "mlp"
        if network_type not in {"mlp", "placement_conv"}:
            raise ValueError("network_type must be 'auto', 'mlp', or 'placement_conv'")
        if network_type == "placement_conv" and (
            config.obs_dim != OBSERVATION_DIM or config.action_dim != PLACEMENT_ACTION_DIM
        ):
            raise ValueError("placement_conv requires obs_dim=254 and action_dim=3200")
        return network_type

    def reset_optimizer(self, learning_rate: Optional[float] = None) -> None:
        """Start a fresh optimizer while retaining model weights."""

        if learning_rate is not None:
            self.config.learning_rate = float(learning_rate)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate,
        )
        self.update_steps = 0

    def set_learning_rate(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(learning_rate)

    def _tensor(self, value: np.ndarray | Tensor, *, dtype: torch.dtype) -> Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _mask_logits(self, logits: Tensor, action_mask: Optional[np.ndarray | Tensor]) -> Tensor:
        if action_mask is None:
            return logits
        mask = self._tensor(action_mask, dtype=torch.float32)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        mask = mask > 0.5
        if mask.shape != logits.shape:
            raise ValueError(f"Action mask shape mismatch: mask={mask.shape}, logits={logits.shape}")
        if torch.any(~mask.any(dim=-1)):
            raise ValueError("Action mask contains no legal actions.")
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    def act(
        self,
        obs: np.ndarray,
        *,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        deterministic: bool = False,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, float, float]:
        batch = obs[np.newaxis, :] if obs.ndim == 1 else obs
        mask_batch = None
        if action_mask is not None:
            mask_batch = action_mask[np.newaxis, :] if action_mask.ndim == 1 else action_mask
        actions, log_probs, values = self.act_batch(
            batch,
            temperature=temperature,
            epsilon=epsilon,
            deterministic=deterministic,
            action_mask=mask_batch,
        )
        return int(actions[0]), float(log_probs[0]), float(values[0])

    def act_batch(
        self,
        obs_batch: np.ndarray,
        *,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        deterministic: bool = False,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_tensor = self._tensor(obs_batch, dtype=torch.float32)
        with torch.no_grad():
            logits, values = self.policy(obs_tensor)
        scaled_logits = self._mask_logits(logits / max(float(temperature), 1e-6), action_mask)
        probs = torch.softmax(scaled_logits, dim=-1)
        if epsilon > 0:
            if action_mask is not None:
                legal = (self._tensor(action_mask, dtype=torch.float32) > 0.5).float()
                if legal.ndim == 1:
                    legal = legal.unsqueeze(0)
                uniform = legal / legal.sum(dim=-1, keepdim=True).clamp(min=1.0)
            else:
                uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = (1.0 - epsilon) * probs + epsilon * uniform
        if deterministic:
            action = torch.argmax(probs, dim=-1)
            log_prob = probs.gather(1, action.unsqueeze(1)).clamp(min=1e-8).log().squeeze(1)
        else:
            dist = Categorical(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return (
            action.cpu().numpy().reshape(-1).astype(np.int64),
            log_prob.cpu().numpy().reshape(-1).astype(np.float32),
            values.cpu().numpy().reshape(-1).astype(np.float32),
        )

    def value_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        obs_tensor = self._tensor(obs_batch, dtype=torch.float32)
        with torch.no_grad():
            if isinstance(self.policy, StructuredActorCritic):
                values = self.policy.critic(obs_tensor)
            else:
                _, values = self.policy(obs_tensor)
        return values.cpu().numpy().reshape(-1).astype(np.float32)

    def behavior_cloning_loss(
        self,
        observation: np.ndarray | Tensor,
        teacher_actions: np.ndarray | Tensor,
        action_mask: np.ndarray | Tensor,
    ) -> Tuple[Tensor, Tensor]:
        obs = self._tensor(observation, dtype=torch.float32)
        targets = self._tensor(teacher_actions, dtype=torch.long)
        masks = self._tensor(action_mask, dtype=torch.float32)
        if isinstance(self.policy, StructuredActorCritic):
            logits = self.policy.actor(obs)
        else:
            logits, _ = self.policy(obs)
        masked_logits = self._mask_logits(logits, masks)
        loss = F.cross_entropy(masked_logits, targets)
        agreement = (masked_logits.argmax(dim=-1) == targets).float().mean()
        return loss, agreement

    def pretrain_expert_batch(
        self,
        batch: Mapping[str, np.ndarray],
        *,
        coefficient: float = 1.0,
    ) -> Dict[str, float]:
        bc_loss, agreement = self.behavior_cloning_loss(
            batch["obs"],
            batch["teacher_best_action"],
            batch["action_mask"],
        )
        total = float(coefficient) * bc_loss
        self.optimizer.zero_grad()
        total.backward()
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.update_steps += 1
        return {
            "bc_loss": float(bc_loss.detach().item()),
            "teacher_top1_agreement": float(agreement.detach().item()),
        }

    def update(
        self,
        buffer: RolloutBuffer,
        batch_size: int,
        epochs: int,
        *,
        expert_dataset: Optional[ExpertDatasetLike] = None,
        expert_batch_size: Optional[int] = None,
        bc_coef: float = 0.0,
        expert_rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, float]:
        if buffer.num_samples() == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "explained_variance": 0.0,
                "bc_loss": 0.0,
                "teacher_top1_agreement": 0.0,
                "early_stopped": 0.0,
            }

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "bc_loss": 0.0,
            "teacher_top1_agreement": 0.0,
        }
        updates = 0
        early_stopped = False
        rng = expert_rng or np.random.default_rng()

        for _ in range(epochs):
            for batch in buffer.iter_minibatches(batch_size):
                obs = self._tensor(batch["obs"], dtype=torch.float32)
                actions = self._tensor(batch["actions"], dtype=torch.long)
                old_log_probs = self._tensor(batch["log_probs"], dtype=torch.float32)
                advantages = self._tensor(batch["advantages"], dtype=torch.float32)
                returns = self._tensor(batch["returns"], dtype=torch.float32)
                old_values = self._tensor(batch["values"], dtype=torch.float32)
                action_masks = self._tensor(batch["action_masks"], dtype=torch.float32)

                logits, values = self.policy(obs)
                dist = Categorical(logits=self._mask_logits(logits, action_masks))
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                log_ratio = log_probs - old_log_probs
                ratios = torch.exp(log_ratio)
                unclipped = ratios * advantages
                clipped = torch.clamp(
                    ratios,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                ) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = clipped_value_loss(
                    values,
                    old_values,
                    returns,
                    self.config.value_clip_range,
                )
                approx_kl = ((ratios - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (torch.abs(ratios - 1.0) > self.config.clip_range).float().mean()
                )

                if (
                    self.config.target_kl is not None
                    and self.config.target_kl > 0
                    and float(approx_kl.detach().item()) > self.config.target_kl
                ):
                    totals["policy_loss"] += float(policy_loss.detach().item())
                    totals["value_loss"] += float(value_loss.detach().item())
                    totals["entropy"] += float(entropy.detach().item())
                    totals["approx_kl"] += float(approx_kl.detach().item())
                    totals["clip_fraction"] += float(clip_fraction.detach().item())
                    updates += 1
                    early_stopped = True
                    break

                bc_loss = torch.zeros((), device=self.device)
                agreement = torch.zeros((), device=self.device)
                if expert_dataset is not None and float(bc_coef) != 0.0:
                    expert_batch = expert_dataset.sample(
                        expert_batch_size or int(actions.shape[0]),
                        rng,
                    )
                    bc_loss, agreement = self.behavior_cloning_loss(
                        expert_batch["obs"],
                        expert_batch["teacher_best_action"],
                        expert_batch["action_mask"],
                    )

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                    + float(bc_coef) * bc_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.max_grad_norm,
                    )
                self.optimizer.step()
                self.update_steps += 1

                totals["policy_loss"] += float(policy_loss.detach().item())
                totals["value_loss"] += float(value_loss.detach().item())
                totals["entropy"] += float(entropy.detach().item())
                totals["approx_kl"] += float(approx_kl.detach().item())
                totals["clip_fraction"] += float(clip_fraction.detach().item())
                totals["bc_loss"] += float(bc_loss.detach().item())
                totals["teacher_top1_agreement"] += float(agreement.detach().item())
                updates += 1

            if early_stopped:
                break

        count = max(1, updates)
        metrics = {name: value / count for name, value in totals.items()}
        total = buffer.num_samples()
        predictions = self.value_batch(buffer.obs[: buffer.step].reshape(total, -1))
        targets = buffer.returns[: buffer.step].reshape(total)
        metrics["explained_variance"] = explained_variance(predictions, targets)
        metrics["early_stopped"] = 1.0 if early_stopped else 0.0
        return metrics

    def save(
        self,
        path: str,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        payload = {
            "algo": "ppo",
            "checkpoint_version": 2,
            "observation_schema": (
                "tetris_v2_254" if self.config.obs_dim == OBSERVATION_DIM else "flat"
            ),
            "action_order": ACTION_ORDER,
            "config": {
                "obs_dim": self.config.obs_dim,
                "action_dim": self.config.action_dim,
                "hidden_sizes": self.config.hidden_sizes,
                "learning_rate": self.config.learning_rate,
                "gamma": self.config.gamma,
                "gae_lambda": self.config.gae_lambda,
                "clip_range": self.config.clip_range,
                "value_clip_range": self.config.value_clip_range,
                "entropy_coef": self.config.entropy_coef,
                "value_coef": self.config.value_coef,
                "max_grad_norm": self.config.max_grad_norm,
                "target_kl": self.config.target_kl,
                "network_type": self.network_type,
            },
            "state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_steps": int(self.update_steps),
            "metadata": dict(metadata or {}),
        }
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        device: Optional[str] = None,
        restore_optimizer: bool = True,
    ) -> Tuple["PPOAgent", Dict[str, Any]]:
        payload = torch.load(path, map_location=device or "cpu", weights_only=False)
        if payload.get("algo") != "ppo":
            raise ValueError("Checkpoint is not a PPO checkpoint.")
        if payload.get("action_order", ACTION_ORDER) != ACTION_ORDER:
            raise ValueError("PPO checkpoint uses an incompatible placement action order.")
        cfg = payload["config"]
        config = PPOConfig(
            obs_dim=int(cfg["obs_dim"]),
            action_dim=int(cfg["action_dim"]),
            hidden_sizes=tuple(int(v) for v in cfg["hidden_sizes"]),
            learning_rate=float(cfg["learning_rate"]),
            gamma=float(cfg["gamma"]),
            gae_lambda=float(cfg["gae_lambda"]),
            clip_range=float(cfg["clip_range"]),
            value_clip_range=(
                None
                if cfg.get("value_clip_range") is None
                else float(cfg["value_clip_range"])
            ),
            entropy_coef=float(cfg["entropy_coef"]),
            value_coef=float(cfg["value_coef"]),
            max_grad_norm=float(cfg["max_grad_norm"]),
            target_kl=(None if cfg.get("target_kl") is None else float(cfg["target_kl"])),
            network_type=str(cfg.get("network_type", "mlp")),
            device=device,
        )
        agent = cls(config)
        agent.policy.load_state_dict(payload["state_dict"])
        if restore_optimizer and "optimizer_state_dict" in payload:
            agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        agent.update_steps = int(payload.get("update_steps", 0)) if restore_optimizer else 0
        return agent, dict(payload.get("metadata", {}))


__all__ = [
    "ACTION_ORDER",
    "ActorCritic",
    "PPOAgent",
    "PPOConfig",
    "PlacementPolicyNetwork",
    "PlacementValueNetwork",
    "RolloutBuffer",
    "StructuredActorCritic",
    "clipped_value_loss",
    "explained_variance",
]
