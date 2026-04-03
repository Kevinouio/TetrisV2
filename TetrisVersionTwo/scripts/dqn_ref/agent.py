from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import DQNRefConfig
from .env_bridge import CandidateAfterstate
from .model import LinearQNet, QTrainer
from .replay import PrioritizedReplay
from .reward import ReferenceReward, RewardTerms


class DQNRefAgent:
    def __init__(
        self,
        genome: Dict[str, float],
        config: DQNRefConfig,
        device: torch.device,
        checkpoint_path: Optional[Path] = None,
    ):
        self.weight = {k: float(v) for k, v in genome.items()}
        self.config = config
        self.device = device

        self.n_games = 0
        self.epsilon = float(config.training.epsilon_start)
        self.gamma = float(config.training.gamma)
        self.min_num = float(config.training.epsilon_min)
        self.epsilon_0 = float(config.training.epsilon_start)
        self.alpha = float(config.training.epsilon_alpha)
        self.total_games = int(config.training.total_games_for_decay)
        self.random = False

        self.memory = PrioritizedReplay(max_size=int(config.replay.max_memory))
        self.model1 = LinearQNet(
            input_size=int(config.model.input_size),
            hidden_sizes=tuple(int(v) for v in config.model.hidden_sizes),
            output_size=int(config.model.output_size),
        ).to(device)
        self.model2 = LinearQNet(
            input_size=int(config.model.input_size),
            hidden_sizes=tuple(int(v) for v in config.model.hidden_sizes),
            output_size=int(config.model.output_size),
        ).to(device)

        if bool(config.runtime.torch_compile) and hasattr(torch, "compile"):
            try:
                self.model1 = torch.compile(self.model1)  # type: ignore[attr-defined]
                self.model2 = torch.compile(self.model2)  # type: ignore[attr-defined]
            except Exception:
                pass

        self.trainer = QTrainer(
            model1=self.model1,
            model2=self.model2,
            lr=float(config.training.learning_rate_start),
            gamma=float(config.training.gamma),
            memory=self.memory,
            train_epochs=int(config.training.train_epochs_per_call),
            batch_size=int(config.replay.batch_size),
            replay_beta=float(config.replay.beta),
            grad_clip_norm=float(config.training.grad_clip_norm),
            device=device,
        )
        self.reward = ReferenceReward(self.weight)

        self.total_steps = 0
        self.losses: List[float] = []
        self.last_loss = 0.0
        self.trained = False
        self.games_without_training = 0

        if checkpoint_path is not None and Path(checkpoint_path).exists():
            self.load_model(checkpoint_path)
        else:
            self.update_target_network()

    def save_model(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model1.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer_state(),
        }
        torch.save(payload, path)

    def load_model(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        self.model1.load_state_dict(checkpoint["model_state_dict"])
        self.model2.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            try:
                self.trainer.load_optimizer_state(checkpoint["optimizer_state_dict"])
            except Exception:
                pass

    def calculate_lr(self, games_played: int) -> float:
        max_games = max(1, int(self.config.training.total_games_for_decay))
        min_lr = float(self.config.training.learning_rate_end)
        max_lr = float(self.config.training.learning_rate_start)
        lr = max_lr - (max_lr - min_lr) * (float(games_played) / float(max_games))
        lr = max(lr, min_lr)
        self.trainer.update_lr(lr)
        return lr

    def decay_epsilon(self, game_number: int) -> float:
        ratio = 1.0 - (float(game_number) / float(max(1, self.total_games)))
        ratio = max(0.0, ratio)
        epsilon_t = self.min_num + (self.epsilon_0 - self.min_num) * (ratio**self.alpha)
        self.epsilon = max(self.min_num, float(epsilon_t))
        return self.epsilon

    def remember(self, state: np.ndarray, next_state: np.ndarray, reward: float, finished: bool) -> None:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)

        current_q_value = float(self.model1(state_tensor).detach().squeeze().item())
        with torch.no_grad():
            if not finished:
                next_q_value = float(self.model2(next_state_tensor).max().item())
                target_q_value = float(reward + self.gamma * next_q_value)
            else:
                target_q_value = float(reward)

        td_error = abs(target_q_value - current_q_value)
        self.memory.add(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(next_state, dtype=np.float32),
                float(reward),
                bool(finished),
            ),
            td_error,
        )

    def train_long_memory(self) -> float:
        self.trainer.clear_q()
        loss = float(self.trainer.train())
        self.last_loss = loss
        self.losses.append(loss)
        return loss

    def update_target_network(self) -> None:
        self.model2.load_state_dict(self.model1.state_dict())

    def get_action(self, candidates: List[CandidateAfterstate]) -> Optional[CandidateAfterstate]:
        if not candidates:
            return None

        if random.random() < self.epsilon:
            self.random = True
            return random.choice(candidates)

        self.random = False
        feature_batch = np.stack([c.feature_vector for c in candidates], axis=0).astype(np.float32, copy=False)
        with torch.no_grad():
            q_values = self.model1(torch.as_tensor(feature_batch, dtype=torch.float32, device=self.device)).view(-1)
        best_idx = int(torch.argmax(q_values).item())
        return candidates[best_idx]

    def check_steps(self) -> Optional[float]:
        self.total_steps += 1
        out_loss: Optional[float] = None
        if self.total_steps % int(self.config.training.target_sync_every_steps) == 0:
            self.update_target_network()
        if self.total_steps % int(self.config.training.train_every_steps) == 0:
            out_loss = self.train_long_memory()
            self.trained = True
        return out_loss

    def check_training(self) -> Optional[float]:
        out_loss: Optional[float] = None
        if not self.trained:
            out_loss = self.train_long_memory()
            self.games_without_training += 1
            if self.games_without_training == int(self.config.training.train_fallback_games):
                self.update_target_network()
                self.games_without_training = 0
        else:
            self.games_without_training = 0
        self.trained = False
        return out_loss

    def calculate_reward(self, feature_vector: np.ndarray, finished: bool) -> RewardTerms:
        return self.reward.compute(feature_vector, finished=finished)

