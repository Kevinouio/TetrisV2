from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from .replay import PrioritizedReplay


class LinearQNet(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: Tuple[int, int, int], output_size: int):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.output_layer = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.output_layer(x)
        return x


class QTrainer:
    def __init__(
        self,
        model1: nn.Module,
        model2: nn.Module,
        lr: float,
        gamma: float,
        memory: PrioritizedReplay,
        train_epochs: int,
        batch_size: int,
        replay_beta: float,
        grad_clip_norm: float,
        device: torch.device,
    ):
        self.memory = memory
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.train_epochs = int(train_epochs)
        self.batch_size = int(batch_size)
        self.replay_beta = float(replay_beta)
        self.grad_clip_norm = float(grad_clip_norm)
        self.model1 = model1
        self.model2 = model2
        self.device = device
        self.optimizer1 = optim.Adam(model1.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()
        self.q_values = []

    def update_lr(self, new_lr: float) -> None:
        self.lr = float(new_lr)
        for param_group in self.optimizer1.param_groups:
            param_group["lr"] = self.lr

    def clear_q(self) -> None:
        self.q_values = []

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x_train = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y_train = (
            y
            if isinstance(y, torch.Tensor)
            else torch.as_tensor(y, dtype=torch.float32, device=self.device).view(-1, 1)
        )

        self.optimizer1.zero_grad(set_to_none=True)
        outputs = self.model1(x_train)
        loss = self.criterion(outputs, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model1.parameters(), max_norm=self.grad_clip_norm)
        self.optimizer1.step()
        return float(loss.item())

    def train_step(self) -> float:
        if self.batch_size > len(self.memory):
            return 0.0

        batch, indices, _weights = self.memory.sample(self.batch_size, beta=self.replay_beta)
        states, next_states, rewards, dones = zip(*batch)

        state_batch = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        next_state_batch = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
        reward_batch = torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=self.device).view(-1, 1)
        done_batch = torch.as_tensor(np.asarray(dones), dtype=torch.bool, device=self.device)

        current_q_values = self.model1(state_batch)
        next_q_values_target = self.model2(next_state_batch)
        next_q_values_primary = self.model1(next_state_batch)

        max_actions = torch.argmax(next_q_values_primary, dim=1)
        target_q_values = reward_batch + (1.0 - done_batch.float().view(-1, 1)) * self.gamma * next_q_values_target.gather(
            1, max_actions.view(-1, 1)
        )

        td_errors = torch.abs(target_q_values - current_q_values.gather(1, max_actions.view(-1, 1)))
        self.memory.update_priority(indices, td_errors.detach().cpu().view(-1).tolist())

        return self.fit(state_batch, target_q_values.detach())

    def train(self) -> float:
        total_loss = 0.0
        for _ in range(self.train_epochs):
            total_loss += self.train_step()
        return float(total_loss)

    def load_optimizer_state(self, state_dict: dict) -> None:
        self.optimizer1.load_state_dict(state_dict)

    def optimizer_state(self) -> dict:
        return self.optimizer1.state_dict()

