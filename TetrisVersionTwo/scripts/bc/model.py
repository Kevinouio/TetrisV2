from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class BCPolicyNet(nn.Module):
    def __init__(
        self,
        action_vocab_size: int,
        aux_dim: int,
        board_height: int = 20,
        board_width: int = 10,
        conv_channels: Sequence[int] = (32, 64, 64),
        mlp_hidden: Sequence[int] = (256, 256),
    ):
        super().__init__()
        if action_vocab_size <= 0:
            raise ValueError(f"action_vocab_size must be positive, got {action_vocab_size}")
        if len(conv_channels) < 2:
            raise ValueError("conv_channels must contain at least two entries.")
        if len(mlp_hidden) < 1:
            raise ValueError("mlp_hidden must contain at least one entry.")

        c1, c2, *rest = [int(v) for v in conv_channels]
        c3 = int(rest[0]) if rest else c2

        self.board_encoder = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, board_height, board_width)
            board_out_dim = int(self.board_encoder(dummy).flatten(start_dim=1).shape[1])

        board_feat_dim = int(mlp_hidden[0])
        self.board_proj = nn.Sequential(
            nn.Linear(board_out_dim, board_feat_dim),
            nn.ReLU(),
        )

        self.aux_dim = int(aux_dim)
        aux_hidden = max(32, min(128, self.aux_dim if self.aux_dim > 0 else 32))
        if self.aux_dim > 0:
            self.aux_encoder = nn.Sequential(
                nn.Linear(self.aux_dim, aux_hidden),
                nn.ReLU(),
            )
            fusion_in = board_feat_dim + aux_hidden
        else:
            self.aux_encoder = None
            fusion_in = board_feat_dim

        layers: list[nn.Module] = []
        prev = fusion_in
        for hidden in [int(v) for v in mlp_hidden]:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            prev = hidden
        layers.append(nn.Linear(prev, action_vocab_size))
        self.head = nn.Sequential(*layers)

    def forward(self, board: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        board = board.float()
        aux = aux.float()

        board_feat = self.board_encoder(board).flatten(start_dim=1)
        board_feat = self.board_proj(board_feat)

        if self.aux_encoder is not None:
            aux_feat = self.aux_encoder(aux)
            fused = torch.cat([board_feat, aux_feat], dim=1)
        else:
            fused = board_feat
        return self.head(fused)

