"""Expert auxiliary losses for placement-action DQN."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def _mask_q(q_values: Tensor, action_mask: Tensor) -> Tensor:
    legal = action_mask > 0.5
    if torch.any(~legal.any(dim=-1)):
        raise ValueError("Action mask contains no legal actions.")
    return q_values.masked_fill(~legal, torch.finfo(q_values.dtype).min)


def behavior_cloning_ce_loss(
    q_values: Tensor,
    teacher_best_action: Tensor,
    action_mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Cross-entropy to teacher top-1 plus top-1 agreement."""

    logits = _mask_q(q_values, action_mask)
    target = teacher_best_action.long()
    loss = F.cross_entropy(logits, target)

    pred = torch.argmax(logits, dim=-1)
    agreement = (pred == target).float().mean()
    return loss, agreement


def pairwise_ranking_loss(
    q_values: Tensor,
    teacher_best_action: Tensor,
    action_mask: Tensor,
    *,
    rng: np.random.Generator,
    pairs_per_sample: int = 4,
) -> Tensor:
    """Rank the teacher's action above sampled legal alternatives."""

    batch = int(q_values.shape[0])
    if batch <= 0 or pairs_per_sample <= 0:
        return q_values.new_zeros(())

    legal = action_mask.detach().cpu().numpy() > 0.5
    rows = np.arange(batch)
    legal[rows, teacher_best_action.detach().cpu().numpy().astype(np.int64)] = False
    has_alternative = legal.any(axis=1)
    if not np.any(has_alternative):
        return q_values.new_zeros(())

    action_dim = int(q_values.shape[1])
    random_scores = rng.random((batch, pairs_per_sample, action_dim), dtype=np.float32)
    random_scores = np.where(legal[:, None, :], random_scores, -1.0)
    alternatives = torch.from_numpy(np.argmax(random_scores, axis=-1)).to(
        q_values.device, dtype=torch.long
    )
    teacher_q = q_values.gather(1, teacher_best_action.long().unsqueeze(1))
    alternative_q = q_values.gather(1, alternatives)
    valid = torch.from_numpy(has_alternative).to(q_values.device)
    return -F.logsigmoid((teacher_q - alternative_q)[valid]).mean()


def expert_aux_losses(
    q_values: Tensor,
    teacher_best_action: Tensor,
    action_mask: Tensor,
    *,
    rng: np.random.Generator,
    pairs_per_sample: int = 4,
) -> Tuple[Tensor, Tensor, Tensor]:
    bc_loss, agreement = behavior_cloning_ce_loss(q_values, teacher_best_action, action_mask)
    pair_loss = pairwise_ranking_loss(
        q_values,
        teacher_best_action,
        action_mask,
        rng=rng,
        pairs_per_sample=pairs_per_sample,
    )
    return bc_loss, pair_loss, agreement


__all__ = [
    "behavior_cloning_ce_loss",
    "expert_aux_losses",
    "pairwise_ranking_loss",
]
