from __future__ import annotations

import heapq
from typing import List, Sequence, Tuple

import numpy as np

Transition = Tuple[np.ndarray, np.ndarray, float, bool]


def fast_sample(batch_size: int, probabilities: np.ndarray) -> np.ndarray:
    cumulative_probs = np.cumsum(probabilities)
    random_vals = np.random.rand(batch_size)
    return np.searchsorted(cumulative_probs, random_vals)


class PrioritizedReplay:
    """Faithful behavior to Version2 PrioritizedMemory."""

    def __init__(self, max_size: int):
        self.max_size = int(max_size)
        self.memory: List[Tuple[np.float32, int, Transition]] = []
        self.counter: int = 0
        self._pending_heapify = False

    def __len__(self) -> int:
        return len(self.memory)

    def add(self, experience: Transition, priority: float) -> None:
        if self._pending_heapify:
            heapq.heapify(self.memory)
            self._pending_heapify = False
        pri = np.float32(max(float(priority), 1e-6))
        item = (pri, self.counter, experience)
        if len(self.memory) < self.max_size:
            heapq.heappush(self.memory, item)
        else:
            heapq.heappushpop(self.memory, item)
        self.counter += 1

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        if len(self.memory) == 0:
            raise RuntimeError("Cannot sample from empty replay.")

        priorities = np.asarray([float(p) for p, _, _ in self.memory], dtype=np.float32)
        if np.any(priorities <= 0):
            priorities = np.maximum(priorities, 1e-6)

        probabilities = priorities ** float(beta)
        prob_sum = float(probabilities.sum())
        if prob_sum <= 0.0:
            probabilities = np.full_like(probabilities, 1.0 / len(probabilities))
        else:
            probabilities /= prob_sum

        indices = fast_sample(int(batch_size), probabilities)
        indices = np.clip(indices, 0, len(probabilities) - 1)
        batch = [self.memory[int(i)][2] for i in indices]

        weights = (1.0 / len(self.memory) / probabilities[indices]) ** float(beta)
        weights /= np.max(weights)
        return batch, indices.astype(np.int64), weights.astype(np.float32)

    def update_priority(
        self,
        indices: Sequence[int],
        td_errors: Sequence[float],
        *,
        defer_heapify: bool = False,
    ) -> None:
        for idx, td_error in zip(indices, td_errors):
            i = int(idx)
            priority = np.float32(max(float(td_error), 1e-6))
            old = self.memory[i]
            self.memory[i] = (priority, old[1], old[2])
        if defer_heapify:
            self._pending_heapify = True
            return
        heapq.heapify(self.memory)
        self._pending_heapify = False

    def flush_priority_updates(self) -> None:
        if not self._pending_heapify:
            return
        heapq.heapify(self.memory)
        self._pending_heapify = False
