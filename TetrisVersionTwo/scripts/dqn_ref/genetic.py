from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .agent import DQNRefAgent
from .config import DQNRefConfig, SEEDED_GENOME, WEIGHT_RANGES


@dataclass
class PopulationEntry:
    genome: Dict[str, float]
    agent: DQNRefAgent
    fitness: float = 0.0


class GeneticPopulation:
    def __init__(
        self,
        config: DQNRefConfig,
        device: torch.device,
        seed: int,
        model_checkpoint: Path,
    ):
        self.config = config
        self.device = device
        self.rng = np.random.default_rng(int(seed))
        self.model_checkpoint = model_checkpoint

    def _new_agent(self, genome: Dict[str, float]) -> DQNRefAgent:
        checkpoint = self.model_checkpoint if self.model_checkpoint.exists() else None
        return DQNRefAgent(
            genome=genome,
            config=self.config,
            device=self.device,
            checkpoint_path=checkpoint,
        )

    def create_population(self, n: int) -> List[PopulationEntry]:
        out: List[PopulationEntry] = []
        n = int(max(1, n))
        for _ in range(max(0, n - 1)):
            genome = {
                key: float(self.rng.uniform(low=low, high=high))
                for key, (low, high) in WEIGHT_RANGES.items()
            }
            out.append(PopulationEntry(genome=genome, agent=self._new_agent(genome), fitness=0.0))

        seeded = {k: float(v) for k, v in SEEDED_GENOME.items()}
        out.append(PopulationEntry(genome=seeded, agent=self._new_agent(seeded), fitness=0.0))
        return out

    def best_elites(self, population: Sequence[PopulationEntry]) -> List[Tuple[Dict[str, float], float]]:
        scored = list(enumerate(population))
        scored.sort(key=lambda x: (-float(x[1].fitness), x[0]))
        top_k = int(min(max(1, self.config.ga.elite_count), len(scored)))
        return [(dict(entry.genome), float(entry.fitness)) for _, entry in scored[:top_k]]

    def selection(
        self,
        population: Sequence[Tuple[Dict[str, float], float]],
        generation_number: int,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        base_pick = int(self.config.ga.size_pick)
        growth = int(self.config.ga.generation_rate)
        pick_n = base_pick + ((int(generation_number) - 1) // max(1, growth))
        pick_n = min(max(1, pick_n), len(population))

        idx1 = self.rng.choice(len(population), size=pick_n, replace=False)
        idx2 = self.rng.choice(len(population), size=pick_n, replace=False)
        parent1 = max((population[int(i)] for i in idx1), key=lambda x: x[1])[0]
        parent2 = max((population[int(i)] for i in idx2), key=lambda x: x[1])[0]
        return dict(parent1), dict(parent2)

    def get_crossover_rates(self, generation_number: int) -> Tuple[float, float]:
        k = float(self.config.ga.crossover_k)
        midpoint = float(self.config.ga.crossover_midpoint)
        uniform_rate = 1.0 / (1.0 + math.exp(k * (float(generation_number - 1) - midpoint)))
        alpha_rate = 1.0 - uniform_rate
        return float(uniform_rate), float(alpha_rate)

    def crossover(
        self,
        parent1: Dict[str, float],
        parent2: Dict[str, float],
        generation_number: int,
    ) -> Dict[str, float]:
        uniform_rate, _alpha_rate = self.get_crossover_rates(generation_number)
        if float(self.rng.random()) < uniform_rate:
            child = {
                key: (float(parent1[key]) if float(self.rng.random()) < 0.5 else float(parent2[key]))
                for key in parent1.keys()
            }
            return child
        alpha = float(self.rng.uniform(0.0, 1.0))
        return {
            key: float(alpha * float(parent1[key]) + (1.0 - alpha) * float(parent2[key]))
            for key in parent1.keys()
        }

    def get_mutate_rate(self, generation_number: int) -> float:
        initial_rate = float(self.config.ga.mutate_initial_rate)
        min_rate = float(self.config.ga.mutate_min_rate)
        decay_start = int(self.config.ga.mutate_decay_start)
        k = float(self.config.ga.mutate_decay_k)
        if generation_number < decay_start:
            return initial_rate
        return min_rate + (initial_rate - min_rate) * math.exp(-k * (generation_number + 1 - decay_start))

    def mutate(self, genome: Dict[str, float], generation_number: int) -> Dict[str, float]:
        rate = self.get_mutate_rate(generation_number)
        out = dict(genome)
        for key in list(out.keys()):
            if float(self.rng.random()) < rate:
                low, high = WEIGHT_RANGES[key]
                out[key] = float(self.rng.uniform(low=low, high=high))
        return out

    def next_population(
        self,
        elites: Sequence[Tuple[Dict[str, float], float]],
        generation_number: int,
    ) -> List[PopulationEntry]:
        if not elites:
            raise ValueError("Cannot create next population with no elites.")

        out: List[PopulationEntry] = []
        for genome, _fitness in elites:
            out.append(PopulationEntry(genome=dict(genome), agent=self._new_agent(dict(genome)), fitness=0.0))

        target_size = int(self.config.ga.population_size)
        while len(out) < target_size:
            p1, p2 = self.selection(elites, generation_number=generation_number)
            child = self.crossover(p1, p2, generation_number=generation_number)
            child = self.mutate(child, generation_number=generation_number)
            out.append(PopulationEntry(genome=child, agent=self._new_agent(child), fitness=0.0))
        return out

