from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


WEIGHT_RANGES: Dict[str, Tuple[float, float]] = {
    "game_over": (-300.0, -50.0),
    "survival_instinct": (20.0, 200.0),
    "total_height": (-20.0, 0.0),
    "lines_removed": (5.0, 50.0),
    "holes": (-40.0, 0.0),
    "bumpiness": (-40.0, 0.0),
    "pillar": (-40.0, 0.0),
    "y_pos_reward": (50.0, 300.0),
    "y_pos_punish": (50.0, 300.0),
}


SEEDED_GENOME: Dict[str, float] = {
    "game_over": 189.27613725914273,
    "survival_instinct": 8.388926084018738,
    "total_height": -0.17634932529980674,
    "lines_removed": 8.594602383216944,
    "holes": -4.743561101942274,
    "bumpiness": -6.683915232551735,
    "pillar": -11.042880500059761,
    "y_pos_reward": 207.81525814829266,
    "y_pos_punish": 117.90325502640637,
}


@dataclass(frozen=True)
class ModelConfig:
    input_size: int = 6
    hidden_sizes: Tuple[int, int, int] = (32, 32, 32)
    output_size: int = 1


@dataclass(frozen=True)
class ReplayConfig:
    max_memory: int = 10_000
    batch_size: int = 128
    beta: float = 0.4


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate_start: float = 0.01
    learning_rate_end: float = 0.001
    gamma: float = 0.999
    train_epochs_per_call: int = 2
    train_every_steps: int = 200
    target_sync_every_steps: int = 1000
    epsilon_start: float = 0.3
    epsilon_min: float = 0.0001
    epsilon_alpha: float = 1.0
    total_games_for_decay: int = 500
    max_steps_per_episode: int = 2_000
    train_fallback_games: int = 5
    grad_clip_norm: float = 1.0


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 50
    generations: int = 100
    total_games_per_agent: int = 500
    size_pick: int = 5
    generation_rate: int = 10
    elite_count: int = 10
    crossover_k: float = 0.1
    crossover_midpoint: float = 50.0
    mutate_initial_rate: float = 0.50
    mutate_min_rate: float = 0.20
    mutate_decay_start: int = 100
    mutate_decay_k: float = 0.08


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 1234
    device: str = "cpu"
    torch_compile: bool = False
    channels_last: bool = False
    log_every_episodes: int = 10


@dataclass(frozen=True)
class DQNRefConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

