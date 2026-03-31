from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence, Tuple


@dataclass
class EncoderConfig:
    board_height: int = 20
    board_width: int = 10
    queue_length: int = 5
    include_scalars: bool = False


@dataclass
class SplitConfig:
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 123

    def validate(self) -> None:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split fractions must sum to 1.0, got {total:.6f} "
                f"(train={self.train_fraction}, val={self.val_fraction}, test={self.test_fraction})."
            )
        for name, value in (
            ("train_fraction", self.train_fraction),
            ("val_fraction", self.val_fraction),
            ("test_fraction", self.test_fraction),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")


@dataclass
class CollectionConfig:
    num_episodes: int = 5_000
    max_steps_per_episode: int = 2_000
    think_ms: int = 20
    seed: int = 1234
    episodes_per_shard: int = 200


@dataclass
class ModelConfig:
    conv_channels: Tuple[int, int, int] = (32, 64, 64)
    mlp_hidden: Tuple[int, int] = (256, 256)


@dataclass
class TrainConfig:
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 6
    seed: int = 123


def parse_int_tuple(raw: str) -> Tuple[int, ...]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError(f"Expected at least one integer in '{raw}'.")
    out = tuple(int(v) for v in values)
    if any(v <= 0 for v in out):
        raise ValueError(f"All values must be positive in '{raw}'.")
    return out


def dataclass_to_dict(obj: object) -> Dict[str, object]:
    return asdict(obj)  # type: ignore[arg-type]


def tuple_to_list(values: Sequence[int]) -> list[int]:
    return [int(v) for v in values]

