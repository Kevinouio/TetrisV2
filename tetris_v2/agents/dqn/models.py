"""Utilities for building DQN agents tailored to Tetris observations."""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Sequence

import numpy as np
from gymnasium import spaces

try:
    import torch as th
    from torch import nn
    from stable_baselines3 import DQN
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except ImportError as exc:  # pragma: no cover - optional dependency
    DQN = None  # type: ignore[assignment]
    BaseFeaturesExtractor = object  # type: ignore[assignment]
    th = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    OPTIONAL_IMPORT_ERROR = exc
else:
    OPTIONAL_IMPORT_ERROR = None


def _flatten_dim(space: spaces.Space) -> int:
    if isinstance(space, spaces.Discrete):
        return 1
    return int(np.prod(space.shape))


if DQN is not None:

    class BoardFeatureExtractor(BaseFeaturesExtractor):
        """Simple feature extractor that flattens dict observations."""

        def __init__(self, observation_space: spaces.Dict):
            total_dim = sum(_flatten_dim(space) for space in observation_space.spaces.values())
            super().__init__(observation_space, features_dim=total_dim)

        def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
            parts = []
            for key, value in observations.items():
                tensor = value
                if tensor.dim() > 2:
                    tensor = tensor.flatten(start_dim=1)
                else:
                    tensor = tensor.float().view(tensor.shape[0], -1)
                # Basic scaling for discrete signals
                if key == "board":
                    tensor = tensor / 7.0
                parts.append(tensor)
            return th.cat(parts, dim=1)

else:  # pragma: no cover - fallback for missing deps

    class BoardFeatureExtractor:  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise OPTIONAL_IMPORT_ERROR  # type: ignore[misc]


def build_dqn_agent(
    env,
    *,
    learning_rate: float = 2.5e-4,
    buffer_size: int = 200_000,
    exploration_fraction: float = 0.2,
    exploration_final_eps: float = 0.05,
    batch_size: int = 256,
    gamma: float = 0.99,
    target_update_interval: int = 10_000,
    net_arch: Sequence[int] = (512, 256),
    device: Optional[str] = None,
    **kwargs: Any,
):
    """Construct a configured SB3 DQN agent.

    Parameters mirror ``stable_baselines3.DQN`` for convenience.
    """
    if DQN is None or OPTIONAL_IMPORT_ERROR is not None:  # pragma: no cover - optional
        raise ImportError(
            "stable-baselines3[DQN] and PyTorch are required for training. "
            "Install them with `pip install stable-baselines3[extra] torch`."
        ) from OPTIONAL_IMPORT_ERROR

    policy_kwargs: Dict[str, Any] = dict(
        features_extractor_class=BoardFeatureExtractor,
        net_arch=list(net_arch),
        activation_fn=nn.ReLU,
    )

    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        batch_size=batch_size,
        gamma=gamma,
        target_update_interval=target_update_interval,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device or ("cuda" if th and th.cuda.is_available() else "cpu"),
        **kwargs,
    )
    return model
