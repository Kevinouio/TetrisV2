"""Top-1 behavioral cloning baseline package (lazy exports)."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "CollectionConfig": ("TetrisVersionTwo.scripts.bc.config", "CollectionConfig"),
    "EncoderConfig": ("TetrisVersionTwo.scripts.bc.config", "EncoderConfig"),
    "ModelConfig": ("TetrisVersionTwo.scripts.bc.config", "ModelConfig"),
    "SplitConfig": ("TetrisVersionTwo.scripts.bc.config", "SplitConfig"),
    "TrainConfig": ("TetrisVersionTwo.scripts.bc.config", "TrainConfig"),
    "encode_state": ("TetrisVersionTwo.scripts.bc.encoders", "encode_state"),
    "flatten_aux_features": ("TetrisVersionTwo.scripts.bc.encoders", "flatten_aux_features"),
    "ActionCodec": ("TetrisVersionTwo.scripts.bc.utils", "ActionCodec"),
    "BCEnvAdapter": ("TetrisVersionTwo.scripts.bc.utils", "BCEnvAdapter"),
    "NativeAction": ("TetrisVersionTwo.scripts.bc.utils", "NativeAction"),
    "BCAgent": ("TetrisVersionTwo.scripts.bc.inference_agent", "BCAgent"),
    "BCPolicyNet": ("TetrisVersionTwo.scripts.bc.model", "BCPolicyNet"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol_name = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
