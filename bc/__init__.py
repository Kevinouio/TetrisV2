"""Lazy compatibility shim for the BC package."""

from __future__ import annotations

import importlib
from typing import Any

_IMPL_MODULE = "TetrisVersionTwo.scripts.bc"


def __getattr__(name: str) -> Any:
    mod = importlib.import_module(_IMPL_MODULE)
    return getattr(mod, name)


def __dir__() -> list[str]:
    mod = importlib.import_module(_IMPL_MODULE)
    return sorted(set(globals().keys()) | set(dir(mod)))

