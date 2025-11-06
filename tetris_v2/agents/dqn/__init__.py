"""Compatibility shim for relocated DQN package."""

from importlib import import_module
import sys

_pkg = "tetris_v2.agents.single_agent.dqn"

train = import_module(f"{_pkg}.train")
eval = import_module(f"{_pkg}.eval")
models = import_module(f"{_pkg}.models")

sys.modules[__name__ + ".train"] = train
sys.modules[__name__ + ".eval"] = eval
sys.modules[__name__ + ".models"] = models

from tetris_v2.agents.single_agent.dqn import *  # noqa: F401,F403,E402
