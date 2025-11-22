"""Compatibility shim for the native evaluator."""

from tetris_v2.agents.single_agent.dqn.eval import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
