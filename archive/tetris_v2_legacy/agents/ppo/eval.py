"""Compatibility shim: route to the native PPO eval script."""

from tetris_v2.agents.single_agent.ppo.eval import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
