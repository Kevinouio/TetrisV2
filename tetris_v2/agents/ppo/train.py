"""Compatibility shim: route legacy entry point to the native PPO trainer."""

from tetris_v2.agents.single_agent.ppo.train import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
