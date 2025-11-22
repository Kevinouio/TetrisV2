"""Convenience wrapper to launch the PPO trainer."""

from __future__ import annotations

import sys

from TetrisVersionOne.agents.ppo.train import main as ppo_main


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    return ppo_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
