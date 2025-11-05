"""Convenience wrapper to train on the modern environment."""

from __future__ import annotations

import sys

from tetris_v2.agents.dqn.train import main as dqn_main


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--env" not in argv:
        argv = ["--env", "modern", *argv]
    return dqn_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
