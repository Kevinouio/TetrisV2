"""Evaluate a trained PPO checkpoint."""

from __future__ import annotations

import sys

from TetrisVersionOne.agents.ppo.eval import main as eval_main


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    return eval_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
