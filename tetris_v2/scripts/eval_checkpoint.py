"""CLI entrypoint for evaluating saved checkpoints."""

from __future__ import annotations

import sys

from tetris_v2.agents.dqn.eval import main as eval_main


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    return eval_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
