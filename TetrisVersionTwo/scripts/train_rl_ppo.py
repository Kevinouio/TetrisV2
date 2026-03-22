"""Lane entrypoint: train VersionTwo PPO agent."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        from TetrisVersionTwo.rl.ppo.train import main as ppo_train_main
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc}. Install with: pip install -r requirements.txt", file=sys.stderr)
        return 1
    return ppo_train_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
