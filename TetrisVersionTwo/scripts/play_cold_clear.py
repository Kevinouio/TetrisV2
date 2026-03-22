"""Lane entrypoint: Cold Clear autoplay viewer."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from TetrisVersionTwo.scripts.play_pygame import main as viewer_main


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--ai" not in argv:
        argv = ["--ai"] + argv
    return viewer_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
