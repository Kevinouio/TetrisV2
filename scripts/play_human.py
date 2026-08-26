"""Lane entrypoint: manual human placement explorer."""

from __future__ import annotations

from scripts.play_pygame import main as viewer_main


def main(argv=None):
    return viewer_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
