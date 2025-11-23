import argparse
import json
import subprocess
import sys
from pathlib import Path

import pygame


COLORS = {
    "I": (0, 255, 255),
    "O": (255, 255, 0),
    "T": (128, 0, 128),
    "L": (255, 165, 0),
    "J": (0, 0, 255),
    "S": (0, 200, 0),
    "Z": (200, 0, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Watch Cold Clear play via pygame.")
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("build/TetrisVersionTwo/cc_play"),
        help="Path to cc_play executable built from the C++ bot.",
    )
    parser.add_argument("--steps", type=int, default=500, help="Frames to play.")
    parser.add_argument("--delay-ms", type=int, default=80, help="Delay between moves in the bot (controls speed).")
    parser.add_argument("--cell", type=int, default=28, help="Cell size in pixels.")
    return parser.parse_args()


def launch_bot(binary: Path, steps: int, delay_ms: int):
    cmd = [str(binary), "--json", "--steps", str(steps), "--delay-ms", str(delay_ms)]
    try:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    except FileNotFoundError:
        sys.stderr.write(f"Could not start {binary}. Did you build it? (cmake --build build)\n")
        sys.exit(1)


def draw_board(surface, board_rows, cell_size):
    # board_rows are top->bottom strings of length 10
    for y, row in enumerate(board_rows):
        for x, c in enumerate(row):
            if c == ".":
                continue
            color = COLORS.get(c, (180, 180, 180))
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, (40, 40, 40), rect, width=1)


def draw_side_text(surface, font, hold, queue, offset_x, cell_size):
    y = 10
    hold_text = font.render(f"Hold: {hold}", True, (255, 255, 255))
    surface.blit(hold_text, (offset_x, y))
    y += hold_text.get_height() + 10

    surface.blit(font.render("Next:", True, (255, 255, 255)), (offset_x, y))
    y += font.get_height() + 6
    for p in queue[:5]:
        text = font.render(p, True, COLORS.get(p, (200, 200, 200)))
        surface.blit(text, (offset_x, y))
        y += text.get_height() + 4


def main():
    args = parse_args()
    proc = launch_bot(args.binary, args.steps, args.delay_ms)

    pygame.init()
    cell = args.cell
    board_width = 10 * cell
    board_height = 20 * cell
    side_width = 160
    screen = pygame.display.set_mode((board_width + side_width, board_height))
    pygame.display.set_caption("Cold Clear (C++ bot)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)

    running = True
    for line in proc.stdout:
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if not running:
            break

        screen.fill((10, 10, 10))
        draw_board(screen, frame["board"], cell)
        draw_side_text(screen, font, frame["hold"], frame["queue"], board_width + 12, cell)

        pygame.display.flip()
        clock.tick(60)

    proc.terminate()
    pygame.quit()


if __name__ == "__main__":
    main()
