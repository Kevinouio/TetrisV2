import argparse
import ctypes
import random
import sys
from pathlib import Path
from typing import Optional

import pygame


BOARD_ROWS = 20
BOARD_COLS = 10

ACTION_CW = 5
ACTION_CCW = 6
ACTION_180 = 7
ACTION_HOLD = 8

PIECE_NAMES = {
    0: "I",
    1: "O",
    2: "T",
    3: "L",
    4: "J",
    5: "S",
    6: "Z",
    7: "None",
}

ROTATION_NAMES = {0: "N", 1: "E", 2: "S", 3: "W"}

PIECE_COLORS = {
    0: (0, 230, 230),
    1: (240, 220, 0),
    2: (170, 70, 230),
    3: (255, 150, 60),
    4: (80, 110, 255),
    5: (80, 220, 100),
    6: (240, 90, 90),
}

BG_COLOR = (14, 16, 22)
PANEL_COLOR = (22, 26, 34)
GRID_LINE = (50, 56, 70)
LOCK_TEXT = (230, 235, 245)
PASS_COLOR = (110, 230, 150)
FAIL_COLOR = (240, 120, 120)
SELECT_COLOR = (80, 130, 240)
BOARD_FILL = (70, 80, 100)


def parse_args():
    parser = argparse.ArgumentParser(description="Pygame Placement + Kick Explorer via ctypes.")
    parser.add_argument("--lib", type=Path, default=None, help="Path to tetris_v2_c_api shared library.")
    parser.add_argument("--cell", type=int, default=28, help="Main board cell size.")
    parser.add_argument("--fps", type=int, default=60, help="Render FPS.")
    parser.add_argument("--seed", type=int, default=1234, help="Initial reset seed.")
    parser.add_argument("--queue-visible", type=int, default=8, help="How many queued pieces to display.")
    return parser.parse_args()


def find_library(explicit_path: Optional[Path]) -> Path:
    if explicit_path is not None:
        if explicit_path.exists():
            return explicit_path
        raise FileNotFoundError(f"Library not found: {explicit_path}")

    candidates = [
        Path("build/TetrisVersionTwo/tetris_v2_c_api.dll"),
        Path("build/TetrisVersionTwo/Debug/tetris_v2_c_api.dll"),
        Path("build/TetrisVersionTwo/Release/tetris_v2_c_api.dll"),
        Path("build/TetrisVersionTwo/libtetris_v2_c_api.so"),
        Path("build/TetrisVersionTwo/libtetris_v2_c_api.dylib"),
        Path("TetrisVersionTwo/build/tetris_v2_c_api.dll"),
        Path("TetrisVersionTwo/build/Debug/tetris_v2_c_api.dll"),
        Path("TetrisVersionTwo/build/Release/tetris_v2_c_api.dll"),
        Path("TetrisVersionTwo/build/libtetris_v2_c_api.so"),
        Path("TetrisVersionTwo/build/libtetris_v2_c_api.dylib"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate tetris_v2_c_api shared library. Build it first.")


class EnvCtypes:
    def __init__(self, lib_path: Path, seed: int):
        self.lib = ctypes.CDLL(str(lib_path))
        self._bind()
        self.handle = self.lib.tetris_env_create(ctypes.c_uint32(seed))
        if not self.handle:
            raise RuntimeError("Failed to create env handle")
        self.seed = int(seed)

    def _bind(self):
        void_p = ctypes.c_void_p
        c_int_p = ctypes.POINTER(ctypes.c_int)

        self.lib.tetris_env_create.argtypes = [ctypes.c_uint32]
        self.lib.tetris_env_create.restype = void_p

        self.lib.tetris_env_destroy.argtypes = [void_p]
        self.lib.tetris_env_destroy.restype = None

        self.lib.tetris_env_reset.argtypes = [void_p, ctypes.c_uint32]
        self.lib.tetris_env_reset.restype = None

        self.lib.tetris_env_step.argtypes = [void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        self.lib.tetris_env_step.restype = ctypes.c_int

        self.lib.tetris_env_hold.argtypes = [void_p, ctypes.POINTER(ctypes.c_float)]
        self.lib.tetris_env_hold.restype = ctypes.c_int

        self.lib.tetris_env_observation_size.argtypes = [void_p, ctypes.c_int]
        self.lib.tetris_env_observation_size.restype = ctypes.c_size_t

        self.lib.tetris_env_board_write.argtypes = [
            void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        self.lib.tetris_env_board_write.restype = ctypes.c_size_t

        self.lib.tetris_env_active_piece.argtypes = [void_p, c_int_p, c_int_p, c_int_p, c_int_p]
        self.lib.tetris_env_active_piece.restype = ctypes.c_int

        self.lib.tetris_env_hold_piece.argtypes = [void_p, c_int_p, c_int_p, c_int_p]
        self.lib.tetris_env_hold_piece.restype = ctypes.c_int

        self.lib.tetris_env_queue_count.argtypes = [void_p]
        self.lib.tetris_env_queue_count.restype = ctypes.c_size_t

        self.lib.tetris_env_queue_get.argtypes = [void_p, ctypes.c_size_t, c_int_p]
        self.lib.tetris_env_queue_get.restype = ctypes.c_int

        self.lib.tetris_env_meta.argtypes = [void_p, c_int_p, c_int_p, c_int_p, c_int_p, c_int_p, c_int_p, c_int_p]
        self.lib.tetris_env_meta.restype = ctypes.c_int

        self.lib.tetris_env_placement_count.argtypes = [void_p]
        self.lib.tetris_env_placement_count.restype = ctypes.c_size_t

        self.lib.tetris_env_placement_get.argtypes = [void_p, ctypes.c_size_t, c_int_p, c_int_p, c_int_p, c_int_p]
        self.lib.tetris_env_placement_get.restype = ctypes.c_int

        self.lib.tetris_env_placement_board_write.argtypes = [
            void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        self.lib.tetris_env_placement_board_write.restype = ctypes.c_size_t

        self.lib.tetris_env_apply_placement_index.argtypes = [
            void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            c_int_p,
            c_int_p,
        ]
        self.lib.tetris_env_apply_placement_index.restype = ctypes.c_int

        self.lib.tetris_env_rotation_trace_count.argtypes = [void_p, ctypes.c_int]
        self.lib.tetris_env_rotation_trace_count.restype = ctypes.c_size_t

        self.lib.tetris_env_rotation_trace_get.argtypes = [
            void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            c_int_p,
            c_int_p,
            c_int_p,
            c_int_p,
            c_int_p,
            c_int_p,
            c_int_p,
            c_int_p,
            c_int_p,
            c_int_p,
        ]
        self.lib.tetris_env_rotation_trace_get.restype = ctypes.c_int

        self.lib.tetris_env_rotation_trace_meta.argtypes = [void_p, ctypes.c_int, c_int_p, c_int_p, c_int_p, c_int_p]
        self.lib.tetris_env_rotation_trace_meta.restype = ctypes.c_int

    def close(self):
        if self.handle:
            self.lib.tetris_env_destroy(self.handle)
            self.handle = None

    def reset(self, seed: int):
        self.seed = int(seed)
        self.lib.tetris_env_reset(self.handle, ctypes.c_uint32(seed))

    def hold(self):
        reward = ctypes.c_float(0.0)
        success = self.lib.tetris_env_hold(self.handle, ctypes.byref(reward))
        return {"success": bool(success), "reward": float(reward.value)}

    def observation_size(self):
        return int(self.lib.tetris_env_observation_size(self.handle, 1))

    def board(self):
        buf = (ctypes.c_uint8 * (BOARD_ROWS * BOARD_COLS))()
        written = self.lib.tetris_env_board_write(self.handle, 0, buf, len(buf))
        if written != len(buf):
            return [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        flat = [int(v) for v in buf]
        return [flat[r * BOARD_COLS : (r + 1) * BOARD_COLS] for r in range(BOARD_ROWS)]

    def active(self):
        piece = ctypes.c_int(-1)
        rotation = ctypes.c_int(-1)
        x = ctypes.c_int(0)
        y = ctypes.c_int(0)
        ok = self.lib.tetris_env_active_piece(
            self.handle,
            ctypes.byref(piece),
            ctypes.byref(rotation),
            ctypes.byref(x),
            ctypes.byref(y),
        )
        if not ok:
            return None
        return {"piece": int(piece.value), "rotation": int(rotation.value), "x": int(x.value), "y": int(y.value)}

    def hold_info(self):
        has_hold = ctypes.c_int(0)
        hold_piece = ctypes.c_int(7)
        hold_available = ctypes.c_int(0)
        ok = self.lib.tetris_env_hold_piece(
            self.handle,
            ctypes.byref(has_hold),
            ctypes.byref(hold_piece),
            ctypes.byref(hold_available),
        )
        if not ok:
            return {"has_hold": False, "hold_piece": 7, "hold_available": False}
        return {
            "has_hold": bool(has_hold.value),
            "hold_piece": int(hold_piece.value),
            "hold_available": bool(hold_available.value),
        }

    def queue(self):
        count = int(self.lib.tetris_env_queue_count(self.handle))
        out = []
        for i in range(count):
            piece = ctypes.c_int(-1)
            if self.lib.tetris_env_queue_get(self.handle, i, ctypes.byref(piece)):
                out.append(int(piece.value))
        return out

    def meta(self):
        vals = [ctypes.c_int(0) for _ in range(7)]
        ok = self.lib.tetris_env_meta(
            self.handle,
            ctypes.byref(vals[0]),
            ctypes.byref(vals[1]),
            ctypes.byref(vals[2]),
            ctypes.byref(vals[3]),
            ctypes.byref(vals[4]),
            ctypes.byref(vals[5]),
            ctypes.byref(vals[6]),
        )
        if not ok:
            return {
                "game_over": True,
                "top_out": False,
                "combo": -1,
                "b2b": False,
                "lines": 0,
                "lock_timer": 0,
                "lock_resets": 0,
            }
        return {
            "game_over": bool(vals[0].value),
            "top_out": bool(vals[1].value),
            "combo": int(vals[2].value),
            "b2b": bool(vals[3].value),
            "lines": int(vals[4].value),
            "lock_timer": int(vals[5].value),
            "lock_resets": int(vals[6].value),
        }

    def placements(self):
        count = int(self.lib.tetris_env_placement_count(self.handle))
        out = []
        for i in range(count):
            x = ctypes.c_int(0)
            y = ctypes.c_int(0)
            rot = ctypes.c_int(0)
            lines = ctypes.c_int(0)
            if self.lib.tetris_env_placement_get(
                self.handle, i, ctypes.byref(x), ctypes.byref(y), ctypes.byref(rot), ctypes.byref(lines)
            ):
                out.append({"index": i, "x": int(x.value), "y": int(y.value), "rotation": int(rot.value), "lines": int(lines.value)})
        return out

    def placement_board(self, index: int):
        buf = (ctypes.c_uint8 * (BOARD_ROWS * BOARD_COLS))()
        written = self.lib.tetris_env_placement_board_write(self.handle, int(index), buf, len(buf))
        if written != len(buf):
            return [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        flat = [int(v) for v in buf]
        return [flat[r * BOARD_COLS : (r + 1) * BOARD_COLS] for r in range(BOARD_ROWS)]

    def apply_placement(self, index: int):
        reward = ctypes.c_float(0.0)
        lines = ctypes.c_int(0)
        game_over = ctypes.c_int(0)
        ok = self.lib.tetris_env_apply_placement_index(
            self.handle,
            int(index),
            ctypes.byref(reward),
            ctypes.byref(lines),
            ctypes.byref(game_over),
        )
        return {
            "success": bool(ok),
            "reward": float(reward.value),
            "lines": int(lines.value),
            "game_over": bool(game_over.value),
        }

    def rotation_trace(self, action: int):
        count = int(self.lib.tetris_env_rotation_trace_count(self.handle, int(action)))
        tests = []
        for i in range(count):
            vals = [ctypes.c_int(0) for _ in range(10)]
            ok = self.lib.tetris_env_rotation_trace_get(
                self.handle,
                int(action),
                i,
                ctypes.byref(vals[0]),
                ctypes.byref(vals[1]),
                ctypes.byref(vals[2]),
                ctypes.byref(vals[3]),
                ctypes.byref(vals[4]),
                ctypes.byref(vals[5]),
                ctypes.byref(vals[6]),
                ctypes.byref(vals[7]),
                ctypes.byref(vals[8]),
                ctypes.byref(vals[9]),
            )
            if not ok:
                continue
            tests.append(
                {
                    "test_index": int(vals[0].value),
                    "phase": int(vals[1].value),
                    "kick_index": int(vals[2].value),
                    "dx": int(vals[3].value),
                    "dy": int(vals[4].value),
                    "passed": bool(vals[5].value),
                    "x": int(vals[6].value),
                    "y": int(vals[7].value),
                    "rotation": int(vals[8].value),
                    "collides": bool(vals[9].value),
                }
            )

        success = ctypes.c_int(0)
        fx = ctypes.c_int(-1)
        fy = ctypes.c_int(-1)
        fr = ctypes.c_int(-1)
        self.lib.tetris_env_rotation_trace_meta(
            self.handle,
            int(action),
            ctypes.byref(success),
            ctypes.byref(fx),
            ctypes.byref(fy),
            ctypes.byref(fr),
        )
        return {
            "success": bool(success.value),
            "final_x": int(fx.value),
            "final_y": int(fy.value),
            "final_rotation": int(fr.value),
            "tests": tests,
        }


def piece_cells(piece: int, rotation: int):
    base = {
        0: [(-1, 0), (0, 0), (1, 0), (2, 0)],
        1: [(0, 0), (1, 0), (0, 1), (1, 1)],
        2: [(-1, 0), (0, 0), (1, 0), (0, 1)],
        3: [(-1, 0), (0, 0), (1, 0), (1, 1)],
        4: [(-1, 0), (0, 0), (1, 0), (-1, 1)],
        5: [(-1, 0), (0, 0), (0, 1), (1, 1)],
        6: [(-1, 1), (0, 1), (0, 0), (1, 0)],
    }.get(piece, [])

    def rot(cell):
        x, y = cell
        if rotation == 0:
            return (x, y)
        if rotation == 1:
            return (y, -x)
        if rotation == 2:
            return (-x, -y)
        return (-y, x)

    return [rot(c) for c in base]


def draw_board(surface, x0, y0, cell, board, active):
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            rect = pygame.Rect(x0 + c * cell, y0 + r * cell, cell, cell)
            if board[r][c]:
                pygame.draw.rect(surface, BOARD_FILL, rect)
            pygame.draw.rect(surface, GRID_LINE, rect, width=1)

    if active is not None and 0 <= active["piece"] <= 6:
        color = PIECE_COLORS.get(active["piece"], (220, 220, 220))
        for dx, dy in piece_cells(active["piece"], active["rotation"]):
            x = active["x"] + dx
            y = active["y"] + dy
            if x < 0 or x >= BOARD_COLS or y < 0 or y >= BOARD_ROWS:
                continue
            row = (BOARD_ROWS - 1) - y
            rect = pygame.Rect(x0 + x * cell, y0 + row * cell, cell, cell)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, (30, 30, 30), rect, width=1)


def draw_small_board(surface, x0, y0, cell, board):
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            rect = pygame.Rect(x0 + c * cell, y0 + r * cell, cell, cell)
            if board[r][c]:
                pygame.draw.rect(surface, (120, 140, 170), rect)
            pygame.draw.rect(surface, (55, 60, 72), rect, width=1)


def action_name(action: int):
    return {ACTION_CW: "CW", ACTION_CCW: "CCW", ACTION_180: "180"}.get(action, "?")


def main():
    args = parse_args()
    try:
        lib_path = find_library(args.lib)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    pygame.init()
    pygame.display.set_caption("Tetris Placement + Kick Explorer")
    font = pygame.font.SysFont("Consolas", 18)
    small_font = pygame.font.SysFont("Consolas", 15)

    cell = max(12, args.cell)
    board_x, board_y = 20, 20
    board_w, board_h = BOARD_COLS * cell, BOARD_ROWS * cell
    right_x = board_x + board_w + 20
    right_w = 520
    screen_w = right_x + right_w + 20
    screen_h = max(board_y + board_h + 20, 840)

    screen = pygame.display.set_mode((screen_w, screen_h))
    clock = pygame.time.Clock()

    env = EnvCtypes(lib_path, args.seed)
    selected_index = 0
    inspector_actions = [ACTION_CW, ACTION_CCW, ACTION_180]
    inspector_idx = 0
    status = f"Loaded {lib_path.name}"
    seed = int(args.seed)

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_UP:
                        selected_index = max(0, selected_index - 1)
                    elif event.key == pygame.K_DOWN:
                        selected_index += 1
                    elif event.key == pygame.K_RETURN:
                        result = env.apply_placement(selected_index)
                        if result["success"]:
                            status = f"Applied placement {selected_index}: reward={result['reward']:.1f} lines={result['lines']}"
                        else:
                            status = "Placement apply failed."
                    elif event.key == pygame.K_h:
                        result = env.hold()
                        status = "Hold used." if result["success"] else "Hold unavailable."
                    elif event.key == pygame.K_LEFTBRACKET:
                        inspector_idx = (inspector_idx - 1) % len(inspector_actions)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        inspector_idx = (inspector_idx + 1) % len(inspector_actions)
                    elif event.key == pygame.K_r:
                        env.reset(seed)
                        selected_index = 0
                        status = f"Reset seed={seed}"
                    elif event.key == pygame.K_n:
                        seed = random.randint(1, 2**31 - 1)
                        env.reset(seed)
                        selected_index = 0
                        status = f"Reset new seed={seed}"

            board = env.board()
            active = env.active()
            hold = env.hold_info()
            queue = env.queue()
            meta = env.meta()
            placements = env.placements()

            if placements:
                selected_index = max(0, min(selected_index, len(placements) - 1))
                preview_board = env.placement_board(selected_index)
            else:
                selected_index = 0
                preview_board = [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]

            inspect_action = inspector_actions[inspector_idx]
            trace = env.rotation_trace(inspect_action)

            screen.fill(BG_COLOR)

            pygame.draw.rect(screen, PANEL_COLOR, (board_x - 4, board_y - 4, board_w + 8, board_h + 8), border_radius=6)
            draw_board(screen, board_x, board_y, cell, board, active)

            # Top-right info panel
            info_y = board_y
            pygame.draw.rect(screen, PANEL_COLOR, (right_x, info_y, right_w, 150), border_radius=8)
            lines = [
                f"Seed: {seed}",
                f"Obs size: {env.observation_size()}",
                f"Hold: {PIECE_NAMES.get(hold['hold_piece'], '?')}  avail={hold['hold_available']}",
                f"Queue: {' '.join(PIECE_NAMES.get(p, '?') for p in queue[:max(0, args.queue_visible)])}",
                f"GameOver={meta['game_over']} TopOut={meta['top_out']}",
                f"Combo={meta['combo']} B2B={meta['b2b']} Lines={meta['lines']}",
                f"LockTimer={meta['lock_timer']} Resets={meta['lock_resets']}",
            ]
            for i, txt in enumerate(lines):
                surface = small_font.render(txt, True, LOCK_TEXT)
                screen.blit(surface, (right_x + 10, info_y + 10 + i * 19))

            # Placement list panel
            list_y = info_y + 160
            list_h = 280
            pygame.draw.rect(screen, PANEL_COLOR, (right_x, list_y, right_w, list_h), border_radius=8)
            title = font.render(f"Placements ({len(placements)})", True, LOCK_TEXT)
            screen.blit(title, (right_x + 10, list_y + 8))

            row_h = 18
            max_rows = max(1, (list_h - 38) // row_h)
            start = 0
            if selected_index >= max_rows:
                start = selected_index - max_rows + 1
            end = min(len(placements), start + max_rows)
            for i in range(start, end):
                p = placements[i]
                y = list_y + 30 + (i - start) * row_h
                text = f"[{p['index']:03d}] x={p['x']:>2} y={p['y']:>2} rot={ROTATION_NAMES.get(p['rotation'], '?')} lines={p['lines']}"
                color = LOCK_TEXT
                if i == selected_index:
                    pygame.draw.rect(screen, SELECT_COLOR, (right_x + 6, y - 1, right_w - 12, row_h - 1), border_radius=4)
                    color = (255, 255, 255)
                screen.blit(small_font.render(text, True, color), (right_x + 12, y))

            # Preview panel
            preview_y = list_y + list_h + 10
            preview_cell = max(8, cell // 2)
            preview_h = BOARD_ROWS * preview_cell + 36
            pygame.draw.rect(screen, PANEL_COLOR, (right_x, preview_y, right_w, preview_h), border_radius=8)
            screen.blit(font.render("Selected Placement Board", True, LOCK_TEXT), (right_x + 10, preview_y + 8))
            draw_small_board(screen, right_x + 10, preview_y + 30, preview_cell, preview_board)

            # Kick inspector panel
            kick_y = preview_y + preview_h + 10
            kick_h = screen_h - kick_y - 20
            pygame.draw.rect(screen, PANEL_COLOR, (right_x, kick_y, right_w, kick_h), border_radius=8)
            header = f"Kick Inspector [{action_name(inspect_action)}] success={trace['success']} final=({trace['final_x']},{trace['final_y']},{ROTATION_NAMES.get(trace['final_rotation'], '?')})"
            screen.blit(small_font.render(header, True, LOCK_TEXT), (right_x + 10, kick_y + 8))

            trace_row_h = 17
            max_trace_rows = max(1, (kick_h - 34) // trace_row_h)
            tests = trace["tests"][:max_trace_rows]
            for i, t in enumerate(tests):
                y = kick_y + 28 + i * trace_row_h
                mark = "PASS" if t["passed"] else "fail"
                color = PASS_COLOR if t["passed"] else FAIL_COLOR
                txt = (
                    f"{t['test_index']:02d} ph{t['phase']} k{t['kick_index']} "
                    f"({t['dx']:+d},{t['dy']:+d}) {mark} -> "
                    f"({t['x']},{t['y']},{ROTATION_NAMES.get(t['rotation'], '?')})"
                )
                screen.blit(small_font.render(txt, True, color), (right_x + 10, y))

            controls = "Controls: Up/Down select | Enter apply | H hold | [ ] inspector | R reset seed | N new seed | Q quit"
            screen.blit(small_font.render(controls, True, LOCK_TEXT), (board_x, screen_h - 22))
            screen.blit(small_font.render(status, True, (180, 210, 255)), (board_x, board_y + board_h + 8))

            pygame.display.flip()
            clock.tick(max(10, args.fps))
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
