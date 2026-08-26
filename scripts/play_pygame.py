import argparse
import random
from pathlib import Path
from typing import Optional

import pygame

from tetris_v2.rl.runtime import (
    ACTION_CCW,
    ACTION_CW,
    BOARD_COLS,
    BOARD_ROWS,
    EMPTY_CELL_ID,
    PIECE_NAMES,
    EnvCtypes,
    find_library,
)

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


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Pygame Placement + Kick Explorer via ctypes.")
    parser.add_argument("--lib", type=Path, default=None, help="Path to tetris_v2_c_api shared library.")
    parser.add_argument("--cell", type=int, default=28, help="Main board cell size.")
    parser.add_argument("--fps", type=int, default=60, help="Render FPS.")
    parser.add_argument("--seed", type=int, default=1234, help="Initial reset seed.")
    parser.add_argument("--queue-visible", type=int, default=8, help="How many queued pieces to display.")
    parser.add_argument("--ai", action="store_true", help="Start with AI autoplay enabled.")
    parser.add_argument("--think-ms", type=int, default=20, help="AI think budget per move in milliseconds.")
    parser.add_argument(
        "--auto-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto reset to next seed on topout in AI mode (default: on).",
    )
    return parser.parse_args(argv)


def draw_board(surface, x0, y0, cell, board_piece_ids):
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            rect = pygame.Rect(x0 + c * cell, y0 + r * cell, cell, cell)
            piece_id = board_piece_ids[r][c]
            if piece_id != EMPTY_CELL_ID:
                pygame.draw.rect(surface, PIECE_COLORS.get(piece_id, BOARD_FILL), rect)
            pygame.draw.rect(surface, GRID_LINE, rect, width=1)


def draw_small_board(surface, x0, y0, cell, board_piece_ids):
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            rect = pygame.Rect(x0 + c * cell, y0 + r * cell, cell, cell)
            piece_id = board_piece_ids[r][c]
            if piece_id != EMPTY_CELL_ID:
                pygame.draw.rect(surface, PIECE_COLORS.get(piece_id, BOARD_FILL), rect)
            pygame.draw.rect(surface, (55, 60, 72), rect, width=1)


def action_name(action: int):
    return {ACTION_CW: "CW", ACTION_CCW: "CCW"}.get(action, "?")


def main(argv: Optional[list[str]] = None):
    args = parse_args(argv)
    lib_path = find_library(args.lib)

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
    screen_h = max(board_y + board_h + 20, 980)

    screen = pygame.display.set_mode((screen_w, screen_h))
    clock = pygame.time.Clock()

    env = EnvCtypes(lib_path, args.seed)
    selected_index = 0
    list_scroll = 0
    inspector_actions = [ACTION_CW, ACTION_CCW]
    inspector_idx = 0
    status = f"Loaded {lib_path.name}"
    seed = int(args.seed)
    ai_enabled = bool(args.ai)
    ai_metrics = {
        "pieces": 0,
        "lines": 0,
        "topouts": 0,
        "last_think_ms": 0.0,
        "avg_think_ms": 0.0,
        "last_nodes": 0,
        "last_nps": 0.0,
        "last_score": 0.0,
        "last_budget_miss": 0,
        "budget_misses": 0,
        "think_sum_ms": 0.0,
        "think_samples": 0,
        "start_ticks": pygame.time.get_ticks(),
    }
    if ai_enabled:
        status = f"AI enabled at startup (think={max(1, int(args.think_ms))}ms)"

    info_h = 240
    list_y = board_y + info_h + 10
    list_h = 280
    list_header_h = 30
    list_footer_h = 30
    row_h = 18
    list_content_h = max(1, list_h - list_header_h - list_footer_h)
    max_rows = max(1, list_content_h // row_h)
    scrollbar_w = 10
    list_rect = pygame.Rect(right_x, list_y, right_w, list_h)
    list_rows_rect = pygame.Rect(right_x + 6, list_y + list_header_h, right_w - 24, row_h * max_rows)
    list_scrollbar_rect = pygame.Rect(
        list_rows_rect.right + 6,
        list_rows_rect.top,
        scrollbar_w,
        list_rows_rect.height,
    )
    apply_button_rect = pygame.Rect(right_x + right_w - 126, list_y + list_h - 26, 116, 20)

    placements = env.placements()
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_a:
                        ai_enabled = not ai_enabled
                        if ai_enabled:
                            ai_metrics["start_ticks"] = pygame.time.get_ticks()
                            env.bot_sync()
                            status = f"AI enabled (think={max(1, int(args.think_ms))}ms)"
                        else:
                            status = "AI disabled."
                    elif event.key == pygame.K_UP:
                        selected_index = max(0, selected_index - 1)
                        if selected_index < list_scroll:
                            list_scroll = selected_index
                    elif event.key == pygame.K_DOWN:
                        selected_index += 1
                        if selected_index >= list_scroll + max_rows:
                            list_scroll = selected_index - max_rows + 1
                    elif event.key == pygame.K_RETURN and not ai_enabled:
                        result = env.apply_placement(selected_index)
                        if result["success"]:
                            status = f"Applied placement {selected_index}: reward={result['reward']:.1f} lines={result['lines']}"
                        else:
                            status = "Placement apply failed."
                    elif event.key == pygame.K_h and not ai_enabled:
                        result = env.hold()
                        status = "Hold used." if result["success"] else "Hold unavailable."
                    elif event.key == pygame.K_LEFTBRACKET:
                        inspector_idx = (inspector_idx - 1) % len(inspector_actions)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        inspector_idx = (inspector_idx + 1) % len(inspector_actions)
                    elif event.key == pygame.K_r:
                        env.reset(seed)
                        selected_index = 0
                        list_scroll = 0
                        status = f"Reset seed={seed}"
                    elif event.key == pygame.K_n:
                        seed = random.randint(1, 2**31 - 1)
                        env.reset(seed)
                        selected_index = 0
                        list_scroll = 0
                        status = f"Reset new seed={seed}"
                elif event.type == pygame.MOUSEWHEEL:
                    if list_rect.collidepoint(pygame.mouse.get_pos()):
                        max_start = max(0, len(placements) - max_rows)
                        list_scroll = max(0, min(max_start, list_scroll - int(event.y)))
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if apply_button_rect.collidepoint(event.pos) and placements and not ai_enabled:
                        result = env.apply_placement(selected_index)
                        if result["success"]:
                            status = f"Applied placement {selected_index}: reward={result['reward']:.1f} lines={result['lines']}"
                        else:
                            status = "Placement apply failed."
                    elif list_scrollbar_rect.collidepoint(event.pos) and placements and len(placements) > max_rows:
                        max_start = len(placements) - max_rows
                        rel = event.pos[1] - list_scrollbar_rect.top
                        ratio = max(0.0, min(1.0, rel / float(max(1, list_scrollbar_rect.height - 1))))
                        list_scroll = int(round(ratio * max_start))
                    elif list_rows_rect.collidepoint(event.pos):
                        row = (event.pos[1] - list_rows_rect.top) // row_h
                        clicked_index = list_scroll + int(row)
                        if 0 <= row < max_rows and 0 <= clicked_index < len(placements):
                            if clicked_index == selected_index and not ai_enabled:
                                result = env.apply_placement(selected_index)
                                if result["success"]:
                                    status = (
                                        f"Applied placement {selected_index}: reward={result['reward']:.1f} "
                                        f"lines={result['lines']}"
                                    )
                                else:
                                    status = "Placement apply failed."
                            else:
                                selected_index = clicked_index
                                status = f"Selected placement {selected_index}"

            pre_meta = env.meta()
            if ai_enabled:
                if pre_meta["game_over"]:
                    ai_metrics["topouts"] += 1
                    if args.auto_reset:
                        seed += 1
                        env.reset(seed)
                        status = f"AI auto-reset to seed={seed}"
                    else:
                        ai_enabled = False
                        status = "AI paused on topout. Press R/N or toggle AI with A."
                else:
                    ai_result = env.bot_choose_and_apply(args.think_ms)
                    if ai_result["success"]:
                        ai_metrics["pieces"] += 1
                        ai_metrics["lines"] += int(ai_result["lines"])
                        ai_metrics["last_think_ms"] = float(ai_result["think_ms"])
                        ai_metrics["think_sum_ms"] += float(ai_result["think_ms"])
                        ai_metrics["think_samples"] += 1
                        ai_metrics["avg_think_ms"] = (
                            ai_metrics["think_sum_ms"] / max(1, ai_metrics["think_samples"])
                        )
                        ai_metrics["last_nodes"] = int(ai_result["nodes"])
                        ai_metrics["last_nps"] = float(ai_result["nps"])
                        ai_metrics["last_score"] = float(ai_result["score"])
                        ai_metrics["last_budget_miss"] = int(ai_result["budget_miss"])
                        ai_metrics["budget_misses"] += int(ai_result["budget_miss"])
                        status = (
                            f"AI move idx={ai_result['placement_index']} hold={ai_result['used_hold']} "
                            f"score={ai_result['score']:.2f} lines+={ai_result['lines']}"
                        )
                        if ai_result["game_over"]:
                            ai_metrics["topouts"] += 1
                            if args.auto_reset:
                                seed += 1
                                env.reset(seed)
                                status = f"AI topout -> auto-reset seed={seed}"
                            else:
                                ai_enabled = False
                                status = "AI topout. Autoplay stopped (auto-reset disabled)."
                    else:
                        ai_enabled = False
                        status = "AI choose/apply failed. Autoplay disabled."

            board_piece_ids = env.board_piece_ids(include_active=True)
            hold = env.hold_info()
            queue = env.queue()
            meta = env.meta()
            placements = env.placements()

            if placements:
                selected_index = max(0, min(selected_index, len(placements) - 1))
                max_start = max(0, len(placements) - max_rows)
                list_scroll = max(0, min(max_start, list_scroll))
                preview_piece_ids = env.placement_piece_ids(selected_index)
            else:
                selected_index = 0
                list_scroll = 0
                preview_piece_ids = [[EMPTY_CELL_ID for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]

            inspect_action = inspector_actions[inspector_idx]
            trace = env.rotation_trace(inspect_action)

            screen.fill(BG_COLOR)

            pygame.draw.rect(screen, PANEL_COLOR, (board_x - 4, board_y - 4, board_w + 8, board_h + 8), border_radius=6)
            draw_board(screen, board_x, board_y, cell, board_piece_ids)

            # Top-right info panel
            info_y = board_y
            pygame.draw.rect(screen, PANEL_COLOR, (right_x, info_y, right_w, info_h), border_radius=8)
            ai_elapsed_s = max(1e-6, (pygame.time.get_ticks() - ai_metrics["start_ticks"]) / 1000.0)
            ai_pps = ai_metrics["pieces"] / ai_elapsed_s
            lines = [
                f"Seed: {seed}",
                f"Obs size: {env.observation_size()}",
                f"Hold: {PIECE_NAMES.get(hold['hold_piece'], '?')}  avail={hold['hold_available']}",
                f"Queue: {' '.join(PIECE_NAMES.get(p, '?') for p in queue[:max(0, args.queue_visible)])}",
                f"GameOver={meta['game_over']} TopOut={meta['top_out']}",
                f"Combo={meta['combo']} B2B={meta['b2b']} Lines={meta['lines']}",
                f"LockTimer={meta['lock_timer']} Resets={meta['lock_resets']}",
                f"AI: {'ON' if ai_enabled else 'OFF'} think={max(1, int(args.think_ms))}ms",
                f"AI pieces={ai_metrics['pieces']} lines={ai_metrics['lines']} topouts={ai_metrics['topouts']}",
                f"AI PPS={ai_pps:.2f} think(last/avg)={ai_metrics['last_think_ms']:.1f}/{ai_metrics['avg_think_ms']:.1f} ms",
                f"AI nodes={ai_metrics['last_nodes']} nps={ai_metrics['last_nps']:.0f} score={ai_metrics['last_score']:.2f}",
                f"AI budget_miss last/total={ai_metrics['last_budget_miss']}/{ai_metrics['budget_misses']}",
            ]
            for i, txt in enumerate(lines):
                surface = small_font.render(txt, True, LOCK_TEXT)
                screen.blit(surface, (right_x + 10, info_y + 10 + i * 19))

            # Placement list panel
            pygame.draw.rect(screen, PANEL_COLOR, (right_x, list_y, right_w, list_h), border_radius=8)
            title = font.render(f"Placements ({len(placements)})", True, LOCK_TEXT)
            screen.blit(title, (right_x + 10, list_y + 8))

            start = list_scroll
            end = min(len(placements), start + max_rows)
            for i in range(start, end):
                p = placements[i]
                y = list_rows_rect.top + (i - start) * row_h
                text = f"[{p['index']:03d}] x={p['x']:>2} y={p['y']:>2} rot={ROTATION_NAMES.get(p['rotation'], '?')} lines={p['lines']}"
                color = LOCK_TEXT
                if i == selected_index:
                    pygame.draw.rect(screen, SELECT_COLOR, (list_rows_rect.left, y - 1, list_rows_rect.width, row_h - 1), border_radius=4)
                    color = (255, 255, 255)
                screen.blit(small_font.render(text, True, color), (list_rows_rect.left + 6, y))
            if placements:
                shown = f"Showing {start + 1}-{end} / {len(placements)}"
                screen.blit(small_font.render(shown, True, LOCK_TEXT), (right_x + right_w - 180, list_y + 9))

            # Scrollbar and apply button
            pygame.draw.rect(screen, (42, 48, 60), list_scrollbar_rect, border_radius=4)
            if placements and len(placements) > max_rows:
                max_start = len(placements) - max_rows
                thumb_h = max(24, int(list_scrollbar_rect.height * (max_rows / float(len(placements)))))
                travel = max(0, list_scrollbar_rect.height - thumb_h)
                thumb_top = list_scrollbar_rect.top + int(travel * (list_scroll / float(max_start)))
                thumb_rect = pygame.Rect(list_scrollbar_rect.left, thumb_top, list_scrollbar_rect.width, thumb_h)
            else:
                thumb_rect = pygame.Rect(
                    list_scrollbar_rect.left,
                    list_scrollbar_rect.top,
                    list_scrollbar_rect.width,
                    list_scrollbar_rect.height,
                )
            pygame.draw.rect(screen, (120, 136, 170), thumb_rect, border_radius=4)

            button_enabled = bool(placements and not ai_enabled)
            button_color = (66, 96, 170) if button_enabled else (58, 64, 76)
            pygame.draw.rect(screen, button_color, apply_button_rect, border_radius=5)
            button_label = "Apply (Enter)" if not ai_enabled else "AI running"
            button_text = small_font.render(button_label, True, (255, 255, 255))
            screen.blit(button_text, (apply_button_rect.x + 12, apply_button_rect.y + 2))

            # Preview panel
            preview_y = list_y + list_h + 10
            preview_cell = max(8, cell // 2)
            preview_h = BOARD_ROWS * preview_cell + 36
            pygame.draw.rect(screen, PANEL_COLOR, (right_x, preview_y, right_w, preview_h), border_radius=8)
            screen.blit(font.render("Selected Placement Board", True, LOCK_TEXT), (right_x + 10, preview_y + 8))
            draw_small_board(screen, right_x + 10, preview_y + 30, preview_cell, preview_piece_ids)

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

            controls = "Controls: A toggle AI | Wheel browse | Click row | Enter apply | H | [ ] | R/N | Q"
            screen.blit(small_font.render(controls, True, LOCK_TEXT), (board_x, screen_h - 22))
            screen.blit(small_font.render(status, True, (180, 210, 255)), (board_x, board_y + board_h + 8))

            pygame.display.flip()
            clock.tick(max(10, args.fps))
    finally:
        env.close()
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
