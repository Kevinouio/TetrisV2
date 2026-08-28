"""Pygame playback for TetrisV2 RL checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pygame

from tetris_v2.rl.actions import decode_action
from tetris_v2.rl.env import CCTetrisEnv
from tetris_v2.rl.policy import load_policy
from tetris_v2.rl.runtime import (
    BOARD_COLS,
    BOARD_ROWS,
    EMPTY_CELL_ID,
    PIECE_NAMES,
)


BG_COLOR = (14, 16, 22)
PANEL_COLOR = (22, 26, 34)
GRID_LINE = (50, 56, 70)
LOCK_TEXT = (230, 235, 245)
BOARD_FILL = (70, 80, 100)
PIECE_COLORS = {
    0: (0, 230, 230),
    1: (240, 220, 0),
    2: (170, 70, 230),
    3: (255, 150, 60),
    4: (80, 110, 255),
    5: (80, 220, 100),
    6: (240, 90, 90),
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a TetrisV2 RL policy in pygame.")
    parser.add_argument("checkpoint", type=Path, help="Checkpoint path (.pt)")
    parser.add_argument("--algo", choices=("ppo", "dqn", "flow_dqn"), required=True)
    parser.add_argument("--lib", type=Path, default=None, help="Path to tetris_v2_c_api shared library.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cell", type=int, default=28)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="PPO sampling temperature or Flow-DQN Gaussian-latent scale.",
    )
    parser.add_argument("--epsilon", type=float, default=0.0, help="Exploration rate when --stochastic")
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=True,
        help="Use greedy policy actions (default).",
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Sample/explore instead of greedy actions.",
    )
    parser.add_argument(
        "--auto-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-reset when an episode ends (default: on).",
    )
    return parser.parse_args(argv)


def draw_board(surface, x0: int, y0: int, cell: int, board_piece_ids):
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            rect = pygame.Rect(x0 + c * cell, y0 + r * cell, cell, cell)
            piece_id = board_piece_ids[r][c]
            if piece_id != EMPTY_CELL_ID:
                pygame.draw.rect(surface, PIECE_COLORS.get(piece_id, BOARD_FILL), rect)
            pygame.draw.rect(surface, GRID_LINE, rect, width=1)


def action_name(action_idx: int) -> str:
    action = decode_action(action_idx)
    prefix = "H+" if action["use_hold"] else ""
    return f"{prefix}x{action['x']} y{action['y']} r{action['rotation']}"


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    policy = load_policy(args.algo, args.checkpoint, device=args.device)
    env = CCTetrisEnv(seed=args.seed, max_steps=args.max_steps, lib_path=args.lib)

    if policy.action_dim != int(env.action_space.n):
        env.close()
        raise SystemExit(
            f"Checkpoint action_dim={policy.action_dim} incompatible with RL action space={env.action_space.n}."
        )
    env_obs_dim = int(np.prod(env.observation_space.shape))
    if policy.obs_dim != env_obs_dim:
        env.close()
        raise SystemExit(
            f"Checkpoint obs_dim={policy.obs_dim} incompatible with RL observation size={env_obs_dim}."
        )

    obs, info = env.reset(seed=args.seed)
    action_mask = np.asarray(info["action_mask"], dtype=np.float32)

    pygame.init()
    pygame.display.set_caption("TetrisV2 RL Placement Playback")
    font = pygame.font.SysFont("Consolas", 18)
    small_font = pygame.font.SysFont("Consolas", 15)
    clock = pygame.time.Clock()

    cell = max(12, int(args.cell))
    board_x, board_y = 20, 20
    board_w, board_h = BOARD_COLS * cell, BOARD_ROWS * cell
    panel_x = board_x + board_w + 20
    panel_w = 520
    screen_w = panel_x + panel_w + 20
    screen_h = max(640, board_y + board_h + 20)
    screen = pygame.display.set_mode((screen_w, screen_h))

    seed = int(args.seed)
    step_count = 0
    total_reward = 0.0
    topouts = 0
    episodes = 1
    last_action = 0
    last_reward = 0.0
    paused = False
    episode_done = False
    episode_topout = False
    status = f"Loaded {args.algo.upper()} checkpoint: {args.checkpoint.name}"
    start_ticks = pygame.time.get_ticks()

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        status = "Paused." if paused else "Running."
                    elif event.key == pygame.K_r:
                        obs, info = env.reset(seed=seed)
                        action_mask = np.asarray(info["action_mask"], dtype=np.float32)
                        step_count = 0
                        total_reward = 0.0
                        last_reward = 0.0
                        episode_done = False
                        episode_topout = False
                        status = f"Reset seed={seed}"
                    elif event.key == pygame.K_n:
                        seed += 1
                        episodes += 1
                        obs, info = env.reset(seed=seed)
                        action_mask = np.asarray(info["action_mask"], dtype=np.float32)
                        step_count = 0
                        total_reward = 0.0
                        last_reward = 0.0
                        episode_done = False
                        episode_topout = False
                        status = f"Reset next seed={seed}"

            if not paused:
                if episode_done:
                    if args.auto_reset:
                        seed += 1
                        episodes += 1
                        obs, info = env.reset(seed=seed)
                        action_mask = np.asarray(info["action_mask"], dtype=np.float32)
                        step_count = 0
                        total_reward = 0.0
                        last_reward = 0.0
                        reason = "Topout" if episode_topout else "Time limit"
                        episode_done = False
                        episode_topout = False
                        status = f"{reason} -> auto-reset seed={seed}"
                    else:
                        paused = True
                        status = "Episode ended. Press R or N."
                else:
                    action_idx = int(
                        policy.act(
                            obs,
                            deterministic=args.deterministic,
                            temperature=args.temperature,
                            epsilon=args.epsilon,
                            action_mask=action_mask,
                        )
                    )
                    obs, reward, terminated, truncated, info = env.step(action_idx)
                    action_mask = np.asarray(info["action_mask"], dtype=np.float32)
                    step_count += 1
                    last_action = action_idx
                    last_reward = float(reward)
                    total_reward += last_reward
                    episode_done = bool(terminated or truncated)
                    episode_topout = bool(terminated and info.get("game_over", False))
                    if episode_topout:
                        topouts += 1
                    status = (
                        f"Step={step_count} action={action_name(last_action)} "
                        f"reward={last_reward:.2f} total={total_reward:.2f}"
                    )

                    if episode_done and not args.auto_reset:
                        paused = True
                        status = "Episode ended. Press R or N."

            runtime = env.runtime
            board_piece_ids = runtime.board_piece_ids(include_active=True)
            hold = runtime.hold_info()
            queue = runtime.queue()
            meta = runtime.meta()

            screen.fill(BG_COLOR)
            pygame.draw.rect(
                screen, PANEL_COLOR, (board_x - 4, board_y - 4, board_w + 8, board_h + 8), border_radius=6
            )
            draw_board(screen, board_x, board_y, cell, board_piece_ids)

            pygame.draw.rect(screen, PANEL_COLOR, (panel_x, board_y, panel_w, screen_h - 40), border_radius=8)
            elapsed_s = max(1e-6, (pygame.time.get_ticks() - start_ticks) / 1000.0)
            pps = step_count / elapsed_s
            lines = [
                f"Algo: {args.algo.upper()}",
                f"Checkpoint: {args.checkpoint.name}",
                f"Seed: {seed}",
                f"Episode: {episodes}",
                f"Step: {step_count}",
                f"PPS: {pps:.2f}",
                f"Last Action: {action_name(last_action)}",
                f"Last Reward: {last_reward:.2f}",
                f"Total Reward: {total_reward:.2f}",
                f"Lines: {meta['lines']}",
                f"Combo/B2B: {meta['combo']} / {meta['b2b']}",
                f"Topouts: {topouts}",
                f"GameOver/TopOut: {meta['game_over']} / {meta['top_out']}",
                f"Placement raw/valid: {int(info.get('placement_count_raw', 0))} / {int(info.get('legal_action_count', 0))}",
                f"Placement overflow: {bool(info.get('placement_overflow', False))}",
                f"Selected index/hold: {int(info.get('selected_placement_index', -1))} / {bool(info.get('selected_is_hold', False))}",
                f"Action success: {bool(info.get('success', False))}",
                f"Hold: {PIECE_NAMES.get(hold['hold_piece'], '?')} avail={hold['hold_available']}",
                f"Queue: {' '.join(PIECE_NAMES.get(p, '?') for p in queue[:8])}",
                f"Policy: {'deterministic' if args.deterministic else 'stochastic'}",
                f"Epsilon: {args.epsilon:.3f}",
                f"Temperature: {args.temperature:.2f}",
                f"AutoReset: {args.auto_reset}",
                f"Paused: {paused}",
            ]
            for i, text in enumerate(lines):
                screen.blit(small_font.render(text, True, LOCK_TEXT), (panel_x + 10, board_y + 10 + i * 22))

            controls = "Controls: Space pause | R reset | N next seed | Q/Esc quit"
            screen.blit(font.render(controls, True, LOCK_TEXT), (board_x, screen_h - 30))
            screen.blit(small_font.render(status, True, (180, 210, 255)), (board_x, board_y + board_h + 8))

            pygame.display.flip()
            clock.tick(max(5, int(args.fps)))
    finally:
        env.close()
        pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
