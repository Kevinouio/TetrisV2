#!/usr/bin/env python3
"""Manual controller for playing board presets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym

try:
    import pygame
except ImportError as exc:  # pragma: no cover - only triggered if pygame missing
    raise SystemExit("pygame is required for scripts/play_preset.py") from exc

from tetris_v2.envs.registration import register_envs
from tetris_v2.envs.state_presets import apply_board_preset, load_board_presets


DAS_DELAY_FRAMES = 8
ARR_FRAMES = 2

NES_CONTINUOUS_ACTIONS = {
    pygame.K_a: (1, True),
    pygame.K_d: (2, True),
    pygame.K_LSHIFT: (4, False),
    pygame.K_RSHIFT: (4, False),
}
NES_DISCRETE_ACTIONS = {
    pygame.K_LEFT: 3,
    pygame.K_RIGHT: 3,
    pygame.K_UP: 3,
    pygame.K_SPACE: 5,
}

MODERN_CONTINUOUS_ACTIONS = {
    pygame.K_a: (1, True),
    pygame.K_d: (2, True),
    pygame.K_LSHIFT: (5, False),
    pygame.K_RSHIFT: (5, False),
}
MODERN_DISCRETE_ACTIONS = {
    pygame.K_LEFT: 4,
    pygame.K_RIGHT: 3,
    pygame.K_UP: 8,
    pygame.K_SPACE: 6,
    pygame.K_c: 7,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a preset board configuration manually.")
    parser.add_argument(
        "--env",
        choices=("nes", "modern", "custom"),
        default="modern",
        help="Preset family to use.",
    )
    parser.add_argument("--env-id", help="Custom Gymnasium id when --env=custom.")
    parser.add_argument("--board-preset-file", type=Path, required=True, help="JSON file of preset boards.")
    parser.add_argument("--preset", required=False, help="Preset name to load when the session starts.")
    parser.add_argument("--seed", type=int, help="Optional RNG seed for env reset.")
    parser.add_argument("--list", action="store_true", help="List available presets and exit.")
    return parser.parse_args()


def _resolve_env_id(env_choice: str, env_id: Optional[str]) -> str:
    if env_choice == "nes":
        return "KevinNES/Tetris-v0"
    if env_choice == "modern":
        return "KevinModern/Tetris-v0"
    if not env_id:
        raise SystemExit("--env-id is required when --env=custom.")
    return env_id


def _action_maps(env_choice: str):
    if env_choice == "nes":
        return NES_CONTINUOUS_ACTIONS, NES_DISCRETE_ACTIONS
    return MODERN_CONTINUOUS_ACTIONS, MODERN_DISCRETE_ACTIONS


def _print_instructions(env_choice: str):
    print("Controls:")
    if env_choice == "nes":
        print("  A / D: Move left/right (DAS w/ delay)")
        print("  Shift: Soft drop")
        print("  Space: Hard drop")
        print("  Arrow Left/Right/Up: Rotate")
    else:
        print("  A / D: Move left/right (DAS w/ delay)")
        print("  Shift: Soft drop")
        print("  Space: Hard drop")
        print("  Arrow Left: Rotate left")
        print("  Arrow Right: Rotate right")
        print("  Arrow Up: Rotate 180°")
        print("  C: Hold")
    print("  R: Reset preset")
    print("  N: Load next preset")
    print("  Esc / Q: Quit")


def _select_action(
    keys: list[int],
    action_table: Dict[int, tuple[int, bool]],
    pressed,
    hold_frames: Dict[int, int],
) -> int:
    for key in keys:
        mapped, uses_das = action_table[key]
        if pressed[key]:
            frames = hold_frames.get(key, 0)
            fire = False
            if not uses_das:
                fire = True
            else:
                if frames == 0:
                    fire = True
                elif frames >= DAS_DELAY_FRAMES and (frames - DAS_DELAY_FRAMES) % ARR_FRAMES == 0:
                    fire = True
            hold_frames[key] = frames + 1
            if fire:
                return mapped
        else:
            hold_frames.pop(key, None)
    return 0


def main() -> int:
    args = _parse_args()
    env_id = _resolve_env_id(args.env, args.env_id)
    register_envs()
    library = load_board_presets(args.board_preset_file)
    if args.list or not args.preset:
        print("Available presets:")
        for name, preset in library.items():
            print(f"- {name} (env={preset.env})")
        if not args.preset:
            return 0
    if args.preset not in library:
        raise SystemExit(f"Preset '{args.preset}' not found in {args.board_preset_file}.")
    preset_names = list(library.keys())
    preset_index = preset_names.index(args.preset)
    env = gym.make(env_id, render_mode="human")
    continuous_actions, discrete_actions = _action_maps(args.env)
    hold_frames: Dict[int, int] = {}
    _print_instructions(args.env)
    move_keys = [key for key, (_, uses_das) in continuous_actions.items() if uses_das]
    misc_keys = [key for key, (_, uses_das) in continuous_actions.items() if not uses_das]
    clock = pygame.time.Clock()
    pending_discrete: Optional[int] = None

    def _load_preset(name: str):
        preset = library[name]
        if args.env != "custom" and preset.env.lower() != args.env:
            print(
                f"Warning: preset '{name}' targets env '{preset.env}', "
                f"but you selected '{args.env}'. Continuing anyway."
            )
        obs, _ = env.reset(seed=args.seed)
        obs = apply_board_preset(env, preset)
        env.render()
        print(f"Loaded preset '{name}'.")
        return obs

    obs = _load_preset(args.preset)
    score = 0.0
    running = True
    while running:
        action = 0
        reset_requested = False
        next_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                    break
                if event.key == pygame.K_r:
                    reset_requested = True
                    break
                if event.key == pygame.K_n:
                    next_requested = True
                    break
                discrete = discrete_actions.get(event.key)
                if discrete is not None:
                    pending_discrete = discrete
        if not running:
            break
        if reset_requested:
            obs = _load_preset(preset_names[preset_index])
            score = 0.0
            pending_discrete = None
            continue
        if next_requested:
            preset_index = (preset_index + 1) % len(preset_names)
            obs = _load_preset(preset_names[preset_index])
            score = 0.0
            pending_discrete = None
            continue

        pressed = pygame.key.get_pressed()
        if pending_discrete is not None:
            action = pending_discrete
            pending_discrete = None
        else:
            action = _select_action(move_keys, continuous_actions, pressed, hold_frames)
            if action == 0:
                action = _select_action(misc_keys, continuous_actions, pressed, hold_frames)

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        score = float(info.get("score", score))
        if reward:
            print(f"Reward: {reward:.2f} | Score: {score:.1f}")
        if terminated or truncated:
            print("Round ended. Press R to reload or N for next preset.")
        clock.tick(60)

    env.close()
    print("Goodbye!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
