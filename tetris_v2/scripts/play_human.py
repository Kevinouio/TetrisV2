"""Real-time human play loop with DAS/ARR and pygame visuals."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import gymnasium as gym

try:  # pragma: no cover - pygame needed only when running script
    import pygame
except Exception:  # pragma: no cover
    pygame = None

from tetris_v2.envs.registration import register_envs


ACTION_SCHEMES: Dict[str, Dict[str, int]] = {
    "nes": {
        "none": 0,
        "left": 1,
        "right": 2,
        "rotate_cw": 3,
        "soft_drop": 4,
        "hard_drop": 5,
    },
    "modern": {
        "none": 0,
        "left": 1,
        "right": 2,
        "rotate_cw": 3,
        "rotate_ccw": 4,
        "soft_drop": 5,
        "hard_drop": 6,
        "hold": 7,
    },
}
ACTION_SCHEMES["versus"] = ACTION_SCHEMES["modern"]


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Tetris with real-time controls.")
    parser.add_argument("--env", choices=("nes", "modern", "versus"), default="nes")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=60, help="Target frames per second.")
    parser.add_argument("--das", type=int, default=10, help="Frames before auto shift activates.")
    parser.add_argument("--arr", type=int, default=2, help="Frames between repeated shifts.")
    parser.add_argument("--quit-key", default="escape", help="Key used to exit (default: escape).")
    return parser.parse_args(argv)


def _resolve_env_id(env_key: str) -> str:
    if env_key == "nes":
        return "KevinNES/Tetris-v0"
    if env_key == "modern":
        return "KevinModern/Tetris-v0"
    return "KevinVersus/Tetris-v0"


@dataclass
class KeyRepeat:
    das: int
    arr: int
    pressed: bool = False
    frames_held: int = 0
    arr_counter: int = 0

    def tick(self, pressed: bool) -> bool:
        if not pressed:
            self.pressed = False
            self.frames_held = 0
            self.arr_counter = 0
            return False
        if not self.pressed:
            self.pressed = True
            self.frames_held = 0
            self.arr_counter = 0
            return True
        self.frames_held += 1
        if self.frames_held < self.das:
            return False
        self.arr_counter += 1
        if self.arr_counter >= max(1, self.arr):
            self.arr_counter = 0
            return True
        return False


class InputController:
    """Maps pygame key events to Gym discrete actions with DAS/ARR."""

    def __init__(self, env_key: str, *, das: int, arr: int) -> None:
        self.env_key = env_key
        self.scheme = ACTION_SCHEMES["modern" if env_key in ("modern", "versus") else "nes"]
        self.left_repeat = KeyRepeat(das=das, arr=arr)
        self.right_repeat = KeyRepeat(das=das, arr=arr)
        self.pending: Deque[str] = deque()
        self.soft_drop_active = False
        self.running = True
        self.reset_requested = False

    def process_events(self) -> None:
        for event in pygame.event.get():  # pragma: no cover - integration path
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif event.key == pygame.K_r:
                    self.reset_requested = True
                elif event.key == pygame.K_SPACE:
                    self.pending.append("hard_drop")
                elif event.key in (pygame.K_c,):
                    if "hold" in self.scheme:
                        self.pending.append("hold")
                elif event.key in (pygame.K_UP, pygame.K_w, pygame.K_x):
                    self.pending.append("rotate_cw")
                elif event.key in (pygame.K_z, pygame.K_LCTRL, pygame.K_RCTRL):
                    if "rotate_ccw" in self.scheme:
                        self.pending.append("rotate_ccw")
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    self.left_repeat.tick(False)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    self.right_repeat.tick(False)

        keys = pygame.key.get_pressed()
        self.soft_drop_active = keys[pygame.K_DOWN] or keys[pygame.K_s]
        self.left_pressed = keys[pygame.K_LEFT] or keys[pygame.K_a]
        self.right_pressed = keys[pygame.K_RIGHT] or keys[pygame.K_d]

    def next_action(self) -> int:
        while self.pending:
            cmd = self.pending.popleft()
            action = self.scheme.get(cmd)
            if action is not None:
                return action
        if self.soft_drop_active and "soft_drop" in self.scheme:
            return self.scheme["soft_drop"]
        move_left = self.left_repeat.tick(self.left_pressed)
        move_right = self.right_repeat.tick(self.right_pressed)
        if move_left and not move_right and "left" in self.scheme:
            return self.scheme["left"]
        if move_right and not move_left and "right" in self.scheme:
            return self.scheme["right"]
        return self.scheme["none"]

    def clear_reset(self) -> None:
        self.reset_requested = False


def main(argv=None) -> int:
    args = _parse_args(argv)
    if pygame is None:
        raise SystemExit("pygame is required for human play. Install it with `pip install pygame`.")
    register_envs()
    env_id = _resolve_env_id(args.env)
    env = gym.make(env_id, render_mode="human")
    controller = InputController(args.env, das=max(1, args.das), arr=max(1, args.arr))
    obs, _ = env.reset(seed=args.seed)
    clock = pygame.time.Clock()
    while controller.running:
        controller.process_events()
        action = controller.next_action()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if controller.reset_requested or terminated or truncated:
            controller.clear_reset()
            obs, _ = env.reset()
        clock.tick(args.fps)
    env.close()
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
