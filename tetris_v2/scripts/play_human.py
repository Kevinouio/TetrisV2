"""Real-time human play loop with DAS/ARR and pygame visuals."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

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
        "rotate_180": 8,
    },
}
ACTION_SCHEMES["versus"] = ACTION_SCHEMES["modern"]

if pygame:
    KEY_BINDINGS: Dict[str, Dict[str, List[int]]] = {
        "modern": {
            "move_left": [pygame.K_a],
            "move_right": [pygame.K_d],
            "soft_drop": [pygame.K_LSHIFT, pygame.K_RSHIFT],
            "hard_drop": [pygame.K_SPACE],
            "rotate_ccw": [pygame.K_LEFT],
            "rotate_cw": [pygame.K_RIGHT],
            "rotate_180": [pygame.K_UP],
            "hold": [pygame.K_c],
        },
        "nes": {
            "move_left": [pygame.K_LEFT, pygame.K_a],
            "move_right": [pygame.K_RIGHT, pygame.K_d],
            "soft_drop": [pygame.K_DOWN, pygame.K_s],
            "hard_drop": [pygame.K_SPACE],
            "rotate_cw": [pygame.K_UP, pygame.K_w],
        },
    }
else:
    KEY_BINDINGS = {}


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Tetris with real-time controls.")
    parser.add_argument("--env", choices=("nes", "modern", "versus"), default="nes")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=60, help="Target frames per second.")
    parser.add_argument("--das", type=int, default=20, help="Frames before auto shift activates.")
    parser.add_argument("--arr", type=int, default=4, help="Frames between repeated shifts.")
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
        scheme_key = "modern" if env_key in ("modern", "versus") else "nes"
        self.scheme = ACTION_SCHEMES[scheme_key]
        self.bindings = KEY_BINDINGS.get(scheme_key, {})
        self.left_repeat = KeyRepeat(das=das, arr=arr)
        self.right_repeat = KeyRepeat(das=das, arr=arr)
        self.pending: Deque[str] = deque()
        self.soft_drop_active = False
        self.left_pressed = False
        self.right_pressed = False
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
                elif self._binding_contains("hard_drop", event.key):
                    self.pending.append("hard_drop")
                elif self._binding_contains("hold", event.key) and "hold" in self.scheme:
                    self.pending.append("hold")
                elif self._binding_contains("rotate_cw", event.key):
                    self.pending.append("rotate_cw")
                elif self._binding_contains("rotate_ccw", event.key) and "rotate_ccw" in self.scheme:
                    self.pending.append("rotate_ccw")
                elif self._binding_contains("rotate_180", event.key) and "rotate_180" in self.scheme:
                    self.pending.append("rotate_180")
            elif event.type == pygame.KEYUP:
                if self._binding_contains("move_left", event.key):
                    self.left_repeat.tick(False)
                elif self._binding_contains("move_right", event.key):
                    self.right_repeat.tick(False)

        keys = pygame.key.get_pressed()
        self.soft_drop_active = any(keys[key] for key in self.bindings.get("soft_drop", ()))
        self.left_pressed = any(keys[key] for key in self.bindings.get("move_left", ()))
        self.right_pressed = any(keys[key] for key in self.bindings.get("move_right", ()))

    def next_actions(self) -> List[int]:
        actions: List[int] = []
        while self.pending:
            cmd = self.pending.popleft()
            action = self.scheme.get(cmd)
            if action is None:
                continue
            actions.append(action)
            if cmd == "hard_drop":
                actions.extend(self._post_hard_drop_actions())
                return actions

        lateral = None
        move_left = self.left_repeat.tick(self.left_pressed)
        move_right = self.right_repeat.tick(self.right_pressed)
        if move_left and not move_right and "left" in self.scheme:
            lateral = self.scheme["left"]
        elif move_right and not move_left and "right" in self.scheme:
            lateral = self.scheme["right"]
        if lateral is not None:
            actions.append(lateral)

        if self.soft_drop_active and "soft_drop" in self.scheme:
            actions.append(self.scheme["soft_drop"])

        if not actions:
            actions.append(self.scheme["none"])
        return actions

    def _post_hard_drop_actions(self) -> List[int]:
        return []

    def clear_reset(self) -> None:
        self.reset_requested = False

    def _binding_contains(self, action: str, key: int) -> bool:
        return key in self.bindings.get(action, ())


def main(argv=None) -> int:
    args = _parse_args(argv)
    if pygame is None:
        raise SystemExit("pygame is required for human play. Install it with `pip install pygame`.")
    if not pygame.get_init():
        pygame.init()
    register_envs()
    env_id = _resolve_env_id(args.env)
    env = gym.make(env_id, render_mode="human")
    controller = InputController(args.env, das=max(1, args.das), arr=max(1, args.arr))
    obs, _ = env.reset(seed=args.seed)
    clock = pygame.time.Clock()
    while controller.running:
        controller.process_events()
        terminated = truncated = False
        actions = controller.next_actions()
        for idx, action in enumerate(actions):
            env.unwrapped._skip_frame_advance = idx > 0
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.unwrapped._skip_frame_advance = False
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
