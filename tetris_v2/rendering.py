"""Utilities for lightweight real-time rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import pygame
except Exception:  # pragma: no cover
    pygame = None


@dataclass
class HudLine:
    label: str
    value: str


class PygameBoardRenderer:
    """Creates a simple HUD similar to Gym classic-control viewers."""

    def __init__(
        self,
        *,
        title: str,
        board_shape: Tuple[int, int],
        scale: int = 2,
        panel_width: int = 220,
    ) -> None:
        if pygame is None:  # pragma: no cover - requires pygame
            raise RuntimeError(
                "pygame is required for human rendering. Install it with `pip install pygame`."
            )
        pygame.init()
        self.scale = scale
        self.board_width = board_shape[1] * scale
        self.board_height = board_shape[0] * scale
        self.panel_width = panel_width
        self.surface = pygame.display.set_mode((self.board_width + panel_width, self.board_height))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("Menlo", 18)
        self.hud_font = pygame.font.SysFont("Menlo", 16)

    def draw(self, board_rgb: np.ndarray, hud_values: Dict[str, str | int | float]) -> None:
        pygame.event.pump()
        if board_rgb.shape[0] != self.board_height or board_rgb.shape[1] != self.board_width:
            board_rgb = self._scale_frame(board_rgb)
        board_surface = pygame.surfarray.make_surface(np.transpose(board_rgb, (1, 0, 2)))
        self.surface.blit(board_surface, (0, 0))
        self._draw_panel(hud_values)
        pygame.display.flip()

    def _scale_frame(self, board_rgb: np.ndarray) -> np.ndarray:
        scale_y = self.board_height // max(board_rgb.shape[0], 1)
        scale_x = self.board_width // max(board_rgb.shape[1], 1)
        scale = max(1, min(scale_x, scale_y))
        return np.repeat(np.repeat(board_rgb, scale, axis=0), scale, axis=1)

    def _draw_panel(self, hud_values: Dict[str, str | int | float]) -> None:
        panel_rect = pygame.Rect(self.board_width, 0, self.panel_width, self.board_height)
        pygame.draw.rect(self.surface, (30, 30, 30), panel_rect)
        header = self.font.render("Tetris HUD", True, (220, 220, 220))
        self.surface.blit(header, (self.board_width + 20, 20))
        lines: Iterable[HudLine] = [
            HudLine(label=str(k), value=str(v)) for k, v in hud_values.items()
        ]
        y = 60
        for line in lines:
            label_surface = self.hud_font.render(f"{line.label}:", True, (200, 200, 200))
            value_surface = self.hud_font.render(str(line.value), True, (120, 200, 255))
            self.surface.blit(label_surface, (self.board_width + 20, y))
            self.surface.blit(value_surface, (self.board_width + 140, y))
            y += 22

    def close(self) -> None:
        if pygame:
            pygame.display.quit()
            pygame.quit()
