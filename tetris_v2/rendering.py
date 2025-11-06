"""Utilities for lightweight real-time rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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

    def draw(
        self,
        board_rgb: np.ndarray,
        hud_values: Dict[str, str | int | float],
        *,
        hold_image: Optional[np.ndarray] = None,
        queue_images: Optional[List[np.ndarray]] = None,
    ) -> None:
        pygame.event.pump()
        if board_rgb.shape[0] != self.board_height or board_rgb.shape[1] != self.board_width:
            board_rgb = self._scale_frame(board_rgb)
        board_surface = pygame.surfarray.make_surface(np.transpose(board_rgb, (1, 0, 2)))
        self.surface.blit(board_surface, (0, 0))
        self._draw_panel(hud_values)
        self._draw_hold_queue(hold_image, queue_images or [])
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

    def _draw_hold_queue(
        self,
        hold_image: Optional[np.ndarray],
        queue_images: List[np.ndarray],
    ) -> None:
        x_origin = self.board_width + 20
        if hold_image is not None:
            hold_surface = self._make_surface(hold_image)
            hold_label = self.hud_font.render("HOLD", True, (200, 200, 200))
            self.surface.blit(hold_label, (x_origin, self.board_height // 2 - 140))
            self.surface.blit(hold_surface, (x_origin, self.board_height // 2 - 110))
        if queue_images:
            qy = self.board_height // 2 + 10
            queue_label = self.hud_font.render("NEXT", True, (200, 200, 200))
            self.surface.blit(queue_label, (x_origin, qy))
            qy += 20
            for img in queue_images:
                q_surface = self._make_surface(img)
                self.surface.blit(q_surface, (x_origin, qy))
                qy += q_surface.get_height() + 10

    def _make_surface(self, image: np.ndarray) -> "pygame.Surface":
        scaled = self._scale_preview(image)
        return pygame.surfarray.make_surface(np.transpose(scaled, (1, 0, 2)))

    def _scale_preview(self, image: np.ndarray) -> np.ndarray:
        target = 64
        if image.shape[0] == target:
            return image
        scale = max(1, target // max(image.shape[0], 1))
        return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)

    def close(self) -> None:
        if pygame:
            pygame.display.quit()
            pygame.quit()
