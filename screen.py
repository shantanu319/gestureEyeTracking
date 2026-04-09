from __future__ import annotations

import cv2
import numpy as np


class Screen:
    def __init__(
        self,
        screen_size: tuple[int, int] = (1280, 720),
        window_name: str = "Calibration",
        fullscreen: bool = True,
    ):
        self.width = int(screen_size[0])
        self.height = int(screen_size[1])
        self.window_name = window_name
        self.fullscreen = fullscreen
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._initialized = False

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def _ensure_window(self) -> None:
        if self._initialized:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            try:
                cv2.setWindowProperty(
                    self.window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN,
                )
            except cv2.error:
                pass
        self._initialized = True

    def _draw_text(self, lines: list[str], start_y: int) -> None:
        for index, line in enumerate(lines):
            cv2.putText(
                self.canvas,
                line,
                (48, start_y + index * 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (220, 220, 220),
                2,
            )

    def render(
        self,
        target: tuple[int, int],
        progress: float = 0.0,
        title: str = "Calibration",
        subtitle: str = "Look at the target.",
    ) -> None:
        self.canvas = np.full((self.height, self.width, 3), 18, dtype=np.uint8)
        x, y = target
        radius = max(int(min(self.width, self.height) * 0.015), 10)

        cv2.circle(self.canvas, (x, y), radius, (255, 255, 255), -1)
        cv2.circle(self.canvas, (x, y), radius + 14, (80, 180, 80), 2)

        if progress > 0.0:
            cv2.ellipse(
                self.canvas,
                (x, y),
                (radius + 20, radius + 20),
                0,
                0,
                360 * min(max(progress, 0.0), 1.0),
                (0, 255, 0),
                4,
            )

        self._draw_text(
            [
                title,
                subtitle,
                "Press ESC or q to cancel calibration.",
            ],
            start_y=72,
        )

    def render_message(self, title: str, subtitle: str = "") -> None:
        self.canvas = np.full((self.height, self.width, 3), 18, dtype=np.uint8)
        lines = [title]
        if subtitle:
            lines.append(subtitle)

        text_sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0] for line in lines]
        total_height = sum(size[1] for size in text_sizes) + 40 * max(len(lines) - 1, 0)
        current_y = (self.height - total_height) // 2

        for line, size in zip(lines, text_sizes):
            x = (self.width - size[0]) // 2
            y = current_y + size[1]
            cv2.putText(
                self.canvas,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 255, 255),
                2,
            )
            current_y += size[1] + 40

    def show(self) -> None:
        self._ensure_window()
        cv2.imshow(self.window_name, self.canvas)

    def close(self) -> None:
        if self._initialized:
            cv2.destroyWindow(self.window_name)
            self._initialized = False
