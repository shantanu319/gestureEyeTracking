from __future__ import annotations

import math

import numpy as np


class AdaptivePointerFilter:
    def __init__(
        self,
        screen_size: tuple[int, int],
        min_alpha: float = 0.18,
        max_alpha: float = 0.82,
        deadzone_ratio: float = 0.002,
        motion_scale_ratio: float = 0.035,
        snap_ratio: float = 0.12,
        hold_frames: int = 4,
    ):
        diagonal = math.hypot(screen_size[0], screen_size[1])
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.deadzone_px = max(2.0, diagonal * deadzone_ratio)
        self.motion_scale_px = max(self.deadzone_px * 2.0, diagonal * motion_scale_ratio)
        self.snap_distance_px = max(self.motion_scale_px, diagonal * snap_ratio)
        self.hold_frames = max(0, int(hold_frames))

        self._value: np.ndarray | None = None
        self._missing_frames = 0

    def reset(self) -> None:
        self._value = None
        self._missing_frames = 0

    def update(self, point: tuple[int, int] | np.ndarray | None):
        if point is None:
            self._missing_frames += 1
            if self._value is not None and self._missing_frames <= self.hold_frames:
                return self.value
            self._value = None
            return None

        raw = np.asarray(point, dtype=np.float64)
        self._missing_frames = 0

        if self._value is None:
            self._value = raw
            return self.value

        delta = raw - self._value
        distance = float(np.linalg.norm(delta))

        if distance <= self.deadzone_px:
            return self.value

        if distance >= self.snap_distance_px:
            alpha = self.max_alpha
        else:
            distance_ratio = min(distance / self.motion_scale_px, 1.0)
            alpha = self.min_alpha + distance_ratio * (self.max_alpha - self.min_alpha)

        self._value = self._value + alpha * delta

        if np.linalg.norm(raw - self._value) <= self.deadzone_px * 0.5:
            self._value = raw

        return self.value

    @property
    def value(self):
        if self._value is None:
            return None
        return tuple(int(round(component)) for component in self._value)

