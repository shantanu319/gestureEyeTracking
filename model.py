from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EyeState:
    center: tuple[int, int]
    bbox: tuple[int, int, int, int]
    openness: float
    ratio: tuple[float, float]


@dataclass
class GazeObservation:
    feature_vector: np.ndarray
    left_eye: EyeState
    right_eye: EyeState
    nose_tip: tuple[int, int]
    face_center: tuple[int, int]
    face_size: tuple[float, float]
    average_eye_openness: float


@dataclass
class GestureFrame:
    label: str = "NONE"
    handedness: Optional[str] = None
    landmarks: list[tuple[int, int]] = field(default_factory=list)
    left_click: bool = False
    right_click: bool = False
    drag_active: bool = False
    scroll_delta: int = 0

