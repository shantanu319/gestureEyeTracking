from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from Eye_Tracker import EyeTracker
from calibration import CalibrationModel
from pointer_filter import AdaptivePointerFilter


class GazeTracker:
    def __init__(
        self,
        screen_size: tuple[int, int],
        calibration_path: str = "artifacts/calibration.json",
        smoothing: float = 0.35,
        minimum_eye_openness: float = 0.16,
    ):
        self.screen_size = (int(screen_size[0]), int(screen_size[1]))
        self.calibration_path = Path(calibration_path)
        self.eye_tracker = EyeTracker()
        self.calibration = CalibrationModel()
        self.minimum_eye_openness = float(minimum_eye_openness)
        self.last_observation = None
        self.last_gaze = None
        self.raw_gaze = None
        self.pointer_filter = AdaptivePointerFilter(
            self.screen_size,
            min_alpha=min(0.5, max(0.08, smoothing * 0.6)),
            max_alpha=min(0.92, max(0.45, smoothing + 0.45)),
        )
        self.load_calibration()

    @property
    def is_calibrated(self) -> bool:
        return self.calibration.is_fitted

    def load_calibration(self) -> bool:
        if not self.calibration_path.exists():
            return False
        self.calibration = CalibrationModel.load(self.calibration_path)
        return True

    def save_calibration(self) -> None:
        if self.calibration.is_fitted:
            self.calibration.save(self.calibration_path)

    def reset_smoothing(self) -> None:
        self.pointer_filter.reset()
        self.last_gaze = None
        self.raw_gaze = None

    def observe(self, frame):
        self.last_observation = self.eye_tracker.process(frame)
        return self.last_observation

    def estimate(self, observation=None):
        observation = observation or self.last_observation
        if not self.is_calibrated:
            self.last_gaze = None
            self.raw_gaze = None
            self.pointer_filter.reset()
            return None
        if observation is None:
            self.raw_gaze = None
            self.last_gaze = self.pointer_filter.update(None)
            return self.last_gaze
        if observation.average_eye_openness < self.minimum_eye_openness:
            self.raw_gaze = None
            self.last_gaze = self.pointer_filter.update(None)
            return self.last_gaze

        self.raw_gaze = self.calibration.predict(observation.feature_vector, self.screen_size)
        self.last_gaze = self.pointer_filter.update(self.raw_gaze)
        return self.last_gaze

    def update(self, frame):
        observation = self.observe(frame)
        return self.estimate(observation)

    def fit(self, samples) -> int:
        kept = self.calibration.fit(samples)
        self.save_calibration()
        self.reset_smoothing()
        return kept

    def decorate_frame(self, frame):
        frame = self.eye_tracker.decorate_frame(frame, self.last_observation)
        if self.raw_gaze is not None:
            cv2.circle(frame, self.raw_gaze, 5, (0, 165, 255), 1)
        return frame

    def close(self) -> None:
        self.eye_tracker.close()
