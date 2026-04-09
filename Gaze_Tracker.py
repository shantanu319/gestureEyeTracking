from __future__ import annotations

from pathlib import Path

import numpy as np

from Eye_Tracker import EyeTracker
from calibration import CalibrationModel


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
        self.smoothing = float(smoothing)
        self.minimum_eye_openness = float(minimum_eye_openness)
        self.last_observation = None
        self.last_gaze = None
        self._smoothed_gaze = None
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
        self._smoothed_gaze = None
        self.last_gaze = None

    def observe(self, frame):
        self.last_observation = self.eye_tracker.process(frame)
        return self.last_observation

    def estimate(self, observation=None):
        observation = observation or self.last_observation
        if observation is None or not self.is_calibrated:
            self.last_gaze = None
            return None
        if observation.average_eye_openness < self.minimum_eye_openness:
            self.last_gaze = None
            return None

        raw_prediction = np.asarray(
            self.calibration.predict(observation.feature_vector, self.screen_size),
            dtype=np.float64,
        )
        if self._smoothed_gaze is None:
            self._smoothed_gaze = raw_prediction
        else:
            self._smoothed_gaze = (
                self.smoothing * raw_prediction + (1.0 - self.smoothing) * self._smoothed_gaze
            )

        self.last_gaze = tuple(int(round(value)) for value in self._smoothed_gaze)
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
        return self.eye_tracker.decorate_frame(frame, self.last_observation)

    def close(self) -> None:
        self.eye_tracker.close()
