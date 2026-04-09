from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class CalibrationError(RuntimeError):
    pass


@dataclass
class CalibrationSample:
    target: tuple[int, int]
    feature_vector: np.ndarray


class CalibrationModel:
    def __init__(self):
        self.coefficients: np.ndarray | None = None
        self.feature_count: int = 0

    @property
    def is_fitted(self) -> bool:
        return self.coefficients is not None

    def _design_matrix(self, features: np.ndarray) -> np.ndarray:
        features = np.atleast_2d(np.asarray(features, dtype=np.float64))
        intercept = np.ones((features.shape[0], 1), dtype=np.float64)
        return np.concatenate((intercept, features), axis=1)

    def _filter_samples(self, samples: list[CalibrationSample]) -> list[CalibrationSample]:
        grouped_samples: dict[tuple[int, int], list[CalibrationSample]] = defaultdict(list)
        for sample in samples:
            grouped_samples[sample.target].append(sample)

        filtered: list[CalibrationSample] = []
        for group in grouped_samples.values():
            features = np.asarray([sample.feature_vector for sample in group], dtype=np.float64)
            if len(group) < 4:
                filtered.extend(group)
                continue

            median = np.median(features, axis=0)
            mad = np.median(np.abs(features - median), axis=0)
            mad[mad < 1e-6] = 1e-6
            keep_mask = np.all(np.abs(features - median) <= 6.0 * mad, axis=1)
            kept = [sample for sample, keep in zip(group, keep_mask) if keep]
            filtered.extend(kept if kept else group)

        return filtered

    def fit(self, samples: list[CalibrationSample]) -> int:
        if len(samples) < 3:
            raise ValueError("At least three calibration samples are required.")

        filtered = self._filter_samples(samples)
        features = np.asarray([sample.feature_vector for sample in filtered], dtype=np.float64)
        targets = np.asarray([sample.target for sample in filtered], dtype=np.float64)

        design = self._design_matrix(features)
        regularization = 1e-4 * np.eye(design.shape[1], dtype=np.float64)
        regularization[0, 0] = 0.0
        self.coefficients = np.linalg.solve(design.T @ design + regularization, design.T @ targets)
        self.feature_count = features.shape[1]
        return len(filtered)

    def predict(self, feature_vector, screen_size: tuple[int, int]) -> tuple[int, int]:
        if not self.is_fitted:
            raise CalibrationError("Calibration model has not been fitted yet.")

        vector = np.asarray(feature_vector, dtype=np.float64)
        if vector.shape[0] != self.feature_count:
            raise CalibrationError(
                f"Expected {self.feature_count} gaze features, received {vector.shape[0]}."
            )

        prediction = self._design_matrix(vector) @ self.coefficients
        max_x = max(screen_size[0] - 1, 0)
        max_y = max(screen_size[1] - 1, 0)
        return (
            int(np.clip(round(prediction[0, 0]), 0, max_x)),
            int(np.clip(round(prediction[0, 1]), 0, max_y)),
        )

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise CalibrationError("Cannot save an unfitted calibration model.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_count": self.feature_count,
            "coefficients": self.coefficients.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationModel":
        payload = json.loads(Path(path).read_text())
        model = cls()
        model.feature_count = int(payload["feature_count"])
        model.coefficients = np.asarray(payload["coefficients"], dtype=np.float64)
        return model


def calibration_targets(screen_size: tuple[int, int], margin_ratio: float = 0.1) -> list[tuple[int, int]]:
    width, height = int(screen_size[0]), int(screen_size[1])
    min_x = int(width * margin_ratio)
    max_x = int(width * (1.0 - margin_ratio))
    mid_x = width // 2
    min_y = int(height * margin_ratio)
    max_y = int(height * (1.0 - margin_ratio))
    mid_y = height // 2

    return [
        (mid_x, mid_y),
        (min_x, min_y),
        (mid_x, min_y),
        (max_x, min_y),
        (min_x, mid_y),
        (max_x, mid_y),
        (min_x, max_y),
        (mid_x, max_y),
        (max_x, max_y),
    ]


def run_calibration(
    camera,
    gaze_tracker,
    screen,
    debug_window_name: str = "Gesture Eye Tracking",
    warmup_frames: int = 10,
    sample_count: int = 24,
) -> bool:
    targets = calibration_targets(screen.size)
    samples: list[CalibrationSample] = []
    total_targets = len(targets)

    for index, target in enumerate(targets, start=1):
        warmup = 0
        collected = 0

        while collected < sample_count:
            ok, frame = camera.read()
            if not ok:
                return False

            frame = cv2.flip(frame, 1)
            observation = gaze_tracker.observe(frame)
            ready = observation is not None and observation.average_eye_openness >= gaze_tracker.minimum_eye_openness

            if ready:
                warmup += 1
                if warmup > warmup_frames:
                    samples.append(
                        CalibrationSample(
                            target=target,
                            feature_vector=np.asarray(observation.feature_vector, dtype=np.float64),
                        )
                    )
                    collected += 1

            screen.render(
                target=target,
                progress=collected / float(sample_count),
                title=f"Calibration {index}/{total_targets}",
                subtitle="Look at the target and keep your head still.",
            )
            screen.show()

            debug_frame = gaze_tracker.eye_tracker.decorate_frame(frame.copy(), observation)
            status = "capturing" if ready else "face/eyes not ready"
            cv2.putText(
                debug_frame,
                f"Calibration {index}/{total_targets} | {collected}/{sample_count} | {status}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if ready else (0, 0, 255),
                2,
            )
            cv2.imshow(debug_window_name, debug_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                screen.close()
                return False

    kept = gaze_tracker.fit(samples)
    screen.render_message(
        title="Calibration complete",
        subtitle=f"Saved {kept} filtered samples to {gaze_tracker.calibration_path}.",
    )
    screen.show()
    cv2.waitKey(400)
    screen.close()
    return True


