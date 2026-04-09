from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from model import EyeState, GazeObservation


LEFT_EYE_LANDMARKS = [33, 133, 159, 145, 160, 144, 158, 153, 173, 157, 163, 154, 155]
RIGHT_EYE_LANDMARKS = [362, 263, 386, 374, 387, 373, 385, 380, 398, 384, 381, 382, 390]
LEFT_IRIS_LANDMARKS = [468, 469, 470, 471, 472]
RIGHT_IRIS_LANDMARKS = [473, 474, 475, 476, 477]
NOSE_TIP_INDEX = 1
LEFT_FACE_INDEX = 234
RIGHT_FACE_INDEX = 454
TOP_FACE_INDEX = 10
BOTTOM_FACE_INDEX = 152
FACE_LANDMARKER_MODEL = Path(__file__).resolve().parent / "models" / "face_landmarker.task"


class EyeTracker:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if not FACE_LANDMARKER_MODEL.exists():
            raise FileNotFoundError(
                f"Missing model asset: {FACE_LANDMARKER_MODEL}. "
                "Download the vendored MediaPipe task models or restore the repo assets."
            )

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._mesh = vision.FaceLandmarker.create_from_options(options)
        self.last_observation: GazeObservation | None = None
        self._timestamp_ms = 0

    def _landmark_to_pixel(self, landmark, width: int, height: int) -> tuple[int, int]:
        x = int(min(max(landmark.x * width, 0), width - 1))
        y = int(min(max(landmark.y * height, 0), height - 1))
        return (x, y)

    def _build_eye_state(
        self,
        landmarks,
        width: int,
        height: int,
        eye_indices: list[int],
        iris_indices: list[int],
    ) -> EyeState:
        eye_points = [self._landmark_to_pixel(landmarks[index], width, height) for index in eye_indices]
        iris_points = np.asarray(
            [self._landmark_to_pixel(landmarks[index], width, height) for index in iris_indices],
            dtype=np.float64,
        )
        iris_center = iris_points.mean(axis=0)

        xs = [point[0] for point in eye_points]
        ys = [point[1] for point in eye_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        box_width = max(max_x - min_x, 1)
        box_height = max(max_y - min_y, 1)

        left_corner = np.asarray(eye_points[0], dtype=np.float64)
        right_corner = np.asarray(eye_points[1], dtype=np.float64)
        top_lid = np.asarray(eye_points[2], dtype=np.float64)
        bottom_lid = np.asarray(eye_points[3], dtype=np.float64)
        openness = float(np.linalg.norm(top_lid - bottom_lid) / max(np.linalg.norm(left_corner - right_corner), 1.0))

        return EyeState(
            center=(int(round(iris_center[0])), int(round(iris_center[1]))),
            bbox=(min_x, min_y, box_width, box_height),
            openness=openness,
            ratio=(
                float((iris_center[0] - min_x) / box_width),
                float((iris_center[1] - min_y) / box_height),
            ),
        )

    def process(self, frame) -> GazeObservation | None:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        now_ms = int(time.monotonic() * 1000)
        self._timestamp_ms = max(self._timestamp_ms + 1, now_ms)
        result = self._mesh.detect_for_video(image, self._timestamp_ms)

        if not result.face_landmarks:
            self.last_observation = None
            return None

        landmarks = result.face_landmarks[0]
        left_eye = self._build_eye_state(landmarks, width, height, LEFT_EYE_LANDMARKS, LEFT_IRIS_LANDMARKS)
        right_eye = self._build_eye_state(landmarks, width, height, RIGHT_EYE_LANDMARKS, RIGHT_IRIS_LANDMARKS)

        nose_tip = self._landmark_to_pixel(landmarks[NOSE_TIP_INDEX], width, height)
        left_face = self._landmark_to_pixel(landmarks[LEFT_FACE_INDEX], width, height)
        right_face = self._landmark_to_pixel(landmarks[RIGHT_FACE_INDEX], width, height)
        top_face = self._landmark_to_pixel(landmarks[TOP_FACE_INDEX], width, height)
        bottom_face = self._landmark_to_pixel(landmarks[BOTTOM_FACE_INDEX], width, height)

        face_width = max(abs(right_face[0] - left_face[0]), 1)
        face_height = max(abs(bottom_face[1] - top_face[1]), 1)
        face_center = (
            int(round((left_face[0] + right_face[0]) / 2)),
            int(round((top_face[1] + bottom_face[1]) / 2)),
        )
        head_offset_x = float((nose_tip[0] - face_center[0]) / face_width)
        head_offset_y = float((nose_tip[1] - face_center[1]) / face_height)

        average_eye_openness = float((left_eye.openness + right_eye.openness) / 2.0)
        average_eye_ratio_x = float((left_eye.ratio[0] + right_eye.ratio[0]) / 2.0)
        average_eye_ratio_y = float((left_eye.ratio[1] + right_eye.ratio[1]) / 2.0)

        feature_vector = np.asarray(
            [
                average_eye_ratio_x,
                average_eye_ratio_y,
                left_eye.ratio[0],
                left_eye.ratio[1],
                right_eye.ratio[0],
                right_eye.ratio[1],
                head_offset_x,
                head_offset_y,
                face_width / float(width),
                face_height / float(height),
                left_eye.openness,
                right_eye.openness,
            ],
            dtype=np.float64,
        )

        self.last_observation = GazeObservation(
            feature_vector=feature_vector,
            left_eye=left_eye,
            right_eye=right_eye,
            nose_tip=nose_tip,
            face_center=face_center,
            face_size=(float(face_width), float(face_height)),
            average_eye_openness=average_eye_openness,
        )
        return self.last_observation

    def update(self, frame) -> GazeObservation | None:
        return self.process(frame)

    def decorate_frame(self, frame, observation: GazeObservation | None = None):
        observation = observation or self.last_observation
        if observation is None:
            return frame

        for eye in (observation.left_eye, observation.right_eye):
            x, y, w, h = eye.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            cv2.circle(frame, eye.center, 3, (0, 255, 0), -1)

        cv2.circle(frame, observation.nose_tip, 4, (0, 0, 255), -1)
        cv2.circle(frame, observation.face_center, 4, (255, 0, 255), -1)
        cv2.line(frame, observation.face_center, observation.nose_tip, (255, 0, 255), 2)
        return frame

    def close(self) -> None:
        self._mesh.close()

