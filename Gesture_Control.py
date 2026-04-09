from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from model import GestureFrame


THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

THUMB_IP = 3
INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18

INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17
HAND_LANDMARKER_MODEL = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


class GestureController:
    def __init__(
        self,
        dominant_hand: str = "Right",
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        activation_frames: int = 3,
        processing_width: int = 512,
        process_every_n_frames: int = 2,
    ):
        self.dominant_hand = dominant_hand.capitalize()
        self.activation_frames = max(1, int(activation_frames))
        self.scroll_threshold = 0.035
        self._candidate_label = "NONE"
        self._candidate_frames = 0
        self._stable_label = "NONE"
        self._scroll_anchor_y = None
        self.processing_width = max(256, int(processing_width))
        self.process_every_n_frames = max(1, int(process_every_n_frames))
        self._frame_index = 0
        self._cached_label = "NONE"
        self._cached_handedness = None
        self._cached_landmarks: list[tuple[int, int]] = []
        self._cached_drag_active = False

        if not HAND_LANDMARKER_MODEL.exists():
            raise FileNotFoundError(
                f"Missing model asset: {HAND_LANDMARKER_MODEL}. "
                "Download the vendored MediaPipe task models or restore the repo assets."
            )

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._hands = vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def _distance(self, hand_landmarks, first_index: int, second_index: int) -> float:
        first = hand_landmarks[first_index]
        second = hand_landmarks[second_index]
        return math.hypot(first.x - second.x, first.y - second.y)

    def _finger_states(self, hand_landmarks, handedness: str) -> dict[str, bool]:
        landmarks = hand_landmarks
        thumb_extended = (
            landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x
            if handedness == "Right"
            else landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x
        )

        return {
            "thumb": thumb_extended,
            "index": landmarks[INDEX_TIP].y < landmarks[INDEX_PIP].y < landmarks[INDEX_MCP].y,
            "middle": landmarks[MIDDLE_TIP].y < landmarks[MIDDLE_PIP].y < landmarks[MIDDLE_MCP].y,
            "ring": landmarks[RING_TIP].y < landmarks[RING_PIP].y < landmarks[RING_MCP].y,
            "pinky": landmarks[PINKY_TIP].y < landmarks[PINKY_PIP].y < landmarks[PINKY_MCP].y,
        }

    def _select_hand(self, results):
        if not results.hand_landmarks or not results.handedness:
            return None, None

        candidates = list(zip(results.hand_landmarks, results.handedness))
        for hand_landmarks, handedness in candidates:
            label = handedness[0].category_name
            if label == self.dominant_hand:
                return hand_landmarks, label

        first_landmarks, first_handedness = candidates[0]
        return first_landmarks, first_handedness[0].category_name

    def _classify(self, hand_landmarks, handedness: str) -> str:
        finger_states = self._finger_states(hand_landmarks, handedness)
        pinch_distance = self._distance(hand_landmarks, THUMB_TIP, INDEX_TIP)

        if pinch_distance < 0.045:
            return "PINCH"
        if not any(finger_states[name] for name in ("index", "middle", "ring", "pinky")):
            return "FIST"
        if finger_states["index"] and not any(finger_states[name] for name in ("middle", "ring", "pinky")):
            return "INDEX"
        if finger_states["index"] and finger_states["middle"] and not finger_states["ring"] and not finger_states["pinky"]:
            return "TWO_FINGER"
        if all(finger_states.values()):
            return "PALM"
        return "PALM"

    def _stabilize(self, label: str) -> tuple[str, bool]:
        changed = False

        if label == self._candidate_label:
            self._candidate_frames += 1
        else:
            self._candidate_label = label
            self._candidate_frames = 1

        release_frames = 2 if label == "NONE" else self.activation_frames
        if self._candidate_frames >= release_frames and label != self._stable_label:
            self._stable_label = label
            changed = True

        return self._stable_label, changed

    def _landmarks_to_pixels(self, hand_landmarks, width: int, height: int) -> list[tuple[int, int]]:
        pixels = []
        for landmark in hand_landmarks:
            pixels.append((int(landmark.x * width), int(landmark.y * height)))
        return pixels

    def _cached_frame(self) -> GestureFrame:
        return GestureFrame(
            label=self._cached_label,
            handedness=self._cached_handedness,
            landmarks=list(self._cached_landmarks),
            drag_active=self._cached_drag_active,
        )

    def _update_cache(self, gesture_frame: GestureFrame) -> None:
        self._cached_label = gesture_frame.label
        self._cached_handedness = gesture_frame.handedness
        self._cached_landmarks = list(gesture_frame.landmarks)
        self._cached_drag_active = gesture_frame.drag_active

    def process(self, frame) -> GestureFrame:
        self._frame_index += 1
        if self.process_every_n_frames > 1 and self._frame_index % self.process_every_n_frames != 1:
            return self._cached_frame()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        if width > self.processing_width:
            processing_height = max(1, int(round(height * self.processing_width / width)))
            rgb = cv2.resize(rgb, (self.processing_width, processing_height), interpolation=cv2.INTER_AREA)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        now_ms = int(time.monotonic() * 1000)
        self._timestamp_ms = max(self._timestamp_ms + 1, now_ms)
        results = self._hands.detect_for_video(image, self._timestamp_ms)
        hand_landmarks, handedness = self._select_hand(results)

        if hand_landmarks is None:
            label, changed = self._stabilize("NONE")
            if changed:
                self._scroll_anchor_y = None
            gesture_frame = GestureFrame(label=label)
            self._update_cache(gesture_frame)
            return gesture_frame

        raw_label = self._classify(hand_landmarks, handedness)
        label, changed = self._stabilize(raw_label)
        gesture_frame = GestureFrame(
            label=label,
            handedness=handedness,
            landmarks=self._landmarks_to_pixels(hand_landmarks, frame.shape[1], frame.shape[0]),
        )

        gesture_frame.drag_active = label == "FIST"

        if changed:
            if label == "INDEX":
                gesture_frame.left_click = True
            elif label == "TWO_FINGER":
                gesture_frame.right_click = True
            if label != "PINCH":
                self._scroll_anchor_y = None

        if label == "PINCH":
            current_y = hand_landmarks[INDEX_TIP].y
            if self._scroll_anchor_y is None:
                self._scroll_anchor_y = current_y
            delta = self._scroll_anchor_y - current_y
            if abs(delta) >= self.scroll_threshold:
                gesture_frame.scroll_delta = 120 if delta > 0 else -120
                self._scroll_anchor_y = current_y
        else:
            self._scroll_anchor_y = None

        self._update_cache(gesture_frame)
        return gesture_frame

    def apply_actions(self, gesture_frame: GestureFrame, mouse_controller) -> None:
        mouse_controller.sync_drag(gesture_frame.drag_active)
        if gesture_frame.left_click:
            mouse_controller.click()
        if gesture_frame.right_click:
            mouse_controller.right_click()
        if gesture_frame.scroll_delta:
            mouse_controller.scroll(gesture_frame.scroll_delta)

    def decorate_frame(self, frame, gesture_frame: GestureFrame):
        for point in gesture_frame.landmarks:
            cv2.circle(frame, point, 3, (255, 160, 0), -1)
        return frame

    def close(self) -> None:
        self._hands.close()
