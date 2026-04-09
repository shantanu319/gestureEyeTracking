from __future__ import annotations

import argparse
import time

import cv2

from mouse_controller import MouseController
from screen import Screen


WINDOW_NAME = "Gesture Eye Tracking"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hands-free desktop control with gaze + gestures.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index to use.")
    parser.add_argument("--frame-width", type=int, default=960, help="Requested capture width.")
    parser.add_argument("--frame-height", type=int, default=540, help="Requested capture height.")
    parser.add_argument(
        "--calibration-file",
        default="artifacts/calibration.json",
        help="Path to the saved calibration model.",
    )
    parser.add_argument(
        "--no-mouse",
        action="store_true",
        help="Run in debug mode without moving or clicking the system mouse.",
    )
    parser.add_argument(
        "--calibrate-on-start",
        action="store_true",
        help="Launch the on-screen calibration flow before the main loop starts.",
    )
    parser.add_argument(
        "--windowed-calibration",
        action="store_true",
        help="Use a normal window instead of a fullscreen calibration overlay.",
    )
    return parser.parse_args()


def open_camera(index: int, frame_width: int, frame_height: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        raise RuntimeError(f"Unable to open camera index {index}.")

    return camera


def draw_status(
    frame,
    fps: float,
    gaze_tracker,
    gesture_frame,
    gaze_point,
    mouse_controller: MouseController,
):
    lines = [
        f"FPS: {fps:.1f}",
        "Calibration: loaded" if gaze_tracker.is_calibrated else "Calibration: required (press c)",
        f"Gaze: {gaze_point if gaze_point is not None else 'n/a'}",
        f"Gesture: {gesture_frame.label}",
        mouse_controller.status_message(),
        "Keys: c calibrate | m toggle mouse | q/esc quit",
    ]

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (16, 28 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 255, 40) if index < 4 else (255, 255, 255),
            2,
        )

    return frame


def main() -> None:
    args = parse_args()

    try:
        from Gaze_Tracker import GazeTracker
        from Gesture_Control import GestureController
        from calibration import run_calibration
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing runtime dependency: {exc.name}. Install requirements with "
            "'python3 -m pip install -r requirements.txt'."
        ) from exc

    mouse_controller = MouseController(enabled=not args.no_mouse)
    overlay = Screen(
        mouse_controller.screen_size,
        fullscreen=not args.windowed_calibration,
    )
    gaze_tracker = GazeTracker(
        screen_size=mouse_controller.screen_size,
        calibration_path=args.calibration_file,
    )
    gesture_controller = GestureController()

    camera = open_camera(args.camera_index, args.frame_width, args.frame_height)

    print("Press 'c' to calibrate, 'm' to toggle mouse control, 'q' or ESC to quit.")

    try:
        if args.calibrate_on_start:
            run_calibration(camera, gaze_tracker, overlay, debug_window_name=WINDOW_NAME)

        previous_frame_time = time.perf_counter()

        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera read failed.")

            frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_frame_time, 1e-6)
            previous_frame_time = now

            gaze_point = gaze_tracker.update(frame)
            gesture_frame = gesture_controller.process(frame)

            if gaze_point is not None:
                mouse_controller.move_to(*gaze_point)

            gesture_controller.apply_actions(gesture_frame, mouse_controller)

            debug_frame = gaze_tracker.decorate_frame(frame.copy())
            debug_frame = gesture_controller.decorate_frame(debug_frame, gesture_frame)
            debug_frame = draw_status(
                debug_frame,
                fps,
                gaze_tracker,
                gesture_frame,
                gaze_point,
                mouse_controller,
            )

            cv2.imshow(WINDOW_NAME, debug_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                mouse_controller.sync_drag(False)
                run_calibration(camera, gaze_tracker, overlay, debug_window_name=WINDOW_NAME)
            if key == ord("m"):
                mouse_controller.set_enabled(not mouse_controller.enabled)

    finally:
        mouse_controller.sync_drag(False)
        gesture_controller.close()
        gaze_tracker.close()
        overlay.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
