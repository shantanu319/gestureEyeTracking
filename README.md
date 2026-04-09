# gestureEyeTracking

This repo is now a single desktop prototype for hands-free control with no frontend. It uses:

- MediaPipe Face Landmarker to extract eye and head-pose features
- A 9-point on-screen calibration step to map those features to screen coordinates
- MediaPipe Hand Landmarker for click, drag, and scroll gestures
- `pyautogui` for optional OS mouse control

The old repo state was a broken flattening of unrelated eye-tracking and virtual-mouse projects. The missing Haar/Purkinje assets, deleted `model.py`, tracked bytecode, and auto-running scripts have been replaced with one coherent runtime.

## What Works

- Gaze-driven cursor motion after calibration
- Saved calibration model in `artifacts/calibration.json`
- Gesture actions:
  - `index finger`: left click
  - `index + middle`: right click
  - `fist`: click-and-drag
  - `pinch`: scroll
- A debug window plus fullscreen calibration overlay
- Headless-safe `--no-mouse` mode for debugging without moving the real cursor

## Repo Layout

- `main.py`: application entrypoint
- `Eye_Tracker.py`: MediaPipe face/iris feature extraction
- `Gaze_Tracker.py`: calibration loading, smoothing, gaze prediction
- `Gesture_Control.py`: gesture recognition and mouse actions
- `calibration.py`: regression model and calibration routine
- `screen.py`: fullscreen calibration overlay
- `mouse_controller.py`: guarded `pyautogui` wrapper
- `models/`: vendored official MediaPipe task bundles
- `datasets/checkerboard/`: small recovered camera-calibration sample set

## Setup

Python `3.11` is the tested target here.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On macOS you may also need to grant camera access and Accessibility permissions so `pyautogui` can move/click the mouse.

## Run

Debug without touching the real cursor:

```bash
python3 main.py --no-mouse --calibrate-on-start
```

Normal run:

```bash
python3 main.py --calibrate-on-start
```

Useful options:

```bash
python3 main.py --help
```

## Runtime Keys

- `c`: recalibrate
- `m`: toggle live mouse control
- `q` or `esc`: quit

## Notes

- Calibration is user-specific. Re-run it if lighting, camera angle, or seating position changes.
- The checkerboard assets under [`datasets/checkerboard`](./datasets/checkerboard) are not required by the main app; they are there as lightweight reference data for camera-calibration experiments.
- Large public gaze datasets were intentionally not vendored into the repo because they are too large to justify here. See [`datasets/README.md`](./datasets/README.md).
- The MediaPipe task bundles used by the runtime are checked into [`models`](./models) so first-run setup does not need any extra model download step.
