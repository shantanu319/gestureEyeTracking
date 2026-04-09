# Model Assets

These files are official MediaPipe task bundles used by the runtime:

- `face_landmarker.task`
- `hand_landmarker.task`

Downloaded from:

- <https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>
- <https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task>

They are intentionally vendored so `main.py` can run after a normal `pip install -r requirements.txt` without another manual model fetch.
