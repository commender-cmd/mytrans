"""Camera -> MediaPipe VIDEO -> raw left/right normalized 21-point hands.

Capture settings and extraction follow ``MediaPipe/hand capture media.py``.
No mirroring, coordinate transforms, temporal windows, or recording is applied.
Provide the original ``hand_landmarker.task`` via ``model_asset_path`` when it
is not in the working directory. Importing this module does not open a camera,
import OpenCV/MediaPipe, or download a model.

Demo (Ctrl+C stops capture)::

    python -m retargeting.inputs.mediapipe --model-asset-path PATH --frames 300
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np


MODEL_PATH = "hand_landmarker.task"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAM_FPS = 30
MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5


def parse_result(
    result, timestamp_ms: int, metadata: dict | None = None,
    *, invalid_hand_as_missing: bool = False,
) -> dict:
    """Extract a single detection result; malformed detections raise ValueError.

``timestamp`` is the exact Unix millisecond integer passed to VIDEO detection.
Missing hands are None. Present hands retain the original normalized x/y/z
values and landmark order, with shape (21, 3) and dtype float32.
An opt-in realtime policy drops malformed landmarks for a known side, records
the error in metadata, and preserves the other side. Repeated known labels
invalidate that side instead of choosing one detection. Unknown labels still
raise; the default remains strict.
"""
    hands = {"left": None, "right": None}
    seen_sides = set()
    invalid_reasons = {}
    try:
        landmarks = result.hand_landmarks
        handedness = result.handedness
        if len(landmarks) != len(handedness) or len(landmarks) > 2:
            raise ValueError("Expected matching landmarks/handedness for at most two hands")
        for index, hand_landmarks in enumerate(landmarks):
            # Match the reference's first-category handedness lookup exactly.
            label = handedness[index][0].category_name
            if label not in ("Left", "Right"):
                raise ValueError(f"Unknown handedness: {label!r}")
            side = label.lower()
            if side in seen_sides:
                message = f"Duplicate handedness: {label}; refusing to overwrite a hand"
                if not invalid_hand_as_missing:
                    raise ValueError(message)
                hands[side] = None
                invalid_reasons[side] = message
                continue
            seen_sides.add(side)
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    points = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32
                    )
                if points.shape != (21, 3):
                    raise ValueError(f"{side} landmarks must have shape (21, 3), got {points.shape}")
                if not np.isfinite(points).all():
                    raise ValueError(f"{side} landmarks must be finite float32 values")
                hands[side] = points
            except (AttributeError, TypeError, ValueError, OverflowError) as error:
                if not invalid_hand_as_missing:
                    raise
                invalid_reasons[side] = str(error)
    except (AttributeError, IndexError, TypeError, OverflowError) as error:
        raise ValueError("Malformed MediaPipe landmarks or handedness") from error
    return {
        **hands,
        "timestamp": timestamp_ms,
        "metadata": {
            **({} if metadata is None else metadata),
            "source_landmark_space": "mediapipe_normalized",
            "timestamp_unit": "unix_ms",
            **({"invalid_reasons": invalid_reasons} if invalid_reasons else {}),
        },
    }


class MediaPipeCameraInput:
    """Read one raw hand frame at a time; use a context manager or release()."""

    def __init__(
        self,
        model_asset_path: str | Path = MODEL_PATH,
        camera_index: int = 0,
        width: int = IMAGE_WIDTH,
        height: int = IMAGE_HEIGHT,
        fps: int = CAM_FPS,
        *,
        invalid_hand_as_missing: bool = False,
    ):
        model_path = Path(model_asset_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"MediaPipe hand model not found: {model_path}")

        import cv2
        import mediapipe as mp

        self._cv2 = cv2
        self._mp = mp
        self._cap = None
        self._landmarker = None
        self._camera_index = camera_index
        self._frame_index = 0
        self._last_timestamp_ms = None
        self._invalid_hand_as_missing = invalid_hand_as_missing
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=MIN_DET_CONF,
            min_hand_presence_confidence=MIN_DET_CONF,
            min_tracking_confidence=MIN_TRK_CONF,
        )
        try:
            self._cap = cv2.VideoCapture(camera_index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS, fps)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open camera {camera_index}")
            self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        except BaseException:
            self.release()
            raise

    def next_frame(self) -> dict:
        """Return a frame even with no hands; capture/detection errors raise.

On failure both resources are released. VIDEO timestamps use the reference's
wall-clock milliseconds, advanced by 1 ms only if the clock repeats or moves
backward, to satisfy the landmarker's increasing-timestamp requirement.
"""
        if self._cap is None or self._landmarker is None:
            raise RuntimeError("MediaPipe camera input is closed")
        try:
            success, frame = self._cap.read()
            if not success or frame is None or frame.size == 0:
                raise RuntimeError(f"Frame read failed for camera {self._camera_index}")
            frame_rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(time.time() * 1000)
            if self._last_timestamp_ms is not None:
                timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
            output = parse_result(result, timestamp_ms, {
                "camera_index": self._camera_index,
                "frame_index": self._frame_index,
                "image_width": int(frame.shape[1]),
                "image_height": int(frame.shape[0]),
            }, invalid_hand_as_missing=self._invalid_hand_as_missing)
            self._last_timestamp_ms = timestamp_ms
            self._frame_index += 1
            return output
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        """Release both resources, including partial initialization; idempotent."""
        cap, landmarker = self._cap, self._landmarker
        self._cap = self._landmarker = None
        try:
            if cap is not None:
                cap.release()
        finally:
            if landmarker is not None:
                landmarker.close()

    def __enter__(self) -> "MediaPipeCameraInput":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-asset-path", default=MODEL_PATH)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=300, help="positive number of frames to capture")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    counts = {"left": 0, "right": 0, "no_hands": 0, "both_hands": 0}
    completed = 0
    started = None
    try:
        with MediaPipeCameraInput(args.model_asset_path, args.camera_index) as camera:
            started = time.perf_counter()
            for _ in range(args.frames):
                frame = camera.next_frame()
                completed += 1
                descriptions = []
                for side in ("left", "right"):
                    points = frame[side]
                    if points is None:
                        descriptions.append(f"{side}=None")
                    else:
                        counts[side] += 1
                        descriptions.append(
                            f"{side}: shape={points.shape} dtype={points.dtype} "
                            f"finite={bool(np.isfinite(points).all())}"
                        )
                counts["no_hands"] += int(frame["left"] is None and frame["right"] is None)
                counts["both_hands"] += int(frame["left"] is not None and frame["right"] is not None)
                fps = completed / max(time.perf_counter() - started, 1e-9)
                print(f"frame={completed} timestamp={frame['timestamp']} unix_ms "
                      f"{' | '.join(descriptions)} FPS={fps:.1f}", flush=True)
    except KeyboardInterrupt:
        print("Capture stopped.")
    except Exception as error:
        print(f"Capture failed: {error}", flush=True)
        return 1
    elapsed = 0.0 if started is None else time.perf_counter() - started
    print(f"Captured={completed} {counts} FPS={completed / max(elapsed, 1e-9):.1f}; resources released")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
