"""MediaPipe raw input -> per-frame palm-local alignment -> three-frame windows.

Identity tracking is disabled: MediaPipe handedness is used directly. Each
invalid side resets immediately and independently; no historical frame or basis
is reused. The demo loads palm_local_v2 and prints 18D outputs without sending
commands or recording data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from retargeting.coordinates import (
    PALM_LOCAL_COORDINATE_ALIGNMENT,
    PALM_LOCAL_METADATA,
    align_palm_local_coordinates,
    build_l21_reference_basis,
)
from retargeting.inputs.mediapipe import MediaPipeCameraInput
from retargeting.tracking import (
    DEFAULT_MAX_CENTER_DISPLACEMENT,
    DEFAULT_MAX_SHAPE_RMSE,
    MEDIAPIPE_APPROX_SOURCE,
    HandWindowBuffer,
    ensure_hand25,
)


PALM_LOCAL_V2_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "checkpoint/models/twohand_h5/linker/palm_local_v2/model_best.pth"
)
SIDES = ("left", "right")


class MediaPipePalmLocalProcessor:
    """Camera-independent processing of the raw input's single-frame mapping."""

    def __init__(self):
        # Fatal robot-reference errors propagate before any camera is opened.
        # Each zero-pose robot basis is computed exactly once for this stream.
        self._robot_bases = {side: build_l21_reference_basis(side) for side in SIDES}
        self._buffer = HandWindowBuffer(receptive_field=3)
        self.current_hands = {side: None for side in SIDES}
        self.valid_streak = {side: 0 for side in SIDES}
        self.invalid_reasons = {side: None for side in SIDES}

    def process_frame(self, frame: dict) -> dict | None:
        """Align each raw21 hand before appending to its own buffer.

        current_hands exposes this frame's aligned (25,3) values for smoke and
        equivalence checks. It never holds an old frame on invalid input.
        """
        current = {side: None for side in SIDES}
        for side in SIDES:
            raw = frame[side]
            reason = frame.get("metadata", {}).get("invalid_reasons", {}).get(side, "missing") if raw is None else None
            if raw is not None:
                try:
                    with np.errstate(over="ignore", invalid="ignore"):
                        points = np.asarray(raw, dtype=np.float32)
                        if points.shape != (21, 3):
                            raise ValueError(f"raw shape must be (21,3), got {points.shape}")
                        if not np.isfinite(points).all():
                            raise ValueError("nonfinite raw landmarks")
                        points25 = ensure_hand25(points)
                        aligned = align_palm_local_coordinates(points25, self._robot_bases[side])
                    if aligned.shape != (25, 3) or not np.isfinite(aligned).all() or not np.any(aligned):
                        reason = "invalid or degenerate palm-local frame"
                    else:
                        current[side] = aligned
                except (TypeError, ValueError, OverflowError) as error:
                    reason = str(error)
            if current[side] is None:
                # update_canonical(None) alone does not clear old history.
                self._buffer.reset_side(side)
                self.valid_streak[side] = 0
            else:
                self.valid_streak[side] = min(self.valid_streak[side] + 1, 3)
            self.invalid_reasons[side] = reason

        self.current_hands = current
        metadata = {
            **frame.get("metadata", {}),
            **PALM_LOCAL_METADATA,
            "identity_tracking": False,
            "hand_valid": {side: current[side] is not None for side in SIDES},
            "invalid_reasons": dict(self.invalid_reasons),
        }
        # Already aligned: never use update(), which reconverts coordinates.
        return self._buffer.update_canonical(
            left_hand=current["left"], right_hand=current["right"],
            timestamp=frame.get("timestamp"), source=MEDIAPIPE_APPROX_SOURCE,
            metadata=metadata,
        )

    def reset(self) -> None:
        self._buffer.reset()
        self.current_hands = {side: None for side in SIDES}
        self.valid_streak = {side: 0 for side in SIDES}
        self.invalid_reasons = {side: None for side in SIDES}


class MediaPipeCameraAdapter:
    """Compose raw camera capture with palm-local realtime processing.

    next_input() retains the optional-window payload and relative seconds
    timestamp. Capture errors propagate after clearing history.
    The old identity thresholds are accepted for caller compatibility and have
    no effect. Scale and confidence must retain the raw-input defaults.
    """

    def __init__(
        self,
        model_asset_path: str | Path = "hand_landmarker.task",
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        scale_factor: float = 1.0,
        min_confidence: float = 0.5,
        max_center_displacement: float = DEFAULT_MAX_CENTER_DISPLACEMENT,
        max_shape_rmse: float = DEFAULT_MAX_SHAPE_RMSE,
    ):
        if scale_factor != 1.0:
            raise ValueError("MediaPipe palm-local realtime requires scale_factor=1.0")
        if min_confidence != 0.5:
            raise ValueError("MediaPipeCameraInput uses fixed min_confidence=0.5")
        self.model_asset_path = Path(model_asset_path)
        self.processor = MediaPipePalmLocalProcessor()
        self.last_raw_frame = None
        self._input = MediaPipeCameraInput(
            model_asset_path=model_asset_path, camera_index=camera_index,
            width=width, height=height, fps=fps,
            invalid_hand_as_missing=True,
        )
        self._start_time = time.time()

    def next_input(self) -> dict | None:
        try:
            raw_frame = self._input.next_frame()
            self.last_raw_frame = raw_frame
            frame = {
                **raw_frame,
                "timestamp": round(time.time() - self._start_time, 3),
                "metadata": {
                    **raw_frame.get("metadata", {}),
                    "raw_timestamp_ms": raw_frame["timestamp"],
                    "timestamp_unit": "relative_seconds",
                },
            }
            return self.processor.process_frame(frame)
        except BaseException:
            self.last_raw_frame = None
            self.release()
            raise

    def release(self) -> None:
        self.processor.reset()
        self._input.release()

    def __enter__(self) -> "MediaPipeCameraAdapter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


def load_palm_local_retargeter(checkpoint_path=PALM_LOCAL_V2_CHECKPOINT, device="cpu"):
    """Load the existing model and enforce the palm-local checkpoint contract."""
    from retargeting.config import L21
    from retargeting.model import create_twohand_retargeter

    return create_twohand_retargeter(
        L21.model_kwargs(), device, checkpoint_path=str(checkpoint_path),
        expected_coordinate_alignment=PALM_LOCAL_COORDINATE_ALIGNMENT,
    ).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-asset-path", default="hand_landmarker.task")
    parser.add_argument("--checkpoint", default=str(PALM_LOCAL_V2_CHECKPOINT))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")

    counts = {side: {"raw": 0, "outputs": 0, "resets": 0, "recoveries": 0} for side in SIDES}
    seen_output = {side: False for side in SIDES}
    awaiting_recovery = {side: False for side in SIDES}
    completed = 0
    started = None
    try:
        import torch

        if args.device == "cpu":
            torch.set_num_threads(1)
        model = load_palm_local_retargeter(args.checkpoint, args.device)
        print(f"checkpoint={args.checkpoint} alignment={PALM_LOCAL_COORDINATE_ALIGNMENT} identity_tracking=False", flush=True)
        print("Move hands into view, out of view, then back into view to check reset/recovery.", flush=True)
        with MediaPipeCameraAdapter(args.model_asset_path, args.camera_index) as camera:
            started = time.perf_counter()
            for _ in range(args.frames):
                previous_streak = dict(camera.processor.valid_streak)
                payload = camera.next_input()
                predictions = {side: None for side in SIDES} if payload is None else model.predict(payload, args.device)["hands"]
                completed += 1
                descriptions = []
                for side in SIDES:
                    raw = camera.last_raw_frame[side]
                    palm = camera.processor.current_hands[side]
                    window = None if payload is None else payload["hands"][side]
                    angles = predictions[side]
                    streak = camera.processor.valid_streak[side]
                    event = ""
                    counts[side]["raw"] += int(raw is not None)
                    if streak == 0 and previous_streak[side] > 0:
                        counts[side]["resets"] += 1
                        awaiting_recovery[side] = seen_output[side]
                        event = " reset"
                    if angles is not None:
                        if angles.shape != (18,) or not np.isfinite(angles).all():
                            raise ValueError(f"Invalid {side} model output")
                        counts[side]["outputs"] += 1
                        seen_output[side] = True
                        if awaiting_recovery[side]:
                            counts[side]["recoveries"] += 1
                            awaiting_recovery[side] = False
                            event = " recovery_after_3_valid_frames"
                    arrays = (raw, palm, window, angles)
                    shapes = [None if value is None else value.shape for value in arrays]
                    finite = all(np.isfinite(value).all() for value in arrays if value is not None)
                    descriptions.append(
                        f"{side}: raw={shapes[0]} palm={shapes[1]} window={shapes[2]} "
                        f"angles={shapes[3]} finite={bool(finite)} streak={streak} "
                        f"invalid={camera.processor.invalid_reasons[side]!r}{event}"
                    )
                fps = completed / max(time.perf_counter() - started, 1e-9)
                print(f"frame={completed} timestamp={camera.last_raw_frame['timestamp']} unix_ms "
                      f"{' | '.join(descriptions)} FPS={fps:.1f}", flush=True)
    except KeyboardInterrupt:
        print("Capture stopped.")
    except Exception as error:
        print(f"Realtime smoke failed: {error}", flush=True)
        return 1
    elapsed = 0.0 if started is None else time.perf_counter() - started
    print(f"Captured={completed} counts={counts} FPS={completed / max(elapsed, 1e-9):.1f}; resources released")
    if not any(counts[side]["outputs"] for side in SIDES):
        print("Hand windows and real-hand inference were not observed; manual verification is still required.")
    elif not any(counts[side]["recoveries"] for side in SIDES):
        print("Real-hand dropout/recovery was not observed; manual verification is still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
