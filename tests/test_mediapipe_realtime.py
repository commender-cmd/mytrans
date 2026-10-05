"""Per-side recovery and offline/realtime equivalence for palm-local v2."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from retargeting.coordinates import (
    PALM_LOCAL_COORDINATE_ALIGNMENT,
    PalmBasisError,
    align_palm_local_coordinates,
    build_l21_reference_basis,
)
from retargeting.mediapipe import (
    MediaPipeCameraAdapter,
    MediaPipePalmLocalProcessor,
    PALM_LOCAL_V2_CHECKPOINT,
    load_palm_local_retargeter,
)
from retargeting.tracking import ensure_hand25
from tests.test_coordinate_contracts import synthetic_hand_pair
from tests.test_mediapipe_input import result as detection_result
from retargeting.inputs.mediapipe import parse_result


SIDES = ("left", "right")


def raw_frame(index, left=True, right=True):
    pair = synthetic_hand_pair()
    frame = {"timestamp": index / 30.0, "metadata": {"frame_index": index}}
    for side, raw, present in zip(SIDES, pair, (left, right)):
        points = raw.astype(np.float32)
        # Change finger shape as well as camera pose, so old/new frames differ.
        points[[4, 8, 12, 16, 20], 2] += index * 0.002
        angle = index * 0.04
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], dtype=np.float32)
        points = points @ rotation.T + np.array([0.003 * index, 0.0, 0.0], dtype=np.float32)
        frame[side] = points if present else None
    return frame


def offline_reference(sequence, bases):
    """Independent per-side reference lists; no realtime processor or buffer."""
    histories = {side: [] for side in SIDES}
    outputs = []
    for frame in sequence:
        current, windows = {}, {}
        for side in SIDES:
            raw = frame[side]
            aligned = None
            if raw is not None:
                array = np.asarray(raw, dtype=np.float32)
                if array.shape == (21, 3) and np.isfinite(array).all():
                    candidate = align_palm_local_coordinates(ensure_hand25(array), bases[side])
                    if np.isfinite(candidate).all() and np.any(candidate):
                        aligned = candidate
            current[side] = aligned
            if aligned is None:
                histories[side] = []
            else:
                histories[side] = (histories[side] + [aligned])[-3:]
            windows[side] = np.stack(histories[side]) if len(histories[side]) == 3 else None
        outputs.append((current, windows))
    return outputs


def sequences():
    left = [raw_frame(i, right=False) for i in range(7)]
    right = [raw_frame(i, left=False) for i in range(7)]
    dual = [raw_frame(i) for i in range(7)]
    missing_left = [raw_frame(i, left=i not in (3, 4)) for i in range(10)]
    missing_right = [raw_frame(i, right=i not in (3, 4)) for i in range(10)]
    regression = [raw_frame(i, left=i != 2, right=False) for i in range(6)]
    degenerate = [raw_frame(i) for i in range(9)]
    degenerate[3]["left"] = np.arange(63, dtype=np.float32).reshape(21, 3) / 100
    degenerate[4]["right"] = np.zeros((21, 3), dtype=np.float32)
    invalid = [raw_frame(i) for i in range(9)]
    invalid[3]["left"][7, 0] = np.nan
    invalid[4]["right"][7, 2] = np.inf
    return {
        "continuous_left": left, "continuous_right": right, "both_hands": dual,
        "left_missing_right_continuous": missing_left,
        "right_missing_left_continuous": missing_right,
        "A1_A2_None_A4_A5_A6": regression,
        "degenerate_palms": degenerate, "nonfinite_sides": invalid,
    }


class MediaPipeRealtimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bases = {side: build_l21_reference_basis(side) for side in SIDES}

    def test_per_frame_coordinates_and_windows_match_offline_reference(self):
        max_error = 0.0
        for name, sequence in sequences().items():
            with self.subTest(sequence=name):
                processor = MediaPipePalmLocalProcessor()
                reference = offline_reference(sequence, self.bases)
                for frame, (expected_current, expected_windows) in zip(sequence, reference):
                    payload = processor.process_frame(frame)
                    for side in SIDES:
                        expected = expected_current[side]
                        actual = processor.current_hands[side]
                        if expected is None:
                            self.assertIsNone(actual)
                            self.assertEqual(processor.valid_streak[side], 0)
                        else:
                            self.assertEqual(actual.shape, (25, 3))
                            self.assertEqual(actual.dtype, np.float32)
                            self.assertTrue(np.isfinite(actual).all())
                            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
                            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
                        window = None if payload is None else payload["hands"][side]
                        if expected_windows[side] is None:
                            self.assertIsNone(window)
                        else:
                            np.testing.assert_array_equal(window, expected_windows[side])
                            self.assertEqual(window.shape, (3, 25, 3))
                            self.assertTrue(np.isfinite(window).all())
                    if payload is not None:
                        self.assertEqual(payload["metadata"]["coordinate_frame"], "l21")
                        self.assertEqual(payload["metadata"]["coordinate_alignment"], PALM_LOCAL_COORDINATE_ALIGNMENT)
                        self.assertEqual(payload["metadata"]["source_landmark_space"], "mediapipe_normalized")
                        self.assertIs(payload["metadata"]["identity_tracking"], False)
                        self.assertEqual(payload["timestamp"], frame["timestamp"])
        print(f"offline/realtime max palm-local coordinate error={max_error:.9g}")

    def test_A1_A2_None_A4_A5_A6_resets_immediately_and_keeps_only_new_frames(self):
        processor = MediaPipePalmLocalProcessor()
        frames = sequences()["A1_A2_None_A4_A5_A6"]
        for frame in frames[:2]:
            self.assertIsNone(processor.process_frame(frame))
        self.assertIsNone(processor.process_frame(frames[2]))
        self.assertEqual(len(processor._buffer._buffers["left"]), 0)
        self.assertIsNone(processor.current_hands["left"])
        self.assertIsNone(processor.process_frame(frames[3]))
        self.assertIsNone(processor.process_frame(frames[4]))
        window = processor.process_frame(frames[5])["hands"]["left"]
        expected = np.stack([
            align_palm_local_coordinates(ensure_hand25(frame["left"]), self.bases["left"])
            for frame in frames[3:6]
        ])
        np.testing.assert_array_equal(window, expected)
        old = align_palm_local_coordinates(ensure_hand25(frames[0]["left"]), self.bases["left"])
        self.assertFalse(np.array_equal(window[0], old))

    def test_invalid_side_resets_without_clearing_other_side_and_recovers_after_three(self):
        for side in SIDES:
            other = "right" if side == "left" else "left"
            invalid_values = [None, np.zeros((21, 3), dtype=np.float32),
                              np.arange(63, dtype=np.float32).reshape(21, 3),
                              np.full((21, 3), np.nan), np.full((21, 3), np.inf),
                              np.full((21, 3), 1e100), np.zeros((25, 3)), "invalid"]
            for invalid in invalid_values:
                with self.subTest(side=side, invalid=str(invalid)[:35]):
                    processor = MediaPipePalmLocalProcessor()
                    for i in range(3):
                        processor.process_frame(raw_frame(i))
                    missing = raw_frame(3)
                    missing[side] = invalid
                    payload = processor.process_frame(missing)
                    self.assertIsNone(payload["hands"][side])
                    self.assertEqual(processor.valid_streak[side], 0)
                    self.assertEqual(len(processor._buffer._buffers[side]), 0)
                    self.assertEqual(payload["hands"][other].shape, (3, 25, 3))
                    self.assertEqual(processor.valid_streak[other], 3)
                    recovered = []
                    for i in (4, 5, 6):
                        frame = raw_frame(i)
                        recovered.append(align_palm_local_coordinates(ensure_hand25(frame[side]), self.bases[side]))
                        payload = processor.process_frame(frame)
                        self.assertEqual(payload["hands"][other].shape, (3, 25, 3))
                        if i < 6:
                            self.assertIsNone(payload["hands"][side])
                        else:
                            np.testing.assert_array_equal(payload["hands"][side], np.stack(recovered))

    def test_palm_basis_exception_is_side_local(self):
        processor = MediaPipePalmLocalProcessor()
        for i in range(3):
            processor.process_frame(raw_frame(i))
        right = align_palm_local_coordinates(ensure_hand25(raw_frame(3)["right"]), self.bases["right"])
        with patch("retargeting.mediapipe.align_palm_local_coordinates",
                   side_effect=[PalmBasisError("degenerate source"), right]):
            payload = processor.process_frame(raw_frame(3))
        self.assertIsNone(payload["hands"]["left"])
        self.assertIsNotNone(payload["hands"]["right"])
        self.assertEqual(processor.valid_streak["left"], 0)

    def test_each_basis_built_once_and_alignment_precedes_buffer_without_tracking(self):
        events = []
        with patch("retargeting.mediapipe.build_l21_reference_basis", side_effect=lambda side: self.bases[side]) as basis:
            processor = MediaPipePalmLocalProcessor()
            self.assertEqual(basis.call_count, 2)
            self.assertEqual([call.args[0] for call in basis.call_args_list], list(SIDES))
            original_append = processor._buffer.update_canonical
            def convert(points):
                events.append("convert")
                return ensure_hand25(points)
            def align(points, robot):
                events.append("align_left" if robot is self.bases["left"] else "align_right")
                return align_palm_local_coordinates(points, robot)
            def append(**kwargs):
                events.append("append")
                for side in SIDES:
                    np.testing.assert_array_equal(kwargs[f"{side}_hand"], processor.current_hands[side])
                return original_append(**kwargs)
            with patch("retargeting.mediapipe.ensure_hand25", side_effect=convert), \
                 patch("retargeting.mediapipe.align_palm_local_coordinates", side_effect=align), \
                 patch.object(processor._buffer, "update_canonical", side_effect=append), \
                 patch("retargeting.tracking.HandIdentityTracker", side_effect=AssertionError("identity tracker called")), \
                 patch("retargeting.coordinates.align_source_hand_coordinates", side_effect=AssertionError("legacy alignment called")):
                for i in range(5):
                    processor.process_frame(raw_frame(i))
            self.assertEqual(events, ["convert", "align_left", "convert", "align_right", "append"] * 5)
            self.assertEqual(basis.call_count, 2)

    def test_media_pipe_sides_are_trusted_after_label_swap_or_position_jump(self):
        processor = MediaPipePalmLocalProcessor()
        for i in range(3):
            processor.process_frame(raw_frame(i))
        frame = raw_frame(3)
        frame["left"], frame["right"] = frame["right"] + 5, frame["left"] - 5
        payload = processor.process_frame(frame)
        for side in SIDES:
            expected = align_palm_local_coordinates(ensure_hand25(frame[side]), self.bases[side])
            np.testing.assert_array_equal(payload["hands"][side][-1], expected)

    def test_adapter_composes_raw_input_and_preserves_relative_timestamp_semantics(self):
        raw = Mock()
        frames = [raw_frame(i) for i in range(3)]
        for i, frame in enumerate(frames):
            frame["timestamp"] = 100000 + i * 33
            frame["metadata"]["timestamp_unit"] = "unix_ms"
        raw.next_frame.side_effect = frames
        with patch("retargeting.mediapipe.MediaPipeCameraInput", return_value=raw) as camera, \
             patch("retargeting.mediapipe.time.time", side_effect=[100, 100.1, 100.2, 100.3]):
            with MediaPipeCameraAdapter("model.task", camera_index=2) as adapter:
                self.assertIsNone(adapter.next_input())
                self.assertIsNone(adapter.next_input())
                payload = adapter.next_input()
                self.assertEqual(payload["timestamp"], 0.3)
                self.assertEqual(payload["metadata"]["raw_timestamp_ms"], 100066)
                self.assertEqual(payload["metadata"]["timestamp_unit"], "relative_seconds")
                self.assertIs(adapter.last_raw_frame, frames[-1])
            camera.assert_called_once_with(model_asset_path="model.task", camera_index=2,
                                           width=640, height=480, fps=30, invalid_hand_as_missing=True)
        raw.release.assert_called_once()
        self.assertEqual(adapter.processor.valid_streak, {"left": 0, "right": 0})

    def test_capture_error_does_not_leave_old_windows(self):
        raw = Mock()
        raw.next_frame.side_effect = [raw_frame(i) for i in range(3)] + [RuntimeError("capture failed")]
        with patch("retargeting.mediapipe.MediaPipeCameraInput", return_value=raw):
            adapter = MediaPipeCameraAdapter("model.task")
            for _ in range(3):
                adapter.next_input()
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                adapter.next_input()
        self.assertEqual(adapter.processor.valid_streak, {"left": 0, "right": 0})
        self.assertIsNone(adapter.last_raw_frame)
        raw.release.assert_called_once()

    def test_duplicate_raw_labels_clear_history_and_restart_three_valid_frames(self):
        processor = MediaPipePalmLocalProcessor()
        for i in range(3):
            processor.process_frame(raw_frame(i))
        duplicate = parse_result(detection_result(["Right", "Right"]), 1000,
                                 invalid_hand_as_missing=True)
        self.assertIsNone(processor.process_frame(duplicate))
        self.assertEqual(processor.valid_streak, {"left": 0, "right": 0})
        self.assertIn("Duplicate handedness", processor.invalid_reasons["right"])
        self.assertIsNone(processor.process_frame(raw_frame(4)))
        self.assertIsNone(processor.process_frame(raw_frame(5)))
        payload = processor.process_frame(raw_frame(6))
        for side in SIDES:
            expected = np.stack([align_palm_local_coordinates(ensure_hand25(raw_frame(i)[side]), self.bases[side])
                                 for i in (4, 5, 6)])
            np.testing.assert_array_equal(payload["hands"][side], expected)

    def test_robot_reference_failure_is_fatal_before_camera_open(self):
        with patch("retargeting.mediapipe.build_l21_reference_basis", side_effect=PalmBasisError("robot reference")), \
             patch("retargeting.mediapipe.MediaPipeCameraInput") as camera:
            with self.assertRaisesRegex(PalmBasisError, "robot reference"):
                MediaPipeCameraAdapter("model.task")
        camera.assert_not_called()

    def test_scale_normalization_and_changed_capture_confidence_are_rejected(self):
        for kwargs in ({"scale_factor": 2}, {"min_confidence": 0.9}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MediaPipeCameraAdapter("model.task", **kwargs)

    def test_checkpoint_loader_always_requests_palm_local_alignment(self):
        with patch("retargeting.model.create_twohand_retargeter") as create:
            load_palm_local_retargeter()
        self.assertEqual(create.call_args.kwargs["checkpoint_path"], str(PALM_LOCAL_V2_CHECKPOINT))
        self.assertEqual(create.call_args.kwargs["expected_coordinate_alignment"], PALM_LOCAL_COORDINATE_ALIGNMENT)

    def test_legacy_missing_and_unknown_checkpoint_alignment_fail_before_state_loading(self):
        for alignment in ("source_to_l21_xyz", None, "palm_local_to_l21_v2"):
            with self.subTest(alignment=alignment), \
                 patch("retargeting.model.build_hand_model", return_value=torch.nn.Linear(1, 1)), \
                 patch("retargeting.model.torch.load", return_value={"coordinate_alignment": alignment}):
                with self.assertRaisesRegex(ValueError, "alignment"):
                    load_palm_local_retargeter()


@unittest.skipUnless(PALM_LOCAL_V2_CHECKPOINT.is_file(), "local palm_local_v2 checkpoint unavailable")
class PalmLocalV2AngleEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.model = load_palm_local_retargeter()
        cls.bases = {side: build_l21_reference_basis(side) for side in SIDES}

    @classmethod
    def tearDownClass(cls):
        del cls.model
        torch.set_num_threads(cls.previous_threads)

    def test_same_checkpoint_batched_offline_and_streaming_realtime_angles(self):
        max_error = 0.0
        compared = 0
        for name, sequence in sequences().items():
            with self.subTest(sequence=name):
                references = offline_reference(sequence, self.bases)
                expected_angles = {side: {} for side in SIDES}
                for side in SIDES:
                    indices = [i for i, (_, windows) in enumerate(references) if windows[side] is not None]
                    if indices:
                        batch = np.stack([references[i][1][side] for i in indices])
                        with torch.no_grad():
                            predicted = self.model.model(torch.from_numpy(batch)).cpu().numpy()
                        self.assertEqual(predicted.shape, (len(indices), 18))
                        self.assertTrue(np.isfinite(predicted).all())
                        expected_angles[side] = dict(zip(indices, predicted))
                processor = MediaPipePalmLocalProcessor()
                for index, frame in enumerate(sequence):
                    payload = processor.process_frame(frame)
                    actual = {side: None for side in SIDES} if payload is None else self.model.predict(payload, "cpu")["hands"]
                    for side in SIDES:
                        if index not in expected_angles[side]:
                            self.assertIsNone(actual[side])
                        else:
                            self.assertEqual(actual[side].shape, (18,))
                            self.assertTrue(np.isfinite(actual[side]).all())
                            error = float(np.max(np.abs(actual[side] - expected_angles[side][index])))
                            max_error = max(max_error, error)
                            np.testing.assert_allclose(actual[side], expected_angles[side][index], atol=1e-5, rtol=1e-5)
                            compared += 1
        self.assertGreater(compared, 0)
        print(f"palm_local_v2 offline/realtime angles compared={compared} max angle error={max_error:.9g} radians")


if __name__ == "__main__":
    unittest.main()
