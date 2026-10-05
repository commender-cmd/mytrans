"""Raw MediaPipe contracts and camera lifecycle without hardware dependencies."""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from retargeting.inputs.mediapipe import MediaPipeCameraInput, parse_result


def hand(offset=0.0):
    return [NS(x=0.1 + i * 0.01 + offset, y=0.2, z=-0.03) for i in range(21)]


def result(labels=(), hands=None):
    return NS(
        hand_landmarks=[hand(i * 0.3) for i in range(len(labels))] if hands is None else hands,
        handedness=[[NS(category_name=label)] for label in labels],
    )


class RawMediaPipeParsingTests(unittest.TestCase):
    def test_no_hand_frame_still_has_timestamp_and_metadata(self):
        frame = parse_result(result(), 1234, {"frame_index": 0})
        self.assertIsNone(frame["left"])
        self.assertIsNone(frame["right"])
        self.assertEqual(frame["timestamp"], 1234)
        self.assertEqual(frame["metadata"], {
            "frame_index": 0, "source_landmark_space": "mediapipe_normalized",
            "timestamp_unit": "unix_ms",
        })

    def test_single_hand_mapping_preserves_raw_xyz(self):
        for label, side, missing in (("Left", "left", "right"), ("Right", "right", "left")):
            with self.subTest(label=label):
                raw = hand()
                frame = parse_result(result([label], [raw]), 1234)
                points = frame[side]
                self.assertEqual(points.shape, (21, 3))
                self.assertEqual(points.dtype, np.float32)
                self.assertTrue(np.isfinite(points).all())
                np.testing.assert_array_equal(points, np.array(
                    [[lm.x, lm.y, lm.z] for lm in raw], dtype=np.float32
                ))
                self.assertIsNone(frame[missing])

    def test_two_hands_use_labels_in_either_detection_order(self):
        for labels in (("Left", "Right"), ("Right", "Left")):
            with self.subTest(labels=labels):
                detected = result(labels)
                frame = parse_result(detected, 1234)
                for index, label in enumerate(labels):
                    points = frame[label.lower()]
                    self.assertEqual(points.shape, (21, 3))
                    self.assertEqual(points.dtype, np.float32)
                    self.assertTrue(np.isfinite(points).all())
                    np.testing.assert_array_equal(points, np.array(
                        [[lm.x, lm.y, lm.z] for lm in detected.hand_landmarks[index]],
                        dtype=np.float32,
                    ))

    def test_wrong_landmark_counts_raise(self):
        for count in (0, 20, 22, 25):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "shape"):
                parse_result(result(["Left"], [[NS(x=0.1, y=0.2, z=0.3)] * count]), 1)

    def test_nonfinite_and_float32_overflow_raise(self):
        for value in (np.nan, np.inf, -np.inf, 1e100):
            for axis in ("x", "y", "z"):
                points = hand()
                setattr(points[7], axis, value)
                with self.subTest(value=value, axis=axis), self.assertRaisesRegex(ValueError, "finite"):
                    parse_result(result(["Right"], [points]), 1)

    def test_unknown_or_duplicate_handedness_raise(self):
        for labels in (("Unknown",), ("left",), ("Left", "Left"), ("Right", "Right")):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "handedness"):
                parse_result(result(labels), 1)

    def test_realtime_opt_in_drops_only_invalid_known_side_with_explicit_reason(self):
        for bad_side, labels in (("left", ("Left", "Right")), ("right", ("Right", "Left"))):
            for invalid in (hand()[:20], [NS(x=np.nan, y=0.1, z=0.2)] * 21):
                with self.subTest(side=bad_side):
                    detected = result(labels, [invalid, hand()])
                    frame = parse_result(detected, 1, invalid_hand_as_missing=True)
                    self.assertIsNone(frame[bad_side])
                    other = "right" if bad_side == "left" else "left"
                    self.assertEqual(frame[other].shape, (21, 3))
                    self.assertTrue(np.isfinite(frame[other]).all())
                    self.assertIn(bad_side, frame["metadata"]["invalid_reasons"])
                    with self.assertRaises(ValueError):
                        parse_result(detected, 1)

    def test_realtime_opt_in_does_not_guess_unknown_handedness(self):
        with self.assertRaisesRegex(ValueError, "handedness"):
            parse_result(result(["Unknown"]), 1, invalid_hand_as_missing=True)

    def test_realtime_duplicate_known_labels_invalidate_side_without_overwriting(self):
        for label in ("Left", "Right"):
            with self.subTest(label=label):
                frame = parse_result(result([label, label]), 1, invalid_hand_as_missing=True)
                self.assertIsNone(frame["left"])
                self.assertIsNone(frame["right"])
                self.assertIn("Duplicate handedness", frame["metadata"]["invalid_reasons"][label.lower()])
                # Even if the first duplicate was malformed, do not take the second.
                frame = parse_result(result([label, label], [hand()[:20], hand()]), 1,
                                     invalid_hand_as_missing=True)
                self.assertIsNone(frame[label.lower()])

    def test_malformed_results_raise_instead_of_returning_partial_data(self):
        missing_xyz = hand()
        missing_xyz[3] = NS(x=0.2, y=0.1)
        nonnumeric_xyz = hand()
        nonnumeric_xyz[3].z = "invalid"
        invalid_results = [
            None, NS(), NS(hand_landmarks=None, handedness=[]),
            NS(hand_landmarks=[hand()], handedness=[]),
            NS(hand_landmarks=[], handedness=[[NS(category_name="Left")]]),
            NS(hand_landmarks=[hand()], handedness=[[]]),
            NS(hand_landmarks=[hand()], handedness=[[NS()]]),
            result(["Left", "Right", "Left"]),
            result(["Left"], [missing_xyz]),
            result(["Left"], [nonnumeric_xyz]),
            result(["Left", "Right"], [hand(), hand()[:20]]),
        ]
        for index, detected in enumerate(invalid_results):
            with self.subTest(index=index), self.assertRaises(ValueError):
                parse_result(detected, 1)


class MediaPipeCameraLifecycleTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.asset = Path(directory.name) / "hand_landmarker.task"
        self.asset.touch()
        self.bgr = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        self.cap = Mock()
        self.cap.isOpened.return_value = True
        self.cap.read.return_value = (True, self.bgr)
        self.landmarker = Mock()
        self.landmarker.detect_for_video.return_value = result()
        self.cv2 = NS(
            VideoCapture=Mock(return_value=self.cap),
            CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4, CAP_PROP_FPS=5,
            COLOR_BGR2RGB=99,
            cvtColor=Mock(side_effect=lambda frame, mode: frame[..., ::-1].copy()),
        )
        self.mp = NS(
            Image=Mock(side_effect=lambda **kwargs: NS(**kwargs)),
            ImageFormat=NS(SRGB="SRGB"),
            tasks=NS(
                BaseOptions=Mock(side_effect=lambda **kwargs: NS(**kwargs)),
                vision=NS(
                    RunningMode=NS(VIDEO="VIDEO"),
                    HandLandmarkerOptions=Mock(side_effect=lambda **kwargs: NS(**kwargs)),
                    HandLandmarker=NS(create_from_options=Mock(return_value=self.landmarker)),
                ),
            ),
        )
        modules = patch.dict(sys.modules, {"cv2": self.cv2, "mediapipe": self.mp})
        modules.start()
        self.addCleanup(modules.stop)

    def camera(self):
        camera = MediaPipeCameraInput(self.asset)
        self.addCleanup(camera.release)
        return camera

    def test_reference_camera_and_landmarker_settings(self):
        self.camera()
        self.cv2.VideoCapture.assert_called_once_with(0)
        self.assertEqual(self.cap.set.call_args_list, [
            unittest.mock.call(3, 640), unittest.mock.call(4, 480), unittest.mock.call(5, 30),
        ])
        options = self.mp.tasks.vision.HandLandmarker.create_from_options.call_args.args[0]
        self.assertEqual(options.base_options.model_asset_path, str(self.asset))
        self.assertEqual(options.running_mode, "VIDEO")
        self.assertEqual(options.num_hands, 2)
        self.assertEqual(options.min_hand_detection_confidence, 0.5)
        self.assertEqual(options.min_hand_presence_confidence, 0.5)
        self.assertEqual(options.min_tracking_confidence, 0.5)

    def test_rgb_video_timestamp_and_frame_metadata(self):
        camera = self.camera()
        with patch("retargeting.inputs.mediapipe.time.time", return_value=100.125):
            output = camera.next_frame()
        self.cv2.cvtColor.assert_called_once_with(self.bgr, 99)
        image, timestamp = self.landmarker.detect_for_video.call_args.args
        np.testing.assert_array_equal(image.data, self.bgr[..., ::-1])
        self.assertEqual(image.image_format, "SRGB")
        self.assertEqual(timestamp, 100125)
        self.assertEqual(output["timestamp"], timestamp)
        self.assertEqual(output["metadata"]["camera_index"], 0)
        self.assertEqual(output["metadata"]["frame_index"], 0)
        self.assertEqual(output["metadata"]["image_width"], 2)
        self.assertEqual(output["metadata"]["image_height"], 1)
        self.assertIsNone(output["left"])
        self.assertIsNone(output["right"])

    def test_first_frame_returns_a_raw_hand_without_window_or_transform(self):
        self.landmarker.detect_for_video.return_value = result(["Left"])
        output = self.camera().next_frame()
        self.assertEqual(output["left"].shape, (21, 3))
        np.testing.assert_array_equal(output["left"][0], np.array([0.1, 0.2, -0.03], dtype=np.float32))

    def test_realtime_invalid_hand_policy_keeps_camera_running_and_other_hand_valid(self):
        invalid = hand()
        invalid[3].x = np.nan
        self.landmarker.detect_for_video.side_effect = [result(["Left", "Right"], [invalid, hand()]), result(["Left", "Right"])]
        with MediaPipeCameraInput(self.asset, invalid_hand_as_missing=True) as camera:
            first = camera.next_frame()
            self.assertIsNone(first["left"])
            self.assertEqual(first["right"].shape, (21, 3))
            self.cap.release.assert_not_called()
            second = camera.next_frame()
            self.assertEqual(second["left"].shape, (21, 3))
            self.assertEqual(second["right"].shape, (21, 3))

    def test_repeated_or_backward_clock_keeps_video_timestamps_increasing(self):
        camera = self.camera()
        with patch("retargeting.inputs.mediapipe.time.time", side_effect=[100, 100, 99, 101]):
            frames = [camera.next_frame() for _ in range(4)]
        self.assertEqual([frame["timestamp"] for frame in frames], [100000, 100001, 100002, 101000])
        self.assertEqual([frame["metadata"]["frame_index"] for frame in frames], [0, 1, 2, 3])

    def test_camera_open_failure_releases_capture(self):
        self.cap.isOpened.return_value = False
        with self.assertRaisesRegex(RuntimeError, "Cannot open camera"):
            MediaPipeCameraInput(self.asset)
        self.cap.release.assert_called_once()
        self.mp.tasks.vision.HandLandmarker.create_from_options.assert_not_called()

    def test_landmarker_initialization_failure_releases_capture(self):
        self.mp.tasks.vision.HandLandmarker.create_from_options.side_effect = RuntimeError("model failure")
        with self.assertRaisesRegex(RuntimeError, "model failure"):
            MediaPipeCameraInput(self.asset)
        self.cap.release.assert_called_once()

    def test_missing_asset_does_not_open_camera(self):
        with self.assertRaises(FileNotFoundError):
            MediaPipeCameraInput(self.asset.with_name("missing.task"))
        self.cv2.VideoCapture.assert_not_called()

    def test_read_failures_release_both_resources(self):
        for read_result in ((False, None), (True, None), (True, np.empty((0, 0, 3), dtype=np.uint8))):
            with self.subTest(read_result=read_result):
                self.cap.release.reset_mock()
                self.landmarker.close.reset_mock()
                camera = self.camera()
                self.cap.read.return_value = read_result
                with self.assertRaisesRegex(RuntimeError, "Frame read failed"):
                    camera.next_frame()
                self.cap.release.assert_called_once()
                self.landmarker.close.assert_called_once()

    def test_detection_failure_releases_both_resources(self):
        camera = self.camera()
        self.landmarker.detect_for_video.side_effect = RuntimeError("detection failure")
        with self.assertRaisesRegex(RuntimeError, "detection failure"):
            camera.next_frame()
        self.cap.release.assert_called_once()
        self.landmarker.close.assert_called_once()

    def test_invalid_detection_releases_both_resources(self):
        camera = self.camera()
        self.landmarker.detect_for_video.return_value = result(["Unknown"])
        with self.assertRaises(ValueError):
            camera.next_frame()
        self.cap.release.assert_called_once()
        self.landmarker.close.assert_called_once()

    def test_context_exit_release_is_idempotent_and_closed_input_raises(self):
        camera = self.camera()
        with camera:
            camera.next_frame()
        camera.release()
        self.cap.release.assert_called_once()
        self.landmarker.close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            camera.next_frame()

    def test_context_exception_releases_both_resources(self):
        with self.assertRaisesRegex(RuntimeError, "consumer failure"):
            with self.camera():
                raise RuntimeError("consumer failure")
        self.cap.release.assert_called_once()
        self.landmarker.close.assert_called_once()

    def test_landmarker_closes_even_if_capture_release_raises(self):
        camera = self.camera()
        self.cap.release.side_effect = RuntimeError("release failure")
        with self.assertRaisesRegex(RuntimeError, "release failure"):
            camera.release()
        self.landmarker.close.assert_called_once()
        camera.release()


if __name__ == "__main__":
    unittest.main()
