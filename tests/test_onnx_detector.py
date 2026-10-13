import unittest
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort

from src.onnx_detector import IoUTracker, ONNXObjectDetector


class ONNXDecodeTests(unittest.TestCase):
    def test_missing_or_pytorch_model_files_are_rejected_with_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing_onnx = Path(directory) / "missing.onnx"
            pytorch_model = Path(directory) / "model.pt"
            pytorch_model.touch()

            with self.assertRaisesRegex(FileNotFoundError, "ONNX model not found"):
                ONNXObjectDetector(missing_onnx)
            with self.assertRaisesRegex(ValueError, "Expected an ONNX model file"):
                ONNXObjectDetector(pytorch_model)

    def test_decodes_yolov8_output_and_restores_letterboxed_coordinates(self) -> None:
        output = np.zeros((1, 84, 1), dtype=np.float32)
        output[0, :4, 0] = [320, 320, 100, 200]
        output[0, 4 + 2, 0] = 0.9

        boxes, class_ids, confidences = ONNXObjectDetector._decode(
            output,
            image_shape=(360, 640),
            scale=1.0,
            pad_x=0,
            pad_y=140,
            confidence_threshold=0.3,
            class_count=80,
        )

        np.testing.assert_array_equal(boxes, [[270, 80, 370, 280]])
        np.testing.assert_array_equal(class_ids, [2])
        np.testing.assert_allclose(confidences, [0.9])

    def test_nms_keeps_overlapping_different_classes_and_suppresses_duplicate(self) -> None:
        output = np.zeros((1, 84, 3), dtype=np.float32)
        output[0, :4, :] = np.array(
            [[320, 320, 321], [320, 320, 320], [100, 100, 100], [100, 100, 100]],
            dtype=np.float32,
        )
        output[0, 4 + 2, :] = [0.9, 0.8, 0.0]
        output[0, 4 + 3, :] = [0.0, 0.0, 0.85]

        boxes, class_ids, confidences = ONNXObjectDetector._decode(
            output,
            image_shape=(640, 640),
            scale=1.0,
            pad_x=0,
            pad_y=0,
            confidence_threshold=0.3,
            class_count=80,
        )

        self.assertEqual(len(boxes), 2)
        self.assertEqual(set(class_ids.tolist()), {2, 3})
        self.assertEqual({round(float(score), 2) for score in confidences}, {0.85, 0.9})

    def test_decodes_export_with_embedded_nms(self) -> None:
        output = np.array([[[10, 20, 40, 60, 0.95, 2]]], dtype=np.float32)

        boxes, class_ids, confidences = ONNXObjectDetector._decode(
            output,
            image_shape=(100, 100),
            scale=1.0,
            pad_x=0,
            pad_y=0,
            confidence_threshold=0.3,
            class_count=80,
        )

        np.testing.assert_array_equal(boxes, [[10, 20, 40, 60]])
        np.testing.assert_array_equal(class_ids, [2])
        np.testing.assert_allclose(confidences, [0.95])

    def test_decodes_yolo26_end_to_end_output_and_discards_padding(self) -> None:
        output = np.zeros((1, 300, 6), dtype=np.float32)
        output[0, 0] = [10, 20, 40, 60, 0.95, 2]
        output[0, 1] = [50, 60, 90, 100, 0.25, 3]

        boxes, class_ids, confidences = ONNXObjectDetector._decode(
            output,
            image_shape=(100, 100),
            scale=1.0,
            pad_x=0,
            pad_y=0,
            confidence_threshold=0.3,
            class_count=80,
        )

        np.testing.assert_array_equal(boxes, [[10, 20, 40, 60]])
        np.testing.assert_array_equal(class_ids, [2])
        np.testing.assert_allclose(confidences, [0.95])

    def test_large_frames_use_full_frame_and_overlapping_crops(self) -> None:
        detector = ONNXObjectDetector.__new__(ONNXObjectDetector)
        detector.input_width = 640
        detector.input_height = 640

        regions = detector._inference_regions((720, 1280))

        self.assertEqual(
            regions,
            [
                (0, 0, 1280, 720),
                (0, 0, 768, 720),
                (512, 0, 1280, 720),
            ],
        )

    def test_small_frames_keep_single_full_frame_pass(self) -> None:
        detector = ONNXObjectDetector.__new__(ONNXObjectDetector)
        detector.input_width = 640
        detector.input_height = 640

        self.assertEqual(detector._inference_regions((480, 640)), [(0, 0, 640, 480)])

    def test_global_nms_suppresses_cross_crop_duplicates_by_class(self) -> None:
        boxes = np.array(
            [[10, 10, 50, 50], [11, 11, 51, 51], [10, 10, 50, 50]]
        )
        class_ids = np.array([2, 2, 3])
        scores = np.array([0.9, 0.8, 0.85], dtype=np.float32)

        kept_indices = ONNXObjectDetector._nms_indices(boxes, class_ids, scores)

        np.testing.assert_array_equal(kept_indices, [0, 2])


class ONNXInferenceTests(unittest.TestCase):
    def test_large_frame_runs_full_frame_and_crop_inference(self) -> None:
        class FakeSession:
            def __init__(self) -> None:
                self.inputs: list[np.ndarray] = []

            def run(self, _, inputs):
                self.inputs.append(inputs["images"])
                return [np.zeros((1, 84, 1), dtype=np.float32)]

        detector = ONNXObjectDetector.__new__(ONNXObjectDetector)
        detector.input_width = 640
        detector.input_height = 640
        detector.input_name = "images"
        detector.names = ["class"] * 80
        detector.tracker = IoUTracker()
        fake_session = FakeSession()
        detector.session = cast(ort.InferenceSession, fake_session)

        detections = detector.infer(np.zeros((720, 1280, 3), dtype=np.uint8), 0.3)

        self.assertEqual(len(fake_session.inputs), 3)
        self.assertTrue(
            all(tensor.shape == (1, 3, 640, 640) for tensor in fake_session.inputs)
        )
        self.assertEqual(detections[0].shape, (0, 4))


class IoUTrackerTests(unittest.TestCase):
    def test_tracks_objects_across_frames_and_assigns_new_ids(self) -> None:
        tracker = IoUTracker()
        first_ids = tracker.update(
            np.array([[10, 10, 40, 40], [100, 100, 140, 140]]),
            np.array([2, 3]),
        )
        next_ids = tracker.update(
            np.array([[12, 10, 42, 40], [100, 100, 140, 140], [200, 200, 240, 240]]),
            np.array([2, 3, 2]),
        )

        np.testing.assert_array_equal(first_ids, [1, 2])
        np.testing.assert_array_equal(next_ids, [1, 2, 3])

    def test_tracks_do_not_match_different_classes(self) -> None:
        tracker = IoUTracker()
        first_id = tracker.update(np.array([[10, 10, 40, 40]]), np.array([2]))
        second_id = tracker.update(np.array([[10, 10, 40, 40]]), np.array([3]))

        self.assertNotEqual(first_id[0], second_id[0])


if __name__ == "__main__":
    unittest.main()
