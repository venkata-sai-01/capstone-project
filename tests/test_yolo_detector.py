import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.yolo_detector import IoUTracker, YOLO26Detector


class FakeTensor:
    def __init__(self, value: np.ndarray):
        self.value = value

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class FakeBoxes:
    def __init__(
        self,
        xyxy: list[list[float]],
        class_ids: list[int],
        scores: list[float],
    ):
        self.xyxy = FakeTensor(np.asarray(xyxy, dtype=np.float32).reshape(-1, 4))
        self.cls = FakeTensor(np.asarray(class_ids, dtype=np.float32))
        self.conf = FakeTensor(np.asarray(scores, dtype=np.float32))


class FakeModel:
    names = {0: "car", 1: "person"}

    def __init__(self):
        self.calls: list[dict] = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        boxes = FakeBoxes([[10, 20, 50, 60]], [0], [0.9])
        return [SimpleNamespace(boxes=boxes)]


class YOLO26InitializationTests(unittest.TestCase):
    def test_requires_an_ultralytics_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            YOLO26Detector("model.onnx")

    def test_reports_when_ultralytics_is_not_installed(self) -> None:
        with patch.dict(sys.modules, {"ultralytics": None}):
            with self.assertRaisesRegex(RuntimeError, "Install project dependencies"):
                YOLO26Detector("model.pt")

    def test_initializes_ultralytics_checkpoint_and_class_names(self) -> None:
        model = FakeModel()
        with patch.dict(
            sys.modules,
            {"ultralytics": SimpleNamespace(YOLO=lambda *args, **kwargs: model)},
        ):
            detector = YOLO26Detector("model.pt")

        self.assertIs(detector.model, model)
        self.assertEqual(detector.names, ["car", "person"])


class YOLO26InferenceTests(unittest.TestCase):
    @staticmethod
    def detector_with_fake_model() -> tuple[YOLO26Detector, FakeModel]:
        detector = YOLO26Detector.__new__(YOLO26Detector)
        detector.input_width = 640
        detector.input_height = 640
        detector.device = "cpu"
        detector.names = ["car", "person"]
        detector.tracker = IoUTracker()
        model = FakeModel()
        detector.model = model
        return detector, model

    def test_runs_model_and_returns_boxes_tracks_classes_and_scores(self) -> None:
        detector, model = self.detector_with_fake_model()

        boxes, track_ids, class_ids, confidences, names = detector.infer(
            np.zeros((100, 100, 3), dtype=np.uint8), 0.3
        )

        np.testing.assert_array_equal(boxes, [[10, 20, 50, 60]])
        np.testing.assert_array_equal(track_ids, [1])
        np.testing.assert_array_equal(class_ids, [0])
        np.testing.assert_allclose(confidences, [0.9])
        self.assertEqual(names, ["car", "person"])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["conf"], 0.3)
        self.assertEqual(model.calls[0]["device"], "cpu")

    def test_large_frames_use_full_frame_and_overlapping_crop_passes(self) -> None:
        detector, model = self.detector_with_fake_model()

        boxes, _, class_ids, _, _ = detector.infer(
            np.zeros((720, 1280, 3), dtype=np.uint8), 0.3
        )

        self.assertEqual(len(model.calls), 3)
        self.assertEqual(
            [call["source"].shape[:2] for call in model.calls],
            [(720, 1280), (720, 768), (720, 768)],
        )
        np.testing.assert_array_equal(boxes, [[10, 20, 50, 60], [522, 20, 562, 60]])
        np.testing.assert_array_equal(class_ids, [0, 0])

    def test_nms_suppresses_same_class_duplicates_but_keeps_other_classes(self) -> None:
        boxes = np.array(
            [[10, 10, 50, 50], [11, 11, 51, 51], [10, 10, 50, 50]]
        )
        class_ids = np.array([0, 0, 1])
        scores = np.array([0.9, 0.8, 0.85], dtype=np.float32)

        kept_indices = YOLO26Detector._nms_indices(boxes, class_ids, scores)

        np.testing.assert_array_equal(kept_indices, [0, 2])

    def test_small_frames_use_single_full_frame_pass(self) -> None:
        detector, _ = self.detector_with_fake_model()

        self.assertEqual(
            detector._inference_regions((480, 640)),
            [(0, 0, 640, 480)],
        )


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
