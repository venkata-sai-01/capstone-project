import tempfile
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_yolo import (
    box_iou,
    evaluate_predictions,
    load_yolo_labels,
)


class YoloLabelTests(unittest.TestCase):
    def test_loads_normalized_yolo_labels_as_pixel_xyxy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label_path = Path(directory) / "frame.txt"
            label_path.write_text("2 0.5 0.5 0.4 0.2\n", encoding="utf-8")

            boxes, class_ids = load_yolo_labels(label_path, (100, 200))

        np.testing.assert_allclose(boxes, [[60, 40, 140, 60]])
        np.testing.assert_array_equal(class_ids, [2])

    def test_rejects_malformed_or_out_of_range_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label_path = Path(directory) / "frame.txt"
            for contents in ("0 0.5 0.5 0.2\n", "0 1.1 0.5 0.2 0.2\n", "x 0.5 0.5 0.2 0.2\n"):
                label_path.write_text(contents, encoding="utf-8")
                with self.subTest(contents=contents):
                    with self.assertRaises(ValueError):
                        load_yolo_labels(label_path, (100, 200))

    def test_empty_label_file_represents_a_negative_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label_path = Path(directory) / "negative.txt"
            label_path.write_text("", encoding="utf-8")

            boxes, class_ids = load_yolo_labels(label_path, (100, 200))

        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(class_ids.shape, (0,))


class YoloMetricsTests(unittest.TestCase):
    def test_iou_handles_overlapping_and_empty_boxes(self) -> None:
        overlaps = box_iou(
            np.array([0, 0, 10, 10]),
            np.array([[0, 0, 10, 10], [10, 10, 20, 20]]),
        )

        np.testing.assert_allclose(overlaps, [1.0, 0.0])
        self.assertEqual(box_iou(np.array([0, 0, 1, 1]), np.empty((0, 4))).shape, (0,))

    def test_reports_precision_recall_f1_and_map_for_matches_and_misses(self) -> None:
        ground_truth = [
            (
                np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32),
                np.array([0, 1]),
            )
        ]
        predictions = [
            (
                np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32),
                np.array([0, 0]),
                np.array([0.9, 0.8]),
            )
        ]

        metrics = evaluate_predictions(
            ground_truth,
            predictions,
            ["car", "person"],
            confidence_threshold=0.25,
        )

        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["map50"], 0.5)
        self.assertAlmostEqual(metrics["map50_95"], 0.5)
        self.assertAlmostEqual(metrics["per_class"]["car"]["map50"], 1.0)
        self.assertAlmostEqual(metrics["per_class"]["person"]["map50"], 0.0)

    def test_confidence_cutoff_applies_to_precision_and_recall(self) -> None:
        ground_truth = [(np.array([[0, 0, 10, 10]], dtype=np.float32), np.array([0]))]
        predictions = [
            (
                np.array([[50, 50, 60, 60], [0, 0, 10, 10]], dtype=np.float32),
                np.array([0, 0]),
                np.array([0.9, 0.2]),
            )
        ]

        metrics = evaluate_predictions(
            ground_truth,
            predictions,
            ["car"],
            confidence_threshold=0.25,
        )

        self.assertEqual(metrics["true_positives"], 0)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)


if __name__ == "__main__":
    unittest.main()
