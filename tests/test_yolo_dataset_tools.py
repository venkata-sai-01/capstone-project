import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.annotate_yolo import normalized_yolo_label
from scripts.prepare_yolo_dataset import sampled_frame_indices


class DatasetPreparationTests(unittest.TestCase):
    def test_samples_evenly_spaced_unique_frames(self) -> None:
        indices = sampled_frame_indices(101, 5)

        self.assertEqual(indices, [10, 30, 50, 70, 90])
        self.assertEqual(len(indices), len(set(indices)))

    def test_samples_unique_indices_near_video_length(self) -> None:
        indices = sampled_frame_indices(4, 3)

        self.assertEqual(indices, [0, 2, 3])
        self.assertEqual(len(indices), len(set(indices)))

    def test_limits_sample_count_to_short_video_length(self) -> None:
        self.assertEqual(sampled_frame_indices(3, 10), [0, 1, 2])

    def test_rejects_empty_video_or_sample_count(self) -> None:
        with self.assertRaises(ValueError):
            sampled_frame_indices(0, 5)
        with self.assertRaises(ValueError):
            sampled_frame_indices(5, 0)


class AnnotationTests(unittest.TestCase):
    def test_converts_dragged_box_to_normalized_yolo_format(self) -> None:
        label = normalized_yolo_label(2, (20, 10), (60, 50), (100, 100))

        self.assertEqual(label, "2 0.400000 0.300000 0.400000 0.400000")

    def test_clips_dragged_box_to_image_bounds(self) -> None:
        label = normalized_yolo_label(0, (-10, 10), (60, 120), (100, 100))

        self.assertEqual(label, "0 0.300000 0.550000 0.600000 0.900000")

    def test_rejects_zero_area_box(self) -> None:
        with self.assertRaises(ValueError):
            normalized_yolo_label(2, (10, 10), (10, 30), (100, 100))


if __name__ == "__main__":
    unittest.main()
