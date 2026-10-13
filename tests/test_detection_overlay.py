import unittest

import numpy as np

from src.detection_overlay import draw_detection_overlay


class DetectionOverlayTests(unittest.TestCase):
    def test_draws_class_confidence_label_and_bounding_box(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        draw_detection_overlay(
            frame,
            (30, 30, 100, 90),
            "Car 0.93",
            "car",
            "Track #1",
        )

        self.assertTrue(np.any(np.all(frame == (0, 230, 0), axis=2)))
        self.assertTrue(np.any(np.all(frame == (255, 255, 255), axis=2)))
        self.assertGreater(int(frame[30, 60, 1]), 0)

    def test_clips_detection_box_to_frame(self) -> None:
        frame = np.zeros((40, 60, 3), dtype=np.uint8)

        draw_detection_overlay(frame, (-20, -10, 100, 80), "Person 0.87", "person")

        self.assertTrue(np.any(np.all(frame == (0, 0, 255), axis=2)))


if __name__ == "__main__":
    unittest.main()
