from pathlib import Path
from typing import Any

import cv2
import numpy as np


class IoUTracker:
    def __init__(self, max_missed_frames: int = 12, minimum_iou: float = 0.25):
        self.max_missed_frames = max_missed_frames
        self.minimum_iou = minimum_iou
        self.next_track_id = 1
        self.tracks: dict[int, tuple[np.ndarray, int, int]] = {}

    @staticmethod
    def _iou(first: np.ndarray, second: np.ndarray) -> float:
        left = max(float(first[0]), float(second[0]))
        top = max(float(first[1]), float(second[1]))
        right = min(float(first[2]), float(second[2]))
        bottom = min(float(first[3]), float(second[3]))
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, float(first[2] - first[0])) * max(
            0.0, float(first[3] - first[1])
        )
        second_area = max(0.0, float(second[2] - second[0])) * max(
            0.0, float(second[3] - second[1])
        )
        union = first_area + second_area - intersection
        return intersection / union if union > 0.0 else 0.0

    def update(self, boxes: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
        for track_id, (box, class_id, missed) in list(self.tracks.items()):
            self.tracks[track_id] = (box, class_id, missed + 1)

        matches: list[tuple[float, int, int]] = []
        for detection_index, (box, class_id) in enumerate(zip(boxes, class_ids)):
            for track_id, (track_box, track_class, _) in self.tracks.items():
                if int(class_id) == track_class:
                    overlap = self._iou(box, track_box)
                    if overlap >= self.minimum_iou:
                        matches.append((overlap, detection_index, track_id))

        track_ids = np.full(len(boxes), -1, dtype=int)
        matched_detections: set[int] = set()
        matched_tracks: set[int] = set()
        for _, detection_index, track_id in sorted(matches, reverse=True):
            if detection_index in matched_detections or track_id in matched_tracks:
                continue
            track_ids[detection_index] = track_id
            self.tracks[track_id] = (
                boxes[detection_index].astype(float),
                int(class_ids[detection_index]),
                0,
            )
            matched_detections.add(detection_index)
            matched_tracks.add(track_id)

        for detection_index, box in enumerate(boxes):
            if detection_index in matched_detections:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            track_ids[detection_index] = track_id
            self.tracks[track_id] = (box.astype(float), int(class_ids[detection_index]), 0)

        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if track[2] <= self.max_missed_frames
        }
        return track_ids


class YOLO26Detector:
    TILE_FRACTION = 0.6
    TILE_SPLIT_THRESHOLD = 1.5

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        model_path = Path(model_path)
        if model_path.suffix.lower() != ".pt":
            raise ValueError(
                f"Expected an Ultralytics YOLO checkpoint (.pt), got {model_path}. "
                "Update YOLO_MODEL_PATH to a YOLO26 .pt checkpoint."
            )

        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            if exc.name != "ultralytics":
                raise
            raise RuntimeError(
                "Ultralytics is required for YOLO26. Install project dependencies "
                "with 'pip install -r requirements.txt'."
            ) from exc

        self.model = YOLO(str(model_path), task="detect")
        raw_names: Any = self.model.names
        if isinstance(raw_names, dict):
            self.names = [
                str(raw_names.get(index, raw_names.get(str(index), f"class_{index}")))
                for index in range(max((int(key) for key in raw_names), default=-1) + 1)
            ]
        else:
            self.names = [str(name) for name in raw_names]
        self.input_width = 640
        self.input_height = 640
        self.device = device
        self.tracker = IoUTracker()

    def infer(
        self, frame: np.ndarray, confidence_threshold: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        frame_height, frame_width = frame.shape[:2]
        all_boxes: list[np.ndarray] = []
        all_class_ids: list[np.ndarray] = []
        all_confidences: list[np.ndarray] = []

        for left, top, right, bottom in self._inference_regions(frame.shape[:2]):
            region = frame[top:bottom, left:right]
            results = self.model.predict(
                source=region,
                conf=float(confidence_threshold),
                imgsz=self.input_height,
                device=self.device,
                verbose=False,
            )
            if not results or results[0].boxes is None:
                continue
            detections = results[0].boxes
            boxes = detections.xyxy.cpu().numpy().astype(np.float32)
            class_ids = detections.cls.cpu().numpy().astype(int)
            confidences = detections.conf.cpu().numpy().astype(np.float32)
            if not len(boxes):
                continue

            boxes[:, [0, 2]] += left
            boxes[:, [1, 3]] += top
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_width)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_height)
            all_boxes.append(boxes)
            all_class_ids.append(class_ids)
            all_confidences.append(confidences)

        if all_boxes:
            boxes = np.concatenate(all_boxes)
            class_ids = np.concatenate(all_class_ids)
            confidences = np.concatenate(all_confidences)
            kept_indices = self._nms_indices(boxes, class_ids, confidences)
            boxes = boxes[kept_indices]
            class_ids = class_ids[kept_indices]
            confidences = confidences[kept_indices]
        else:
            boxes = np.empty((0, 4), dtype=int)
            class_ids = np.empty((0,), dtype=int)
            confidences = np.empty((0,), dtype=float)

        track_ids = self.tracker.update(boxes, class_ids)
        return boxes.astype(int), track_ids, class_ids, confidences, self.names

    def _inference_regions(
        self, image_shape: tuple[int, int]
    ) -> list[tuple[int, int, int, int]]:
        height, width = image_shape

        def axis_regions(length: int, input_length: int) -> tuple[int, list[int]]:
            if length <= input_length * self.TILE_SPLIT_THRESHOLD:
                return length, [0]
            tile_length = min(
                length,
                max(input_length, int(round(length * self.TILE_FRACTION))),
            )
            return tile_length, [0, length - tile_length]

        tile_width, x_starts = axis_regions(width, self.input_width)
        tile_height, y_starts = axis_regions(height, self.input_height)
        if len(x_starts) == 1 and len(y_starts) == 1:
            return [(0, 0, width, height)]

        regions = [(0, 0, width, height)]
        regions.extend(
            (left, top, left + tile_width, top + tile_height)
            for top in y_starts
            for left in x_starts
        )
        return regions

    @staticmethod
    def _nms_indices(
        boxes: np.ndarray,
        class_ids: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float = 0.45,
    ) -> np.ndarray:
        indices: list[int] = []
        for class_id in np.unique(class_ids):
            class_indices = np.flatnonzero(class_ids == class_id)
            boxes_for_nms = [
                [
                    float(boxes[index, 0]),
                    float(boxes[index, 1]),
                    float(boxes[index, 2] - boxes[index, 0]),
                    float(boxes[index, 3] - boxes[index, 1]),
                ]
                for index in class_indices
            ]
            keep = cv2.dnn.NMSBoxes(
                boxes_for_nms,
                scores[class_indices].tolist(),
                0.0,
                iou_threshold,
            )
            kept_local_indices = np.asarray(keep).reshape(-1).astype(int)
            indices.extend(class_indices[kept_local_indices].tolist())
        indices.sort(key=lambda index: float(scores[index]), reverse=True)
        return np.asarray(indices, dtype=np.int64)
