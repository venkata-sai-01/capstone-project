import ast
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


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


class ONNXObjectDetector:
    TILE_FRACTION = 0.6
    TILE_SPLIT_THRESHOLD = 1.5

    def __init__(self, model_path: str | Path):
        model_path = Path(model_path)
        if model_path.suffix.lower() != ".onnx":
            raise ValueError(
                f"Expected an ONNX model file, got {model_path}. Convert or provide "
                "a compatible YOLOv8 detection model with the .onnx extension."
            )
        if not model_path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}. Provide a YOLOv8 detection ONNX "
                "model at this path (input: NCHW RGB, output: YOLOv8 detection tensor)."
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 2) // 2))
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        model_input = self.session.get_inputs()[0]
        shape = model_input.shape
        self.input_name = model_input.name
        self.input_height = shape[2] if isinstance(shape[2], int) else 640
        self.input_width = shape[3] if isinstance(shape[3], int) else 640
        self.names = self._load_names()
        self.tracker = IoUTracker()

    def _load_names(self) -> list[str]:
        raw_names = self.session.get_modelmeta().custom_metadata_map.get("names")
        if raw_names:
            try:
                parsed: Any = json.loads(raw_names)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(raw_names)
                except (ValueError, SyntaxError):
                    parsed = None
            if isinstance(parsed, dict):
                try:
                    return [str(parsed.get(str(index), parsed.get(index))) for index in range(len(parsed))]
                except (TypeError, ValueError):
                    pass
            if isinstance(parsed, list) and parsed:
                return [str(name) for name in parsed]
        return COCO_CLASSES.copy()

    def infer(
        self, frame: np.ndarray, confidence_threshold: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        frame_height, frame_width = frame.shape[:2]
        all_boxes: list[np.ndarray] = []
        all_class_ids: list[np.ndarray] = []
        all_confidences: list[np.ndarray] = []
        for left, top, right, bottom in self._inference_regions(frame.shape[:2]):
            region = frame[top:bottom, left:right]
            input_tensor, scale, pad_x, pad_y = self._preprocess(region)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            raw_output = outputs[0]
            if not isinstance(raw_output, np.ndarray):
                raise TypeError(
                    f"Unsupported ONNX output type: {type(raw_output).__name__}; "
                    "expected a dense tensor."
                )
            boxes, class_ids, confidences = self._decode(
                raw_output,
                region.shape[:2],
                scale,
                pad_x,
                pad_y,
                confidence_threshold,
                len(self.names),
            )
            if len(boxes):
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
        return boxes, track_ids, class_ids, confidences, self.names

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

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = frame.shape[:2]
        scale = min(self.input_width / width, self.input_height / height)
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.input_width - resized_width) // 2
        pad_y = (self.input_height - resized_height) // 2
        canvas = np.full(
            (self.input_height, self.input_width, 3), 114, dtype=np.uint8
        )
        canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
        tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(tensor), scale, pad_x, pad_y

    @staticmethod
    def _decode(
        output: np.ndarray,
        image_shape: tuple[int, int],
        scale: float,
        pad_x: int,
        pad_y: int,
        confidence_threshold: float,
        class_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prediction = np.asarray(output)
        if prediction.ndim == 3:
            prediction = prediction[0]
        if prediction.ndim != 2:
            raise ValueError(f"Unsupported YOLO ONNX output shape: {prediction.shape}")

        expected_channels = class_count + 4
        if prediction.shape[1] == 6:
            pass
        elif prediction.shape[0] == 6 or prediction.shape[0] == expected_channels or (
            prediction.shape[0] < prediction.shape[1] and prediction.shape[0] <= 512
        ):
            prediction = prediction.T
        if prediction.shape[1] == 6:
            candidate_boxes = prediction[:, :4].astype(np.float32)
            scores = prediction[:, 4].astype(np.float32)
            class_ids = prediction[:, 5].astype(int)
        elif prediction.shape[1] >= 5:
            class_scores = prediction[:, 4:]
            class_ids = np.argmax(class_scores, axis=1).astype(int)
            scores = class_scores[np.arange(len(class_ids)), class_ids].astype(np.float32)
            center_x, center_y, width, height = prediction[:, :4].T
            candidate_boxes = np.column_stack(
                (
                    center_x - width / 2,
                    center_y - height / 2,
                    center_x + width / 2,
                    center_y + height / 2,
                )
            ).astype(np.float32)
        else:
            raise ValueError(f"Unsupported YOLO ONNX output shape: {prediction.shape}")

        valid = (
            np.isfinite(scores)
            & np.all(np.isfinite(candidate_boxes), axis=1)
            & (scores >= confidence_threshold)
            & (class_ids >= 0)
            & (class_ids < class_count)
            & (candidate_boxes[:, 2] > candidate_boxes[:, 0])
            & (candidate_boxes[:, 3] > candidate_boxes[:, 1])
        )
        candidate_boxes = candidate_boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]
        if not len(candidate_boxes):
            return (
                np.empty((0, 4), dtype=int),
                np.empty((0,), dtype=int),
                np.empty((0,), dtype=float),
            )

        kept_indices = ONNXObjectDetector._nms_indices(
            candidate_boxes, class_ids, scores
        )
        boxes = candidate_boxes[kept_indices]
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        image_height, image_width = image_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_height)
        return boxes.astype(int), class_ids[kept_indices], scores[kept_indices]

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
