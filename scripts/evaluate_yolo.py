import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.onnx_detector import ONNXObjectDetector


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IOU_THRESHOLDS = tuple(round(0.5 + index * 0.05, 2) for index in range(10))


def load_yolo_labels(label_path: Path, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    boxes: list[list[float]] = []
    class_ids: list[int] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if len(values) != 5:
            raise ValueError(
                f"{label_path}:{line_number}: expected class_id x_center y_center width height."
            )
        try:
            class_value, center_x, center_y, box_width, box_height = map(float, values)
        except ValueError as error:
            raise ValueError(f"{label_path}:{line_number}: label values must be numeric.") from error
        if not all(math.isfinite(value) for value in (class_value, center_x, center_y, box_width, box_height)):
            raise ValueError(f"{label_path}:{line_number}: label values must be finite.")
        if (
            class_value < 0
            or not class_value.is_integer()
            or not 0.0 <= center_x <= 1.0
            or not 0.0 <= center_y <= 1.0
            or not 0.0 < box_width <= 1.0
            or not 0.0 < box_height <= 1.0
            or center_x - box_width / 2 < 0.0
            or center_y - box_height / 2 < 0.0
            or center_x + box_width / 2 > 1.0
            or center_y + box_height / 2 > 1.0
        ):
            raise ValueError(f"{label_path}:{line_number}: invalid normalized YOLO label.")

        class_ids.append(int(class_value))
        boxes.append(
            [
                (center_x - box_width / 2) * width,
                (center_y - box_height / 2) * height,
                (center_x + box_width / 2) * width,
                (center_y + box_height / 2) * height,
            ]
        )

    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(class_ids, dtype=np.int64)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty((0,), dtype=np.float32)
    box = np.asarray(box, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0), axis=1)
    box_area = np.prod(np.maximum(box[2:] - box[:2], 0))
    boxes_area = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    union = box_area + boxes_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    if not len(recall):
        return 0.0
    interpolated = [
        float(np.max(precision[recall >= point], initial=0.0))
        for point in np.linspace(0.0, 1.0, 101)
    ]
    return float(np.mean(interpolated))


def _class_average_precision(
    class_id: int,
    iou_threshold: float,
    ground_truth: list[tuple[np.ndarray, np.ndarray]],
    ranked_predictions: list[tuple[float, int, np.ndarray]],
) -> float:
    ground_truth_count = sum(int(np.count_nonzero(classes == class_id)) for _, classes in ground_truth)
    if ground_truth_count == 0:
        return 0.0

    matched = [np.zeros(len(classes), dtype=bool) for _, classes in ground_truth]
    true_positives = np.zeros(len(ranked_predictions), dtype=np.float64)
    false_positives = np.zeros(len(ranked_predictions), dtype=np.float64)
    for index, (_, image_index, predicted_box) in enumerate(ranked_predictions):
        boxes, classes = ground_truth[image_index]
        candidate_indices = np.flatnonzero((classes == class_id) & ~matched[image_index])
        if len(candidate_indices):
            overlaps = box_iou(predicted_box, boxes[candidate_indices])
            best_candidate = int(np.argmax(overlaps))
            if overlaps[best_candidate] >= iou_threshold:
                matched[image_index][candidate_indices[best_candidate]] = True
                true_positives[index] = 1.0
                continue
        false_positives[index] = 1.0

    if not ranked_predictions:
        return 0.0
    cumulative_true = np.cumsum(true_positives)
    cumulative_false = np.cumsum(false_positives)
    recall = cumulative_true / ground_truth_count
    precision = cumulative_true / np.maximum(cumulative_true + cumulative_false, 1e-12)
    return _average_precision(recall, precision)


def evaluate_predictions(
    ground_truth: list[tuple[np.ndarray, np.ndarray]],
    predictions: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    class_names: list[str],
    confidence_threshold: float = 0.25,
    precision_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    if len(ground_truth) != len(predictions):
        raise ValueError("Ground-truth and prediction image counts must match.")
    if not ground_truth:
        raise ValueError("The evaluation dataset contains no images.")
    class_ids = sorted(
        {int(class_id) for _, classes in ground_truth for class_id in classes.tolist()}
    )
    if not class_ids:
        raise ValueError("The evaluation dataset contains no labeled objects.")
    if class_ids[-1] >= len(class_names):
        raise ValueError(
            f"Label class {class_ids[-1]} exceeds the model's {len(class_names)} class names."
        )

    ranked_by_class: dict[int, list[tuple[float, int, np.ndarray]]] = {
        class_id: [] for class_id in class_ids
    }
    for image_index, (boxes, predicted_classes, scores) in enumerate(predictions):
        for box, predicted_class, score in zip(boxes, predicted_classes, scores):
            class_predictions = ranked_by_class.get(int(predicted_class))
            if class_predictions is not None:
                class_predictions.append((float(score), image_index, box))
    for class_predictions in ranked_by_class.values():
        class_predictions.sort(key=lambda item: item[0], reverse=True)

    total_true = total_false = total_ground_truth = 0
    for (truth_boxes, truth_classes), (pred_boxes, pred_classes, scores) in zip(
        ground_truth, predictions
    ):
        total_ground_truth += len(truth_boxes)
        eligible = np.flatnonzero(scores >= confidence_threshold)
        eligible = eligible[np.argsort(scores[eligible])[::-1]]
        matched = np.zeros(len(truth_boxes), dtype=bool)
        for prediction_index in eligible:
            same_class = np.flatnonzero(
                (truth_classes == pred_classes[prediction_index]) & ~matched
            )
            if len(same_class):
                overlaps = box_iou(pred_boxes[prediction_index], truth_boxes[same_class])
                best_candidate = int(np.argmax(overlaps))
                if overlaps[best_candidate] >= precision_iou_threshold:
                    matched[same_class[best_candidate]] = True
                    total_true += 1
                    continue
            total_false += 1

    total_missed = total_ground_truth - total_true
    precision = total_true / (total_true + total_false) if total_true + total_false else 0.0
    recall = total_true / total_ground_truth
    aps = {
        class_id: {
            threshold: _class_average_precision(
                class_id,
                threshold,
                ground_truth,
                ranked_by_class[class_id],
            )
            for threshold in IOU_THRESHOLDS
        }
        for class_id in class_ids
    }
    per_class = {
        class_names[class_id]: {
            "ground_truth_objects": sum(
                int(np.count_nonzero(classes == class_id)) for _, classes in ground_truth
            ),
            "map50": aps[class_id][0.5],
            "map50_95": float(np.mean(list(aps[class_id].values()))),
        }
        for class_id in class_ids
    }
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "map50": float(np.mean([aps[class_id][0.5] for class_id in class_ids])),
        "map50_95": float(
            np.mean([np.mean(list(aps[class_id].values())) for class_id in class_ids])
        ),
        "true_positives": total_true,
        "false_positives": total_false,
        "false_negatives": total_missed,
        "per_class": per_class,
    }


def collect_dataset(
    image_directory: Path, label_directory: Path
) -> list[tuple[Path, Path]]:
    image_paths = sorted(
        path
        for path in image_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported images found under {image_directory}.")
    samples = []
    for image_path in image_paths:
        label_path = label_directory / image_path.relative_to(image_directory).with_suffix(".txt")
        if not label_path.is_file():
            raise FileNotFoundError(
                f"Missing YOLO label file for {image_path}: expected {label_path}. "
                "Create an empty .txt file for images with no labeled objects."
            )
        samples.append((image_path, label_path))
    return samples


def benchmark_model(
    model_path: Path,
    samples: list[tuple[Path, Path]],
    confidence_threshold: float,
    metric_confidence: float,
    warmup_count: int,
    repeats: int,
) -> tuple[dict[str, Any], list[str], int]:
    load_start = time.perf_counter()
    detector = ONNXObjectDetector(model_path)
    model_load_seconds = time.perf_counter() - load_start
    names = detector.names
    ground_truth = []
    predictions = []
    latencies_ms = []

    for image_index, (image_path, label_path) in enumerate(samples):
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Could not read evaluation image: {image_path}")
        truth = load_yolo_labels(label_path, frame.shape[:2])
        if image_index < warmup_count:
            detector.infer(frame, confidence_threshold)

        measured = None
        for _ in range(repeats):
            start = time.perf_counter()
            measured = detector.infer(frame, confidence_threshold)
            latencies_ms.append((time.perf_counter() - start) * 1000)
        if measured is None:
            raise RuntimeError("Model inference did not produce a result.")
        boxes, _, class_ids, scores, _ = measured
        ground_truth.append(truth)
        predictions.append((boxes.astype(np.float32), class_ids, scores))

    metrics = evaluate_predictions(
        ground_truth,
        predictions,
        names,
        confidence_threshold=metric_confidence,
    )
    model_result = {
        "model": str(model_path),
        "model_load_seconds": model_load_seconds,
        "latency_ms": {
            "mean": statistics.fmean(latencies_ms),
            "median": statistics.median(latencies_ms),
            "p95": float(np.percentile(latencies_ms, 95)),
        },
        "mean_fps": 1000 / statistics.fmean(latencies_ms),
        **metrics,
    }
    return model_result, names, sum(len(boxes) for boxes, _ in ground_truth)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare compatible YOLO ONNX detectors against images with standard "
            "YOLO-format labels."
        )
    )
    parser.add_argument("--images", type=Path, required=True, help="Root directory of evaluation images.")
    parser.add_argument("--labels", type=Path, required=True, help="Mirrored root directory of YOLO .txt labels.")
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        required=True,
        help="YOLO detection ONNX model; repeat this option to compare models.",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/yolo-evaluation.json"))
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.001,
        help="Lowest confidence retained during inference and mAP calculation.",
    )
    parser.add_argument(
        "--metric-confidence",
        type=float,
        default=0.25,
        help="Confidence cutoff for precision, recall, and F1.",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Initial dataset images used for warmup.")
    parser.add_argument("--repeats", type=int, default=1, help="Measured inferences per image.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be non-negative and --repeats must be at least 1.")
    if not 0.0 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence must be between 0 and 1.")
    if not args.min_confidence <= args.metric_confidence <= 1.0:
        parser.error("--metric-confidence must be between --min-confidence and 1.")

    samples = collect_dataset(args.images, args.labels)
    results = []
    reference_names: list[str] | None = None
    labeled_objects = 0
    for model_path in args.model:
        result, model_names, model_labeled_objects = benchmark_model(
            model_path,
            samples,
            args.min_confidence,
            args.metric_confidence,
            min(args.warmup, len(samples)),
            args.repeats,
        )
        if reference_names is None:
            reference_names = model_names
            labeled_objects = model_labeled_objects
        elif model_names != reference_names:
            raise ValueError(
                "Compared models do not have the same class names in the same order."
            )
        results.append(result)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "images": len(samples),
            "labeled_objects": labeled_objects,
        },
        "settings": {
            "minimum_inference_confidence": args.min_confidence,
            "metric_confidence": args.metric_confidence,
            "precision_recall_iou": 0.5,
            "map_iou_thresholds": list(IOU_THRESHOLDS),
            "warmup_images": min(args.warmup, len(samples)),
            "repeats_per_image": args.repeats,
        },
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved evaluation report to {args.output}")


if __name__ == "__main__":
    main()
