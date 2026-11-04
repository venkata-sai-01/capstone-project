import argparse
from pathlib import Path

import cv2
import numpy as np

from scripts.evaluate_yolo import IMAGE_EXTENSIONS, load_yolo_labels


TRAFFIC_CLASSES = [
    (0, "person"),
    (1, "bicycle"),
    (2, "car"),
    (3, "motorcycle"),
    (5, "bus"),
    (7, "truck"),
]
WINDOW_NAME = "YOLO Ground-Truth Annotator"


def normalized_yolo_label(
    class_id: int,
    first_point: tuple[int, int],
    second_point: tuple[int, int],
    image_size: tuple[int, int],
) -> str:
    width, height = image_size
    x1, x2 = sorted((max(0, min(width, first_point[0])), max(0, min(width, second_point[0]))))
    y1, y2 = sorted((max(0, min(height, first_point[1])), max(0, min(height, second_point[1]))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("Bounding box must be at least 2 pixels wide and high.")
    center_x = (x1 + x2) / (2 * width)
    center_y = (y1 + y2) / (2 * height)
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return (
        f"{class_id} {center_x:.6f} {center_y:.6f} "
        f"{box_width:.6f} {box_height:.6f}"
    )


def _label_path(image_path: Path, images_root: Path, labels_root: Path) -> Path:
    return labels_root / image_path.relative_to(images_root).with_suffix(".txt")


def _draw_box(
    image: np.ndarray,
    box: np.ndarray,
    class_id: int,
    scale_x: float,
    scale_y: float,
) -> None:
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
    class_name = dict(TRAFFIC_CLASSES).get(class_id, f"class {class_id}")
    cv2.rectangle(image, (x1, y1), (x2, y2), (60, 220, 60), 2)
    cv2.putText(
        image,
        class_name,
        (x1, max(18, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (60, 220, 60),
        2,
    )


def annotate_dataset(images_root: Path, labels_root: Path) -> None:
    image_paths = sorted(
        path
        for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported images found under {images_root}.")

    state: dict[str, object] = {
        "drawing": False,
        "start": (0, 0),
        "current": (0, 0),
        "boxes": [],
        "class_index": 0,
    }

    def mouse_callback(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"] = (x, y)
            state["current"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["current"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["current"] = (x, y)
            state["drawing"] = False
            start = state["start"]
            current = state["current"]
            class_index = int(state["class_index"])
            boxes = state["boxes"]
            if isinstance(start, tuple) and isinstance(current, tuple) and isinstance(boxes, list):
                if abs(current[0] - start[0]) >= 2 and abs(current[1] - start[1]) >= 2:
                    boxes.append((TRAFFIC_CLASSES[class_index][0], start, current))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    index = 0
    try:
        while index < len(image_paths):
            image_path = image_paths[index]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            original_height, original_width = image.shape[:2]
            scale = min(1280 / original_width, 780 / original_height, 1.0)
            display_width = max(1, round(original_width * scale))
            display_height = max(1, round(original_height * scale))
            scale_x = display_width / original_width
            scale_y = display_height / original_height
            label_path = _label_path(image_path, images_root, labels_root)
            if label_path.is_file():
                boxes, class_ids = load_yolo_labels(
                    label_path, (original_height, original_width)
                )
                annotations = [
                    (
                        int(class_id),
                        (
                            int(round(box[0] * scale_x)),
                            int(round(box[1] * scale_y)),
                        ),
                        (
                            int(round(box[2] * scale_x)),
                            int(round(box[3] * scale_y)),
                        ),
                    )
                    for box, class_id in zip(boxes, class_ids)
                ]
            else:
                annotations = []
            state["boxes"] = annotations

            while True:
                canvas = cv2.resize(image, (display_width, display_height))
                boxes = state["boxes"]
                if isinstance(boxes, list):
                    for class_id, first, second in boxes:
                        x1, x2 = sorted((int(first[0]), int(second[0])))
                        y1, y2 = sorted((int(first[1]), int(second[1])))
                        _draw_box(
                            canvas,
                            np.array([x1, y1, x2, y2]),
                            int(class_id),
                            1.0,
                            1.0,
                        )
                if state["drawing"]:
                    start = state["start"]
                    current = state["current"]
                    if isinstance(start, tuple) and isinstance(current, tuple):
                        cv2.rectangle(canvas, start, current, (0, 220, 255), 2)

                class_index = int(state["class_index"])
                class_id, class_name = TRAFFIC_CLASSES[class_index]
                status = (
                    f"{index + 1}/{len(image_paths)}  {image_path.name}  "
                    f"Class: {class_name} (key {class_index + 1}, ID {class_id})"
                )
                cv2.putText(
                    canvas, status, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )
                cv2.putText(
                    canvas, "1-6 class | drag box | z undo | c clear | s save/next | b save/back | q quit",
                    (12, display_height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1
                )
                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    return
                if ord("1") <= key <= ord("6"):
                    state["class_index"] = key - ord("1")
                elif key == ord("z"):
                    active_boxes = state["boxes"]
                    if isinstance(active_boxes, list) and active_boxes:
                        active_boxes.pop()
                elif key == ord("c"):
                    state["boxes"] = []
                elif key in (ord("s"), ord("b")):
                    output_path = _label_path(image_path, images_root, labels_root)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    boxes_to_save = state["boxes"]
                    if not isinstance(boxes_to_save, list):
                        raise RuntimeError("Annotation state is invalid.")
                    lines = []
                    for class_id, first, second in boxes_to_save:
                        display_label = normalized_yolo_label(
                            int(class_id),
                            first,
                            second,
                            (display_width, display_height),
                        )
                        lines.append(display_label)
                    output_path.write_text(
                        "\n".join(lines) + ("\n" if lines else ""),
                        encoding="utf-8",
                    )
                    index = max(0, index - 1) if key == ord("b") else index + 1
                    break
    finally:
        cv2.destroyWindow(WINDOW_NAME)
    print(f"Finished reviewing images. Labels are in {labels_root}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw human-reviewed YOLO labels for sampled traffic images."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help=r"Image root, such as datasets\traffic\images\test.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help=r"Mirrored label root, such as datasets\traffic\labels\test.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    annotate_dataset(args.images, args.labels)


if __name__ == "__main__":
    main()
