import cv2
import numpy as np


CLASS_COLORS = {
    "person": (0, 0, 255),
    "car": (0, 230, 0),
    "truck": (255, 150, 0),
    "bus": (0, 165, 255),
    "motorcycle": (0, 220, 255),
    "bicycle": (220, 0, 220),
}


def detection_color(class_name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(class_name.lower(), (60, 210, 255))


def draw_detection_overlay(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    class_name: str,
    subtitle: str = "",
    color_override: tuple[int, int, int] | None = None,
) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1, min(width - 1, x2))
    y2 = max(y1, min(height - 1, y2))
    color = color_override or detection_color(class_name)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_top = max(0, y1 - text_size[1] - baseline - 8)
    label_right = min(width - 1, x1 + text_size[0] + 10)
    cv2.rectangle(frame, (x1, label_top), (label_right, y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 5, max(text_size[1], y1 - baseline - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if subtitle:
        cv2.putText(
            frame,
            subtitle,
            (x1, min(height - 2, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )
