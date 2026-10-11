import argparse
import json
import math
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _label_file(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    return labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")


def _validate_label_file(path: Path, class_count: int) -> None:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"{path}:{line_number}: expected a detection label with 5 values "
                "(class x_center y_center width height)."
            )
        try:
            class_value = float(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: labels must contain numeric values.") from exc
        if not class_value.is_integer() or not 0 <= int(class_value) < class_count:
            raise ValueError(
                f"{path}:{line_number}: class id must be an integer from 0 to {class_count - 1}."
            )
        if any(not 0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"{path}:{line_number}: normalized coordinates must be between 0 and 1.")
        if coordinates[2] == 0.0 or coordinates[3] == 0.0:
            raise ValueError(f"{path}:{line_number}: bounding-box width and height must be positive.")
        x_center, y_center, width, height = coordinates
        if (
            x_center - width / 2 < 0.0
            or x_center + width / 2 > 1.0
            or y_center - height / 2 < 0.0
            or y_center + height / 2 > 1.0
        ):
            raise ValueError(f"{path}:{line_number}: bounding box must fit within the image.")


def _yaml_string(value: str) -> str:
    return json.dumps(value)


def prepare_dataset(
    source: Path,
    output: Path,
    classes: list[str],
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Path:
    source = source.resolve()
    output = output.resolve()
    images_dir = source / "images"
    labels_dir = source / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError("Source must contain images/ and labels/ directories.")
    classes = [name.strip() for name in classes]
    if not classes or any(not name for name in classes):
        raise ValueError("Provide at least one non-empty class name.")
    if len(set(classes)) != len(classes):
        raise ValueError("Class names must be unique.")
    if (
        not math.isfinite(val_ratio)
        or not math.isfinite(test_ratio)
        or not 0.0 < val_ratio < 1.0
        or not 0.0 < test_ratio < 1.0
    ):
        raise ValueError("Validation and test ratios must each be between 0 and 1.")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("Validation and test ratios must sum to less than 1.")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output directories must not contain one another.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")

    images = sorted(
        image for image in images_dir.rglob("*")
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(images) < 3:
        raise ValueError("At least three annotated images are required to create train/val/test splits.")

    label_paths = []
    for image in images:
        label = _label_file(image, images_dir, labels_dir)
        if not label.is_file():
            raise ValueError(f"Missing YOLO label file for image: {image}")
        _validate_label_file(label, len(classes))
        label_paths.append(label)

    expected_labels = {_label_file(image, images_dir, labels_dir).resolve() for image in images}
    unexpected_labels = [
        path for path in labels_dir.rglob("*.txt")
        if path.resolve() not in expected_labels
    ]
    if unexpected_labels:
        raise ValueError(f"Label file has no matching image: {unexpected_labels[0]}")

    shuffled_indices = list(range(len(images)))
    random.Random(seed).shuffle(shuffled_indices)
    test_count = max(1, round(len(images) * test_ratio))
    val_count = max(1, round(len(images) * val_ratio))
    while test_count + val_count >= len(images):
        if val_count >= test_count:
            val_count -= 1
        else:
            test_count -= 1
    test_indices = set(shuffled_indices[:test_count])
    val_indices = set(shuffled_indices[test_count:test_count + val_count])

    for index, (image, label) in enumerate(zip(images, label_paths)):
        split = "test" if index in test_indices else "val" if index in val_indices else "train"
        relative_image = image.relative_to(images_dir)
        for root, path in (("images", image), ("labels", label)):
            target = output / root / split / relative_image.with_suffix(path.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    yaml_lines = [
        f"path: {_yaml_string(str(output))}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(classes)}",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {_yaml_string(name)}" for index, name in enumerate(classes))
    output.mkdir(parents=True, exist_ok=True)
    yaml_path = output / "dataset.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate YOLO detection labels and create reproducible train/val/test splits."
    )
    parser.add_argument("--source", type=Path, required=True, help="Folder containing images/ and labels/.")
    parser.add_argument("--output", type=Path, required=True, help="New output folder for the split dataset.")
    parser.add_argument(
        "--classes", nargs="+", required=True,
        help="Class names in the same order as the numeric IDs in the YOLO labels.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        dataset_yaml = prepare_dataset(
            args.source, args.output, args.classes, args.val_ratio, args.test_ratio, args.seed
        )
    except (FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Prepared YOLO dataset configuration: {dataset_yaml}")


if __name__ == "__main__":
    main()
