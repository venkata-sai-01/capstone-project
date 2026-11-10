import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO26n on the Indian Driving Dataset (IDD-D)."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "datasets/indian_driving_dataset_idd/"
            "IDDDetectionsYOLODataset/data.yaml"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("assets/models/yolo26n.pt"),
        help="Starting YOLO26n checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="0", help="CUDA device, or 'cpu'.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--project", type=Path, default=Path("runs/indian_vehicle_training")
    )
    parser.add_argument("--name", default="yolo26n")
    return parser


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_dataset(data_yaml: Path) -> dict[str, Any]:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid dataset YAML: {data_yaml}")
    names = config.get("names")
    if not isinstance(names, (list, dict)) or len(names) != config.get("nc"):
        raise ValueError("Dataset YAML must define matching 'nc' and 'names'.")

    configured_root = Path(config.get("path", "."))
    dataset_root = (
        configured_root
        if configured_root.is_absolute()
        else data_yaml.parent / configured_root
    ).resolve()
    counts = {}
    for split in ("train", "val", "test"):
        split_path = config.get(split)
        if not isinstance(split_path, str):
            raise ValueError(f"Dataset YAML must define the '{split}' image split.")
        image_dir = Path(split_path)
        if not image_dir.is_absolute():
            image_dir = dataset_root / image_dir
        label_dir = image_dir.parent / "labels"
        images = sorted(
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ) if image_dir.is_dir() else []
        if not images:
            raise FileNotFoundError(
                f"No {split} images found in {image_dir}. Download and extract "
                "the Kaggle dataset before training."
            )
        if not label_dir.is_dir():
            raise FileNotFoundError(f"{split} labels directory not found: {label_dir}")
        missing_labels = [
            image.relative_to(image_dir).with_suffix(".txt")
            for image in images
            if not (label_dir / image.relative_to(image_dir).with_suffix(".txt")).is_file()
        ]
        if missing_labels:
            raise FileNotFoundError(
                f"{len(missing_labels)} {split} image(s) are missing YOLO labels; "
                f"first missing label: {label_dir / missing_labels[0]}"
            )
        counts[split] = len(images)
    return counts


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.epochs < 1 or args.imgsz < 1 or args.workers < 0 or args.patience < 0:
        parser.error("--epochs and --imgsz must be positive; workers/patience non-negative.")
    if args.batch != -1 and args.batch < 1:
        parser.error("--batch must be -1 (auto) or a positive integer.")
    if args.device.lower() != "cpu" and not torch.cuda.is_available():
        parser.error(
            "CUDA is not available in this Python environment. Install a "
            "CUDA-enabled PyTorch build or pass --device cpu."
        )

    data_yaml = _project_path(args.data).resolve()
    model_path = _project_path(args.model).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Starting model checkpoint not found: {model_path}")
    counts = _validate_dataset(data_yaml)
    print(
        "Training on "
        + ", ".join(f"{split}={count}" for split, count in counts.items())
        + " images."
    )

    model = YOLO(str(model_path))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        project=str(_project_path(args.project)),
        name=args.name,
    )


if __name__ == "__main__":
    main()
