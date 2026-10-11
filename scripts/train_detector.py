import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def _metric_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    elif hasattr(value, "item"):
        value = value.item()
    if isinstance(value, tuple):
        return [_metric_value(item) for item in value]
    if isinstance(value, list):
        return [_metric_value(item) for item in value]
    if isinstance(value, (int, float)):
        return float(value)
    return value


def train_detector(
    data: Path,
    weights: str,
    epochs: int,
    image_size: int,
    batch_size: int,
    device: str,
    workers: int,
    project: Path,
    name: str,
    evaluation_split: str,
) -> Path:
    data = data.resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data}")
    if epochs < 1 or image_size < 32 or (batch_size < 1 and batch_size != -1) or workers < 0:
        raise ValueError(
            "epochs must be positive, image size must be at least 32, batch size must be "
            "positive or -1, and workers cannot be negative."
        )

    model = YOLO(weights)
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=workers,
        project=str(project),
        name=name,
        exist_ok=False,
    )
    trainer = model.trainer
    if trainer is None:
        raise RuntimeError("YOLO training completed without exposing its trainer output.")
    save_dir = trainer.save_dir
    if save_dir is None:
        raise RuntimeError("YOLO training completed without exposing its run directory.")
    run_directory = Path(save_dir)
    best_weights = run_directory / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training finished without producing best weights: {best_weights}")

    trained_model = YOLO(str(best_weights))
    validation = trained_model.val(
        data=str(data),
        split=evaluation_split,
        imgsz=image_size,
        batch=batch_size,
        device=device,
    )
    metrics = {
        name: _metric_value(value)
        for name, value in validation.results_dict.items()
    }
    metrics["timing_ms_per_image"] = {
        name: _metric_value(value)
        for name, value in validation.speed.items()
    }
    metrics_path = run_directory / f"{evaluation_split}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "weights": str(best_weights),
                "dataset": str(data),
                "evaluation_split": evaluation_split,
                "metrics": metrics,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune an Ultralytics YOLO detector and save split metrics."
    )
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML.")
    parser.add_argument("--weights", default="yolov8n.pt", help="Pretrained checkpoint or model file.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="Use -1 for automatic batch sizing.")
    parser.add_argument("--device", default="cpu", help="cpu, 0, or another Ultralytics device selector.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", default="traffic-safety")
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    args = parser.parse_args()

    try:
        metrics_path = train_detector(
            args.data,
            args.weights,
            args.epochs,
            args.imgsz,
            args.batch,
            args.device,
            args.workers,
            args.project,
            args.name,
            args.eval_split,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Training complete. Evaluation metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
