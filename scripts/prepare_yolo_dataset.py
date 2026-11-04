import argparse
import re
from pathlib import Path

import cv2


def sampled_frame_indices(frame_count: int, requested_count: int) -> list[int]:
    if frame_count < 1:
        raise ValueError("Video must contain at least one frame.")
    if requested_count < 1:
        raise ValueError("Requested frame count must be at least one.")
    count = min(frame_count, requested_count)
    return [
        ((2 * index + 1) * frame_count) // (2 * count)
        for index in range(count)
    ]


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_").lower()
    return stem or "video"


def extract_video_frames(
    video_path: Path,
    split: str,
    output_root: Path,
    requested_count: int,
) -> int:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sampled_frame_indices(frame_count, requested_count)
        output_dir = output_root / "images" / split
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = 0

        for sample_number, frame_index in enumerate(indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success or frame is None:
                raise ValueError(
                    f"Could not read frame {frame_index} from {video_path}."
                )
            filename = (
                f"{_safe_stem(video_path)}_frame_{sample_number:04d}.jpg"
            )
            output_path = output_dir / filename
            if output_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing frame: {output_path}"
                )
            if not cv2.imwrite(str(output_path), frame):
                raise OSError(f"Could not write frame: {output_path}")
            saved += 1
        return saved
    finally:
        capture.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample video frames into train/val/test image folders. "
            "Assign whole videos to splits to avoid frame leakage."
        )
    )
    parser.add_argument("--train-video", type=Path, action="append", default=[])
    parser.add_argument("--val-video", type=Path, action="append", default=[])
    parser.add_argument("--test-video", type=Path, action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/traffic"),
        help="Dataset root containing images/{train,val,test} and labels/{train,val,test}.",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=20,
        help="Approximately evenly-spaced frames to sample from each video.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    splits = {
        "train": args.train_video,
        "val": args.val_video,
        "test": args.test_video,
    }
    if not args.test_video:
        parser.error("Provide at least one held-out video with --test-video.")
    if args.frames_per_video < 1:
        parser.error("--frames-per-video must be at least 1.")

    total = 0
    for split, videos in splits.items():
        for video in videos:
            count = extract_video_frames(
                video, split, args.output, args.frames_per_video
            )
            total += count
            print(f"{split}: sampled {count} frames from {video.name}")
    print(
        f"Prepared {total} images under {args.output}. "
        "Annotate every image before running evaluation."
    )


if __name__ == "__main__":
    main()
