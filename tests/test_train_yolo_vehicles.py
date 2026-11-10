import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.train_yolo_vehicles import _validate_dataset


class VehicleTrainingDatasetTests(unittest.TestCase):
    def create_dataset(self, root: Path) -> Path:
        config = {
            "path": ".",
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": 1,
            "names": ["car"],
        }
        data_yaml = root / "data.yaml"
        data_yaml.write_text(yaml.safe_dump(config), encoding="utf-8")
        for split in ("train", "val", "test"):
            image_dir = root / split / "images"
            label_dir = root / split / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            (image_dir / "sample.jpg").write_bytes(b"image")
            (label_dir / "sample.txt").write_text(
                "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
            )
        return data_yaml

    def test_validates_all_splits_and_resolves_paths_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_yaml = self.create_dataset(Path(temporary_directory))

            counts = _validate_dataset(data_yaml)

        self.assertEqual(counts, {"train": 1, "val": 1, "test": 1})

    def test_rejects_images_without_label_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_yaml = self.create_dataset(Path(temporary_directory))
            (Path(temporary_directory) / "test" / "labels" / "sample.txt").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "missing YOLO labels"):
                _validate_dataset(data_yaml)


if __name__ == "__main__":
    unittest.main()
