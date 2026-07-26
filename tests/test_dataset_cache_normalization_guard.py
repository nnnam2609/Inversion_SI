from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.utils.datasets import _load_or_build_contour_dataset  # noqa: E402


class DatasetCacheNormalizationGuardTests(unittest.TestCase):
    def write_split_cache(self, cache_dir: Path, std_value: float) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": torch.zeros((1, 2, 3), dtype=torch.float32),
                "labels": torch.zeros((1, 2, 1, 2), dtype=torch.float32),
                "frames": torch.zeros((1, 2, 3), dtype=torch.float32),
                "phonemes": torch.zeros((1, 2, 4), dtype=torch.float32),
                "std": torch.full((1, 1, 1, 2), std_value, dtype=torch.float32),
                "mean": torch.zeros((1, 1, 1, 2), dtype=torch.float32),
                "mean_datas": torch.zeros((1, 2), dtype=torch.float32),
                "length_datas": [2],
                "sequences_length": [2],
            },
            cache_dir / "test_sequences.pt",
        )

    def base_config(self, cache_dir: Path) -> dict:
        return {
            "session_cache_dir": str(cache_dir.parent / "raw_sessions"),
            "split_cache_dir": str(cache_dir),
            "normalization_contour_std_floor": 0.1,
            "_split_cache_ready": True,
        }

    def test_training_dataset_loader_accepts_current_std_floor_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "splits"
            self.write_split_cache(cache_dir, 0.1)

            dataset = _load_or_build_contour_dataset(
                self.base_config(cache_dir),
                "test_sequences",
                rank=0,
                world_size=1,
            )

        self.assertEqual(len(dataset), 1)

    def test_training_dataset_loader_rejects_stale_low_std_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "splits"
            self.write_split_cache(cache_dir, 1e-8)

            with self.assertRaisesRegex(RuntimeError, "below the configured normalization floor"):
                _load_or_build_contour_dataset(
                    self.base_config(cache_dir),
                    "test_sequences",
                    rank=0,
                    world_size=1,
                )


if __name__ == "__main__":
    unittest.main()
