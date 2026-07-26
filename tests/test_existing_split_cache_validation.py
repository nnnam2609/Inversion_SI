from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.preprocessing.session_cache import assemble_split  # noqa: E402
from src.train.split_cache import assemble_split_direct, split_normalization_plan  # noqa: E402


def write_split_cache(cache_path: Path, std_value: float) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
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
        cache_path,
    )


def base_config(cache_dir: Path) -> dict:
    return {
        "split_cache_dir": str(cache_dir),
        "normalization_contour_std_floor": 0.1,
        "input_layer": 3,
        "sequence_length": 2,
        "classes": ["tongue"],
        "output_layer": 2,
        "test_sequences": {"P2": ["S1"]},
    }


class ExistingSplitCacheValidationTests(unittest.TestCase):
    def test_split_cache_defaults_to_train_global_unseen_speaker_plan(self) -> None:
        fit_splits, mode = split_normalization_plan({}, ("train_sequences", "valid_sequences", "test_sequences"))

        self.assertEqual(fit_splits, ("train_sequences",))
        self.assertEqual(mode, "train_global")

    def test_split_cache_all_splits_normalization_requires_explicit_mode(self) -> None:
        fit_splits, mode = split_normalization_plan(
            {"normalization_mode": "all_splits_global"},
            ("train_sequences", "valid_sequences", "test_sequences"),
        )

        self.assertEqual(fit_splits, ("train_sequences", "valid_sequences", "test_sequences"))
        self.assertEqual(mode, "all_splits_global")

    def test_split_cache_rejects_invalid_fit_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "normalization_fit_split"):
            split_normalization_plan(
                {
                    "normalization_mode": "train_global",
                    "normalization_fit_split": "dev_sequences",
                },
                ("train_sequences", "valid_sequences", "test_sequences"),
            )

    def test_split_cache_builder_reuses_valid_existing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "splits"
            write_split_cache(cache_dir / "test_sequences.pt", 0.1)

            result = assemble_split_direct(
                base_config(cache_dir),
                "test_sequences",
                norm_stats={},
                rebuild=False,
            )

        self.assertEqual(result["status"], "existing")
        self.assertTrue(result["cache_contour_std_floor_ok"])

    def test_split_cache_builder_rejects_stale_existing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "splits"
            write_split_cache(cache_dir / "test_sequences.pt", 1e-8)

            with self.assertRaisesRegex(RuntimeError, "below the configured normalization floor"):
                assemble_split_direct(
                    base_config(cache_dir),
                    "test_sequences",
                    norm_stats={},
                    rebuild=False,
                )

    def test_preprocess_assembler_rejects_stale_existing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "splits"
            write_split_cache(cache_dir / "test_sequences.pt", 1e-8)

            with self.assertRaisesRegex(RuntimeError, "below the configured normalization floor"):
                assemble_split(
                    base_config(cache_dir),
                    cache_dir,
                    "test_sequences",
                    rebuild=False,
                    norm_stats={},
                )


if __name__ == "__main__":
    unittest.main()
