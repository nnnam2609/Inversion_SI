from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.utils.split_cache_overrides import (
    feature_change_summary,
    feature_motion_summary,
    prepare_override_cache_dir,
    split_filename,
)


class SplitCacheOverrideTests(unittest.TestCase):
    def test_prepare_override_cache_dir_copies_support_and_non_target_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_dir = root / "base"
            output_dir = root / "out"
            base_dir.mkdir()
            for split in ("train_sequences", "valid_sequences", "test_sequences"):
                (base_dir / split_filename(split)).write_text(split, encoding="utf-8")
            (base_dir / "normalization_stats.npz").write_bytes(b"stats")
            (base_dir / "session_cache_validation.json").write_text("{}", encoding="utf-8")

            output_split = prepare_override_cache_dir(base_dir, output_dir, "test_sequences", force=False)

            self.assertEqual(output_split, output_dir / "test_sequences.pt")
            self.assertFalse((output_dir / "test_sequences.pt").exists())
            self.assertEqual((output_dir / "train_sequences.pt").read_text(encoding="utf-8"), "train_sequences")
            self.assertEqual((output_dir / "valid_sequences.pt").read_text(encoding="utf-8"), "valid_sequences")
            self.assertTrue((output_dir / "normalization_stats.npz").exists())
            self.assertTrue((output_dir / "session_cache_validation.json").exists())

    def test_prepare_override_cache_dir_refuses_existing_target_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_dir = root / "base"
            output_dir = root / "out"
            base_dir.mkdir()
            output_dir.mkdir()
            for split in ("train_sequences", "valid_sequences", "test_sequences"):
                (base_dir / split_filename(split)).write_text(split, encoding="utf-8")
            (output_dir / "test_sequences.pt").write_text("old", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "pass --force"):
                prepare_override_cache_dir(base_dir, output_dir, "test_sequences", force=False)

            self.assertEqual(
                prepare_override_cache_dir(base_dir, output_dir, "test_sequences", force=True),
                output_dir / "test_sequences.pt",
            )

    def test_feature_summaries(self) -> None:
        old = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        new = np.asarray([[1.0, 1.0], [4.0, 3.0]], dtype=np.float32)

        change = feature_change_summary(old, new)
        motion = feature_motion_summary(old, new)

        self.assertAlmostEqual(change["old_feature_mean"], 1.5)
        self.assertAlmostEqual(change["new_feature_mean"], 2.25)
        self.assertAlmostEqual(change["mean_abs_feature_delta"], 0.75)
        self.assertAlmostEqual(motion["old_feature_diff_mean_abs"], 2.0)
        self.assertAlmostEqual(motion["new_feature_diff_mean_abs"], 2.5)


if __name__ == "__main__":
    unittest.main()
