from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.utils.gridnorm_rendering import mri_for_frame, transform_predictions, write_mode_contours


class GridnormRenderingTests(unittest.TestCase):
    def test_transform_predictions_raw_copies_predicted_into_mode_prediction(self) -> None:
        predicted = np.arange(2 * 100, dtype=np.float32).reshape(2, 100)
        rows = [{"predicted": predicted, "held": False, "frame": "0001", "frame_number": 1.0}]

        transformed = transform_predictions(rows, transform=None, mode="raw", class_count=2)

        self.assertIsNot(transformed[0], rows[0])
        np.testing.assert_allclose(transformed[0]["mode_prediction"], predicted)

    def test_transform_predictions_rejects_unknown_mode(self) -> None:
        rows = [{"predicted": np.zeros((1, 100), dtype=np.float32)}]

        with self.assertRaisesRegex(ValueError, "Unsupported mode"):
            transform_predictions(rows, transform=None, mode="bad", class_count=1)

    def test_write_mode_contours_skips_held_frames(self) -> None:
        rows = [
            {
                "mode_prediction": np.zeros((1, 100), dtype=np.float32),
                "held": False,
                "frame": "0001",
            },
            {
                "mode_prediction": np.ones((1, 100), dtype=np.float32),
                "held": True,
                "frame": "0002",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_mode_contours(rows, ["tongue"], output_dir)
            files = sorted(path.name for path in output_dir.glob("*.npy"))

        self.assertEqual(files, ["0001_tongue.npy"])

    def test_write_mode_contours_skips_missing_nan_contour(self) -> None:
        rows = [
            {
                "mode_prediction": np.stack(
                    [np.zeros(100, dtype=np.float32), np.full(100, np.nan, dtype=np.float32)]
                ),
                "held": False,
                "frame": "0001",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_mode_contours(rows, ["tongue", "upper-incisor"], output_dir)
            files = sorted(path.name for path in output_dir.glob("*.npy"))

        self.assertEqual(files, ["0001_tongue.npy"])

    def test_mri_for_frame_rejects_half_frames(self) -> None:
        cache = {
            1: np.zeros((2, 2), dtype=np.uint8),
            2: np.full((2, 2), 10, dtype=np.uint8),
        }

        with self.assertRaisesRegex(ValueError, "NEVER render fractional"):
            mri_for_frame(1.5, cache)


if __name__ == "__main__":
    unittest.main()
