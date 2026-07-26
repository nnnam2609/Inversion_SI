from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.utils.video_rendering import (
    MM_PER_PIXEL,
    draw_dashed_polyline,
    mean_finite,
    rgb_to_bgr255,
    rmse_px,
    scale_points,
)


class VideoRenderingTests(unittest.TestCase):
    def test_mean_finite_ignores_none_and_nan(self) -> None:
        self.assertAlmostEqual(mean_finite([1.0, None, float("nan"), 3.0]), 2.0)
        self.assertTrue(math.isnan(mean_finite([None, float("nan")])))

    def test_rmse_px_and_scale_constant(self) -> None:
        self.assertEqual(MM_PER_PIXEL, 1.62)
        first = np.asarray([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)
        second = np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        self.assertAlmostEqual(rmse_px(first, second), math.sqrt(2.0))

    def test_color_and_point_helpers(self) -> None:
        self.assertEqual(rgb_to_bgr255("red"), (0, 0, 255))
        points = np.asarray([[0.2, 1.2], [2.7, 3.1]], dtype=np.float32)
        np.testing.assert_array_equal(scale_points(points, scale=2), np.asarray([[0, 2], [5, 6]], dtype=np.int32))

    def test_draw_dashed_polyline_writes_pixels(self) -> None:
        canvas = np.zeros((10, 20, 3), dtype=np.uint8)
        points = np.asarray([[1, 5], [18, 5]], dtype=np.int32)

        draw_dashed_polyline(canvas, points, color=(0, 255, 0), thickness=1, dash_length=4, gap_length=3)

        self.assertGreater(int(canvas[:, :, 1].sum()), 0)


if __name__ == "__main__":
    unittest.main()
