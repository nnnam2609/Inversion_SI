from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

from src.adaption_pipeline.legacy import (  # noqa: E402
    render_analyze_asd2_epoch211_selected_experiment as post,
)


class Asd2SelectedPostprocessTests(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic_and_frame_weighted(self) -> None:
        pairs = ((1, 1), (2, 1))
        left = {
            (1, 1): np.asarray([2.0, 2.0]),
            (2, 1): np.asarray([5.0, 5.0, 5.0, 5.0]),
        }
        right = {
            (1, 1): np.asarray([1.0, 1.0]),
            (2, 1): np.asarray([3.0, 3.0, 3.0, 3.0]),
        }
        first = post.paired_bootstrap(
            left, right, pairs, replicates=500, seed=17
        )
        second = post.paired_bootstrap(
            left, right, pairs, replicates=500, seed=17
        )
        self.assertEqual(first, second)
        self.assertTrue(
            np.isclose(first["delta_mm"], (2 * 1.0 + 4 * 2.0) / 6)
        )
        self.assertLessEqual(first["ci95_low_mm"], first["delta_mm"])
        self.assertLessEqual(first["delta_mm"], first["ci95_high_mm"])

    def test_exact_50_fps_rational_check(self) -> None:
        self.assertTrue(post.rational_is_50("50/1"))
        self.assertTrue(post.rational_is_50("100/2"))
        self.assertFalse(post.rational_is_50("25/1"))
        self.assertFalse(post.rational_is_50("2997/100"))

    def test_fixed_protocol_separates_p10_control(self) -> None:
        self.assertEqual(len(post.core.SELECTION), 9)
        self.assertEqual(len(post.COHORTS["UNSEEN_8"]), 8)
        self.assertEqual(
            post.COHORTS["P10_SAME_SPEAKER_CONTROL"], ((10, 14),)
        )
        self.assertNotIn((10, 14), post.COHORTS["UNSEEN_8"])


if __name__ == "__main__":
    unittest.main()
