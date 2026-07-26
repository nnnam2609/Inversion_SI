from __future__ import annotations

import argparse
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.rendering.gridnorm_session import RAW_ROOT, draw_frame, resolve_session_media
from src.utils.session_rendering import build_ground_truth_timeline


class GroundTruthVideoRenderingTests(unittest.TestCase):
    def test_session_media_defaults_follow_numeric_speaker_and_session(self) -> None:
        args = argparse.Namespace(
            speaker=7,
            session=15,
            speaker_name=None,
            session_name=None,
            mri_dicom_dir=None,
            mri_npy_dir=None,
            audio=None,
        )

        resolved = resolve_session_media(args)

        self.assertEqual(resolved.speaker_name, "P7")
        self.assertEqual(resolved.session_name, "S15")
        self.assertEqual(resolved.mri_dicom_dir, RAW_ROOT / "P7/DCM_2D/S15")
        self.assertEqual(
            resolved.audio,
            RAW_ROOT / "P7/OTHER/S15/DENOISED_SOUND_P7_S15.wav",
        )

    def test_session_media_rejects_mismatched_display_names(self) -> None:
        args = argparse.Namespace(
            speaker=7,
            session=15,
            speaker_name="P2",
            session_name="S1",
            mri_dicom_dir=None,
            mri_npy_dir=None,
            audio=None,
        )

        with self.assertRaisesRegex(ValueError, "requires --speaker-name P7"):
            resolve_session_media(args)

    def test_ground_truth_timeline_renders_integer_frames_only_at_step_one(self) -> None:
        classes = ["tongue", "upper-incisor"]
        with tempfile.TemporaryDirectory() as directory:
            contour_dir = Path(directory)
            for frame, value in ((199, 10.0), (200, 14.0)):
                for class_index, articulator in enumerate(classes):
                    contour = np.full((50, 2), value + class_index, dtype=np.float32)
                    np.save(contour_dir / f"{frame:04d}_{articulator}.npy", contour)

            rows = build_ground_truth_timeline(
                contour_dir,
                classes,
                start_frame=199.0,
                end_frame=200.0,
                step=1.0,
                max_frames=None,
            )

        self.assertEqual([row["frame"] for row in rows], ["0199", "0200"])
        np.testing.assert_array_equal(rows[0]["labels"], np.full((2, 100), [[10.0], [11.0]]))
        np.testing.assert_array_equal(rows[1]["labels"], np.full((2, 100), [[14.0], [15.0]]))
        self.assertFalse(any(row["held"] for row in rows))
        self.assertEqual(rows[1]["ground_truth_source"], "direct frame 0200")

    def test_ground_truth_timeline_reports_missing_contours_without_holding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            contour_dir = Path(directory)
            np.save(contour_dir / "0199_tongue.npy", np.zeros((50, 2), dtype=np.float32))
            rows = build_ground_truth_timeline(
                contour_dir,
                ["tongue"],
                start_frame=199.0,
                end_frame=200.0,
                step=1.0,
                max_frames=None,
            )

        self.assertEqual(rows[0]["missing_ground_truth_contours"], [])
        self.assertEqual(rows[1]["missing_ground_truth_contours"], ["tongue"])
        self.assertTrue(np.isnan(rows[1]["labels"]).all())
        self.assertFalse(rows[1]["held"])

    def test_ground_truth_only_draws_without_prediction_or_rmse(self) -> None:
        classes = [f"contour-{index}" for index in range(11)]
        points = np.column_stack(
            [
                np.linspace(12.0, 24.0, 50, dtype=np.float32),
                np.linspace(18.0, 30.0, 50, dtype=np.float32),
            ]
        ).reshape(100)
        labels = np.stack([points + float(index) for index in range(11)], axis=0)
        row = {
            "labels": labels,
            "frame": "0001",
            "phoneme": "i",
            "held": False,
        }

        canvas, metrics = draw_frame(
            row,
            classes,
            primary_indices=[],
            image=np.zeros((136, 136), dtype=np.uint8),
            scale=4,
            mode_label="ground truth only",
            display_label="P7/S15",
            ground_truth_only=True,
        )

        self.assertEqual(canvas.shape, (662, 544, 3))
        self.assertTrue(math.isnan(metrics["rmse_px"]))
        self.assertTrue(math.isnan(metrics["primary_rmse_px"]))
        self.assertGreater(int(canvas.sum()), 0)


if __name__ == "__main__":
    unittest.main()
