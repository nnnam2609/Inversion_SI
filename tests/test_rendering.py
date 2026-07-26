"""Regression tests for the rendering domain."""

from __future__ import annotations

# --- Consolidated from test_gridnorm_rendering.py ---

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.utils.gridnorm_rendering import (
    mri_for_frame as gridnorm_mri_for_frame,
    transform_predictions,
    write_mode_contours,
)


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
            gridnorm_mri_for_frame(1.5, cache)

# --- Consolidated from test_ground_truth_video_rendering.py ---

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

# --- Consolidated from test_mri_rendering.py ---

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.utils.mri_rendering import (
    dicom_filename_sort_key,
    load_or_build_mri_cache,
    mri_for_frame,
    needed_integer_frames,
    normalize_mri_frame,
)


class MriRenderingTests(unittest.TestCase):
    def test_dicom_cache_is_not_reused_for_a_different_source_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "P2" / "DCM_2D" / "S1"
            source_b = root / "P7" / "DCM_2D" / "S15"
            source_a.mkdir(parents=True)
            source_b.mkdir(parents=True)
            cache_path = root / "mri_frames_cache.npz"
            np.savez(
                cache_path,
                frame_numbers=np.array([199], dtype=np.int32),
                images=np.zeros((1, 2, 2), dtype=np.uint8),
                source_dir=np.array(str(source_a.resolve())),
            )

            with self.assertRaisesRegex(RuntimeError, "Missing DICOM InstanceNumber"):
                load_or_build_mri_cache(
                    source_b,
                    dicom_index={},
                    frame_numbers=[199],
                    cache_path=cache_path,
                    workers=1,
                )

    def test_dicom_cache_is_reused_for_the_same_source_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "P7" / "DCM_2D" / "S15"
            source.mkdir(parents=True)
            cache_path = Path(directory) / "mri_frames_cache.npz"
            expected = np.full((2, 2), 17, dtype=np.uint8)
            np.savez(
                cache_path,
                frame_numbers=np.array([199], dtype=np.int32),
                images=expected[None, ...],
                source_dir=np.array(str(source.resolve())),
            )

            cache = load_or_build_mri_cache(
                source,
                dicom_index={},
                frame_numbers=[199],
                cache_path=cache_path,
                workers=1,
            )

        np.testing.assert_array_equal(cache[199], expected)

    def test_dicom_filename_sort_key_uses_timestamp_then_suffix(self) -> None:
        names = ["IMG_2020010100000010", "IMG_2020010100000002", "abc"]

        self.assertEqual(sorted(names, key=dicom_filename_sort_key), ["IMG_2020010100000002", "IMG_2020010100000010", "abc"])

    def test_needed_integer_frames_applies_offset(self) -> None:
        items = [(1.0, 0, 0), (2.0, 0, 0)]

        self.assertEqual(needed_integer_frames(items, frame_offset=10), [11, 12])

    def test_needed_integer_frames_rejects_half_frames(self) -> None:
        with self.assertRaisesRegex(ValueError, "NEVER render fractional"):
            needed_integer_frames([(2.5, 0, 0)], frame_offset=10)

    def test_normalize_mri_frame_returns_uint8_range(self) -> None:
        image = np.asarray([[0, 5], [10, 15]], dtype=np.uint16)

        normalized = normalize_mri_frame(image)

        self.assertEqual(normalized.dtype, np.uint8)
        self.assertEqual(normalized.shape, image.shape)
        self.assertGreater(int(normalized.max()), int(normalized.min()))

    def test_mri_for_frame_rejects_half_frame_with_offset(self) -> None:
        cache = {
            11: np.zeros((2, 2), dtype=np.uint8),
            12: np.full((2, 2), 10, dtype=np.uint8),
        }

        with self.assertRaisesRegex(ValueError, "NEVER render fractional"):
            mri_for_frame(1.5, frame_offset=10, cache=cache)

# --- Consolidated from test_session_rendering.py ---

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.utils.session_rendering import aggregate_state, build_timeline, frame_token


class SessionRenderingTests(unittest.TestCase):
    def test_frame_token_handles_integer_and_rejects_half_frames(self) -> None:
        self.assertEqual(frame_token(12.0), "0012")
        with self.assertRaisesRegex(ValueError, "NEVER render fractional"):
            frame_token(12.5)

    def test_aggregate_state_filters_session_and_averages_duplicate_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            phoneme_path = Path(tmpdir) / "phonemes.json"
            phoneme_path.write_text(json.dumps(["UNK", "i"]), encoding="utf-8")
            config = {"phonemesdir": str(phoneme_path)}
            state = {
                "predicted_raw": torch.tensor(
                    [
                        [
                            [[1.0, 3.0, 5.0, 7.0]],
                            [[2.0, 4.0, 6.0, 8.0]],
                        ],
                    ]
                ),
                "labels_raw": torch.tensor(
                    [
                        [
                            [[10.0, 12.0, 14.0, 16.0]],
                            [[20.0, 22.0, 24.0, 26.0]],
                        ],
                    ]
                ),
                "frames": torch.tensor([[[2.0, 1.0, 10.0], [2.0, 1.0, 10.0]]]),
                "lengths": torch.tensor([2]),
                "phonemes": torch.tensor([[[[0.0, 1.0]], [[0.0, 1.0]]]]),
            }

            rows = aggregate_state(state, config, speaker=2, session=1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["frame"], "0010")
        self.assertEqual(rows[0]["phoneme"], "i")
        np.testing.assert_allclose(rows[0]["predicted"], np.asarray([[1.5, 3.5, 5.5, 7.5]], dtype=np.float32))
        np.testing.assert_allclose(rows[0]["labels"], np.asarray([[15.0, 17.0, 19.0, 21.0]], dtype=np.float32))

    def test_build_timeline_rejects_half_frame_step(self) -> None:
        rows = [
            {"frame_number": 1.0, "frame": "0001", "predicted": np.zeros((1, 4)), "held": False},
            {"frame_number": 2.0, "frame": "0002", "predicted": np.ones((1, 4)), "held": False},
        ]

        with self.assertRaisesRegex(ValueError, "NEVER render fractional"):
            build_timeline(rows, step=0.5, max_frames=None)

# --- Consolidated from test_video_rendering.py ---

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
