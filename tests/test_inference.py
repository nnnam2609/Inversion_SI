"""Regression tests for the inference domain."""

from __future__ import annotations

# --- Consolidated from test_dense_audio_integer_inference.py ---

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import textgrid
import torch

from src.inference.dense_audio import (
    dense_feature_selection,
    infer_full_sequence,
    infer_nonoverlapping_chunks,
    infer_presegmented_sequences,
    integer_frame_feature_selection,
    legacy_interval_feature_selection,
    window_starts,
)


class EchoModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def forward(self, inputs, lengths):
        self.call_count += 1
        return inputs.reshape(inputs.shape[0], inputs.shape[1], 2, 2), None, None


class DenseAudioIntegerInferenceTests(unittest.TestCase):
    def test_full_sequence_uses_one_direct_model_forward(self) -> None:
        model = EchoModel()
        features = np.arange(7 * 4, dtype=np.float32).reshape(7, 4)

        predicted, coverage = infer_full_sequence(
            model,
            features,
            class_count=2,
            output_layer=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(model.call_count, 1)
        self.assertEqual(predicted.shape, (7, 2, 2))
        np.testing.assert_array_equal(predicted.reshape(7, 4), features)
        np.testing.assert_array_equal(coverage, np.ones(7, dtype=np.int32))

    def test_dense_feature_selection_covers_every_integer_mri_frame(self) -> None:
        config = {
            "added_frames": 20,
            "ms_image": 19.98,
            "hop_length_ratio": 10,
        }
        features = np.zeros((4396, 39), dtype=np.float32)

        selected, global_indices, first_by_frame = dense_feature_selection(
            features,
            sample_rate=16000,
            window_length_samples=400,
            config=config,
            frame_min=143,
            frame_max=1606,
        )

        self.assertEqual(selected.shape, (2925, 39))
        self.assertEqual(global_indices[0], 325)
        self.assertEqual(global_indices[-1], 3249)
        self.assertEqual(sorted(first_by_frame), list(range(143, 1607)))
        self.assertEqual(first_by_frame[143], 0)
        self.assertEqual(first_by_frame[199], 112)
        self.assertEqual(first_by_frame[1606], 2923)

    def test_integer_frame_selection_uses_one_center_nearest_mfcc_per_frame(self) -> None:
        config = {
            "added_frames": 20,
            "ms_image": 19.98,
            "hop_length_ratio": 10,
        }
        features = np.zeros((4396, 39), dtype=np.float32)

        selected, global_indices, local_by_frame = integer_frame_feature_selection(
            features,
            sample_rate=16000,
            window_length_samples=400,
            config=config,
            frame_min=143,
            frame_max=1606,
        )

        self.assertEqual(selected.shape, (1464, 39))
        self.assertEqual(global_indices[0], 325)
        self.assertEqual(global_indices[56], 437)
        self.assertEqual(global_indices[-1], 3248)
        self.assertEqual(local_by_frame[143], 0)
        self.assertEqual(local_by_frame[199], 56)
        self.assertEqual(local_by_frame[1606], 1463)
        self.assertTrue(np.all(np.diff(global_indices) > 0))

    def test_nonoverlapping_chunks_reset_context_and_copy_each_output_once(self) -> None:
        model = EchoModel()
        features = np.arange(7 * 4, dtype=np.float32).reshape(7, 4)

        predicted, coverage, starts = infer_nonoverlapping_chunks(
            model,
            features,
            chunk_size=3,
            batch_size=2,
            class_count=2,
            output_layer=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(starts, [0, 3, 6])
        self.assertEqual(model.call_count, 2)
        np.testing.assert_array_equal(predicted.reshape(7, 4), features)
        np.testing.assert_array_equal(coverage, np.ones(7, dtype=np.int32))

    def test_legacy_interval_selection_preserves_boundaries_and_skips_hash_silence(self) -> None:
        config = {
            "added_frames": 0,
            "ms_image": 20.0,
            "hop_length_ratio": 10,
        }
        features = np.zeros((30, 4), dtype=np.float32)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "session.TextGrid"
            grid = textgrid.TextGrid(minTime=0.0, maxTime=0.14)
            tier = textgrid.IntervalTier(name="words", minTime=0.0, maxTime=0.14)
            tier.add(0.0, 0.04, "word")
            tier.add(0.04, 0.10, "#")
            tier.add(0.10, 0.14, "word2")
            grid.append(tier)
            grid.write(str(path))

            selected, indices, local_by_frame, slices, metadata = legacy_interval_feature_selection(
                features,
                sample_rate=16000,
                window_length_samples=400,
                config=config,
                textgrid_path=path,
                frame_min=0,
                frame_max=7,
                chunk_size=2,
            )

        self.assertEqual(selected.shape, (6, 4))
        np.testing.assert_array_equal(indices, np.asarray([0, 2, 3, 10, 12, 13]))
        self.assertEqual(sorted(local_by_frame), [0, 1, 2, 5, 6, 7])
        self.assertEqual(slices, [(0, 2), (2, 3), (3, 5), (5, 6)])
        self.assertEqual(metadata["num_selected_textgrid_intervals"], 2)
        self.assertEqual(metadata["num_skipped_silence_intervals"], 1)

    def test_presegmented_sequences_keep_each_interval_independent(self) -> None:
        model = EchoModel()
        features = np.arange(7 * 4, dtype=np.float32).reshape(7, 4)

        predicted, coverage = infer_presegmented_sequences(
            model,
            features,
            sequence_slices=[(0, 2), (2, 7)],
            batch_size=1,
            class_count=2,
            output_layer=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(model.call_count, 2)
        np.testing.assert_array_equal(predicted.reshape(7, 4), features)
        np.testing.assert_array_equal(coverage, np.ones(7, dtype=np.int32))

    def test_overlapping_window_starts_cover_the_dense_sequence(self) -> None:
        starts = window_starts(length=2925, window_size=80, stride=40)
        coverage = np.zeros(2925, dtype=np.int32)
        for start in starts:
            coverage[start : start + 80] += 1

        self.assertEqual(len(starts), 73)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 2845)
        self.assertEqual(coverage.min(), 1)
        self.assertEqual(coverage.max(), 3)

# --- Consolidated from test_diagnose_prediction_motion_script.py ---

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_diagnostic_payload(static_prediction: bool) -> dict[str, torch.Tensor]:
    labels = torch.tensor(
        [
            [
                [[0.0, 0.0, 1.0, 1.0]],
                [[1.0, 1.0, 2.0, 2.0]],
                [[2.0, 2.0, 3.0, 3.0]],
            ]
        ],
        dtype=torch.float32,
    )
    predicted = labels[:, :1].repeat(1, labels.shape[1], 1, 1) if static_prediction else labels + 0.25
    return {
        "predicted_raw": predicted,
        "labels_raw": labels,
        "frames": torch.tensor([[[2.0, 1.0, 1.0], [2.0, 1.0, 2.0], [2.0, 1.0, 3.0]]], dtype=torch.float32),
        "lengths": torch.tensor([3]),
    }


class DiagnosePredictionMotionScriptTests(unittest.TestCase):
    def run_script(self, payload: dict[str, torch.Tensor], *extra_args: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            payload_path = Path(tmpdir) / "payload.pt"
            torch.save(payload, payload_path)
            return subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/inversion_si.py"),
                    "audit",
                    "motion",
                    "--prediction-payload",
                    str(payload_path),
                    *extra_args,
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

    def test_fail_on_static_exits_nonzero_for_frozen_payload(self) -> None:
        result = self.run_script(
            make_diagnostic_payload(static_prediction=True), "--fail-on-static"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Prediction payload appears frozen", result.stderr)

    def test_fail_on_static_accepts_moving_payload(self) -> None:
        result = self.run_script(
            make_diagnostic_payload(static_prediction=False), "--fail-on-static"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"is_frozen_prediction": false', result.stdout)

    def test_diagnostic_bypass_allows_static_payload(self) -> None:
        result = self.run_script(
            make_diagnostic_payload(static_prediction=True),
            "--fail-on-static",
            "--allow-static-prediction-diagnostic",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"is_frozen_prediction": true', result.stdout)

# --- Consolidated from test_prediction_motion.py ---

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.utils.prediction_motion import assert_prediction_motion, prediction_motion_report


def make_payload(
    static_prediction: bool,
    prediction_only: bool = False,
    prediction_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    labels = torch.tensor(
        [
            [
                [[0.0, 0.0, 1.0, 1.0], [5.0, 5.0, 6.0, 6.0]],
                [[1.0, 1.0, 2.0, 2.0], [6.0, 6.0, 7.0, 7.0]],
                [[2.0, 2.0, 3.0, 3.0], [7.0, 7.0, 8.0, 8.0]],
            ]
        ],
        dtype=torch.float32,
    )
    if static_prediction:
        predicted = labels[:, :1].repeat(1, labels.shape[1], 1, 1)
    else:
        predicted = labels * float(prediction_scale) + 0.25
    frames = torch.tensor([[[2.0, 1.0, 1.0], [2.0, 1.0, 2.0], [2.0, 1.0, 3.0]]], dtype=torch.float32)
    payload = {
        "predicted_raw": predicted,
        "frames": frames,
        "lengths": torch.tensor([3]),
        "prediction_only": bool(prediction_only),
    }
    if not prediction_only:
        payload["labels_raw"] = labels
    return payload


class PredictionMotionTests(unittest.TestCase):
    def test_static_prediction_is_flagged_and_rejected(self) -> None:
        report = prediction_motion_report(
            make_payload(static_prediction=True),
            classes=["a", "b"],
            static_motion_ratio_threshold=0.10,
        )
        self.assertTrue(report["is_frozen_prediction"])
        self.assertTrue(report["is_static_prediction"])
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            assert_prediction_motion(report)
        assert_prediction_motion(report, allow_static_prediction_diagnostic=True)

    def test_moving_prediction_passes(self) -> None:
        report = prediction_motion_report(
            make_payload(static_prediction=False),
            classes=["a", "b"],
            static_motion_ratio_threshold=0.10,
        )
        self.assertFalse(report["is_static_prediction"])
        self.assertFalse(report["is_under_moving_coord_std"])
        self.assertFalse(report["is_frozen_prediction"])
        assert_prediction_motion(report)

    def test_under_moving_coord_std_is_flagged_and_rejected(self) -> None:
        report = prediction_motion_report(
            make_payload(static_prediction=False, prediction_scale=0.15),
            classes=["a", "b"],
            static_motion_ratio_threshold=0.10,
            static_coord_std_ratio_threshold=0.20,
        )
        self.assertFalse(report["is_frozen_prediction"])
        self.assertFalse(report["is_static_prediction"])
        self.assertTrue(report["is_under_moving_coord_std"])
        with self.assertRaisesRegex(RuntimeError, "under-moving"):
            assert_prediction_motion(report)
        assert_prediction_motion(report, allow_static_prediction_diagnostic=True)

    def test_prediction_only_frozen_payload_is_rejected(self) -> None:
        report = prediction_motion_report(
            make_payload(static_prediction=True, prediction_only=True),
            classes=["a", "b"],
        )
        self.assertTrue(report["prediction_only"])
        self.assertTrue(report["is_frozen_prediction"])
        self.assertNotIn("is_static_prediction", report)
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            assert_prediction_motion(report)
        assert_prediction_motion(report, allow_static_prediction_diagnostic=True)

    def test_prediction_only_moving_payload_passes(self) -> None:
        report = prediction_motion_report(
            make_payload(static_prediction=False, prediction_only=True),
            classes=["a", "b"],
        )
        self.assertTrue(report["prediction_only"])
        self.assertFalse(report["is_frozen_prediction"])
        assert_prediction_motion(report)

# --- Consolidated from test_prediction_motion_cli.py ---

import argparse
import unittest

import torch

from src.utils.prediction_motion_cli import (
    add_prediction_motion_arguments,
    assert_prediction_motion_from_args,
    prediction_motion_report_from_args,
)


def make_static_payload() -> dict[str, torch.Tensor]:
    labels = torch.tensor(
        [
            [
                [[0.0, 0.0, 1.0, 1.0]],
                [[1.0, 1.0, 2.0, 2.0]],
                [[2.0, 2.0, 3.0, 3.0]],
            ]
        ],
        dtype=torch.float32,
    )
    return {
        "predicted_raw": labels[:, :1].repeat(1, labels.shape[1], 1, 1),
        "labels_raw": labels,
        "frames": torch.tensor([[[2.0, 1.0, 1.0], [2.0, 1.0, 2.0], [2.0, 1.0, 3.0]]], dtype=torch.float32),
        "lengths": torch.tensor([3]),
    }


class PredictionMotionCliTests(unittest.TestCase):
    def test_add_prediction_motion_arguments_without_diagnostic_flag(self) -> None:
        parser = argparse.ArgumentParser()
        add_prediction_motion_arguments(parser, include_diagnostic_flag=False, action_word="Flag predictions")

        args = parser.parse_args([])

        self.assertEqual(args.static_motion_ratio_threshold, 0.20)
        self.assertEqual(args.static_coord_std_ratio_threshold, 0.20)
        self.assertFalse(hasattr(args, "allow_static_prediction_diagnostic"))

    def test_report_from_args_uses_thresholds_and_diagnostic_bypass(self) -> None:
        parser = argparse.ArgumentParser()
        add_prediction_motion_arguments(parser, include_diagnostic_flag=True, action_word="Fail render")
        args = parser.parse_args(
            [
                "--static-motion-ratio-threshold",
                "0.05",
                "--static-coord-std-ratio-threshold",
                "0.30",
                "--allow-static-prediction-diagnostic",
            ]
        )

        report = prediction_motion_report_from_args(make_static_payload(), ["tongue"], args)

        self.assertEqual(report["static_motion_ratio_threshold"], 0.05)
        self.assertEqual(report["static_coord_std_ratio_threshold"], 0.30)
        self.assertTrue(report["is_static_prediction"])
        assert_prediction_motion_from_args(report, args)
