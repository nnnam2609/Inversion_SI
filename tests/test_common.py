"""Regression tests for the common domain."""

from __future__ import annotations

# --- Consolidated from test_common_foundation.py ---

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.common.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    load_mapping,
    sha256_file,
)
from src.common.frames import (
    frame_token,
    require_identical_frame_numbers,
    require_integer_frames,
)


class ArtifactHelpersTest(unittest.TestCase):
    def test_json_csv_and_hash_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "nested" / "payload.json"
            csv_path = root / "rows.csv"
            atomic_write_json(json_path, {"path": root, "values": (1, 2)})
            atomic_write_csv(csv_path, [{"speaker": "P1", "session": "S16"}])

            self.assertEqual(load_mapping(json_path)["values"], [1, 2])
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8"))["path"],
                str(root),
            )
            self.assertEqual(len(sha256_file(csv_path)), 64)

    def test_csv_rejects_non_rectangular_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "fields differ"):
                atomic_write_csv(
                    Path(directory) / "rows.csv",
                    [{"speaker": "P1"}, {"session": "S16"}],
                )


class IntegerFrameHelpersTest(unittest.TestCase):
    def test_integer_frames_and_tokens(self) -> None:
        require_integer_frames([1.0, 2.0, 3])
        self.assertEqual(frame_token(7.0), "0007")
        np.testing.assert_array_equal(
            require_identical_frame_numbers([1, 2], [1.0, 2.0]),
            np.asarray([1, 2]),
        )

    def test_fractional_frames_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Fractional"):
            require_integer_frames([1.5])

# --- Consolidated from test_temporal_loss.py ---

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.utils.temporal_loss import contour_velocity_loss, valid_velocity_mask  # noqa: E402


class TemporalLossTests(unittest.TestCase):
    def test_valid_velocity_mask_uses_lengths_minus_one(self) -> None:
        mask = valid_velocity_mask(torch.tensor([1, 3, 5]), time_steps=5)

        self.assertEqual(
            mask.tolist(),
            [
                [False, False, False, False],
                [True, True, False, False],
                [True, True, True, True],
            ],
        )

    def test_contour_velocity_loss_ignores_padding(self) -> None:
        target = torch.tensor([[[[0.0]], [[1.0]], [[3.0]], [[100.0]]]])
        predicted = torch.tensor([[[[0.0]], [[2.0]], [[4.0]], [[999.0]]]])
        loss = contour_velocity_loss(target, predicted, torch.tensor([3]))

        self.assertAlmostEqual(float(loss), 0.5)

    def test_contour_velocity_loss_sum_reduction(self) -> None:
        target = torch.tensor([[[[0.0]], [[1.0]], [[3.0]], [[100.0]]]])
        predicted = torch.tensor([[[[0.0]], [[2.0]], [[4.0]], [[999.0]]]])
        loss = contour_velocity_loss(target, predicted, torch.tensor([3]), reduction="sum")

        self.assertAlmostEqual(float(loss), 1.0)

    def test_contour_velocity_loss_returns_zero_for_single_frame(self) -> None:
        predicted = torch.ones((2, 1, 1, 2), dtype=torch.float32)
        target = torch.zeros_like(predicted)

        loss = contour_velocity_loss(target, predicted, torch.tensor([1, 1]))

        self.assertEqual(float(loss), 0.0)
