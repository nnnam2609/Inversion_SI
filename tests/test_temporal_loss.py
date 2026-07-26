from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
