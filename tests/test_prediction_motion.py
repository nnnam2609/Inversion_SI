from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
