from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
