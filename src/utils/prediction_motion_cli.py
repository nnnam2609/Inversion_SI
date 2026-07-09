from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from src.utils.prediction_motion import (
    DEFAULT_STATIC_COORD_STD_RATIO_THRESHOLD,
    DEFAULT_STATIC_MOTION_RATIO_THRESHOLD,
    assert_prediction_motion,
    prediction_motion_report,
)


def add_prediction_motion_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_diagnostic_flag: bool,
    action_word: str,
) -> None:
    parser.add_argument(
        "--static-motion-ratio-threshold",
        type=float,
        default=DEFAULT_STATIC_MOTION_RATIO_THRESHOLD,
        help=f"{action_word} when pred/label frame-diff ratio is below this threshold.",
    )
    parser.add_argument(
        "--static-coord-std-ratio-threshold",
        type=float,
        default=DEFAULT_STATIC_COORD_STD_RATIO_THRESHOLD,
        help=f"{action_word} when pred/label coordinate std ratio is below this threshold.",
    )
    if include_diagnostic_flag:
        parser.add_argument(
            "--allow-static-prediction-diagnostic",
            action="store_true",
            help="Allow rendering a frozen, nearly static, or under-moving prediction payload for diagnostic inspection.",
        )


def prediction_motion_report_from_args(
    payload: dict[str, Any],
    classes: list[str],
    args: argparse.Namespace,
    *,
    mean_contour: np.ndarray | None = None,
    prediction_payload: str | None = None,
) -> dict[str, Any]:
    return prediction_motion_report(
        payload,
        classes=classes,
        static_motion_ratio_threshold=float(args.static_motion_ratio_threshold),
        static_coord_std_ratio_threshold=float(args.static_coord_std_ratio_threshold),
        mean_contour=mean_contour,
        prediction_payload=prediction_payload,
    )


def assert_prediction_motion_from_args(report: dict[str, Any], args: argparse.Namespace) -> None:
    assert_prediction_motion(
        report,
        allow_static_prediction_diagnostic=bool(getattr(args, "allow_static_prediction_diagnostic", False)),
    )
