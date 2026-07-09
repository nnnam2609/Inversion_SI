#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.utils.prediction_motion_cli import (  # noqa: E402
    add_prediction_motion_arguments,
    assert_prediction_motion_from_args,
    prediction_motion_report_from_args,
)


DEFAULT_CLASSES = [
    "arytenoid-cartilage",
    "epiglottis",
    "lower-lip",
    "pharynx",
    "soft-palate-midline",
    "tongue",
    "upper-lip",
    "vocal-folds",
    "thyroid-cartilage",
    "lower-incisor",
    "upper-incisor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose whether cached contour predictions move over time.")
    parser.add_argument("--prediction-payload", type=Path, required=True)
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        default=None,
        help="Optional normalization_stats.npz containing mean_contour for train-mean baseline.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument(
        "--fail-on-static",
        action="store_true",
        help="Exit with an error when the payload is frozen, nearly static, or under-moving.",
    )
    add_prediction_motion_arguments(parser, include_diagnostic_flag=True, action_word="Flag predictions")
    parser.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES, help="Class names in payload order.")
    return parser.parse_args()


def load_mean_contour(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    stats = np.load(path)
    if "mean_contour" not in stats:
        raise KeyError(f"Missing mean_contour in {path}")
    return np.asarray(stats["mean_contour"], dtype=np.float32)


def make_report(args: argparse.Namespace) -> dict[str, Any]:
    payload = torch.load(args.prediction_payload, map_location="cpu", weights_only=False)
    report = prediction_motion_report_from_args(
        payload,
        classes=list(args.classes),
        args=args,
        mean_contour=load_mean_contour(args.normalization_stats),
        prediction_payload=str(args.prediction_payload),
    )
    report["normalization_stats"] = None if args.normalization_stats is None else str(args.normalization_stats)
    return report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Prediction Motion Diagnostic",
        "",
        f"- prediction_payload: `{report['prediction_payload']}`",
        f"- normalization_stats: `{report.get('normalization_stats')}`",
        f"- prediction_only: `{report['prediction_only']}`",
        f"- unique_frames: `{report['num_unique_frames']}`",
        f"- frame_range: `{report['frame_min']}` to `{report['frame_max']}`",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| pred frame diff mean abs | {report['predicted_raw_motion']['frame_diff_mean_abs']:.6f} |",
        f"| pred coord std mean | {report['predicted_raw_motion']['coord_std_mean']:.6f} |",
        f"| frozen prediction flag | `{report['is_frozen_prediction']}` |",
    ]
    if "labels_raw_motion" in report:
        lines.extend(
            [
                f"| label frame diff mean abs | {report['labels_raw_motion']['frame_diff_mean_abs']:.6f} |",
                f"| label coord std mean | {report['labels_raw_motion']['coord_std_mean']:.6f} |",
                f"| pred/label frame-diff ratio | {report['motion_ratio_frame_diff']:.6f} |",
                f"| pred/label coord-std ratio | {report['motion_ratio_coord_std']:.6f} |",
                f"| static prediction flag | `{report['is_static_prediction']}` |",
                f"| under-moving coord-std flag | `{report['is_under_moving_coord_std']}` |",
                f"| model RMSE | {report['model_rmse']:.6f} |",
            ]
        )
    if "train_mean_contour_rmse" in report:
        lines.extend(
            [
                f"| train-mean contour RMSE | {report['train_mean_contour_rmse']:.6f} |",
                f"| model - train-mean RMSE delta | {report['model_vs_train_mean_delta_rmse']:.6f} |",
                f"| model-to-train-mean RMSE | {report['model_to_train_mean_rmse']:.6f} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Per Articulator",
            "",
            "| Class | Pred Diff | Label Diff | Ratio | RMSE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["per_articulator"]:
        lines.append(
            "| {class_name} | {pred:.6f} | {label} | {ratio} | {rmse_value} |".format(
                class_name=row["class"],
                pred=row["pred_frame_diff_mean_abs"],
                label=(
                    f"{row['label_frame_diff_mean_abs']:.6f}"
                    if "label_frame_diff_mean_abs" in row
                    else "N/A"
                ),
                ratio=(
                    f"{row['motion_ratio_frame_diff']:.6f}"
                    if "motion_ratio_frame_diff" in row
                    else "N/A"
                ),
                rmse_value=f"{row['rmse']:.6f}" if "rmse" in row else "N/A",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = make_report(args)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_md:
        write_markdown(report, args.output_md)
    if not args.output_json and not args.output_md:
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_static:
        assert_prediction_motion_from_args(report, args)


if __name__ == "__main__":
    main()
