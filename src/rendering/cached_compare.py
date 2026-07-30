#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.utils.colors import COLORS
from src.utils.config_validation import load_yaml_config as load_config
from src.utils.mri_rendering import (
    load_or_build_dicom_index,
    load_or_build_mri_cache,
    load_or_build_npy_mri_cache,
    mri_for_frame,
    needed_integer_frames,
)
from src.utils.prediction_motion_cli import (
    add_prediction_motion_arguments,
    prediction_motion_report_from_args,
)
from src.utils.video_rendering import MM_PER_PIXEL, draw_dashed_polyline, rgb_to_bgr255, scale_points

INFO_BAND_HEIGHT = 86


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render cached GT/pred contour comparison video.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--background", choices=("dark", "gray"), default="gray")
    parser.add_argument("--mri-dicom-dir", type=Path, default=None)
    parser.add_argument("--mri-npy-dir", type=Path, default=None)
    parser.add_argument("--dicom-index-cache", type=Path, default=None)
    parser.add_argument("--mri-frame-cache", type=Path, default=None)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--dicom-index-method", choices=("filename", "header"), default="filename")
    parser.add_argument("--dicom-index-workers", type=int, default=24)
    parser.add_argument("--dicom-read-workers", type=int, default=8)
    parser.add_argument(
        "--exclude-rmse-classes",
        nargs="*",
        default=[],
        help="Class names to exclude from RMSE computation while still drawing them.",
    )
    add_prediction_motion_arguments(parser, include_diagnostic_flag=True, action_word="Report")
    return parser.parse_args()


def frame_label(frame: np.ndarray) -> str:
    speaker = int(round(float(frame[0])))
    session = int(round(float(frame[1])))
    frame_number = float(frame[2])
    suffix = f"{int(round(frame_number))}.0" if frame_number.is_integer() else f"{frame_number:.1f}"
    return f"P{speaker}/S{session}/{suffix}"


def make_items(state: dict[str, Any], max_frames: int | None) -> list[tuple[float, int, int]]:
    frames = state["frames"]
    lengths = state["lengths"]
    items: list[tuple[float, int, int]] = []
    seen = set()
    for seq_idx in range(frames.shape[0]):
        length = int(lengths[seq_idx])
        for offset in range(length):
            frame = frames[seq_idx, offset].detach().cpu().numpy()
            frame_number = float(frame[2])
            rounded_frame = int(round(frame_number))
            if not math.isclose(frame_number, rounded_frame, rel_tol=0.0, abs_tol=1e-4):
                continue
            label = tuple(float(x) for x in frame.tolist())
            if label in seen:
                continue
            seen.add(label)
            items.append((float(rounded_frame), seq_idx, offset))
    items.sort(key=lambda item: item[0])
    if max_frames is not None:
        return items[: max(1, int(max_frames))]
    return items


def default_cache_path(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}_{suffix}")


def render_one(
    state: dict[str, Any],
    classes: list[str],
    rmse_class_indices: list[int],
    excluded_rmse_classes: list[str],
    item: tuple[float, int, int],
    image_size: int,
    scale: int,
    background: str,
    mri_cache: dict[int, np.ndarray] | None,
    frame_offset: int,
) -> tuple[np.ndarray, float | None]:
    frame_number, seq_idx, offset = item
    has_gt = "labels_raw" in state
    gt = state["labels_raw"][seq_idx, offset].detach().cpu().numpy() if has_gt else None
    pred = state["predicted_raw"][seq_idx, offset].detach().cpu().numpy()
    frame = state["frames"][seq_idx, offset].detach().cpu().numpy()
    rmse_mm = None
    if gt is not None:
        metric_pred = pred[rmse_class_indices]
        metric_gt = gt[rmse_class_indices]
        rmse_mm = float(np.sqrt(np.mean((metric_pred - metric_gt) ** 2))) * MM_PER_PIXEL

    if mri_cache is not None:
        mri = mri_for_frame(frame_number, frame_offset, mri_cache)
        image_canvas = cv2.cvtColor(mri, cv2.COLOR_GRAY2BGR)
        image_canvas = cv2.resize(image_canvas, (image_size * scale, image_size * scale), interpolation=cv2.INTER_CUBIC)
    else:
        fill = 44 if background == "gray" else 12
        image_canvas = np.full((image_size * scale, image_size * scale, 3), fill, dtype=np.uint8)
    canvas = np.full(
        (image_canvas.shape[0] + INFO_BAND_HEIGHT, image_canvas.shape[1], 3),
        15,
        dtype=np.uint8,
    )
    canvas[INFO_BAND_HEIGHT : INFO_BAND_HEIGHT + image_canvas.shape[0], :] = image_canvas

    for articulator_index, articulator in enumerate(classes):
        color = rgb_to_bgr255(COLORS.get(articulator, "white"))
        pred_points = scale_points(pred[articulator_index], scale)
        pred_points[:, 1] += INFO_BAND_HEIGHT
        if gt is not None:
            gt_points = scale_points(gt[articulator_index], scale)
            gt_points[:, 1] += INFO_BAND_HEIGHT
            cv2.polylines(canvas, [gt_points], isClosed=False, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [gt_points], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)
            draw_dashed_polyline(canvas, pred_points, color=(0, 0, 0), thickness=2)
            draw_dashed_polyline(canvas, pred_points, color=color, thickness=1)
        else:
            cv2.polylines(canvas, [pred_points], isClosed=False, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [pred_points], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)

    lines = [f"Frame: {frame_label(frame)}"]
    if rmse_mm is None:
        lines.extend(["Prediction only", "No target labels/std/mean used"])
    else:
        lines.extend([f"RMSE: {rmse_mm:.3f} mm", "Solid = Annotation | Dashed = Prediction"])
        if excluded_rmse_classes:
            lines.append("RMSE excludes: " + ", ".join(excluded_rmse_classes))
    for line_index, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (16, 23 + line_index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return canvas, rmse_mm


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    state = torch.load(args.predictions, map_location="cpu")
    classes = list(config["classes"])
    motion_report = prediction_motion_report_from_args(
        state,
        classes,
        args,
        prediction_payload=str(args.predictions),
    )
    excluded_rmse_classes = [name for name in args.exclude_rmse_classes if name in classes]
    excluded_set = set(excluded_rmse_classes)
    rmse_class_indices = [idx for idx, name in enumerate(classes) if name not in excluded_set]
    if not rmse_class_indices:
        raise ValueError("RMSE exclusion removed all classes")
    items = make_items(state, args.max_frames)
    if not items:
        raise RuntimeError(f"No frames to render from {args.predictions}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mri_cache = None
    dicom_index_cache = None
    mri_frame_cache = None
    if args.mri_dicom_dir is not None:
        if args.mri_npy_dir is not None:
            raise ValueError("Use only one of --mri-dicom-dir or --mri-npy-dir")
        dicom_index_cache = args.dicom_index_cache or default_cache_path(args.output, "dicom_index.json")
        mri_frame_cache = args.mri_frame_cache or default_cache_path(args.output, "mri_frames.npz")
        dicom_index = load_or_build_dicom_index(
            args.mri_dicom_dir,
            dicom_index_cache,
            args.dicom_index_method,
            int(args.dicom_index_workers),
        )
        frame_numbers = needed_integer_frames(items, int(args.frame_offset))
        mri_cache = load_or_build_mri_cache(
            args.mri_dicom_dir,
            dicom_index,
            frame_numbers,
            mri_frame_cache,
            int(args.dicom_read_workers),
        )
    elif args.mri_npy_dir is not None:
        mri_frame_cache = args.mri_frame_cache or default_cache_path(args.output, "mri_frames.npz")
        frame_numbers = needed_integer_frames(items, int(args.frame_offset))
        mri_cache = load_or_build_npy_mri_cache(args.mri_npy_dir, frame_numbers, mri_frame_cache)

    coordinate_maxes = [float(state["predicted_raw"].max()), 136.0]
    if "labels_raw" in state:
        coordinate_maxes.append(float(state["labels_raw"].max()))
    image_size = int(math.ceil(max(coordinate_maxes)))
    width = image_size * int(args.scale)
    height = width + INFO_BAND_HEIGHT
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {args.output}")

    rmses = []
    try:
        for item in items:
            frame, rmse_mm = render_one(
                state,
                classes,
                rmse_class_indices,
                excluded_rmse_classes,
                item,
                image_size,
                int(args.scale),
                args.background,
                mri_cache,
                int(args.frame_offset),
            )
            writer.write(frame)
            if rmse_mm is not None:
                rmses.append(rmse_mm)
    finally:
        writer.release()

    summary = {
        "predictions": str(args.predictions),
        "output": str(args.output),
        "fps": int(args.fps),
        "scale": int(args.scale),
        "frames": len(items),
        "mean_frame_rmse_mm": None if not rmses else float(np.mean(rmses)),
        "excluded_rmse_classes": excluded_rmse_classes,
        "prediction_only": "labels_raw" not in state,
        "min_frame_number": float(items[0][0]),
        "max_frame_number": float(items[-1][0]),
        "background": args.background,
        "mri_dicom_dir": str(args.mri_dicom_dir) if args.mri_dicom_dir is not None else None,
        "mri_npy_dir": str(args.mri_npy_dir) if args.mri_npy_dir is not None else None,
        "dicom_index_cache": str(dicom_index_cache) if dicom_index_cache is not None else None,
        "mri_frame_cache": str(mri_frame_cache) if mri_frame_cache is not None else None,
        "frame_offset": int(args.frame_offset),
        "dicom_index_method": args.dicom_index_method,
        "dicom_index_workers": int(args.dicom_index_workers),
        "dicom_read_workers": int(args.dicom_read_workers),
        "prediction_motion": motion_report,
        "motion_guard_enabled": False,
        "frame_policy": "NEVER render fractional MRI frames; integer frames only",
        "rendered_fractional_frame_count": 0,
    }
    with args.output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
