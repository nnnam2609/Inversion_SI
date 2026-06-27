#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pydicom
import torch
import yaml
from matplotlib import colors as mcolors

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.utils.colors import COLORS

MM_PER_PIXEL = 1.62


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
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config did not parse to a mapping: {path}")
    return config


def rgb_to_bgr255(color_name: str) -> tuple[int, int, int]:
    red, green, blue = mcolors.to_rgb(color_name)
    return int(blue * 255), int(green * 255), int(red * 255)


def scale_points(points: np.ndarray, scale: int) -> np.ndarray:
    coords = points.reshape(-1, 2).astype(np.float32) * float(scale)
    return np.round(coords).astype(np.int32)


def draw_dashed_polyline(
    canvas: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    dash_length: int = 12,
    gap_length: int = 16,
) -> None:
    pattern_pos = 0.0
    cycle = float(dash_length + gap_length)
    for start, end in zip(points[:-1], points[1:]):
        start_float = start.astype(np.float32)
        end_float = end.astype(np.float32)
        segment_length = float(np.linalg.norm(end_float - start_float))
        if segment_length == 0:
            continue
        direction = (end_float - start_float) / segment_length
        cursor = 0.0
        while cursor < segment_length:
            phase = pattern_pos % cycle
            remaining = segment_length - cursor
            if phase < dash_length:
                step = min(float(dash_length) - phase, remaining)
                seg_start = start_float + direction * cursor
                seg_end = start_float + direction * (cursor + step)
                cv2.line(
                    canvas,
                    tuple(np.round(seg_start).astype(int)),
                    tuple(np.round(seg_end).astype(int)),
                    color,
                    thickness,
                    lineType=cv2.LINE_AA,
                )
            else:
                step = min(cycle - phase, remaining)
            cursor += step
            pattern_pos += step


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
            label = tuple(float(x) for x in frame.tolist())
            if label in seen:
                continue
            seen.add(label)
            items.append((float(frame[2]), seq_idx, offset))
    items.sort(key=lambda item: item[0])
    if max_frames is not None:
        return items[: max(1, int(max_frames))]
    return items


def default_cache_path(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}_{suffix}")


def dicom_value(value: Any) -> Any:
    if hasattr(value, "value"):
        value = value.value
    if hasattr(value, "original_string"):
        return value.original_string
    if isinstance(value, (list, tuple)):
        return [dicom_value(item) for item in value]
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)


def read_dicom_header(path: Path) -> tuple[int, str, dict[str, Any]]:
    dataset = pydicom.dcmread(
        str(path),
        stop_before_pixels=True,
        force=True,
        specific_tags=["InstanceNumber", "Rows", "Columns", "PixelSpacing"],
    )
    instance_number = getattr(dataset, "InstanceNumber", None)
    if instance_number is None:
        raise ValueError("missing InstanceNumber")
    metadata = {
        "rows": dicom_value(getattr(dataset, "Rows", None)),
        "columns": dicom_value(getattr(dataset, "Columns", None)),
        "pixel_spacing": dicom_value(getattr(dataset, "PixelSpacing", None)),
    }
    return int(instance_number), path.name, metadata


def dicom_filename_sort_key(name: str) -> tuple[int, str, int, str]:
    match = re.search(r"(\d{14})(\d+)$", name)
    if match is None:
        return (1, name, 0, name)
    return (0, match.group(1), int(match.group(2)), name)


def build_filename_dicom_index(dicom_dir: Path) -> tuple[dict[int, str], dict[str, Any]]:
    names = [name for name in os.listdir(dicom_dir) if not name.startswith(".")]
    sorted_names = sorted(names, key=dicom_filename_sort_key)
    return {index: name for index, name in enumerate(sorted_names, start=1)}, {
        "index_method": "filename_timestamp_rank",
        "dicom_files": len(sorted_names),
    }


def build_header_dicom_index(dicom_dir: Path, workers: int) -> tuple[dict[int, str], dict[str, Any]]:
    names = sorted(os.listdir(dicom_dir))
    paths = [dicom_dir / name for name in names if not name.startswith(".")]
    metadata: dict[str, Any] = {}
    frames: dict[int, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        future_to_path = {executor.submit(read_dicom_header, path): path for path in paths}
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                frame_number, filename, header_metadata = future.result()
            except Exception as exc:
                print(f"Skipping unreadable DICOM header {path}: {exc}", file=sys.stderr)
                continue
            frames[frame_number] = filename
            if not metadata:
                metadata = header_metadata
    metadata["index_method"] = "dicom_header_instance_number"
    metadata["dicom_files"] = len(frames)
    return frames, metadata


def load_or_build_dicom_index(
    dicom_dir: Path,
    cache_path: Path,
    method: str,
    workers: int,
) -> dict[int, str]:
    dicom_dir = dicom_dir.resolve()
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("dicom_dir") == str(dicom_dir):
            return {int(key): value for key, value in payload["frames"].items()}

    start = time.monotonic()
    if method == "filename":
        frames, metadata = build_filename_dicom_index(dicom_dir)
    elif method == "header":
        frames, metadata = build_header_dicom_index(dicom_dir, workers)
    else:
        raise ValueError(f"Unsupported DICOM index method: {method}")

    if not frames:
        raise RuntimeError(f"No InstanceNumber-indexed DICOM files found in {dicom_dir}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dicom_dir": str(dicom_dir),
        "frames": {str(key): value for key, value in sorted(frames.items())},
        "metadata": metadata,
        "build_seconds": time.monotonic() - start,
    }
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return frames


def normalize_mri_frame(pixel_array: np.ndarray) -> np.ndarray:
    image = pixel_array.astype(np.float32)
    low, high = np.percentile(image, [1.0, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(image))
        high = float(np.max(image))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    image = np.clip((image - low) / (high - low), 0.0, 1.0)
    return np.round(image * 255.0).astype(np.uint8)


def needed_integer_frames(items: list[tuple[float, int, int]], frame_offset: int) -> list[int]:
    needed: set[int] = set()
    for frame_number, _, _ in items:
        lower = math.floor(frame_number)
        upper = math.ceil(frame_number)
        needed.add(int(lower) + frame_offset)
        needed.add(int(upper) + frame_offset)
    return sorted(needed)


def load_or_build_mri_cache(
    dicom_dir: Path,
    dicom_index: dict[int, str],
    frame_numbers: list[int],
    cache_path: Path,
    workers: int,
) -> dict[int, np.ndarray]:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_frames = [int(value) for value in cached["frame_numbers"].tolist()]
        if cached_frames == frame_numbers:
            images = cached["images"]
            return {frame_number: images[index] for index, frame_number in enumerate(cached_frames)}

    def read_one(frame_number: int) -> tuple[int, np.ndarray | None]:
        filename = dicom_index.get(frame_number)
        if filename is None:
            return frame_number, None
        dataset = pydicom.dcmread(str(dicom_dir / filename), force=True)
        return frame_number, normalize_mri_frame(dataset.pixel_array)

    images_by_frame: dict[int, np.ndarray] = {}
    missing = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        future_to_frame = {executor.submit(read_one, frame_number): frame_number for frame_number in frame_numbers}
        completed = 0
        total = len(future_to_frame)
        start = time.monotonic()
        for future in concurrent.futures.as_completed(future_to_frame):
            frame_number, image = future.result()
            completed += 1
            if image is None:
                missing.append(frame_number)
            else:
                images_by_frame[frame_number] = image
            if completed == 1 or completed % 250 == 0 or completed == total:
                elapsed = time.monotonic() - start
                print(f"Loaded DICOM MRI frames: {completed}/{total} in {elapsed:.1f}s", flush=True)

    if missing:
        raise RuntimeError(
            "Missing DICOM InstanceNumber(s): "
            + ", ".join(str(value) for value in missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )

    images = [images_by_frame[frame_number] for frame_number in frame_numbers]
    stack = np.stack(images, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, frame_numbers=np.array(frame_numbers, dtype=np.int32), images=stack)
    return {frame_number: stack[index] for index, frame_number in enumerate(frame_numbers)}


def load_or_build_npy_mri_cache(
    npy_dir: Path,
    frame_numbers: list[int],
    cache_path: Path,
) -> dict[int, np.ndarray]:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_frames = [int(value) for value in cached["frame_numbers"].tolist()]
        if cached_frames == frame_numbers:
            images = cached["images"]
            return {frame_number: images[index] for index, frame_number in enumerate(cached_frames)}

    images = []
    missing = []
    for frame_number in frame_numbers:
        path = npy_dir / f"{frame_number:04d}.npy"
        if not path.exists():
            missing.append(frame_number)
            continue
        image = np.load(path, allow_pickle=False)
        if image.ndim != 2:
            raise ValueError(f"Expected 2D MRI frame in {path}, got shape {image.shape}")
        if image.dtype != np.uint8:
            image = normalize_mri_frame(image)
        images.append(image)

    if missing:
        raise RuntimeError(
            "Missing NPY MRI frame(s): "
            + ", ".join(str(value) for value in missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )

    stack = np.stack(images, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, frame_numbers=np.array(frame_numbers, dtype=np.int32), images=stack)
    return {frame_number: stack[index] for index, frame_number in enumerate(frame_numbers)}


def mri_for_frame(frame_number: float, frame_offset: int, cache: dict[int, np.ndarray]) -> np.ndarray:
    lower = int(math.floor(frame_number)) + frame_offset
    upper = int(math.ceil(frame_number)) + frame_offset
    if lower == upper:
        return cache[lower]
    alpha = float(frame_number - math.floor(frame_number))
    blended = (1.0 - alpha) * cache[lower].astype(np.float32) + alpha * cache[upper].astype(np.float32)
    return np.round(blended).astype(np.uint8)


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
        canvas = cv2.cvtColor(mri, cv2.COLOR_GRAY2BGR)
        canvas = cv2.resize(canvas, (image_size * scale, image_size * scale), interpolation=cv2.INTER_CUBIC)
    else:
        fill = 44 if background == "gray" else 12
        canvas = np.full((image_size * scale, image_size * scale, 3), fill, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 86), (15, 15, 15), thickness=-1)

    for articulator_index, articulator in enumerate(classes):
        color = rgb_to_bgr255(COLORS.get(articulator, "white"))
        pred_points = scale_points(pred[articulator_index], scale)
        if gt is not None:
            gt_points = scale_points(gt[articulator_index], scale)
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
    width = height = image_size * int(args.scale)
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
    }
    with args.output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
