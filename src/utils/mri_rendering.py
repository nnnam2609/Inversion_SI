from __future__ import annotations

import concurrent.futures
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pydicom


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
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Built DICOM index with {len(frames)} frames at {cache_path}", flush=True)
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
        rounded = int(round(frame_number))
        if not math.isclose(frame_number, rounded, rel_tol=0.0, abs_tol=1e-4):
            raise ValueError(f"NEVER render fractional MRI frame {frame_number}")
        needed.add(rounded + frame_offset)
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
        cached_source_dir = str(cached["source_dir"].item()) if "source_dir" in cached.files else None
        requested_source_dir = str(dicom_dir.resolve())
        if cached_frames == frame_numbers and cached_source_dir == requested_source_dir:
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
    np.savez(
        cache_path,
        frame_numbers=np.array(frame_numbers, dtype=np.int32),
        images=stack,
        source_dir=np.array(str(dicom_dir.resolve())),
    )
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
    rounded = int(round(frame_number))
    if not math.isclose(frame_number, rounded, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"NEVER render fractional MRI frame {frame_number}")
    return cache[rounded + frame_offset]
