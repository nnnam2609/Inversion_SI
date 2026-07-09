from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY_GRIDNORM_CLASSES = {
    "tongue",
    "pharynx",
    "soft-palate-midline",
    "upper-incisor",
    "lower-incisor",
}


def transform_predictions(
    rows: list[dict[str, Any]],
    transform: dict[str, Any] | None,
    mode: str,
    class_count: int,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        new_row = dict(row)
        pred_points = row["predicted"].reshape(class_count, 50, 2)
        if mode == "raw":
            transformed = pred_points
        elif mode == "affine":
            if transform is None:
                raise ValueError("affine mode requires a grid transform")
            from grid_transform.transform_helpers import apply_transform

            transformed = apply_transform(transform["step1_affine"], pred_points.reshape(-1, 2)).reshape(
                class_count,
                50,
                2,
            )
        elif mode == "affine_tps":
            if transform is None:
                raise ValueError("affine_tps mode requires a grid transform")
            transformed = transform["apply_two_step"](pred_points.reshape(-1, 2)).reshape(class_count, 50, 2)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        new_row["mode_prediction"] = np.asarray(transformed, dtype=np.float32).reshape(class_count, 100)
        output.append(new_row)
    return output


def write_mode_contours(rows: list[dict[str, Any]], classes: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row["held"]:
            continue
        pred = row["mode_prediction"].reshape(len(classes), 50, 2)
        for idx, name in enumerate(classes):
            np.save(output_dir / f"{row['frame']}_{name}.npy", pred[idx].astype(np.float32))


def needed_integer_frames(rows: list[dict[str, Any]]) -> list[int]:
    needed: set[int] = set()
    for row in rows:
        frame_number = float(row["frame_number"])
        needed.add(int(math.floor(frame_number)))
        needed.add(int(math.ceil(frame_number)))
    return sorted(needed)


def build_dicom_mri_cache(dicom_dir: Path, rows: list[dict[str, Any]], output_dir: Path) -> dict[int, np.ndarray]:
    from src.utils.mri_rendering import build_filename_dicom_index, load_or_build_mri_cache

    dicom_index, _ = build_filename_dicom_index(dicom_dir)
    return load_or_build_mri_cache(
        dicom_dir,
        dicom_index,
        needed_integer_frames(rows),
        output_dir / "mri_frames_cache.npz",
        workers=8,
    )


def mri_for_frame(frame_number: float, cache: dict[int, np.ndarray]) -> np.ndarray:
    lower = int(math.floor(frame_number))
    upper = int(math.ceil(frame_number))
    if lower == upper:
        return cache[lower]
    alpha = frame_number - math.floor(frame_number)
    blended = (1.0 - alpha) * cache[lower].astype(np.float32) + alpha * cache[upper].astype(np.float32)
    return np.round(blended).astype(np.uint8)
