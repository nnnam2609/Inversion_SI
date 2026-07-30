"""Array-level contour errors with explicit frame aggregation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def frame_rmse_mm(
    predicted: np.ndarray,
    target: np.ndarray,
    indices: Sequence[int],
    *,
    coordinate_scale_mm: float = 1.62,
) -> np.ndarray:
    difference = (
        np.asarray(predicted)[:, list(indices)].astype(np.float64)
        - np.asarray(target)[:, list(indices)].astype(np.float64)
    )
    return (
        np.sqrt(np.mean(difference * difference, axis=(1, 2, 3)))
        * coordinate_scale_mm
    )


def per_class_frame_rmse_mm(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    coordinate_scale_mm: float = 1.62,
) -> np.ndarray:
    difference = (
        np.asarray(predicted).astype(np.float64)
        - np.asarray(target).astype(np.float64)
    )
    return (
        np.sqrt(np.mean(difference * difference, axis=(2, 3)))
        * coordinate_scale_mm
    )


def static_contour_rmse_mm(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    coordinate_scale_mm: float = 1.62,
) -> np.ndarray:
    difference = (
        np.asarray(predicted).astype(np.float64)
        - np.asarray(target).astype(np.float64)
    )
    return (
        np.sqrt(np.mean(difference * difference, axis=(1, 2)))
        * coordinate_scale_mm
    )
