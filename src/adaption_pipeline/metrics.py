"""Evaluation metrics with explicit units and pairing semantics."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from .contracts import ContractError


def _check_contours(predicted: np.ndarray, target: np.ndarray) -> None:
    if predicted.shape != target.shape:
        raise ContractError(
            f"Contour shape mismatch: predicted={predicted.shape}, target={target.shape}"
        )
    if predicted.ndim != 4 or predicted.shape[-1] != 2:
        raise ContractError(
            "Contours must have shape [frames, articulators, points, xy]"
        )
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ContractError("Contours contain NaN or infinity")


def coordinate_rmse_mm(
    predicted: np.ndarray, target: np.ndarray, coordinate_scale_mm: float = 1.0
) -> np.ndarray:
    """Per-frame/articulator coordinate RMSE converted to millimetres."""

    _check_contours(predicted, target)
    return (
        np.sqrt(np.mean(np.square(predicted - target), axis=(-1, -2)))
        * coordinate_scale_mm
    )


def _directed_point_to_curve(source: np.ndarray, curve: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(
        source[..., :, None, :] - curve[..., None, :, :], axis=-1
    )
    return distances.min(axis=-1).mean(axis=-1)


def p2cp_mm(
    predicted: np.ndarray,
    target: np.ndarray,
    symmetric: bool = True,
    coordinate_scale_mm: float = 1.0,
) -> np.ndarray:
    """Per-frame/articulator point-to-curve distance.

    With ``symmetric=True`` this is the mean of prediction-to-target and
    target-to-prediction directed distances. The output shape is
    ``[frames, articulators]``; aggregation across frames happens only after
    this calculation.
    """

    _check_contours(predicted, target)
    forward = _directed_point_to_curve(predicted, target)
    if not symmetric:
        return forward * coordinate_scale_mm
    backward = _directed_point_to_curve(target, predicted)
    return 0.5 * (forward + backward) * coordinate_scale_mm


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ContractError(
            f"Correlation pairing mismatch: {left.shape} != {right.shape}"
        )
    if left.size < 2:
        raise ContractError("At least two paired observations are required")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ContractError("Correlation inputs contain NaN or infinity")
    if np.std(left) == 0 or np.std(right) == 0:
        raise ContractError("Pearson correlation is undefined for constant input")
    return float(np.corrcoef(left, right)[0, 1])


def correlation_change(
    before: float, after: float, *, use_absolute_magnitude: bool = True
) -> Dict[str, Optional[float]]:
    """Report correlation change and percentage reduction after normalization.

    The default compares ``|r|`` because the objective is to reduce speaker
    dependence irrespective of sign.
    """

    if not math.isfinite(before) or not math.isfinite(after):
        raise ContractError("Correlation values must be finite")
    base = abs(before) if use_absolute_magnitude else before
    final = abs(after) if use_absolute_magnitude else after
    denominator = abs(base)
    reduction = base - final
    return {
        "correlation_before": before,
        "correlation_after": after,
        "absolute_change": after - before,
        "magnitude_reduction": reduction,
        "magnitude_reduction_percent": (
            None if denominator == 0 else 100.0 * reduction / denominator
        ),
    }


def correlation_increase(before: float, after: float) -> Dict[str, Optional[float]]:
    """Report target-to-reference correlation increase after normalization."""

    if not math.isfinite(before) or not math.isfinite(after):
        raise ContractError("Correlation values must be finite")
    increase = after - before
    denominator = abs(before)
    return {
        "correlation_before": before,
        "correlation_after": after,
        "absolute_increase": increase,
        "relative_increase_percent": (
            None if denominator == 0 else 100.0 * increase / denominator
        ),
    }


def signed_correlation_change(
    before: float, after: float
) -> Dict[str, Optional[float]]:
    """Report a direction-neutral signed correlation change.

    Positive values mean correlation increased after normalization, negative
    values mean it decreased, and zero means it was unchanged.
    """

    if not math.isfinite(before) or not math.isfinite(after):
        raise ContractError("Correlation values must be finite")
    change = after - before
    denominator = abs(before)
    tolerance = 1e-15
    direction = (
        "increased"
        if change > tolerance
        else "decreased"
        if change < -tolerance
        else "unchanged"
    )
    return {
        "correlation_before": before,
        "correlation_after": after,
        "signed_change": change,
        "signed_change_percent": (
            None if denominator == 0 else 100.0 * change / denominator
        ),
        "direction": direction,
    }


def contour_metric_summary(
    predicted: np.ndarray, target: np.ndarray, coordinate_scale_mm: float = 1.0
) -> Dict[str, Any]:
    rmse = coordinate_rmse_mm(predicted, target, coordinate_scale_mm)
    p2cp = p2cp_mm(
        predicted, target, symmetric=True, coordinate_scale_mm=coordinate_scale_mm
    )
    return {
        "unit": "mm",
        "aggregation_unit": "frame",
        "coordinate_rmse_mean": float(rmse.mean()),
        "coordinate_rmse_std": float(rmse.mean(axis=1).std(ddof=1))
        if rmse.shape[0] > 1
        else 0.0,
        "symmetric_p2cp_mean": float(p2cp.mean()),
        "symmetric_p2cp_std": float(p2cp.mean(axis=1).std(ddof=1))
        if p2cp.shape[0] > 1
        else 0.0,
        "num_frames": int(predicted.shape[0]),
        "num_articulators": int(predicted.shape[1]),
    }
