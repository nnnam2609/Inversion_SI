"""Masked position, velocity, and motion metrics for contour rollouts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch


def frame_mask(
    lengths: torch.Tensor, time_steps: int, *, start: int = 1, device: torch.device | None = None
) -> torch.Tensor:
    device = device or lengths.device
    steps = torch.arange(time_steps, device=device).unsqueeze(0)
    return (steps >= int(start)) & (
        steps < lengths.to(device=device, dtype=torch.long).unsqueeze(1)
    )


def velocity_mask(
    lengths: torch.Tensor, time_steps: int, *, device: torch.device | None = None
) -> torch.Tensor:
    device = device or lengths.device
    if time_steps <= 1:
        return torch.zeros((lengths.numel(), 0), dtype=torch.bool, device=device)
    steps = torch.arange(1, time_steps, device=device).unsqueeze(0)
    return steps < lengths.to(device=device, dtype=torch.long).unsqueeze(1)


def masked_position_mse(
    predicted: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    time_steps = min(predicted.shape[1], target.shape[1])
    mask = frame_mask(lengths, time_steps, start=1, device=predicted.device)
    if not bool(mask.any()):
        return predicted.sum() * 0.0
    return ((predicted[:, :time_steps] - target[:, :time_steps]) ** 2)[mask].mean()


def masked_velocity_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    time_steps = min(predicted.shape[1], target.shape[1])
    mask = velocity_mask(lengths, time_steps, device=predicted.device)
    if not bool(mask.any()):
        return predicted.sum() * 0.0
    error = (
        (predicted[:, 1:time_steps] - predicted[:, : time_steps - 1])
        - (target[:, 1:time_steps] - target[:, : time_steps - 1])
    ) ** 2
    selected = error[mask]
    if class_weights is None:
        return selected.mean()
    weights = class_weights.to(device=selected.device, dtype=selected.dtype)
    if selected.ndim < 3 or selected.shape[-2] != weights.numel():
        raise ValueError("class weights must match the articulator dimension")
    shape = [1] * selected.ndim
    shape[-2] = weights.numel()
    weights = weights.reshape(shape)
    denominator = weights.sum() * selected.shape[0] * selected.shape[-1]
    return (selected * weights).sum() / denominator.clamp_min(1e-12)


def autoregressive_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    *,
    velocity_weight: float,
    class_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    position = masked_position_mse(predicted, target, lengths)
    velocity = masked_velocity_mse(predicted, target, lengths, class_weights)
    return {
        "loss": position + float(velocity_weight) * velocity,
        "position_mse": position,
        "velocity_mse": velocity,
    }


def train_velocity_rms(
    labels: torch.Tensor, lengths: Iterable[int], *, batch_size: int = 128
) -> torch.Tensor:
    """Calculate train-only per-articulator normalized velocity RMS."""
    squared_sum = torch.zeros(labels.shape[-2], dtype=torch.float64)
    count = torch.zeros(labels.shape[-2], dtype=torch.float64)
    length_tensor = torch.as_tensor(list(lengths), dtype=torch.long)
    for start in range(0, labels.shape[0], batch_size):
        stop = min(start + batch_size, labels.shape[0])
        batch = labels[start:stop]
        mask = velocity_mask(length_tensor[start:stop], batch.shape[1], device=batch.device)
        if not bool(mask.any()):
            continue
        delta = batch[:, 1:] - batch[:, :-1]
        valid = delta[mask].double()
        squared_sum += valid.square().sum(dim=(0, 2)).cpu()
        count += valid.shape[0] * valid.shape[-1]
    return torch.sqrt(squared_sum / count.clamp_min(1))


def velocity_class_weights(
    velocity_rms: torch.Tensor,
    mode: str,
    *,
    mobile_threshold_ratio: float = 0.25,
) -> torch.Tensor:
    rms = velocity_rms.float()
    if mode in {"none", "uniform", "all"}:
        return torch.ones_like(rms)
    if mode == "rms":
        return rms / rms.mean().clamp_min(1e-12)
    if mode == "mobile":
        threshold = rms.max() * float(mobile_threshold_ratio)
        result = (rms >= threshold).float()
        if not bool(result.any()):
            raise RuntimeError("mobile class selection produced no classes")
        return result / result.mean()
    raise ValueError(f"Unknown velocity class weighting mode: {mode!r}")


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    first = first.reshape(-1)
    second = second.reshape(-1)
    finite = np.isfinite(first) & np.isfinite(second)
    if finite.sum() < 2:
        return float("nan")
    first = first[finite]
    second = second[finite]
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _point_errors(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    difference = predicted - target
    if difference.shape[-1] % 2:
        return np.sqrt(np.mean(difference**2, axis=-1))
    pairs = difference.reshape(*difference.shape[:-1], -1, 2)
    return np.sqrt(np.sum(pairs**2, axis=-1))


def evaluate_arrays(
    predicted: np.ndarray,
    target: np.ndarray,
    lengths: np.ndarray,
    *,
    std: np.ndarray | None = None,
    mean: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate sequences without ever joining non-contiguous cache chunks."""
    predicted = np.asarray(predicted)
    target = np.asarray(target)
    lengths = np.asarray(lengths, dtype=int)
    if std is None:
        predicted_raw = predicted
        target_raw = target
    else:
        predicted_raw = predicted * np.asarray(std) + np.asarray(mean)
        target_raw = target * np.asarray(std) + np.asarray(mean)

    pred_frames: list[np.ndarray] = []
    target_frames: list[np.ndarray] = []
    pred_velocities: list[np.ndarray] = []
    target_velocities: list[np.ndarray] = []
    pred_accelerations: list[np.ndarray] = []
    target_accelerations: list[np.ndarray] = []
    pred_dynamic: list[np.ndarray] = []
    target_dynamic: list[np.ndarray] = []
    pred_temporal_stds: list[np.ndarray] = []
    target_temporal_stds: list[np.ndarray] = []
    normalized_errors: list[np.ndarray] = []
    for index, length in enumerate(lengths):
        length = min(int(length), predicted.shape[1], target.shape[1])
        if length <= 1:
            continue
        pred = predicted_raw[index, :length]
        truth = target_raw[index, :length]
        pred_frames.append(pred[1:])
        target_frames.append(truth[1:])
        pred_velocities.append(np.diff(pred, axis=0))
        target_velocities.append(np.diff(truth, axis=0))
        if length > 2:
            pred_accelerations.append(np.diff(pred, n=2, axis=0))
            target_accelerations.append(np.diff(truth, n=2, axis=0))
        # Every method is evaluated against the declared known contour C0.
        # This matters for the audio-only baseline, whose own prediction[0] is
        # not the supplied reference.
        pred_dynamic.append(pred[1:] - truth[0])
        target_dynamic.append(truth[1:] - truth[0])
        pred_temporal_stds.append(np.std(pred[1:].astype(np.float64), axis=0))
        target_temporal_stds.append(np.std(truth[1:].astype(np.float64), axis=0))
        normalized_errors.append((predicted[index, 1:length] - target[index, 1:length]) ** 2)
    if not pred_frames:
        raise ValueError("No sequence contains a predicted frame after the anchor")

    pred_f = np.concatenate(pred_frames)
    target_f = np.concatenate(target_frames)
    pred_v = np.concatenate(pred_velocities)
    target_v = np.concatenate(target_velocities)
    pred_d = np.concatenate(pred_dynamic)
    target_d = np.concatenate(target_dynamic)
    pred_a = np.concatenate(pred_accelerations) if pred_accelerations else np.zeros((1,))
    target_a = np.concatenate(target_accelerations) if target_accelerations else np.zeros((1,))
    point_error = _point_errors(pred_f, target_f)
    frame_diff_pred = float(np.mean(np.abs(pred_v)))
    frame_diff_target = float(np.mean(np.abs(target_v)))
    # Temporal variability must be calculated within each rollout. Computing a
    # global std would incorrectly treat differing static anchors as motion.
    coord_std_pred = float(np.mean(np.stack(pred_temporal_stds)))
    coord_std_target = float(np.mean(np.stack(target_temporal_stds)))
    result: dict[str, Any] = {
        "normalized_mse": float(np.mean(np.concatenate(normalized_errors))),
        "rmse_mm": float(np.sqrt(np.mean(point_error**2))),
        "median_error_mm": float(np.median(point_error)),
        "predicted_frame_diff_mean_abs": frame_diff_pred,
        "target_frame_diff_mean_abs": frame_diff_target,
        "frame_diff_motion_ratio": frame_diff_pred / (frame_diff_target + 1e-12),
        "predicted_coordinate_temporal_std": coord_std_pred,
        "target_coordinate_temporal_std": coord_std_target,
        "coordinate_std_motion_ratio": coord_std_pred / (coord_std_target + 1e-12),
        "velocity_rmse": float(np.sqrt(np.mean((pred_v - target_v) ** 2))),
        "velocity_pearson": _pearson(pred_v, target_v),
        "acceleration_rmse": float(np.sqrt(np.mean((pred_a - target_a) ** 2))),
        "dynamic_residual_rmse": float(np.sqrt(np.mean((pred_d - target_d) ** 2))),
        "num_valid_predicted_frames": int(sum(max(0, int(item) - 1) for item in lengths)),
    }
    return result
