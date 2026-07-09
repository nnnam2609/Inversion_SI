from __future__ import annotations

import torch


def valid_velocity_mask(lengths: torch.Tensor, time_steps: int) -> torch.Tensor:
    """Return mask for valid frame-to-frame differences."""
    if time_steps <= 1:
        return torch.zeros((lengths.numel(), 0), dtype=torch.bool, device=lengths.device)
    steps = torch.arange(time_steps - 1, device=lengths.device).unsqueeze(0)
    return steps < (lengths.to(device=lengths.device).long().unsqueeze(1) - 1)


def contour_velocity_loss(
    target: torch.Tensor,
    predicted: torch.Tensor,
    lengths: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE between valid consecutive-frame contour deltas."""
    time_steps = min(int(target.shape[1]), int(predicted.shape[1]))
    if time_steps <= 1:
        return predicted.sum() * 0.0
    target_delta = target[:, 1:time_steps] - target[:, : time_steps - 1]
    pred_delta = predicted[:, 1:time_steps] - predicted[:, : time_steps - 1]
    mask = valid_velocity_mask(lengths, time_steps).to(device=predicted.device)
    if not bool(torch.any(mask)):
        return predicted.sum() * 0.0
    error = (pred_delta - target_delta) ** 2
    valid_error = error[mask]
    if reduction == "mean":
        return valid_error.mean()
    if reduction == "sum":
        return valid_error.sum()
    raise ValueError(f"Unsupported contour velocity loss reduction: {reduction!r}")
