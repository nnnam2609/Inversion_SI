from __future__ import annotations

import numpy as np
import pytest
import torch

from src.utils.motion_metrics import (
    evaluate_arrays,
    masked_velocity_mse,
    velocity_class_weights,
    velocity_mask,
)


def test_velocity_mask_requires_both_adjacent_frames():
    mask = velocity_mask(torch.tensor([5, 2, 1, 0]), 5)
    assert mask.tolist() == [
        [True, True, True, True],
        [True, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]


def test_velocity_loss_ignores_padded_values():
    predicted = torch.zeros(2, 5, 2, 2)
    target = torch.ones_like(predicted)
    lengths = torch.tensor([5, 2])
    before = masked_velocity_mse(predicted, target, lengths)
    target[1, 2:] = 1e8
    after = masked_velocity_mse(predicted, target, lengths)
    assert torch.equal(before, after)


def test_class_weighting_can_exclude_static_velocity_only():
    rms = torch.tensor([0.01, 0.5, 1.0])
    weights = velocity_class_weights(rms, "mobile", mobile_threshold_ratio=0.25)
    assert weights[0].item() == 0.0
    assert weights[1].item() > 0
    assert weights[2].item() > 0
    assert weights.mean().item() == pytest.approx(1.0)


def test_evaluate_arrays_static_anchor_motion_and_dynamic_error():
    target = np.zeros((1, 4, 1, 2), dtype=np.float32)
    target[0, :, 0, 0] = [2, 3, 4, 5]
    static = np.repeat(target[:, :1], 4, axis=1)
    result = evaluate_arrays(static, target, np.array([4]))
    assert result["frame_diff_motion_ratio"] == 0.0
    assert result["coordinate_std_motion_ratio"] == 0.0
    assert result["velocity_pearson"] == 0.0
    assert result["dynamic_residual_rmse"] > 0


def test_denormalized_rmse_uses_xy_point_distance():
    target = np.zeros((1, 2, 1, 2), dtype=np.float32)
    predicted = target.copy()
    predicted[:, 1, 0] = [3, 4]
    result = evaluate_arrays(
        predicted,
        target,
        np.array([2]),
        std=np.ones((1, 1, 1, 2), dtype=np.float32),
        mean=np.zeros((1, 1, 1, 2), dtype=np.float32),
    )
    assert result["rmse_mm"] == pytest.approx(5.0)
