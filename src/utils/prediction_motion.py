from __future__ import annotations

from typing import Any

import numpy as np
import torch


DEFAULT_STATIC_MOTION_RATIO_THRESHOLD = 0.20
DEFAULT_STATIC_COORD_STD_RATIO_THRESHOLD = 0.20
DEFAULT_FROZEN_FRAME_DIFF_THRESHOLD = 1e-5


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def select_payload_array(payload: dict[str, Any], raw: bool, labels: bool) -> np.ndarray | None:
    key = ("labels_raw" if raw else "labels") if labels else ("predicted_raw" if raw else "predicted")
    if key not in payload:
        return None
    return to_numpy(payload[key])


def flatten_valid_frames(payload: dict[str, Any], raw: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    predicted = select_payload_array(payload, raw=raw, labels=False)
    if predicted is None:
        raise KeyError("Payload must contain predicted/predicted_raw tensors")
    labels = select_payload_array(payload, raw=raw, labels=True)
    frames = to_numpy(payload["frames"])
    lengths = to_numpy(payload.get("lengths", np.full((predicted.shape[0],), predicted.shape[1]))).astype(int)

    pred_items = []
    label_items = []
    frame_items = []
    for seq_idx, length in enumerate(lengths):
        if length <= 0:
            continue
        pred_items.append(predicted[seq_idx, :length])
        if labels is not None:
            label_items.append(labels[seq_idx, :length])
        seq_frames = frames[seq_idx, :length]
        if seq_frames.ndim == 2 and seq_frames.shape[-1] >= 3:
            seq_frames = seq_frames[:, 2]
        frame_items.append(seq_frames.astype(float))

    if not pred_items:
        raise RuntimeError("Prediction payload has no valid frames after applying lengths")
    pred_flat = np.concatenate(pred_items, axis=0)
    label_flat = np.concatenate(label_items, axis=0) if label_items else None
    frame_flat = np.concatenate(frame_items, axis=0)
    return frame_flat, pred_flat, label_flat


def aggregate_unique_frames(
    frames: np.ndarray,
    predicted: np.ndarray,
    labels: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    unique_frames = np.unique(frames)
    pred_items = []
    label_items = []
    for frame in unique_frames:
        mask = frames == frame
        pred_items.append(np.nanmean(predicted[mask], axis=0))
        if labels is not None:
            label_items.append(np.nanmean(labels[mask], axis=0))
    return unique_frames, np.asarray(pred_items), np.asarray(label_items) if label_items else None


def motion_stats(values: np.ndarray) -> dict[str, float]:
    diffs = np.abs(np.diff(values, axis=0)) if len(values) > 1 else np.asarray([np.nan])
    return {
        "coord_std_mean": float(np.nanmean(np.nanstd(values, axis=0))),
        "frame_diff_mean_abs": float(np.nanmean(diffs)),
        "frame_diff_p95_abs": float(np.nanpercentile(diffs, 95)),
    }


def rmse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((first - second) ** 2)))


def per_articulator_rows(
    predicted: np.ndarray,
    labels: np.ndarray | None,
    classes: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for class_idx, class_name in enumerate(classes[: predicted.shape[1]]):
        row: dict[str, Any] = {"class": class_name}
        pred_motion = motion_stats(predicted[:, class_idx])
        row.update({f"pred_{key}": value for key, value in pred_motion.items()})
        if labels is not None:
            label_motion = motion_stats(labels[:, class_idx])
            row.update({f"label_{key}": value for key, value in label_motion.items()})
            row["motion_ratio_frame_diff"] = pred_motion["frame_diff_mean_abs"] / (
                label_motion["frame_diff_mean_abs"] + 1e-12
            )
            row["motion_ratio_coord_std"] = pred_motion["coord_std_mean"] / (
                label_motion["coord_std_mean"] + 1e-12
            )
            row["rmse"] = rmse(predicted[:, class_idx], labels[:, class_idx])
        rows.append(row)
    return rows


def prediction_motion_report(
    payload: dict[str, Any],
    classes: list[str],
    static_motion_ratio_threshold: float = DEFAULT_STATIC_MOTION_RATIO_THRESHOLD,
    static_coord_std_ratio_threshold: float = DEFAULT_STATIC_COORD_STD_RATIO_THRESHOLD,
    frozen_frame_diff_threshold: float = DEFAULT_FROZEN_FRAME_DIFF_THRESHOLD,
    mean_contour: np.ndarray | None = None,
    prediction_payload: str | None = None,
) -> dict[str, Any]:
    frames, predicted_raw_flat, labels_raw_flat = flatten_valid_frames(payload, raw=True)
    unique_frames, predicted_raw, labels_raw = aggregate_unique_frames(
        frames,
        predicted_raw_flat,
        labels_raw_flat,
    )
    predicted_raw_tensor = select_payload_array(payload, raw=True, labels=False)
    if predicted_raw_tensor is None:
        raise KeyError("Payload must contain predicted_raw")
    pred_motion = motion_stats(predicted_raw)
    is_frozen_prediction = bool(
        np.isfinite(pred_motion["frame_diff_mean_abs"])
        and pred_motion["frame_diff_mean_abs"] <= float(frozen_frame_diff_threshold)
    )
    report: dict[str, Any] = {
        "prediction_payload": prediction_payload,
        "prediction_only": bool(payload.get("prediction_only", False)),
        "num_sequences": int(predicted_raw_tensor.shape[0]),
        "num_unique_frames": int(len(unique_frames)),
        "frame_min": float(np.nanmin(unique_frames)),
        "frame_max": float(np.nanmax(unique_frames)),
        "predicted_raw_motion": pred_motion,
        "is_frozen_prediction": is_frozen_prediction,
        "frozen_frame_diff_threshold": float(frozen_frame_diff_threshold),
        "per_articulator": per_articulator_rows(predicted_raw, labels_raw, classes),
    }
    if labels_raw is not None:
        label_motion = motion_stats(labels_raw)
        pred_motion = report["predicted_raw_motion"]
        motion_ratio = pred_motion["frame_diff_mean_abs"] / (
            label_motion["frame_diff_mean_abs"] + 1e-12
        )
        coord_std_ratio = pred_motion["coord_std_mean"] / (
            label_motion["coord_std_mean"] + 1e-12
        )
        report.update(
            {
                "labels_raw_motion": label_motion,
                "motion_ratio_frame_diff": float(motion_ratio),
                "motion_ratio_coord_std": float(coord_std_ratio),
                "is_static_prediction": bool(motion_ratio < static_motion_ratio_threshold),
                "is_under_moving_coord_std": bool(coord_std_ratio < static_coord_std_ratio_threshold),
                "static_motion_ratio_threshold": float(static_motion_ratio_threshold),
                "static_coord_std_ratio_threshold": float(static_coord_std_ratio_threshold),
                "model_rmse": rmse(predicted_raw, labels_raw),
            }
        )
        if mean_contour is not None:
            mean_prediction = np.broadcast_to(mean_contour, predicted_raw.shape)
            mean_rmse = rmse(mean_prediction, labels_raw)
            report.update(
                {
                    "train_mean_contour_rmse": mean_rmse,
                    "model_vs_train_mean_delta_rmse": report["model_rmse"] - mean_rmse,
                    "model_to_train_mean_rmse": rmse(predicted_raw, mean_prediction),
                }
            )
    return report


def assert_prediction_motion(
    report: dict[str, Any],
    allow_static_prediction_diagnostic: bool = False,
) -> None:
    if allow_static_prediction_diagnostic:
        return
    if bool(report.get("is_frozen_prediction", False)):
        raise RuntimeError(
            "Prediction payload appears frozen. "
            f"pred_frame_diff={report['predicted_raw_motion']['frame_diff_mean_abs']:.8g} "
            f"threshold={report['frozen_frame_diff_threshold']:.8g}. "
            "This usually means the payload was built with stale normalization or a "
            "mismatched audio/frontend path. Re-run inference with the current std-floor "
            "policy, or pass the script's static-prediction diagnostic flag only for inspection."
        )
    if not bool(report.get("is_static_prediction", False)):
        if not bool(report.get("is_under_moving_coord_std", False)):
            return
        raise RuntimeError(
            "Prediction payload appears under-moving. "
            f"pred_coord_std={report['predicted_raw_motion']['coord_std_mean']:.8g} "
            f"label_coord_std={report['labels_raw_motion']['coord_std_mean']:.8g} "
            f"ratio={report['motion_ratio_coord_std']:.8g} "
            f"threshold={report['static_coord_std_ratio_threshold']:.8g}. "
            "This usually means the payload was built with stale normalization or a "
            "mismatched audio/frontend path. Re-run inference with the current std-floor "
            "policy, or pass the script's static-prediction diagnostic flag only for inspection."
        )
    raise RuntimeError(
        "Prediction payload appears nearly static. "
        f"pred_frame_diff={report['predicted_raw_motion']['frame_diff_mean_abs']:.8g} "
        f"label_frame_diff={report['labels_raw_motion']['frame_diff_mean_abs']:.8g} "
        f"ratio={report['motion_ratio_frame_diff']:.8g} "
        f"threshold={report['static_motion_ratio_threshold']:.8g}. "
        "This usually means the payload was built with stale normalization or a "
        "mismatched audio/frontend path. Re-run inference with the current std-floor "
        "policy, or pass the script's static-prediction diagnostic flag only for inspection."
    )
