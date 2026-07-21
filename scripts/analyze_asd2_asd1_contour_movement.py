#!/usr/bin/env python3
"""Compare integer-frame ASD2-model contour movement with ASD1 ground truth."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "results/asd2_model_to_asd1_selected_sessions_integer_only_20260718"
DEFAULT_OUTPUT = REPO_ROOT / "results/asd2_model_to_asd1_9session_contour_movement_integer_only_20260718"
SELECTION = ((1, 16), (2, 9), (3, 14), (4, 4), (5, 6), (6, 8), (8, 2), (9, 5), (10, 14))
PACK_NAME = "asd2_model_predictions_and_ground_truth_integer_only.npz"
EXCLUDED_CLASS = "lower-incisor"
MM_PER_PIXEL = 1.62
FPS = 50.0
MAX_LAG_FRAMES = 10
FRAME_POLICY = "NEVER save, score, or analyze fractional frames; integer frames only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def finite_corr(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    mask = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    first = first[mask]
    second = second[mask]
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def speed_trace(delta: np.ndarray) -> np.ndarray:
    squared_norm = np.sum(np.asarray(delta, dtype=np.float64) ** 2, axis=-1)
    reduce_axes = tuple(range(1, squared_norm.ndim))
    return np.sqrt(np.mean(squared_norm, axis=reduce_axes))


def active_threshold(gt_speed: np.ndarray) -> float:
    median = float(np.median(gt_speed))
    mad = float(np.median(np.abs(gt_speed - median)))
    threshold = median + mad
    if np.count_nonzero(gt_speed > threshold) < min(5, len(gt_speed)):
        threshold = float(np.quantile(gt_speed, 0.75))
    return threshold


def direction_trace(predicted_delta: np.ndarray, gt_delta: np.ndarray) -> np.ndarray:
    predicted = predicted_delta.reshape(len(predicted_delta), -1).astype(np.float64)
    ground_truth = gt_delta.reshape(len(gt_delta), -1).astype(np.float64)
    numerator = np.sum(predicted * ground_truth, axis=1)
    denominator = np.linalg.norm(predicted, axis=1) * np.linalg.norm(ground_truth, axis=1)
    output = np.full(len(predicted), np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    output[valid] = numerator[valid] / denominator[valid]
    return output


def lag_correlation(
    predicted_speed: np.ndarray,
    gt_speed: np.ndarray,
    pair_end_frames: np.ndarray,
) -> tuple[float, int]:
    frame_to_index = {int(frame): index for index, frame in enumerate(pair_end_frames)}
    best_correlation = float("nan")
    best_lag = 0
    for lag in range(-MAX_LAG_FRAMES, MAX_LAG_FRAMES + 1):
        predicted_values = []
        gt_values = []
        for gt_index, frame in enumerate(pair_end_frames):
            predicted_index = frame_to_index.get(int(frame) + lag)
            if predicted_index is None:
                continue
            predicted_values.append(predicted_speed[predicted_index])
            gt_values.append(gt_speed[gt_index])
        correlation = finite_corr(np.asarray(predicted_values), np.asarray(gt_values))
        if np.isfinite(correlation) and (not np.isfinite(best_correlation) or correlation > best_correlation):
            best_correlation = correlation
            best_lag = lag
    return best_correlation, best_lag


def movement_metrics(
    predicted_delta: np.ndarray,
    gt_delta: np.ndarray,
    pair_end_frames: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if len(predicted_delta) == 0:
        raise ValueError("Movement metrics require at least one adjacent frame pair")
    predicted_speed = speed_trace(predicted_delta)
    gt_speed = speed_trace(gt_delta)
    threshold = active_threshold(gt_speed)
    active = gt_speed > threshold
    rest = ~active
    difference = predicted_delta.astype(np.float64) - gt_delta.astype(np.float64)
    vector_rmse = float(np.sqrt(np.mean(difference**2)))
    gt_vector_rms = float(np.sqrt(np.mean(gt_delta.astype(np.float64) ** 2)))
    direction = direction_trace(predicted_delta, gt_delta)
    active_vector_rmse = (
        float(np.sqrt(np.mean(difference[active] ** 2))) if np.any(active) else float("nan")
    )
    consecutive_motion = np.diff(pair_end_frames) == 1
    if np.any(consecutive_motion):
        predicted_acceleration = np.diff(predicted_delta, axis=0)[consecutive_motion]
        gt_acceleration = np.diff(gt_delta, axis=0)[consecutive_motion]
        predicted_acceleration_rms = float(np.sqrt(np.mean(predicted_acceleration.astype(np.float64) ** 2)))
        gt_acceleration_rms = float(np.sqrt(np.mean(gt_acceleration.astype(np.float64) ** 2)))
        acceleration_error = float(
            np.sqrt(np.mean((predicted_acceleration.astype(np.float64) - gt_acceleration.astype(np.float64)) ** 2))
        )
    else:
        predicted_acceleration_rms = gt_acceleration_rms = acceleration_error = float("nan")
    best_correlation, best_lag = lag_correlation(predicted_speed, gt_speed, pair_end_frames)
    gt_path = float(np.sum(gt_speed))
    predicted_path = float(np.sum(predicted_speed))
    metrics = {
        "valid_frame_pairs": int(len(pair_end_frames)),
        "active_frame_pairs": int(np.count_nonzero(active)),
        "rest_frame_pairs": int(np.count_nonzero(rest)),
        "active_threshold_gt_mm_per_frame": threshold,
        "movement_vector_rmse_mm_per_frame": vector_rmse,
        "active_movement_vector_rmse_mm_per_frame": active_vector_rmse,
        "gt_movement_vector_rms_mm_per_frame": gt_vector_rms,
        "movement_nrmse": vector_rmse / (gt_vector_rms + 1e-12),
        "gt_mean_speed_mm_per_frame": float(np.mean(gt_speed)),
        "pred_mean_speed_mm_per_frame": float(np.mean(predicted_speed)),
        "gt_mean_speed_mm_per_s": float(np.mean(gt_speed) * FPS),
        "pred_mean_speed_mm_per_s": float(np.mean(predicted_speed) * FPS),
        "speed_mae_mm_per_frame": float(np.mean(np.abs(predicted_speed - gt_speed))),
        "gt_path_length_mm": gt_path,
        "pred_path_length_mm": predicted_path,
        "path_ratio_pred_over_gt": predicted_path / (gt_path + 1e-12),
        "speed_correlation_zero_lag": finite_corr(predicted_speed, gt_speed),
        "speed_correlation_best": best_correlation,
        "best_lag_frames_positive_means_prediction_lags": int(best_lag),
        "best_lag_ms_positive_means_prediction_lags": float(best_lag * 1000.0 / FPS),
        "active_direction_cosine": float(np.nanmean(direction[active])) if np.any(active) else float("nan"),
        "rest_gt_mean_speed_mm_per_frame": float(np.mean(gt_speed[rest])) if np.any(rest) else float("nan"),
        "rest_pred_mean_speed_mm_per_frame": float(np.mean(predicted_speed[rest])) if np.any(rest) else float("nan"),
        "rest_excess_pred_minus_gt_mm_per_frame": (
            float(np.mean(predicted_speed[rest] - gt_speed[rest])) if np.any(rest) else float("nan")
        ),
        "gt_acceleration_rms_mm_per_frame2": gt_acceleration_rms,
        "pred_acceleration_rms_mm_per_frame2": predicted_acceleration_rms,
        "acceleration_error_rmse_mm_per_frame2": acceleration_error,
        "jitter_ratio_pred_over_gt": predicted_acceleration_rms / (gt_acceleration_rms + 1e-12),
    }
    traces = {
        "predicted_speed": predicted_speed,
        "gt_speed": gt_speed,
        "active": active,
        "direction": direction,
        "per_pair_vector_rmse": np.sqrt(
            np.mean(difference**2, axis=tuple(range(1, difference.ndim)))
        ),
    }
    return metrics, traces


def load_session(pack_path: Path) -> dict[str, Any]:
    with np.load(pack_path, allow_pickle=False) as payload:
        frames = np.asarray(payload["frame_numbers"])
        predicted = np.asarray(payload["predicted"], dtype=np.float32)
        ground_truth = np.asarray(payload["ground_truth"], dtype=np.float32)
        classes = [str(value) for value in payload["classes"].tolist()]
    if not np.allclose(frames, np.rint(frames), atol=0.0):
        raise ValueError(f"Fractional frame found in {pack_path}")
    if predicted.shape != ground_truth.shape or predicted.shape[1:] != (len(classes), 50, 2):
        raise ValueError(f"Unexpected contour shapes in {pack_path}: {predicted.shape}, {ground_truth.shape}")
    keep_indices = [index for index, name in enumerate(classes) if name != EXCLUDED_CLASS]
    kept_classes = [classes[index] for index in keep_indices]
    if len(keep_indices) != len(classes) - 1:
        raise ValueError(f"Expected exactly one {EXCLUDED_CLASS} class in {pack_path}")
    frames = np.rint(frames).astype(np.int32)
    adjacent = np.diff(frames) == 1
    return {
        "frames": frames,
        "pair_end_frames": frames[1:][adjacent],
        "predicted_delta": np.diff(predicted[:, keep_indices], axis=0)[adjacent] * MM_PER_PIXEL,
        "ground_truth_delta": np.diff(ground_truth[:, keep_indices], axis=0)[adjacent] * MM_PER_PIXEL,
        "classes": kept_classes,
        "source_classes": classes,
        "num_gaps": int(np.count_nonzero(~adjacent)),
        "pack": str(pack_path.resolve()),
    }


def numeric_macro(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    result = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = float(np.nanmean(values))
    return result


def write_speed_plot(
    output: Path,
    label: str,
    frames: np.ndarray,
    traces: dict[str, np.ndarray],
) -> None:
    fig, axis = plt.subplots(figsize=(13, 4.2))
    segment_starts = np.r_[0, np.flatnonzero(np.diff(frames) != 1) + 1]
    segment_stops = np.r_[segment_starts[1:], len(frames)]
    for segment_index, (start, stop) in enumerate(zip(segment_starts, segment_stops)):
        axis.plot(
            frames[start:stop],
            traces["gt_speed"][start:stop],
            color="#222222",
            linewidth=1.25,
            label="ground truth" if segment_index == 0 else None,
        )
        axis.plot(
            frames[start:stop],
            traces["predicted_speed"][start:stop],
            color="#e76f51",
            linewidth=1.0,
            alpha=0.9,
            label="ASD2 model" if segment_index == 0 else None,
        )
    active = traces["active"]
    if np.any(active):
        axis.scatter(frames[active], traces["gt_speed"][active], s=5, color="#2a9d8f", alpha=0.55, label="GT active")
    axis.set_title(f"{label}: 10-contour movement speed (lower-incisor excluded)")
    axis.set_xlabel("integer MRI frame")
    axis.set_ylabel("RMS point movement (mm/frame)")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def heatmap(
    frame: pd.DataFrame,
    value: str,
    title: str,
    output: Path,
    center: float | None = None,
) -> None:
    pivot = frame.pivot(index="session_label", columns="contour", values=value)
    fig, axis = plt.subplots(figsize=(14, 6))
    values = pivot.to_numpy(dtype=float)
    if center is None:
        image = axis.imshow(values, aspect="auto", cmap="viridis")
    else:
        span = max(abs(float(np.nanmin(values)) - center), abs(float(np.nanmax(values)) - center), 1e-6)
        image = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=center - span, vmax=center + span)
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=7,
                      color="white" if abs(values[row, column] - np.nanmean(values)) > np.nanstd(values) * 0.5 else "black")
    axis.set_title(title)
    fig.colorbar(image, ax=axis, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def session_summary_plot(session_frame: pd.DataFrame, output: Path) -> None:
    labels = session_frame["session_label"].tolist()
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].bar(positions, session_frame["movement_nrmse"], color="#457b9d")
    axes[0].set_title("Movement NRMSE (lower is better)")
    axes[1].bar(positions, session_frame["path_ratio_pred_over_gt"], color="#e9c46a")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Predicted / GT path ratio")
    axes[2].bar(positions, session_frame["speed_correlation_zero_lag"], color="#2a9d8f")
    axes[2].set_ylim(-1, 1)
    axes[2].set_title("Movement-speed correlation")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=40, ha="right")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("ASD2-model movement versus ASD1 ground truth: 10 contours")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def contour_summary_plot(contour_frame: pd.DataFrame, output: Path) -> None:
    positions = np.arange(len(contour_frame))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].bar(positions, contour_frame["movement_nrmse"], color="#457b9d")
    axes[0].set_title("Macro movement NRMSE")
    axes[1].bar(positions, contour_frame["path_ratio_pred_over_gt"], color="#e9c46a")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Macro path ratio")
    axes[2].bar(positions, contour_frame["speed_correlation_zero_lag"], color="#2a9d8f")
    axes[2].set_ylim(-1, 1)
    axes[2].set_title("Macro speed correlation")
    for axis in axes:
        axis.set_xticks(positions, contour_frame["contour"], rotation=40, ha="right")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Per-contour movement agreement across 9 unseen ASD1 speakers")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def format_value(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "N/A" if not np.isfinite(number) else f"{number:.{digits}f}"


def write_report(
    path: Path,
    session_frame: pd.DataFrame,
    contour_frame: pd.DataFrame,
    overall_macro: dict[str, Any],
    overall_micro: dict[str, Any],
    total_frames: int,
    total_pairs: int,
    total_gaps: int,
) -> None:
    best_session = session_frame.loc[session_frame["movement_nrmse"].idxmin()]
    worst_session = session_frame.loc[session_frame["movement_nrmse"].idxmax()]
    best_contour = contour_frame.loc[contour_frame["movement_nrmse"].idxmin()]
    worst_contour = contour_frame.loc[contour_frame["movement_nrmse"].idxmax()]
    under = contour_frame[contour_frame["path_ratio_pred_over_gt"] < 0.90]
    over = contour_frame[contour_frame["path_ratio_pred_over_gt"] > 1.10]
    lines = [
        "# ASD2-model contour movement vs ASD1 ground truth",
        "",
        f"- Frame policy: **{FRAME_POLICY}**.",
        "- Evaluated speakers: P1, P2, P3, P4, P5, P6, P8, P9, P10 (one selected session each).",
        f"- Excluded contour: **{EXCLUDED_CLASS}**; evaluated contours: **10**.",
        f"- Integer source frames: **{total_frames}**; adjacent frame pairs scored: **{total_pairs}**.",
        f"- Timeline gaps skipped without interpolation: **{total_gaps}**.",
        "- Fractional frames saved/scored/analyzed: **0**.",
        "- Units: millimetres using 1.62 mm/pixel; speed conversion uses 50 fps.",
        "",
        "## Main result",
        "",
        "| Aggregation | Movement RMSE (mm/frame) | Movement NRMSE | GT speed | Pred speed | Path ratio | Speed corr. | Active direction | Jitter ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| Macro (equal speaker weight) | {format_value(overall_macro['movement_vector_rmse_mm_per_frame'])} | "
            f"{format_value(overall_macro['movement_nrmse'])} | {format_value(overall_macro['gt_mean_speed_mm_per_frame'])} | "
            f"{format_value(overall_macro['pred_mean_speed_mm_per_frame'])} | {format_value(overall_macro['path_ratio_pred_over_gt'])} | "
            f"{format_value(overall_macro['speed_correlation_zero_lag'])} | {format_value(overall_macro['active_direction_cosine'])} | "
            f"{format_value(overall_macro['jitter_ratio_pred_over_gt'])} |"
        ),
        (
            f"| Micro (all valid pairs) | {format_value(overall_micro['movement_vector_rmse_mm_per_frame'])} | "
            f"{format_value(overall_micro['movement_nrmse'])} | {format_value(overall_micro['gt_mean_speed_mm_per_frame'])} | "
            f"{format_value(overall_micro['pred_mean_speed_mm_per_frame'])} | {format_value(overall_micro['path_ratio_pred_over_gt'])} | "
            f"{format_value(overall_micro['speed_correlation_zero_lag'])} | {format_value(overall_micro['active_direction_cosine'])} | "
            f"{format_value(overall_micro['jitter_ratio_pred_over_gt'])} |"
        ),
        "",
        "Movement NRMSE is the pointwise displacement-vector RMSE divided by GT displacement RMS. Path ratio below 1 means under-moving; above 1 means over-moving or jitter. Positive best lag means prediction occurs later than GT.",
        "",
        "## Per speaker/session",
        "",
        "| Session | Pairs | RMSE | NRMSE | GT speed | Pred speed | Path ratio | Corr. | Best lag | Direction | Rest excess | Jitter |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in session_frame.iterrows():
        lines.append(
            f"| {row['session_label']} | {int(row['valid_frame_pairs'])} | "
            f"{format_value(row['movement_vector_rmse_mm_per_frame'])} | {format_value(row['movement_nrmse'])} | "
            f"{format_value(row['gt_mean_speed_mm_per_frame'])} | {format_value(row['pred_mean_speed_mm_per_frame'])} | "
            f"{format_value(row['path_ratio_pred_over_gt'])} | {format_value(row['speed_correlation_zero_lag'])} | "
            f"{int(row['best_lag_frames_positive_means_prediction_lags']):+d} | {format_value(row['active_direction_cosine'])} | "
            f"{format_value(row['rest_excess_pred_minus_gt_mm_per_frame'])} | {format_value(row['jitter_ratio_pred_over_gt'])} |"
        )
    lines.extend(
        [
            "",
            "## Per contour (macro across 9 speakers)",
            "",
            "| Contour | RMSE | NRMSE | GT speed | Pred speed | Path ratio | Corr. | Best lag | Direction | Rest excess | Jitter |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in contour_frame.iterrows():
        lines.append(
            f"| {row['contour']} | {format_value(row['movement_vector_rmse_mm_per_frame'])} | "
            f"{format_value(row['movement_nrmse'])} | {format_value(row['gt_mean_speed_mm_per_frame'])} | "
            f"{format_value(row['pred_mean_speed_mm_per_frame'])} | {format_value(row['path_ratio_pred_over_gt'])} | "
            f"{format_value(row['speed_correlation_zero_lag'])} | "
            f"{format_value(row['best_lag_frames_positive_means_prediction_lags'], 1)} | "
            f"{format_value(row['active_direction_cosine'])} | {format_value(row['rest_excess_pred_minus_gt_mm_per_frame'])} | "
            f"{format_value(row['jitter_ratio_pred_over_gt'])} |"
        )
    under_names = ", ".join(under["contour"].tolist()) if len(under) else "none"
    over_names = ", ".join(over["contour"].tolist()) if len(over) else "none"
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Best session by movement NRMSE: **{best_session['session_label']}** ({best_session['movement_nrmse']:.3f}); worst: **{worst_session['session_label']}** ({worst_session['movement_nrmse']:.3f}).",
            f"- Best contour by movement NRMSE: **{best_contour['contour']}** ({best_contour['movement_nrmse']:.3f}); worst: **{worst_contour['contour']}** ({worst_contour['movement_nrmse']:.3f}).",
            f"- Systematically under-moving contours (macro path ratio < 0.90): **{under_names}**.",
            f"- Over-moving/jitter candidates (macro path ratio > 1.10): **{over_names}**.",
            "- Rest excess measures prediction movement on low-GT-movement frames using only the contour-derived GT threshold; it is not an audio/TextGrid silence label.",
            "- These are temporal contour-movement metrics. They intentionally do not measure absolute contour position or static anatomical offset.",
            "",
            "## Figures",
            "",
            "- [Session metric summary](plots/session_metric_summary.png)",
            "- [Movement NRMSE heatmap](plots/movement_nrmse_heatmap.png)",
            "- [Path ratio heatmap](plots/path_ratio_heatmap.png)",
            "- [Speed correlation heatmap](plots/speed_correlation_heatmap.png)",
            "- [Per-contour macro metrics](plots/per_contour_macro_metrics.png)",
            "- Per-session overall speed traces are in `plots/speed_traces/`.",
            "",
            "## Data files",
            "",
            "- [Per-session summary](per_session_movement_metrics.csv)",
            "- [Per-session/per-contour metrics](per_session_per_contour_movement_metrics.csv)",
            "- [Per-contour macro metrics](per_contour_macro_movement_metrics.csv)",
            "- [Per-frame-pair long metrics](per_frame_pair_per_contour_movement.csv)",
            "- [Machine-readable summary](movement_summary.json)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_root / "plots"
    trace_dir = plot_dir / "speed_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    session_rows: list[dict[str, Any]] = []
    contour_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    all_predicted_delta = []
    all_gt_delta = []
    all_pair_end_frames = []
    pair_frame_cursor = 0
    total_frames = total_gaps = 0
    expected_classes: list[str] | None = None

    for speaker, session in SELECTION:
        label = f"P{speaker}/S{session}"
        pack_path = args.input_root / f"P{speaker}/S{session}/{PACK_NAME}"
        if not pack_path.is_file():
            raise FileNotFoundError(pack_path)
        data = load_session(pack_path)
        if expected_classes is None:
            expected_classes = data["classes"]
        elif data["classes"] != expected_classes:
            raise ValueError(f"Class mismatch in {pack_path}")
        total_frames += len(data["frames"])
        total_gaps += data["num_gaps"]
        overall_metrics, overall_traces = movement_metrics(
            data["predicted_delta"], data["ground_truth_delta"], data["pair_end_frames"]
        )
        session_row = {
            "speaker": f"P{speaker}",
            "session": f"S{session}",
            "session_label": label,
            "source_frames": int(len(data["frames"])),
            "timeline_gaps_skipped": data["num_gaps"],
            "source_pack": data["pack"],
            **overall_metrics,
        }
        session_rows.append(session_row)
        write_speed_plot(
            trace_dir / f"p{speaker}_s{session}_overall_speed.png",
            label,
            data["pair_end_frames"],
            overall_traces,
        )
        all_predicted_delta.append(data["predicted_delta"])
        all_gt_delta.append(data["ground_truth_delta"])
        remapped_pair_frames = (
            data["pair_end_frames"] - int(data["pair_end_frames"][0]) + pair_frame_cursor
        )
        all_pair_end_frames.append(remapped_pair_frames)
        pair_frame_cursor = int(remapped_pair_frames[-1]) + MAX_LAG_FRAMES + 2

        for class_index, class_name in enumerate(data["classes"]):
            metrics, traces = movement_metrics(
                data["predicted_delta"][:, class_index],
                data["ground_truth_delta"][:, class_index],
                data["pair_end_frames"],
            )
            contour_rows.append(
                {
                    "speaker": f"P{speaker}",
                    "session": f"S{session}",
                    "session_label": label,
                    "contour": class_name,
                    **metrics,
                }
            )
            for pair_index, frame_end in enumerate(data["pair_end_frames"]):
                frame_rows.append(
                    {
                        "speaker": f"P{speaker}",
                        "session": f"S{session}",
                        "session_label": label,
                        "contour": class_name,
                        "frame_start": int(frame_end - 1),
                        "frame_end": int(frame_end),
                        "gt_speed_mm_per_frame": float(traces["gt_speed"][pair_index]),
                        "pred_speed_mm_per_frame": float(traces["predicted_speed"][pair_index]),
                        "speed_error_pred_minus_gt_mm_per_frame": float(
                            traces["predicted_speed"][pair_index] - traces["gt_speed"][pair_index]
                        ),
                        "movement_vector_rmse_mm_per_frame": float(traces["per_pair_vector_rmse"][pair_index]),
                        "direction_cosine": float(traces["direction"][pair_index]),
                        "gt_active_from_contour_threshold": bool(traces["active"][pair_index]),
                        "active_threshold_gt_mm_per_frame": metrics["active_threshold_gt_mm_per_frame"],
                    }
                )
        print(
            f"DONE {label}: frames={len(data['frames'])}, pairs={len(data['pair_end_frames'])}, "
            f"NRMSE={overall_metrics['movement_nrmse']:.3f}, "
            f"path={overall_metrics['path_ratio_pred_over_gt']:.3f}, "
            f"corr={overall_metrics['speed_correlation_zero_lag']:.3f}",
            flush=True,
        )

    session_frame = pd.DataFrame(session_rows)
    contour_session_frame = pd.DataFrame(contour_rows)
    frame_pair_frame = pd.DataFrame(frame_rows)
    metric_keys = [
        "movement_vector_rmse_mm_per_frame",
        "active_movement_vector_rmse_mm_per_frame",
        "gt_movement_vector_rms_mm_per_frame",
        "movement_nrmse",
        "gt_mean_speed_mm_per_frame",
        "pred_mean_speed_mm_per_frame",
        "gt_mean_speed_mm_per_s",
        "pred_mean_speed_mm_per_s",
        "speed_mae_mm_per_frame",
        "path_ratio_pred_over_gt",
        "speed_correlation_zero_lag",
        "speed_correlation_best",
        "best_lag_frames_positive_means_prediction_lags",
        "best_lag_ms_positive_means_prediction_lags",
        "active_direction_cosine",
        "rest_gt_mean_speed_mm_per_frame",
        "rest_pred_mean_speed_mm_per_frame",
        "rest_excess_pred_minus_gt_mm_per_frame",
        "gt_acceleration_rms_mm_per_frame2",
        "pred_acceleration_rms_mm_per_frame2",
        "acceleration_error_rmse_mm_per_frame2",
        "jitter_ratio_pred_over_gt",
    ]
    overall_macro = numeric_macro(session_rows, metric_keys)

    concatenated_predicted = np.concatenate(all_predicted_delta, axis=0)
    concatenated_gt = np.concatenate(all_gt_delta, axis=0)
    concatenated_pair_frames = np.concatenate(all_pair_end_frames).astype(np.int32)
    overall_micro, _ = movement_metrics(
        concatenated_predicted, concatenated_gt, concatenated_pair_frames
    )
    overall_micro["valid_frame_pairs"] = int(sum(row["valid_frame_pairs"] for row in session_rows))

    contour_macro_rows = []
    for contour in expected_classes or []:
        rows = [row for row in contour_rows if row["contour"] == contour]
        contour_macro_rows.append({"contour": contour, "speakers": len(rows), **numeric_macro(rows, metric_keys)})
    contour_macro_frame = pd.DataFrame(contour_macro_rows)

    session_frame.to_csv(args.output_root / "per_session_movement_metrics.csv", index=False)
    contour_session_frame.to_csv(
        args.output_root / "per_session_per_contour_movement_metrics.csv", index=False
    )
    contour_macro_frame.to_csv(args.output_root / "per_contour_macro_movement_metrics.csv", index=False)
    frame_pair_frame.to_csv(args.output_root / "per_frame_pair_per_contour_movement.csv", index=False)

    session_summary_plot(session_frame, plot_dir / "session_metric_summary.png")
    heatmap(
        contour_session_frame,
        "movement_nrmse",
        "Movement NRMSE by session and contour (lower is better)",
        plot_dir / "movement_nrmse_heatmap.png",
    )
    heatmap(
        contour_session_frame,
        "path_ratio_pred_over_gt",
        "Predicted / GT movement path ratio",
        plot_dir / "path_ratio_heatmap.png",
        center=1.0,
    )
    heatmap(
        contour_session_frame,
        "speed_correlation_zero_lag",
        "Predicted-vs-GT movement-speed correlation",
        plot_dir / "speed_correlation_heatmap.png",
        center=0.0,
    )
    contour_summary_plot(contour_macro_frame, plot_dir / "per_contour_macro_metrics.png")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_root": str(args.input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "selection": [f"P{speaker}/S{session}" for speaker, session in SELECTION],
        "model_training_dataset": "ASD2",
        "target_and_ground_truth_dataset": "ASD1",
        "frame_policy": FRAME_POLICY,
        "fractional_saved_frame_count": 0,
        "fractional_scored_frame_count": 0,
        "fractional_analyzed_frame_count": 0,
        "excluded_classes": [EXCLUDED_CLASS],
        "evaluated_classes": expected_classes,
        "mm_per_pixel": MM_PER_PIXEL,
        "fps": FPS,
        "max_lag_frames": MAX_LAG_FRAMES,
        "total_source_integer_frames": total_frames,
        "total_adjacent_frame_pairs": int(sum(row["valid_frame_pairs"] for row in session_rows)),
        "total_timeline_gaps_skipped": total_gaps,
        "overall_macro_equal_speaker_weight": overall_macro,
        "overall_micro_all_valid_pairs": overall_micro,
        "files": {
            "report": str((args.output_root / "contour_movement_report.md").resolve()),
            "per_session": str((args.output_root / "per_session_movement_metrics.csv").resolve()),
            "per_session_per_contour": str(
                (args.output_root / "per_session_per_contour_movement_metrics.csv").resolve()
            ),
            "per_contour_macro": str(
                (args.output_root / "per_contour_macro_movement_metrics.csv").resolve()
            ),
            "per_frame_pair_per_contour": str(
                (args.output_root / "per_frame_pair_per_contour_movement.csv").resolve()
            ),
        },
    }
    (args.output_root / "movement_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8"
    )
    write_report(
        args.output_root / "contour_movement_report.md",
        session_frame,
        contour_macro_frame,
        overall_macro,
        overall_micro,
        total_frames,
        summary["total_adjacent_frame_pairs"],
        total_gaps,
    )
    print(json.dumps({"report": summary["files"]["report"], "sessions": len(session_rows)}, indent=2))


if __name__ == "__main__":
    main()
