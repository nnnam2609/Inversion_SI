#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.inference.session_inference import load_config, load_model, resolve_device  # noqa: E402
from preprocessing.main_preprocessing import Corpus  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run prediction-only dense audio inference for every requested integer MRI frame. "
            "No TextGrid speech/silence filtering and no target contours are used."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speaker", type=int, required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--frame-min", type=int, required=True)
    parser.add_argument("--frame-max", type=int, required=True)
    parser.add_argument(
        "--inference-mode",
        choices=("full_sequence", "overlapping_windows"),
        default="full_sequence",
        help=(
            "full_sequence runs one direct model forward over the complete selected audio sequence. "
            "overlapping_windows is retained for diagnostics and can introduce window-boundary jumps."
        ),
    )
    parser.add_argument("--window-size", type=int, default=80)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--normalization-stats", type=Path, default=None)
    parser.add_argument("--compare-contour-dir", type=Path, default=None)
    return parser.parse_args()


def mfcc_helper(config: dict[str, Any]) -> Corpus:
    helper = object.__new__(Corpus)
    helper.config = config
    helper.window_length_ms = float(config["window_length_ms"])
    helper.hop_length_ratio = float(config["hop_length_ratio"])
    helper.n_mfcc = int(config["n_mfcc"])
    return helper


def extract_full_mfcc(
    audio_path: Path,
    config: dict[str, Any],
) -> tuple[np.ndarray, int, int, int]:
    audio_signal, sample_rate = librosa.load(audio_path, sr=None)
    features, window_length_samples, hop_length_samples = mfcc_helper(config).compute_mfcc(
        audio_signal,
        sample_rate,
    )
    if features.ndim != 2 or features.shape[1] != int(config["input_layer"]):
        raise ValueError(
            f"Expected dense audio features shaped (T, {config['input_layer']}), got {features.shape}"
        )
    return np.asarray(features), sample_rate, window_length_samples, hop_length_samples


def mfcc_index_to_mri_frame(
    index: int,
    sample_rate: int,
    window_length_samples: int,
    config: dict[str, Any],
) -> float:
    sample_rate_ms = sample_rate / 1000.0
    frame_duration_ms = (window_length_samples / sample_rate) * 1000.0
    mid_frame_sample = int(
        index * sample_rate_ms * float(config["hop_length_ratio"])
        + (frame_duration_ms / 2.0) * sample_rate_ms
    )
    skip_ms = float(config["added_frames"]) * float(config["ms_image"])
    skip_samples = sample_rate_ms * skip_ms
    return (mid_frame_sample - skip_samples) / (float(config["ms_image"]) * sample_rate_ms)


def dense_feature_selection(
    features: np.ndarray,
    sample_rate: int,
    window_length_samples: int,
    config: dict[str, Any],
    frame_min: int,
    frame_max: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    mapped_frames = np.asarray(
        [
            math.floor(mfcc_index_to_mri_frame(index, sample_rate, window_length_samples, config))
            for index in range(features.shape[0])
        ],
        dtype=np.int32,
    )
    selected_global_indices = np.flatnonzero(
        (mapped_frames >= frame_min) & (mapped_frames <= frame_max)
    )
    if selected_global_indices.size == 0:
        raise RuntimeError(f"No MFCC frames map to integer MRI range {frame_min}-{frame_max}")
    expected = np.arange(selected_global_indices[0], selected_global_indices[-1] + 1)
    if not np.array_equal(selected_global_indices, expected):
        raise RuntimeError("Dense MFCC selection is unexpectedly non-contiguous")

    first_local_index_by_frame: dict[int, int] = {}
    for local_index, global_index in enumerate(selected_global_indices.tolist()):
        frame = int(mapped_frames[global_index])
        first_local_index_by_frame.setdefault(frame, local_index)
    missing = sorted(set(range(frame_min, frame_max + 1)) - set(first_local_index_by_frame))
    if missing:
        raise RuntimeError(f"Dense audio mapping missed {len(missing)} integer MRI frames: {missing[:20]}")
    return features[selected_global_indices], selected_global_indices, first_local_index_by_frame


def window_starts(length: int, window_size: int, stride: int) -> list[int]:
    if length <= 0:
        raise ValueError("Dense feature sequence is empty")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    final_start = length - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def infer_overlapping_windows(
    model: torch.nn.Module,
    normalized_features: np.ndarray,
    window_size: int,
    stride: int,
    batch_size: int,
    class_count: int,
    output_layer: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    starts = window_starts(len(normalized_features), window_size, stride)
    prediction_sum = np.zeros(
        (len(normalized_features), class_count, output_layer),
        dtype=np.float64,
    )
    prediction_count = np.zeros(len(normalized_features), dtype=np.int32)

    for batch_start in range(0, len(starts), batch_size):
        batch_starts = starts[batch_start : batch_start + batch_size]
        arrays = [normalized_features[start : start + window_size] for start in batch_starts]
        lengths = torch.tensor([len(array) for array in arrays], dtype=torch.long)
        max_length = int(lengths.max())
        padded = np.zeros((len(arrays), max_length, normalized_features.shape[1]), dtype=np.float32)
        for index, array in enumerate(arrays):
            padded[index, : len(array)] = array
        with torch.no_grad():
            predicted, _, _ = model(torch.from_numpy(padded).to(device), lengths)
        predicted_np = predicted.detach().cpu().numpy()
        for index, start in enumerate(batch_starts):
            length = int(lengths[index])
            prediction_sum[start : start + length] += predicted_np[index, :length]
            prediction_count[start : start + length] += 1

    if np.any(prediction_count == 0):
        missing = np.flatnonzero(prediction_count == 0).tolist()
        raise RuntimeError(f"Overlapping inference left {len(missing)} MFCC positions uncovered")
    averaged = prediction_sum / prediction_count[:, None, None]
    return averaged.astype(np.float32), prediction_count, starts


def infer_full_sequence(
    model: torch.nn.Module,
    normalized_features: np.ndarray,
    class_count: int,
    output_layer: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if len(normalized_features) == 0:
        raise ValueError("Dense feature sequence is empty")
    inputs = torch.from_numpy(normalized_features[None, ...]).to(device)
    lengths = torch.tensor([len(normalized_features)], dtype=torch.long)
    with torch.no_grad():
        predicted, _, _ = model(inputs, lengths)
    predicted_np = predicted.detach().cpu().numpy()
    expected_shape = (1, len(normalized_features), class_count, output_layer)
    if predicted_np.shape != expected_shape:
        raise RuntimeError(
            f"Full-sequence model output has shape {predicted_np.shape}, expected {expected_shape}"
        )
    if not np.isfinite(predicted_np).all():
        raise RuntimeError("Full-sequence model output contains NaN or infinity")
    return predicted_np[0].astype(np.float32), np.ones(len(normalized_features), dtype=np.int32)


def compare_existing_predictions(
    new_contours: dict[int, np.ndarray],
    classes: list[str],
    existing_dir: Path | None,
) -> dict[str, Any] | None:
    if existing_dir is None:
        return None
    per_class_squared: dict[str, list[np.ndarray]] = {name: [] for name in classes}
    compared_frames = []
    for frame, contours in new_contours.items():
        paths = [existing_dir / f"{frame:04d}_{name}.npy" for name in classes]
        if not all(path.is_file() for path in paths):
            continue
        existing = np.stack([np.load(path).reshape(50, 2) for path in paths], axis=0)
        compared_frames.append(frame)
        squared = (contours - existing) ** 2
        for class_index, name in enumerate(classes):
            per_class_squared[name].append(squared[class_index])
    if not compared_frames:
        return {
            "existing_contour_dir": str(existing_dir),
            "num_compared_integer_frames": 0,
        }
    all_squared = np.concatenate(
        [np.stack(per_class_squared[name], axis=0).reshape(-1) for name in classes]
    )
    return {
        "existing_contour_dir": str(existing_dir),
        "num_compared_integer_frames": len(compared_frames),
        "frame_min": min(compared_frames),
        "frame_max": max(compared_frames),
        "coordinate_rmse_px": float(np.sqrt(np.mean(all_squared))),
        "per_class_coordinate_rmse_px": {
            name: float(np.sqrt(np.mean(np.stack(values, axis=0))))
            for name, values in per_class_squared.items()
        },
    }


def main() -> None:
    args = parse_args()
    if args.frame_max < args.frame_min:
        raise ValueError("frame-max must be greater than or equal to frame-min")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    config = load_config(args.config)
    classes = list(config["classes"])
    if len(classes) != 11:
        raise RuntimeError(f"Dense P7 contour inference requires 11 classes, got {len(classes)}")
    stats_path = args.normalization_stats
    if stats_path is None:
        split_cache_dir = Path(config.get("split_cache_dir") or config["dataset_cache_dir"])
        stats_path = split_cache_dir / "normalization_stats.npz"
    stats_path = stats_path.resolve()
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing train-global normalization stats: {stats_path}")
    with np.load(stats_path, allow_pickle=False) as stats:
        mean_mfcc = stats["mean_mfcc"].astype(np.float32)
        std_mfcc = stats["std_mfcc"].astype(np.float32)
        mean_contour = stats["mean_contour"].astype(np.float32)
        std_contour = stats["std_contour"].astype(np.float32)
    expected_mfcc_shape = (int(config["input_layer"]),)
    expected_contour_shape = (len(classes), int(config["output_layer"]))
    if mean_mfcc.shape != expected_mfcc_shape or std_mfcc.shape != expected_mfcc_shape:
        raise ValueError(
            "Invalid MFCC normalization shapes: "
            f"mean={mean_mfcc.shape}, std={std_mfcc.shape}, expected={expected_mfcc_shape}"
        )
    if mean_contour.shape != expected_contour_shape or std_contour.shape != expected_contour_shape:
        raise ValueError(
            "Invalid contour normalization shapes: "
            f"mean={mean_contour.shape}, std={std_contour.shape}, expected={expected_contour_shape}"
        )
    if not all(
        np.isfinite(values).all()
        for values in (mean_mfcc, std_mfcc, mean_contour, std_contour)
    ):
        raise ValueError("Normalization statistics contain NaN or infinity")
    if np.any(std_mfcc <= 0) or np.any(std_contour <= 0):
        raise ValueError("Normalization standard deviations must be strictly positive")

    print(f"Extracting full-session MFCC from {args.audio}", flush=True)
    full_features, sample_rate, window_length_samples, hop_length_samples = extract_full_mfcc(
        args.audio,
        config,
    )
    selected_features, selected_global_indices, first_local_index_by_frame = dense_feature_selection(
        full_features,
        sample_rate,
        window_length_samples,
        config,
        args.frame_min,
        args.frame_max,
    )
    normalized_features = ((selected_features - mean_mfcc) / std_mfcc).astype(np.float32)

    device = resolve_device(args.device)
    print(
        f"Dense inference device={device} mfcc={len(selected_features)} "
        f"integer_frames={len(first_local_index_by_frame)}",
        flush=True,
    )
    model = load_model(config, args.checkpoint, device)
    if args.inference_mode == "full_sequence":
        predicted_normalized, coverage = infer_full_sequence(
            model,
            normalized_features,
            len(classes),
            int(config["output_layer"]),
            device,
        )
        starts: list[int] = []
    else:
        predicted_normalized, coverage, starts = infer_overlapping_windows(
            model,
            normalized_features,
            args.window_size,
            args.stride,
            args.batch_size,
            len(classes),
            int(config["output_layer"]),
            device,
        )
    predicted_raw = predicted_normalized * std_contour[None, :, :] + mean_contour[None, :, :]

    output_dir = args.output_dir.resolve()
    contour_dir = output_dir / "predicted_contours"
    contour_dir.mkdir(parents=True, exist_ok=True)
    per_frame: dict[int, np.ndarray] = {}
    frame_rows = []
    expected_contour_filenames: set[str] = set()
    for frame in range(args.frame_min, args.frame_max + 1):
        local_index = first_local_index_by_frame[frame]
        contour = predicted_raw[local_index].reshape(len(classes), 50, 2).astype(np.float32)
        if not np.isfinite(contour).all():
            raise RuntimeError(f"Model produced NaN or infinity for integer MRI frame {frame}")
        per_frame[frame] = contour
        for class_index, name in enumerate(classes):
            filename = f"{frame:04d}_{name}.npy"
            expected_contour_filenames.add(filename)
            np.save(contour_dir / filename, contour[class_index])
        global_index = int(selected_global_indices[local_index])
        frame_rows.append(
            {
                "frame": f"{frame:04d}",
                "mfcc_global_index": global_index,
                "mfcc_center_seconds": (
                    global_index * hop_length_samples + window_length_samples / 2.0
                )
                / sample_rate,
                "overlap_prediction_count": int(coverage[local_index]),
                "inference_mode": args.inference_mode,
            }
        )

    actual_contour_filenames = {path.name for path in contour_dir.glob("*.npy")}
    missing_contour_files = sorted(expected_contour_filenames - actual_contour_filenames)
    unexpected_contour_files = sorted(actual_contour_filenames - expected_contour_filenames)
    if missing_contour_files or unexpected_contour_files:
        raise RuntimeError(
            "Dense contour output validation failed: "
            f"missing={missing_contour_files[:20]}, unexpected={unexpected_contour_files[:20]}"
        )

    with (output_dir / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)

    stacked = np.stack([per_frame[frame] for frame in sorted(per_frame)], axis=0)
    comparison = compare_existing_predictions(per_frame, classes, args.compare_contour_dir)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "dense_audio_prediction_only",
        "inference_mode": args.inference_mode,
        "temporal_inference": (
            "one_direct_full_sequence_forward"
            if args.inference_mode == "full_sequence"
            else "uniform_average_of_overlapping_windows_diagnostic_only"
        ),
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "audio": str(args.audio.resolve()),
        "speaker": args.speaker,
        "session": args.session,
        "device": str(device),
        "uses_target_labels": False,
        "uses_textgrid_speech_filter": False,
        "uses_silence_filter": False,
        "normalization_stats": str(stats_path),
        "normalization_mode": "train_global",
        "frame_min": args.frame_min,
        "frame_max": args.frame_max,
        "num_integer_frames": len(per_frame),
        "num_half_frames": 0,
        "num_classes": len(classes),
        "classes": classes,
        "num_contour_files": len(actual_contour_filenames),
        "contour_dir": str(contour_dir),
        "full_mfcc_shape": list(full_features.shape),
        "selected_mfcc_shape": list(selected_features.shape),
        "selected_mfcc_global_index_min": int(selected_global_indices[0]),
        "selected_mfcc_global_index_max": int(selected_global_indices[-1]),
        "sample_rate": sample_rate,
        "window_length_samples": window_length_samples,
        "hop_length_samples": hop_length_samples,
        "window_size": args.window_size if args.inference_mode == "overlapping_windows" else None,
        "stride": args.stride if args.inference_mode == "overlapping_windows" else None,
        "num_inference_windows": len(starts),
        "num_full_sequence_forwards": int(args.inference_mode == "full_sequence"),
        "prediction_coverage_min": int(coverage.min()),
        "prediction_coverage_max": int(coverage.max()),
        "prediction_coordinate_min": float(stacked.min()),
        "prediction_coordinate_max": float(stacked.max()),
        "mean_abs_integer_frame_diff_px": float(np.mean(np.abs(np.diff(stacked, axis=0)))),
        "existing_speech_prediction_comparison": comparison,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
