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
import textgrid
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
        choices=(
            "legacy_interval_chunks",
            "nonoverlapping_chunks",
            "full_sequence",
            "overlapping_windows",
        ),
        default="legacy_interval_chunks",
        help=(
            "legacy_interval_chunks reproduces classic inversion prediction boundaries from "
            "TextGrid tier 0, selects one MFCC per integer MRI frame, and directly concatenates "
            "independent max-train-length outputs. nonoverlapping_chunks uses dense fixed-size "
            "chunks. full_sequence and overlapping_windows are retained as diagnostic modes."
        ),
    )
    parser.add_argument(
        "--textgrid",
        type=Path,
        default=None,
        help="Required by legacy_interval_chunks; used only for sequence boundaries/silence intervals.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Chunk/window size. Defaults to the training config sequence_length.",
    )
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


def integer_frame_feature_selection(
    features: np.ndarray,
    sample_rate: int,
    window_length_samples: int,
    config: dict[str, Any],
    frame_min: int,
    frame_max: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Select exactly one acoustic feature nearest each integer MRI-frame center.

    The classic ``inversion/`` preprocessing first reduced the roughly 100 Hz
    acoustic stream to one feature per roughly 50 Hz MRI frame, then formed
    recurrent sequences. This selection reproduces that temporal rate without
    reading target contours, TextGrid labels, or silence masks.
    """
    best_by_frame: dict[int, tuple[float, int]] = {}
    for global_index in range(features.shape[0]):
        mapped_position = mfcc_index_to_mri_frame(
            global_index,
            sample_rate,
            window_length_samples,
            config,
        )
        frame = math.floor(mapped_position)
        if frame < frame_min or frame > frame_max:
            continue
        distance_to_center = abs(mapped_position - (frame + 0.5))
        previous = best_by_frame.get(frame)
        if previous is None or distance_to_center < previous[0]:
            best_by_frame[frame] = (distance_to_center, global_index)

    missing = sorted(set(range(frame_min, frame_max + 1)) - set(best_by_frame))
    if missing:
        raise RuntimeError(
            f"Integer-aligned audio mapping missed {len(missing)} MRI frames: {missing[:20]}"
        )

    selected_global_indices = np.asarray(
        [best_by_frame[frame][1] for frame in range(frame_min, frame_max + 1)],
        dtype=np.int32,
    )
    if np.any(np.diff(selected_global_indices) <= 0):
        raise RuntimeError("Integer-aligned MFCC indices are not strictly increasing")
    local_index_by_frame = {
        frame: local_index
        for local_index, frame in enumerate(range(frame_min, frame_max + 1))
    }
    return features[selected_global_indices], selected_global_indices, local_index_by_frame


def legacy_interval_feature_selection(
    features: np.ndarray,
    sample_rate: int,
    window_length_samples: int,
    config: dict[str, Any],
    textgrid_path: Path,
    frame_min: int,
    frame_max: int,
    chunk_size: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[int, int],
    list[tuple[int, int]],
    dict[str, Any],
]:
    """Build classic tier-0 interval sequences without loading contour labels."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not textgrid_path.is_file():
        raise FileNotFoundError(f"Missing TextGrid: {textgrid_path}")
    grid = textgrid.TextGrid.fromFile(str(textgrid_path))
    if len(grid) == 0 or not hasattr(grid[0], "intervals"):
        raise ValueError(f"TextGrid tier 0 is not an interval tier: {textgrid_path}")

    selected_records: list[tuple[int, int]] = []
    sequence_slices: list[tuple[int, int]] = []
    seen_frames: set[int] = set()
    selected_interval_count = 0
    skipped_silence_interval_count = 0
    duplicate_frame_count = 0
    legacy_hop_samples = 160.06
    previous_index_end = 0

    for interval_index, interval in enumerate(grid[0].intervals):
        mark = str(interval.mark)
        if mark in {"#", ""}:
            skipped_silence_interval_count += 1
            previous_index_end = int(math.ceil(float(interval.maxTime) * sample_rate / legacy_hop_samples))
            continue

        index_begin = int(math.floor(float(interval.minTime) * sample_rate / legacy_hop_samples))
        index_end = int(math.ceil(float(interval.maxTime) * sample_rate / legacy_hop_samples))
        if interval_index > 0:
            index_begin = previous_index_end
        previous_index_end = index_end
        index_begin = max(0, min(index_begin, len(features)))
        index_end = max(index_begin, min(index_end, len(features)))

        best_by_frame: dict[int, tuple[float, int]] = {}
        for global_index in range(index_begin, index_end):
            mapped_position = mfcc_index_to_mri_frame(
                global_index,
                sample_rate,
                window_length_samples,
                config,
            )
            frame = math.floor(mapped_position)
            if frame < frame_min or frame > frame_max:
                continue
            distance_to_center = abs(mapped_position - (frame + 0.5))
            previous = best_by_frame.get(frame)
            if previous is None or distance_to_center < previous[0]:
                best_by_frame[frame] = (distance_to_center, global_index)

        interval_records = []
        for frame in sorted(best_by_frame):
            if frame in seen_frames:
                duplicate_frame_count += 1
                continue
            seen_frames.add(frame)
            interval_records.append((frame, best_by_frame[frame][1]))
        if not interval_records:
            continue
        selected_interval_count += 1
        for offset in range(0, len(interval_records), chunk_size):
            chunk_records = interval_records[offset : offset + chunk_size]
            start = len(selected_records)
            selected_records.extend(chunk_records)
            sequence_slices.append((start, len(selected_records)))

    if not selected_records:
        raise RuntimeError("Legacy TextGrid interval selection produced no integer MRI frames")
    frame_ids = [record[0] for record in selected_records]
    if any(right <= left for left, right in zip(frame_ids, frame_ids[1:])):
        raise RuntimeError("Legacy interval frame ids are not strictly increasing")
    selected_global_indices = np.asarray(
        [record[1] for record in selected_records],
        dtype=np.int32,
    )
    local_index_by_frame = {
        frame: local_index for local_index, frame in enumerate(frame_ids)
    }
    metadata = {
        "textgrid": str(textgrid_path.resolve()),
        "textgrid_tier_index": 0,
        "textgrid_tier_name": str(getattr(grid[0], "name", "")),
        "num_textgrid_intervals": len(grid[0].intervals),
        "num_selected_textgrid_intervals": selected_interval_count,
        "num_skipped_silence_intervals": skipped_silence_interval_count,
        "num_duplicate_interval_boundary_frames_dropped": duplicate_frame_count,
        "num_sequence_chunks": len(sequence_slices),
        "sequence_length_min": min(end - start for start, end in sequence_slices),
        "sequence_length_max": max(end - start for start, end in sequence_slices),
    }
    return (
        features[selected_global_indices],
        selected_global_indices,
        local_index_by_frame,
        sequence_slices,
        metadata,
    )


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


def infer_nonoverlapping_chunks(
    model: torch.nn.Module,
    normalized_features: np.ndarray,
    chunk_size: int,
    batch_size: int,
    class_count: int,
    output_layer: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if len(normalized_features) == 0:
        raise ValueError("Integer-aligned feature sequence is empty")
    if chunk_size <= 0 or batch_size <= 0:
        raise ValueError("chunk_size and batch_size must be positive")

    starts = list(range(0, len(normalized_features), chunk_size))
    predicted_direct = np.empty(
        (len(normalized_features), class_count, output_layer),
        dtype=np.float32,
    )
    prediction_count = np.zeros(len(normalized_features), dtype=np.int32)

    for batch_start in range(0, len(starts), batch_size):
        batch_starts = starts[batch_start : batch_start + batch_size]
        arrays = [normalized_features[start : start + chunk_size] for start in batch_starts]
        lengths = torch.tensor([len(array) for array in arrays], dtype=torch.long)
        max_length = int(lengths.max())
        padded = np.zeros(
            (len(arrays), max_length, normalized_features.shape[1]),
            dtype=np.float32,
        )
        for index, array in enumerate(arrays):
            padded[index, : len(array)] = array

        with torch.no_grad():
            predicted, _, _ = model(torch.from_numpy(padded).to(device), lengths)
        predicted_np = predicted.detach().cpu().numpy()
        expected_shape = (len(arrays), max_length, class_count, output_layer)
        if predicted_np.shape != expected_shape:
            raise RuntimeError(
                f"Non-overlapping model output has shape {predicted_np.shape}, "
                f"expected {expected_shape}"
            )
        if not np.isfinite(predicted_np).all():
            raise RuntimeError("Non-overlapping model output contains NaN or infinity")

        for index, start in enumerate(batch_starts):
            length = int(lengths[index])
            predicted_direct[start : start + length] = predicted_np[index, :length]
            prediction_count[start : start + length] += 1

    if not np.all(prediction_count == 1):
        invalid = np.flatnonzero(prediction_count != 1).tolist()
        raise RuntimeError(
            "Non-overlapping inference must predict every position exactly once; "
            f"invalid positions={invalid[:20]}"
        )
    return predicted_direct, prediction_count, starts


def infer_presegmented_sequences(
    model: torch.nn.Module,
    normalized_features: np.ndarray,
    sequence_slices: list[tuple[int, int]],
    batch_size: int,
    class_count: int,
    output_layer: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if len(normalized_features) == 0 or not sequence_slices:
        raise ValueError("Presegmented feature sequences are empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    expected_start = 0
    for start, end in sequence_slices:
        if start != expected_start or end <= start or end > len(normalized_features):
            raise ValueError(f"Invalid contiguous sequence slice ({start}, {end})")
        expected_start = end
    if expected_start != len(normalized_features):
        raise ValueError("Presegmented sequence slices do not cover all selected features")

    predicted_direct = np.empty(
        (len(normalized_features), class_count, output_layer),
        dtype=np.float32,
    )
    prediction_count = np.zeros(len(normalized_features), dtype=np.int32)
    for batch_start in range(0, len(sequence_slices), batch_size):
        batch_slices = sequence_slices[batch_start : batch_start + batch_size]
        arrays = [normalized_features[start:end] for start, end in batch_slices]
        lengths = torch.tensor([len(array) for array in arrays], dtype=torch.long)
        max_length = int(lengths.max())
        padded = np.zeros(
            (len(arrays), max_length, normalized_features.shape[1]),
            dtype=np.float32,
        )
        for index, array in enumerate(arrays):
            padded[index, : len(array)] = array
        with torch.no_grad():
            predicted, _, _ = model(torch.from_numpy(padded).to(device), lengths)
        predicted_np = predicted.detach().cpu().numpy()
        expected_shape = (len(arrays), max_length, class_count, output_layer)
        if predicted_np.shape != expected_shape:
            raise RuntimeError(
                f"Presegmented model output has shape {predicted_np.shape}, expected {expected_shape}"
            )
        if not np.isfinite(predicted_np).all():
            raise RuntimeError("Presegmented model output contains NaN or infinity")
        for index, (start, end) in enumerate(batch_slices):
            length = end - start
            predicted_direct[start:end] = predicted_np[index, :length]
            prediction_count[start:end] += 1

    if not np.all(prediction_count == 1):
        invalid = np.flatnonzero(prediction_count != 1).tolist()
        raise RuntimeError(
            "Presegmented inference must predict every selected position exactly once; "
            f"invalid positions={invalid[:20]}"
        )
    return predicted_direct, prediction_count


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
    training_sequence_length = int(config["sequence_length"])
    window_size = int(args.window_size or training_sequence_length)
    if args.inference_mode in {"legacy_interval_chunks", "nonoverlapping_chunks"} and window_size != training_sequence_length:
        raise ValueError(
            f"{args.inference_mode} must match the checkpoint training sequence length: "
            f"window_size={window_size}, sequence_length={training_sequence_length}"
        )
    if args.inference_mode == "legacy_interval_chunks" and args.textgrid is None:
        raise ValueError("legacy_interval_chunks requires --textgrid")
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
    sequence_slices: list[tuple[int, int]] = []
    legacy_selection_metadata: dict[str, Any] | None = None
    if args.inference_mode == "legacy_interval_chunks":
        (
            selected_features,
            selected_global_indices,
            first_local_index_by_frame,
            sequence_slices,
            legacy_selection_metadata,
        ) = legacy_interval_feature_selection(
            full_features,
            sample_rate,
            window_length_samples,
            config,
            args.textgrid,
            args.frame_min,
            args.frame_max,
            window_size,
        )
    elif args.inference_mode == "nonoverlapping_chunks":
        selected_features, selected_global_indices, first_local_index_by_frame = (
            integer_frame_feature_selection(
                full_features,
                sample_rate,
                window_length_samples,
                config,
                args.frame_min,
                args.frame_max,
            )
        )
    else:
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
    if args.inference_mode == "legacy_interval_chunks":
        predicted_normalized, coverage = infer_presegmented_sequences(
            model,
            normalized_features,
            sequence_slices,
            args.batch_size,
            len(classes),
            int(config["output_layer"]),
            device,
        )
        starts = [start for start, _ in sequence_slices]
    elif args.inference_mode == "nonoverlapping_chunks":
        predicted_normalized, coverage, starts = infer_nonoverlapping_chunks(
            model,
            normalized_features,
            window_size,
            args.batch_size,
            len(classes),
            int(config["output_layer"]),
            device,
        )
    elif args.inference_mode == "full_sequence":
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
            window_size,
            args.stride,
            args.batch_size,
            len(classes),
            int(config["output_layer"]),
            device,
        )
    if args.inference_mode == "legacy_interval_chunks":
        effective_sequence_slices = sequence_slices
    elif args.inference_mode == "nonoverlapping_chunks":
        effective_sequence_slices = [
            (start, min(start + window_size, len(normalized_features))) for start in starts
        ]
    elif args.inference_mode == "full_sequence":
        effective_sequence_slices = [(0, len(normalized_features))]
    else:
        effective_sequence_slices = []
    sequence_info_by_local: dict[int, tuple[int, int, int]] = {}
    for sequence_index, (start, end) in enumerate(effective_sequence_slices):
        for local_index in range(start, end):
            sequence_info_by_local[local_index] = (
                sequence_index,
                local_index - start,
                end - start,
            )
    predicted_raw = predicted_normalized * std_contour[None, :, :] + mean_contour[None, :, :]

    output_dir = args.output_dir.resolve()
    contour_dir = output_dir / "predicted_contours"
    contour_dir.mkdir(parents=True, exist_ok=True)
    per_frame: dict[int, np.ndarray] = {}
    frame_rows = []
    expected_contour_filenames: set[str] = set()
    for frame in range(args.frame_min, args.frame_max + 1):
        if frame not in first_local_index_by_frame:
            continue
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
        sequence_info = sequence_info_by_local.get(local_index)
        frame_rows.append(
            {
                "frame": f"{frame:04d}",
                "mfcc_global_index": global_index,
                "mfcc_center_seconds": (
                    global_index * hop_length_samples + window_length_samples / 2.0
                )
                / sample_rate,
                "overlap_prediction_count": int(coverage[local_index]),
                "sequence_index": sequence_info[0] if sequence_info is not None else None,
                "sequence_position": sequence_info[1] if sequence_info is not None else None,
                "sequence_length": sequence_info[2] if sequence_info is not None else None,
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

    sorted_predicted_frames = sorted(per_frame)
    stacked = np.stack([per_frame[frame] for frame in sorted_predicted_frames], axis=0)
    consecutive_pair_indices = [
        index
        for index, (left, right) in enumerate(
            zip(sorted_predicted_frames, sorted_predicted_frames[1:])
        )
        if right == left + 1
    ]
    consecutive_diffs = (
        np.stack(
            [np.abs(stacked[index + 1] - stacked[index]) for index in consecutive_pair_indices],
            axis=0,
        )
        if consecutive_pair_indices
        else np.empty((0,), dtype=np.float32)
    )
    sequence_index_by_frame = {
        frame: sequence_info_by_local[first_local_index_by_frame[frame]][0]
        for frame in sorted_predicted_frames
        if first_local_index_by_frame[frame] in sequence_info_by_local
    }
    boundary_pair_indices = [
        index
        for index in consecutive_pair_indices
        if sequence_index_by_frame.get(sorted_predicted_frames[index])
        != sequence_index_by_frame.get(sorted_predicted_frames[index + 1])
    ]
    boundary_pair_index_set = set(boundary_pair_indices)
    within_sequence_pair_indices = [
        index for index in consecutive_pair_indices if index not in boundary_pair_index_set
    ]
    boundary_diffs = (
        np.stack(
            [np.abs(stacked[index + 1] - stacked[index]) for index in boundary_pair_indices],
            axis=0,
        )
        if boundary_pair_indices
        else np.empty((0,), dtype=np.float32)
    )
    within_sequence_diffs = (
        np.stack(
            [
                np.abs(stacked[index + 1] - stacked[index])
                for index in within_sequence_pair_indices
            ],
            axis=0,
        )
        if within_sequence_pair_indices
        else np.empty((0,), dtype=np.float32)
    )
    boundary_mean = float(np.mean(boundary_diffs)) if boundary_diffs.size else None
    within_sequence_mean = (
        float(np.mean(within_sequence_diffs)) if within_sequence_diffs.size else None
    )
    missing_integer_frames = sorted(
        set(range(args.frame_min, args.frame_max + 1)) - set(per_frame)
    )
    comparison = compare_existing_predictions(per_frame, classes, args.compare_contour_dir)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "dense_audio_prediction_only",
        "inference_mode": args.inference_mode,
        "temporal_inference": (
            "classic_textgrid_tier0_intervals_one_mfcc_per_integer_mri_frame_direct_chunks"
            if args.inference_mode == "legacy_interval_chunks"
            else "one_integer_aligned_mfcc_per_mri_frame_nonoverlapping_train_length_chunks"
            if args.inference_mode == "nonoverlapping_chunks"
            else "one_direct_full_sequence_forward_diagnostic_only"
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
        "uses_textgrid_speech_filter": args.inference_mode == "legacy_interval_chunks",
        "uses_silence_filter": args.inference_mode == "legacy_interval_chunks",
        "mfcc_selection": (
            "nearest_mfcc_to_each_integer_mri_frame_center"
            if args.inference_mode in {"legacy_interval_chunks", "nonoverlapping_chunks"}
            else "contiguous_all_mfcc_positions_then_first_position_per_integer_mri_frame"
        ),
        "legacy_interval_selection": legacy_selection_metadata,
        "normalization_stats": str(stats_path),
        "normalization_mode": "train_global",
        "frame_min": args.frame_min,
        "frame_max": args.frame_max,
        "num_integer_frames": len(per_frame),
        "num_requested_integer_frames": args.frame_max - args.frame_min + 1,
        "num_missing_integer_frames": len(missing_integer_frames),
        "missing_integer_frames": missing_integer_frames,
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
        "training_sequence_length": training_sequence_length,
        "window_size": window_size if args.inference_mode != "full_sequence" else None,
        "stride": args.stride if args.inference_mode == "overlapping_windows" else None,
        "num_inference_windows": len(starts),
        "num_nonoverlapping_chunks": (
            len(starts)
            if args.inference_mode in {"legacy_interval_chunks", "nonoverlapping_chunks"}
            else 0
        ),
        "num_model_forward_calls": (
            math.ceil(len(starts) / args.batch_size)
            if args.inference_mode in {"legacy_interval_chunks", "nonoverlapping_chunks"}
            else 1
            if args.inference_mode == "full_sequence"
            else math.ceil(len(starts) / args.batch_size)
        ),
        "num_full_sequence_forwards": int(args.inference_mode == "full_sequence"),
        "prediction_coverage_min": int(coverage.min()),
        "prediction_coverage_max": int(coverage.max()),
        "prediction_coordinate_min": float(stacked.min()),
        "prediction_coordinate_max": float(stacked.max()),
        "mean_abs_integer_frame_diff_px": float(np.mean(np.abs(np.diff(stacked, axis=0)))),
        "num_consecutive_integer_frame_pairs": len(consecutive_pair_indices),
        "mean_abs_consecutive_integer_frame_diff_px": (
            float(np.mean(consecutive_diffs)) if consecutive_diffs.size else None
        ),
        "num_consecutive_sequence_boundary_pairs": len(boundary_pair_indices),
        "mean_abs_sequence_boundary_frame_diff_px": boundary_mean,
        "mean_abs_within_sequence_frame_diff_px": within_sequence_mean,
        "sequence_boundary_to_within_motion_ratio": (
            boundary_mean / within_sequence_mean
            if boundary_mean is not None
            and within_sequence_mean is not None
            and within_sequence_mean > 0
            else None
        ),
        "existing_speech_prediction_comparison": comparison,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
