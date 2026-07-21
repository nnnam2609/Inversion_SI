#!/usr/bin/env python3
"""Compare two ASD2 checkpoints on the same refreshed test labels, integer frames only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.model.baseline_5 import BaselineModel  # noqa: E402


CLASSES = (
    "arytenoid-cartilage",
    "epiglottis",
    "lower-lip",
    "pharynx",
    "soft-palate-midline",
    "tongue",
    "upper-lip",
    "vocal-folds",
    "thyroid-cartilage",
    "lower-incisor",
    "upper-incisor",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--target-normalization", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-normalization", type=Path, required=True)
    parser.add_argument("--reference-name", default="epoch211_reference")
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-normalization", type=Path, required=True)
    parser.add_argument("--candidate-name", default="current_incisor_retrain")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mm-per-pixel", type=float, default=1.62)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Config is not a mapping: {path}")
    if tuple(data.get("classes", ())) != CLASSES:
        raise ValueError(f"Unexpected contour order in {path}")
    return data


def load_normalization(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as pack:
        std = np.asarray(pack["std_contour"], dtype=np.float32)
        mean = np.asarray(pack["mean_contour"], dtype=np.float32)
    if std.shape != (11, 100) or mean.shape != (11, 100):
        raise ValueError(f"Unexpected normalization shapes at {path}: {std.shape}, {mean.shape}")
    if not np.isfinite(std).all() or not np.isfinite(mean).all() or not (std > 0).all():
        raise ValueError(f"Invalid normalization at {path}")
    return std, mean


def load_model(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> tuple[BaselineModel, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise TypeError(f"Invalid checkpoint: {checkpoint_path}")
    model = BaselineModel(
        int(config["input_layer"]),
        int(config["hidden_layer"]),
        int(config["num_layers"]),
        int(config["output_layer"]),
        len(config["classes"]),
        int(config["nbr_phonemes"]),
        config["phonemesdir"],
    )
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, int(checkpoint.get("epoch", -1))


def metric_payload(values_px: np.ndarray, mm_per_pixel: float) -> dict[str, Any]:
    return {
        "overall_px": float(values_px.mean()),
        "overall_mm": float(values_px.mean() * mm_per_pixel),
        "per_contour_px": {
            name: float(values_px[index]) for index, name in enumerate(CLASSES)
        },
        "per_contour_mm": {
            name: float(values_px[index] * mm_per_pixel) for index, name in enumerate(CLASSES)
        },
    }


def evaluate_checkpoint(
    *,
    name: str,
    config: dict[str, Any],
    checkpoint_path: Path,
    prediction_stats_path: Path,
    target_std: np.ndarray,
    target_mean: np.ndarray,
    state: dict[str, Any],
    device: torch.device,
    batch_size: int,
    mm_per_pixel: float,
) -> dict[str, Any]:
    prediction_std, prediction_mean = load_normalization(prediction_stats_path)
    model, checkpoint_epoch = load_model(config, checkpoint_path, device)

    features = state["features"]
    labels = state["labels"]
    frames = state["frames"]
    lengths = np.asarray(state["sequences_length"], dtype=np.int64)
    if features.shape[:2] != labels.shape[:2] or frames.shape[:2] != labels.shape[:2]:
        raise ValueError("Feature/label/frame split-cache shapes disagree")

    frame_rmse_sum = np.zeros(len(CLASSES), dtype=np.float64)
    point_sse = np.zeros((len(CLASSES), 100), dtype=np.float64)
    sequence_rmse_sum = np.zeros(len(CLASSES), dtype=np.float64)
    global_sse = 0.0
    integer_rows = 0
    fractional_rows_excluded = 0
    scored_sequences = 0

    pred_std = prediction_std[None, None, :, :]
    pred_mean = prediction_mean[None, None, :, :]
    gt_std = target_std[None, None, :, :]
    gt_mean = target_mean[None, None, :, :]

    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            batch_features = features[start:stop].to(device, non_blocking=True)
            batch_lengths = torch.as_tensor(lengths[start:stop], dtype=torch.long, device=device)
            predicted, _, _ = model(batch_features, batch_lengths)
            predicted_np = predicted.detach().cpu().numpy().astype(np.float32, copy=False)
            labels_np = labels[start:stop, : predicted_np.shape[1]].numpy().astype(np.float32, copy=False)
            predicted_raw = predicted_np * pred_std + pred_mean
            target_raw = labels_np * gt_std + gt_mean

            for local_index, length in enumerate(lengths[start:stop]):
                valid_length = min(int(length), predicted_raw.shape[1])
                frame_values = frames[start + local_index, :valid_length, 2].numpy()
                integer_mask = np.isclose(frame_values, np.rint(frame_values), atol=1e-6, rtol=0.0)
                fractional_rows_excluded += int((~integer_mask).sum())
                if not np.any(integer_mask):
                    continue
                error = (
                    predicted_raw[local_index, :valid_length][integer_mask]
                    - target_raw[local_index, :valid_length][integer_mask]
                ).astype(np.float64, copy=False)
                squared = np.square(error)
                row_count = int(squared.shape[0])
                integer_rows += row_count
                scored_sequences += 1
                frame_rmse_sum += np.sqrt(np.mean(squared, axis=2)).sum(axis=0)
                point_sse += squared.sum(axis=0)
                sequence_rmse_sum += np.sqrt(np.mean(squared, axis=0)).mean(axis=1)
                global_sse += float(squared.sum())

    if integer_rows <= 0 or scored_sequences <= 0:
        raise RuntimeError("No integer rows were scored")
    image_values_px = frame_rmse_sum / integer_rows
    point_values_px = np.sqrt(point_sse / integer_rows).mean(axis=1)
    sequence_values_px = sequence_rmse_sum / scored_sequences
    global_coordinate_rmse_px = float(
        np.sqrt(global_sse / (integer_rows * len(CLASSES) * 100))
    )
    del model
    torch.cuda.empty_cache()

    return {
        "name": name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch_zero_based": checkpoint_epoch,
        "checkpoint_epoch_one_based": checkpoint_epoch + 1,
        "prediction_normalization": str(prediction_stats_path),
        "prediction_normalization_sha256": file_sha256(prediction_stats_path),
        "integer_rows_scored": integer_rows,
        "fractional_rows_excluded": fractional_rows_excluded,
        "scored_sequences": scored_sequences,
        "sequence_coordinate_rmse": metric_payload(sequence_values_px, mm_per_pixel),
        "image_coordinate_rmse": metric_payload(image_values_px, mm_per_pixel),
        "point_coordinate_rmse": metric_payload(point_values_px, mm_per_pixel),
        "global_coordinate_rmse_px": global_coordinate_rmse_px,
        "global_coordinate_rmse_mm": global_coordinate_rmse_px * mm_per_pixel,
    }


def metric_delta(candidate: dict[str, Any], reference: dict[str, Any], key: str) -> dict[str, Any]:
    candidate_metric = candidate[key]
    reference_metric = reference[key]
    return {
        "overall_px": candidate_metric["overall_px"] - reference_metric["overall_px"],
        "overall_mm": candidate_metric["overall_mm"] - reference_metric["overall_mm"],
        "per_contour_px": {
            name: candidate_metric["per_contour_px"][name] - reference_metric["per_contour_px"][name]
            for name in CLASSES
        },
        "per_contour_mm": {
            name: candidate_metric["per_contour_mm"][name] - reference_metric["per_contour_mm"][name]
            for name in CLASSES
        },
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.mm_per_pixel <= 0:
        raise ValueError("batch-size and mm-per-pixel must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation")
    config = load_yaml(args.config.resolve())
    state = torch.load(args.test_cache.resolve(), map_location="cpu")
    target_std, target_mean = load_normalization(args.target_normalization.resolve())

    common = {
        "config": config,
        "target_std": target_std,
        "target_mean": target_mean,
        "state": state,
        "device": device,
        "batch_size": args.batch_size,
        "mm_per_pixel": args.mm_per_pixel,
    }
    reference = evaluate_checkpoint(
        name=args.reference_name,
        checkpoint_path=args.reference_checkpoint.resolve(),
        prediction_stats_path=args.reference_normalization.resolve(),
        **common,
    )
    candidate = evaluate_checkpoint(
        name=args.candidate_name,
        checkpoint_path=args.candidate_checkpoint.resolve(),
        prediction_stats_path=args.candidate_normalization.resolve(),
        **common,
    )
    for key in ("integer_rows_scored", "fractional_rows_excluded", "scored_sequences"):
        if reference[key] != candidate[key]:
            raise AssertionError(f"Model comparison row mismatch for {key}")

    report = {
        "status": "complete",
        "protocol": "paired_same_features_same_new_targets_integer_frames_only",
        "config": str(args.config.resolve()),
        "test_cache": str(args.test_cache.resolve()),
        "test_cache_sha256": file_sha256(args.test_cache.resolve()),
        "target_normalization": str(args.target_normalization.resolve()),
        "target_normalization_sha256": file_sha256(args.target_normalization.resolve()),
        "mm_per_pixel": float(args.mm_per_pixel),
        "integer_rows_scored": reference["integer_rows_scored"],
        "fractional_input_rows_excluded": reference["fractional_rows_excluded"],
        "fractional_saved_rows": 0,
        "fractional_scored_rows": 0,
        "fractional_rendered_rows": 0,
        "reference": reference,
        "candidate": candidate,
        "candidate_minus_reference": {
            key: metric_delta(candidate, reference, key)
            for key in (
                "sequence_coordinate_rmse",
                "image_coordinate_rmse",
                "point_coordinate_rmse",
            )
        },
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
