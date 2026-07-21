#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.model.baseline_5 import BaselineModel  # noqa: E402
from src.utils.config_validation import load_yaml_config  # noqa: E402
from src.utils.normalization import (  # noqa: E402
    DENORM_SPLIT_CACHE_KEYS,
    INFERENCE_SPLIT_CACHE_KEYS,
    describe_split_denorm,
    load_validated_split_cache_state,
)
from src.utils.video_rendering import MM_PER_PIXEL  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one cached test session.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--speaker", required=True, help="Numeric speaker id in cached frame ids, e.g. 2 for P2")
    parser.add_argument("--session", required=True, help="Numeric session id in cached frame ids, e.g. 1 for S1")
    parser.add_argument("--split", default="test_sequences", choices=("train_sequences", "valid_sequences", "test_sequences"))
    parser.add_argument(
        "--split-cache-path",
        type=Path,
        default=None,
        help="Optional explicit split cache, useful for cross-dataset evaluation without changing the training config.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--skip-predictions", action="store_true", help="Do not write the large prediction tensor artifact.")
    parser.add_argument("--skip-frame-csv", action="store_true", help="Do not write per-frame RMSE CSV.")
    parser.add_argument("--write-contours", action="store_true", help="Write predicted contours as per-frame .npy files.")
    parser.add_argument(
        "--contour-output-dir",
        type=Path,
        default=None,
        help="Directory for --write-contours. Defaults to <output-dir>/predicted_contours.",
    )
    parser.add_argument(
        "--contour-output-format",
        choices=("xy50", "flat100"),
        default="xy50",
        help="Saved .npy contour format for --write-contours.",
    )
    parser.add_argument(
        "--exclude-rmse-classes",
        nargs="*",
        default=[],
        help="Class names to exclude from RMSE computation while still saving/rendering predictions.",
    )
    parser.add_argument(
        "--prediction-only",
        action="store_true",
        help="Do not use target labels/std/mean; write predictions de-normalized by --denorm-cache.",
    )
    parser.add_argument(
        "--denorm-cache",
        type=Path,
        default=None,
        help="Split cache whose train-set std/mean tensors are averaged for prediction-only de-normalization.",
    )
    parser.add_argument(
        "--prediction-denorm-cache",
        type=Path,
        default=None,
        help=(
            "Split cache whose std/mean tensors are averaged to de-normalize predictions "
            "while still loading target labels for compare/metrics."
        ),
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return load_yaml_config(path)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def split_cache_path(config: dict[str, Any], split: str) -> Path:
    split_files = {
        "train_sequences": "train_sequences.pt",
        "valid_sequences": "valid_sequences.pt",
        "test_sequences": "test_sequences.pt",
    }
    cache_dir = config.get("split_cache_dir") or config.get("dataset_cache_dir")
    if not cache_dir:
        raise KeyError("Train config must define split_cache_dir or dataset_cache_dir")
    return Path(cache_dir) / split_files[split]


def load_model(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> BaselineModel:
    model = BaselineModel(
        config["input_layer"],
        config["hidden_layer"],
        config["num_layers"],
        config["output_layer"],
        len(config["classes"]),
        config["nbr_phonemes"],
        config["phonemesdir"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    cleaned = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


def load_phonemes(config: dict[str, Any]) -> list[str]:
    with open(config["phonemesdir"], "r", encoding="utf-8") as handle:
        return json.load(handle)


def decode_phoneme(row: torch.Tensor, phonemes: list[str]) -> str:
    vector = row.detach().cpu().numpy()
    if vector.size == 0 or np.allclose(vector, 0):
        return "UNK"
    return str(phonemes[int(vector.argmax())])


def select_session_indices(state: dict[str, Any], speaker: int, session: int) -> list[int]:
    indices = []
    frames = state["frames"]
    lengths = state["sequences_length"]
    for idx in range(frames.shape[0]):
        length = int(lengths[idx])
        if length <= 0:
            continue
        valid_frames = frames[idx, :length]
        speaker_match = torch.round(valid_frames[:, 0]).to(torch.int64) == speaker
        session_match = torch.round(valid_frames[:, 1]).to(torch.int64) == session
        if bool(torch.any(speaker_match & session_match)):
            indices.append(idx)
    return indices


def load_reference_denorm(
    cache_path: Path,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing de-normalization cache: {cache_path}")
    state, floor_summary = load_validated_split_cache_state(
        cache_path,
        config,
        required_keys=DENORM_SPLIT_CACHE_KEYS,
    )
    if "std" not in state or "mean" not in state:
        raise KeyError(f"De-normalization cache must contain std and mean tensors: {cache_path}")
    std_cpu = state["std"].float()
    mean_cpu = state["mean"].float()
    if std_cpu.ndim != 4 or mean_cpu.ndim != 4:
        raise ValueError(f"Expected std/mean tensors shaped (N, 1, C, P), got {tuple(std_cpu.shape)} / {tuple(mean_cpu.shape)}")
    std_ref = std_cpu.mean(dim=0, keepdim=True).to(device)
    mean_ref = mean_cpu.mean(dim=0, keepdim=True).to(device)
    metadata = {
        "denorm_cache": str(cache_path),
        "denorm_method": "mean_std_and_mean_tensors_over_denorm_cache_sequences",
        "denorm_num_sequences": int(std_cpu.shape[0]),
        "denorm_std_shape": list(std_cpu.shape),
        "denorm_mean_shape": list(mean_cpu.shape),
        "uses_target_std_mean": False,
        **floor_summary,
    }
    return std_ref, mean_ref, metadata


def rmse_class_mask(config: dict[str, Any], excluded_classes: list[str], device: torch.device) -> tuple[torch.Tensor | None, list[str]]:
    classes = list(config["classes"])
    excluded = [name for name in excluded_classes if name in classes]
    if not excluded:
        return None, []
    keep = [name not in set(excluded) for name in classes]
    if not any(keep):
        raise ValueError("RMSE exclusion removed all classes")
    return torch.tensor(keep, dtype=torch.bool, device=device), excluded


def frame_token(value: float) -> str:
    rounded = int(round(value))
    if not np.isclose(value, rounded, atol=1e-4):
        raise ValueError(f"NEVER save fractional contour frame {value}")
    return f"{rounded:04d}"


def contour_array(values: np.ndarray, output_format: str) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if output_format == "flat100":
        return flat
    return flat.reshape(50, 2)


def write_predicted_contours(
    output_dir: Path,
    config: dict[str, Any],
    predicted_raw: torch.Tensor,
    frames: torch.Tensor,
    lengths: torch.Tensor,
    speaker: int,
    session: int,
    output_format: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    accum: dict[tuple[str, str], list[np.ndarray]] = {}
    predicted_cpu = predicted_raw.detach().cpu().numpy()
    classes = list(config["classes"])
    discarded_fractional_rows = 0
    for seq_idx in range(predicted_cpu.shape[0]):
        length = int(lengths[seq_idx].item())
        for frame_offset in range(length):
            frame = frames[seq_idx, frame_offset]
            frame_speaker = int(round(float(frame[0])))
            frame_session = int(round(float(frame[1])))
            if frame_speaker != speaker or frame_session != session:
                continue
            frame_number = float(frame[2])
            if not np.isclose(frame_number, round(frame_number), atol=1e-4):
                discarded_fractional_rows += 1
                continue
            frame_name = frame_token(frame_number)
            for class_idx, class_name in enumerate(classes):
                accum.setdefault((frame_name, class_name), []).append(predicted_cpu[seq_idx, frame_offset, class_idx])

    saved = 0
    for (frame_name, class_name), values in accum.items():
        mean_values = np.mean(np.stack(values, axis=0), axis=0)
        np.save(output_dir / f"{frame_name}_{class_name}.npy", contour_array(mean_values, output_format))
        saved += 1
    manifest = {
        "contour_output_dir": str(output_dir),
        "contour_output_format": output_format,
        "num_unique_frame_articulators": saved,
        "num_unique_frames": len({key[0] for key in accum}),
        "classes": classes,
        "averaged_overlapping_predictions": True,
        "frame_policy": "NEVER save fractional contour frames; integer frames only",
        "discarded_fractional_rows": discarded_fractional_rows,
        "saved_fractional_frame_count": 0,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def main() -> None:
    args = parse_args()
    if args.prediction_only and args.denorm_cache is None:
        raise ValueError("--prediction-only requires --denorm-cache")
    if args.prediction_only and args.prediction_denorm_cache is not None:
        raise ValueError("--prediction-denorm-cache is for compare/metric mode, not --prediction-only")
    config = load_config(args.config)
    cache_path = args.split_cache_path or split_cache_path(config, args.split)
    state, cache_floor_summary = load_validated_split_cache_state(
        cache_path,
        config,
        required_keys=INFERENCE_SPLIT_CACHE_KEYS,
    )
    speaker = int(args.speaker)
    session = int(args.session)
    indices = select_session_indices(state, speaker, session)
    if not indices:
        raise RuntimeError(f"No cached samples found for speaker={speaker} session={session} in {cache_path}")

    device = resolve_device(args.device)
    model = load_model(config, args.checkpoint, device)
    phonemes = load_phonemes(config)
    class_mask, excluded_rmse_classes = rmse_class_mask(config, list(args.exclude_rmse_classes), device)

    index_tensor = torch.tensor(indices, dtype=torch.long)
    features = state["features"].index_select(0, index_tensor).to(device)
    frames = state["frames"].index_select(0, index_tensor)
    phoneme_vectors = state["phonemes"].index_select(0, index_tensor)
    lengths = torch.as_tensor([int(state["sequences_length"][idx]) for idx in indices], dtype=torch.long, device=device)

    with torch.no_grad():
        predicted, _, _ = model(features, lengths)
        if args.prediction_only:
            denorm_std, denorm_mean, denorm_metadata = load_reference_denorm(args.denorm_cache, device, config)
            predicted_raw = (predicted * denorm_std) + denorm_mean
            labels = None
            labels_raw = None
            per_frame_rmse = None
        else:
            labels = state["labels"].index_select(0, index_tensor).to(device)
            std = state["std"].index_select(0, index_tensor).to(device)
            mean = state["mean"].index_select(0, index_tensor).to(device)
            labels = labels[:, : predicted.shape[1]]
            labels_raw = (labels * std) + mean
            if args.prediction_denorm_cache is not None:
                denorm_std, denorm_mean, denorm_metadata = load_reference_denorm(
                    args.prediction_denorm_cache,
                    device,
                    config,
                )
                predicted_raw = (predicted * denorm_std) + denorm_mean
                denorm_metadata.update(
                    {
                        "prediction_denorm_cache": str(args.prediction_denorm_cache),
                        "denorm_method": "prediction_from_reference_cache_labels_from_target_annotations",
                        "uses_target_std_mean": False,
                        "uses_target_std_mean_for_prediction": False,
                        "uses_target_std_mean_for_labels": True,
                    }
                )
            else:
                predicted_raw = (predicted * std) + mean
                denorm_metadata = describe_split_denorm(cache_path, config)
                denorm_metadata.update(
                    {
                        "uses_target_std_mean_for_prediction": bool(
                            denorm_metadata.get("uses_target_std_mean", False)
                        ),
                        "uses_target_std_mean_for_labels": True,
                        **cache_floor_summary,
                    }
                )
            metric_predicted_raw = predicted_raw if class_mask is None else predicted_raw[:, :, class_mask]
            metric_labels_raw = labels_raw if class_mask is None else labels_raw[:, :, class_mask]
            per_point_rmse = torch.sqrt(torch.mean((metric_predicted_raw - metric_labels_raw) ** 2, dim=-1))
            per_frame_rmse = per_point_rmse.mean(dim=-1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "cached_session_predictions.pt"
    if not args.skip_predictions:
        prediction_payload = {
            "indices": indices,
            "features": features.detach().cpu(),
            "predicted": predicted.detach().cpu(),
            "predicted_raw": predicted_raw.detach().cpu(),
            "frames": frames,
            "phonemes": phoneme_vectors,
            "lengths": lengths.detach().cpu(),
            "prediction_only": bool(args.prediction_only),
            **denorm_metadata,
            "excluded_rmse_classes": excluded_rmse_classes,
        }
        if labels is not None and labels_raw is not None:
            prediction_payload["labels"] = labels.detach().cpu()
            prediction_payload["labels_raw"] = labels_raw.detach().cpu()
        torch.save(prediction_payload, predictions_path)

    rows = []
    fractional_input_rows_excluded_from_metrics = 0
    for seq_idx, source_idx in enumerate(indices):
        length = int(lengths[seq_idx].item())
        for frame_offset in range(length):
            frame = frames[seq_idx, frame_offset]
            frame_speaker = int(round(float(frame[0])))
            frame_session = int(round(float(frame[1])))
            if frame_speaker != speaker or frame_session != session:
                continue
            frame_number = float(frame[2])
            if not np.isclose(frame_number, round(frame_number), atol=1e-4):
                fractional_input_rows_excluded_from_metrics += 1
                continue
            phoneme = decode_phoneme(phoneme_vectors[seq_idx, frame_offset, 0], phonemes)
            rows.append(
                {
                    "source_index": source_idx,
                    "frame": f"{frame_speaker}_S{frame_session}_{frame_number:.1f}",
                    "phoneme": phoneme,
                    "rmse_raw": None
                    if per_frame_rmse is None
                    else float(per_frame_rmse[seq_idx, frame_offset].detach().cpu()),
                    "rmse_mm": None
                    if per_frame_rmse is None
                    else float(per_frame_rmse[seq_idx, frame_offset].detach().cpu()) * MM_PER_PIXEL,
                }
            )

    rmse_csv_path = args.output_dir / ("frames.csv" if args.prediction_only else "rmse_per_frame.csv")
    if not args.skip_frame_csv:
        with rmse_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_index", "frame", "phoneme", "rmse_raw", "rmse_mm"])
            writer.writeheader()
            writer.writerows(rows)

    contour_manifest = None
    if args.write_contours:
        contour_dir = args.contour_output_dir or (args.output_dir / "predicted_contours")
        contour_manifest = write_predicted_contours(
            contour_dir,
            config,
            predicted_raw,
            frames,
            lengths,
            speaker,
            session,
            args.contour_output_format,
        )

    if args.prediction_only:
        mean_rmse_raw = None
        mean_rmse_mm = None
    else:
        mean_rmse_raw = float(np.mean([row["rmse_raw"] for row in rows]))
        mean_rmse_mm = mean_rmse_raw * MM_PER_PIXEL
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "cache_path": str(cache_path),
        "split": args.split,
        "speaker": speaker,
        "session": session,
        "device": str(device),
        "num_sequences": len(indices),
        "num_frames": len(rows),
        "fractional_input_rows_excluded_from_metrics": fractional_input_rows_excluded_from_metrics,
        "fractional_scored_rows": 0,
        "fractional_saved_rows": 0 if contour_manifest is None else contour_manifest["saved_fractional_frame_count"],
        "mean_rmse_raw": mean_rmse_raw,
        "mean_rmse_mm": mean_rmse_mm,
        "prediction_only": bool(args.prediction_only),
        "uses_target_labels": not bool(args.prediction_only),
        "uses_target_labels_for_metrics": not bool(args.prediction_only),
        "uses_target_std_mean": bool(denorm_metadata.get("uses_target_std_mean", False)),
        "excluded_rmse_classes": excluded_rmse_classes,
        "predicted_raw_min": float(predicted_raw.detach().cpu().min()),
        "predicted_raw_max": float(predicted_raw.detach().cpu().max()),
        "predictions": None if args.skip_predictions else str(predictions_path),
        "frame_csv": None if args.skip_frame_csv else str(rmse_csv_path),
        "contours": None if contour_manifest is None else contour_manifest,
        **denorm_metadata,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    with (args.output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Cached Session Inference\n\n")
        for key, value in report.items():
            handle.write(f"- {key}: `{value}`\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
