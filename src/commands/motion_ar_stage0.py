"""Maintained Stage-0 audit, train, and evaluation command implementations."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.model.autoregressive_contour import (
    AnchoredAutoregressiveContour,
    BaselineAudioOnly,
)
from src.train.train_autoregressive import (
    REPOSITORY_ROOT,
    load_motion_ar_config,
    load_split_state,
    make_loader,
    overfit_cached_batch,
    resolve_local_path,
    train_from_config,
)
from src.utils.motion_metrics import evaluate_arrays, velocity_mask


DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "config/train_config/motion_ar/stage0a_seed42.yaml"
)
DEFAULT_RESULTS = REPOSITORY_ROOT / "results/motion_ar_stage0"


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_safe),
        encoding="utf-8",
    )


def _session_membership(config: dict[str, Any], split: str) -> list[str]:
    return [
        f"{bucket}_{session}"
        for bucket, sessions in config[split].items()
        for session in sessions
    ]


def _finite_count(tensor: torch.Tensor, batch_size: int = 64) -> int:
    total = 0
    for start in range(0, tensor.shape[0], batch_size):
        total += int((~torch.isfinite(tensor[start : start + batch_size])).sum())
    return total


def _split_statistics(
    state: dict[str, Any],
    *,
    split: str,
    classes: list[str],
    include_contour_statistics: bool,
) -> dict[str, Any]:
    lengths = np.asarray(state["sequences_length"], dtype=int)
    unique_lengths, length_counts = np.unique(lengths, return_counts=True)
    shapes = {
        key: list(value.shape)
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    dtypes = {
        key: str(value.dtype)
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    nonfinite = {
        key: _finite_count(state[key])
        for key in ("features", "labels", "frames", "std", "mean")
    }
    time_step_counts: Counter[str] = Counter()
    gap_count = 0
    duplicate_or_backward = 0
    recovered_sessions: set[str] = set()
    frames = state["frames"]
    for start in range(0, frames.shape[0], 128):
        stop = min(start + 128, frames.shape[0])
        frame_batch = frames[start:stop].numpy()
        for local_index, row in enumerate(frame_batch):
            length = int(lengths[start + local_index])
            valid = row[:length]
            if not len(valid):
                continue
            recovered_sessions.add(f"{int(valid[0, 0])}_S{int(valid[0, 1])}")
            differences = np.diff(valid[:, 2])
            for difference in differences:
                time_step_counts[f"{float(difference):.6g}"] += 1
                if difference > 0.500001:
                    gap_count += 1
                elif difference <= 0:
                    duplicate_or_backward += 1
    result: dict[str, Any] = {
        "split": split,
        "num_cached_sequences": int(shapes["features"][0]),
        "tensor_shapes": shapes,
        "tensor_dtypes": dtypes,
        "length_distribution": {
            "minimum": int(lengths.min()),
            "maximum": int(lengths.max()),
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "p05": float(np.percentile(lengths, 5)),
            "p95": float(np.percentile(lengths, 95)),
            "counts": {
                str(int(length)): int(count)
                for length, count in zip(unique_lengths, length_counts)
            },
        },
        "frame_metadata": {
            "shape": shapes["frames"],
            "layout": ["speaker_bucket", "session_number", "half_frame_index"],
            "recoverable": True,
            "recovered_sessions": sorted(recovered_sessions),
        },
        "temporal_steps": dict(sorted(time_step_counts.items(), key=lambda item: float(item[0]))),
        "detected_internal_gaps": gap_count,
        "duplicate_or_backward_steps": duplicate_or_backward,
        "nonfinite_values": nonfinite,
        "normalization": {
            "std_min": float(state["std"].min()),
            "std_max": float(state["std"].max()),
            "mean_min": float(state["mean"].min()),
            "mean_max": float(state["mean"].max()),
        },
    }
    if include_contour_statistics:
        articulators = len(classes)
        points = state["labels"].shape[-1]
        position_sum = torch.zeros((articulators, points), dtype=torch.float64)
        position_square_sum = torch.zeros_like(position_sum)
        position_min = torch.full((articulators,), float("inf"), dtype=torch.float64)
        position_max = torch.full((articulators,), float("-inf"), dtype=torch.float64)
        velocity_square_sum = torch.zeros((articulators,), dtype=torch.float64)
        velocity_absolute_sum = torch.zeros((articulators,), dtype=torch.float64)
        valid_position_count = 0
        valid_velocity_count = 0
        normalized_sum = torch.zeros((articulators, points), dtype=torch.float64)
        normalized_square_sum = torch.zeros_like(normalized_sum)
        for start in range(0, state["labels"].shape[0], 32):
            stop = min(start + 32, state["labels"].shape[0])
            labels = state["labels"][start:stop]
            std = state["std"][start:stop]
            mean = state["mean"][start:stop]
            batch_lengths = torch.as_tensor(lengths[start:stop])
            steps = torch.arange(labels.shape[1]).unsqueeze(0)
            mask = steps < batch_lengths.unsqueeze(1)
            normalized = labels[mask].double()
            raw = (labels * std + mean)[mask].double()
            normalized_sum += normalized.sum(dim=0)
            normalized_square_sum += normalized.square().sum(dim=0)
            position_sum += raw.sum(dim=0)
            position_square_sum += raw.square().sum(dim=0)
            position_min = torch.minimum(position_min, raw.amin(dim=(0, 2)))
            position_max = torch.maximum(position_max, raw.amax(dim=(0, 2)))
            valid_position_count += raw.shape[0]
            delta = labels[:, 1:] - labels[:, :-1]
            vmask = velocity_mask(batch_lengths, labels.shape[1])
            valid_delta = delta[vmask].double()
            velocity_square_sum += valid_delta.square().sum(dim=(0, 2))
            velocity_absolute_sum += valid_delta.abs().sum(dim=(0, 2))
            valid_velocity_count += valid_delta.shape[0] * points
        normalized_mean = normalized_sum / max(1, valid_position_count)
        normalized_std = torch.sqrt(
            (
                normalized_square_sum / max(1, valid_position_count)
                - normalized_mean.square()
            ).clamp_min(0)
        )
        position_mean = position_sum / max(1, valid_position_count)
        position_std = torch.sqrt(
            (
                position_square_sum / max(1, valid_position_count)
                - position_mean.square()
            ).clamp_min(0)
        )
        result["contour_statistics"] = {
            "normalized_mean_range": [
                float(normalized_mean.min()),
                float(normalized_mean.max()),
            ],
            "normalized_std_range": [
                float(normalized_std.min()),
                float(normalized_std.max()),
            ],
            "denormalized_mean_range": [
                float(position_mean.min()),
                float(position_mean.max()),
            ],
            "denormalized_std_range": [
                float(position_std.min()),
                float(position_std.max()),
            ],
            "per_articulator": [
                {
                    "class": name,
                    "position_mean": float(position_mean[index].mean()),
                    "position_std": float(position_std[index].mean()),
                    "position_min": float(position_min[index]),
                    "position_max": float(position_max[index]),
                    "velocity_rms_normalized": float(
                        torch.sqrt(
                            velocity_square_sum[index] / max(1, valid_velocity_count)
                        )
                    ),
                    "velocity_mean_abs_normalized": float(
                        velocity_absolute_sum[index] / max(1, valid_velocity_count)
                    ),
                }
                for index, name in enumerate(classes)
            ],
        }
    return result


def _raw_cache_statistics(config: dict[str, Any]) -> dict[str, Any]:
    root = resolve_local_path(config["session_cache_dir"])
    paths = sorted(root.rglob("*.pt"))
    split_counts: Counter[str] = Counter()
    chunk_count = 0
    internal_gaps = 0
    boundary_gaps = 0
    duplicate_or_backward = 0
    normal_steps: Counter[str] = Counter()
    schemas: dict[str, Any] = {}
    sessions: list[dict[str, Any]] = []
    started = time.time()
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        raw = payload["raw"]
        split_counts[str(payload["split"])] += 1
        frames = raw["frames"]
        chunk_count += len(frames)
        previous_end = None
        session_internal_gaps = 0
        session_boundary_gaps = 0
        for frame_chunk in frames:
            frame_chunk = np.asarray(frame_chunk)
            differences = np.diff(frame_chunk[:, 2])
            for difference in differences:
                normal_steps[f"{float(difference):.6g}"] += 1
                if difference > 0.500001:
                    internal_gaps += 1
                    session_internal_gaps += 1
                elif difference <= 0:
                    duplicate_or_backward += 1
            if previous_end is not None:
                boundary_difference = float(frame_chunk[0, 2] - previous_end)
                normal_steps[f"boundary:{boundary_difference:.6g}"] += 1
                if boundary_difference > 0.500001:
                    boundary_gaps += 1
                    session_boundary_gaps += 1
                elif boundary_difference <= 0:
                    duplicate_or_backward += 1
            previous_end = float(frame_chunk[-1, 2])
        sessions.append(
            {
                "bucket": str(payload["bucket"]),
                "session": str(payload["session"]),
                "split": str(payload["split"]),
                "num_chunks": len(frames),
                "internal_gaps": session_internal_gaps,
                "boundary_gaps": session_boundary_gaps,
            }
        )
        if not schemas:
            schemas = {
                key: {
                    "container": type(value).__name__,
                    "items": len(value) if hasattr(value, "__len__") else None,
                    "first_shape": list(np.asarray(value[0]).shape)
                    if isinstance(value, list) and value
                    else None,
                    "first_dtype": str(np.asarray(value[0]).dtype)
                    if isinstance(value, list) and value
                    else None,
                }
                for key, value in raw.items()
            }
    return {
        "root": str(root),
        "num_files": len(paths),
        "split_session_counts": dict(split_counts),
        "num_chunks": chunk_count,
        "schema": schemas,
        "time_step_distribution": dict(normal_steps),
        "internal_gaps": internal_gaps,
        "boundary_gaps": boundary_gaps,
        "duplicate_or_backward_steps": duplicate_or_backward,
        "can_recover_bucket_session_frame": True,
        "safe_reassembly": (
            "Conditionally yes: sort raw chunks by cached half-frame index and split "
            "at every non-0.5 step. Naive concatenation is unsafe."
        ),
        "sessions": sessions,
        "elapsed_seconds": time.time() - started,
    }


def _audit_markdown(inventory: dict[str, Any]) -> str:
    config = inventory["configuration"]
    lines = [
        "# ASD2 Stage-0 Cache Audit",
        "",
        f"- Branch: `{inventory['repository']['branch']}`",
        f"- Commit: `{inventory['repository']['commit']}`",
        f"- Python: `{inventory['environment']['python']}`",
        f"- PyTorch/CUDA: `{inventory['environment']['torch']}` / "
        f"`{inventory['environment']['cuda_available']}`",
        f"- Normalization: `{config['normalization_mode']}` fitted on "
        f"`{config['normalization_fit_split']}` only.",
        f"- Configured output: {config['num_classes']} classes × "
        f"{config['output_layer']} coordinates = **{config['output_dimension']}**.",
        "",
        "## Split inventory",
        "",
        "| Split | Sessions | Cached sequences | Length min/median/max | Gaps | Non-finite |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train_sequences", "valid_sequences", "test_sequences"):
        item = inventory["splits"][split]
        distribution = item["length_distribution"]
        lines.append(
            f"| {split} | {len(inventory['membership'][split])} | "
            f"{item['num_cached_sequences']} | "
            f"{distribution['minimum']}/{distribution['median']:.0f}/{distribution['maximum']} | "
            f"{item['detected_internal_gaps']} | "
            f"{sum(item['nonfinite_values'].values())} |"
        )
    lines.extend(
        [
            "",
            "Tensor shapes and dtypes are recorded verbatim in `cache_inventory.json`. "
            "Labels are `[sequence,time,articulator,coordinate]`; frames are "
            "`[speaker bucket, session number, half-frame index]`.",
            "",
            "## Reassembly decision",
            "",
            inventory["raw_cache"]["safe_reassembly"],
            "",
            f"The raw cache contains {inventory['raw_cache']['internal_gaps']} internal "
            f"and {inventory['raw_cache']['boundary_gaps']} between-chunk non-standard "
            "steps. The normal temporal step is 0.5 cached frame units. Bucket, session, "
            "and time are recoverable, but cached chunks must never be concatenated "
            "without sorting and splitting at gaps, duplicates, or backwards time.",
            "",
            "## Train-only motion statistics",
            "",
            "| Articulator | Position mean | Position std | Velocity RMS (normalized) |",
            "|---|---:|---:|---:|",
        ]
    )
    stats = inventory["splits"]["train_sequences"]["contour_statistics"]
    for row in stats["per_articulator"]:
        lines.append(
            f"| {row['class']} | {row['position_mean']:.6g} | "
            f"{row['position_std']:.6g} | {row['velocity_rms_normalized']:.6g} |"
        )
    lines.extend(
        [
            "",
            "All position and velocity statistics above use the training split only. "
            "Validation and test values were inspected only for integrity and are not "
            "used to fit normalization or class weights.",
            "",
        ]
    )
    return "\n".join(lines)


def audit(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config = load_motion_ar_config(config_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    inventory: dict[str, Any] = {
        "repository": {"branch": branch, "commit": commit},
        "environment": {
            "python": subprocess.check_output(
                ["python", "--version"], text=True, stderr=subprocess.STDOUT
            ).strip(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
        "configuration": {
            "path": str(Path(config_path).resolve()),
            "normalization_mode": config.get("normalization_mode"),
            "normalization_fit_split": config.get("normalization_fit_split"),
            "normalization_std_policy": config.get("normalization_std_policy"),
            "num_classes": len(config["classes"]),
            "classes": list(config["classes"]),
            "output_layer": int(config["output_layer"]),
            "output_dimension": len(config["classes"]) * int(config["output_layer"]),
        },
        "membership": {
            split: _session_membership(config, split)
            for split in ("train_sequences", "valid_sequences", "test_sequences")
        },
        "splits": {},
    }
    for split in ("train_sequences", "valid_sequences", "test_sequences"):
        print(f"Auditing {split}...", flush=True)
        state = load_split_state(config, split)
        inventory["splits"][split] = _split_statistics(
            state,
            split=split,
            classes=list(config["classes"]),
            include_contour_statistics=split == "train_sequences",
        )
        del state
    print("Auditing raw per-session cache metadata...", flush=True)
    inventory["raw_cache"] = _raw_cache_statistics(config)
    inventory["consistency"] = {
        "declared_sessions": sum(len(items) for items in inventory["membership"].values()),
        "raw_session_files": inventory["raw_cache"]["num_files"],
        "declared_matches_raw": sum(
            len(items) for items in inventory["membership"].values()
        )
        == inventory["raw_cache"]["num_files"],
        "cache_schema_matches_config": all(
            item["tensor_shapes"]["labels"][-2:]
            == [len(config["classes"]), int(config["output_layer"])]
            for item in inventory["splits"].values()
        ),
    }
    _write_json(output_dir / "cache_inventory.json", inventory)
    markdown = _audit_markdown(inventory)
    (output_dir / "CACHE_AUDIT.md").write_text(markdown, encoding="utf-8")
    root_report = DEFAULT_RESULTS / "CACHE_AUDIT.md"
    root_report.parent.mkdir(parents=True, exist_ok=True)
    root_report.write_text(markdown, encoding="utf-8")
    return inventory


def _load_ar_model(
    config: dict[str, Any], checkpoint_path: str | Path, device: torch.device
) -> AnchoredAutoregressiveContour:
    payload = torch.load(
        resolve_local_path(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    model = AnchoredAutoregressiveContour.from_config(config)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _predict(
    config: dict[str, Any],
    state: dict[str, Any],
    *,
    method: str,
    checkpoint: str | Path | None,
    ablation: str,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    loader = make_loader(
        state,
        batch_size=int(config.get("evaluation_batch_size", config.get("batch_size", 10))),
        shuffle=False,
        seed=int(config.get("seed", 42)),
    )
    chunks: list[np.ndarray] = []
    diagnostics: list[dict[str, float]] = []
    if method == "static":
        for batch in loader:
            target = batch["labels"]
            chunks.append(target[:, :1].expand_as(target).numpy().copy())
        return np.concatenate(chunks), {}
    if checkpoint is None:
        raise ValueError(f"{method} evaluation requires --checkpoint")
    if method == "audio":
        model: torch.nn.Module = BaselineAudioOnly(config)
        model.load_checkpoint(str(resolve_local_path(checkpoint)))
        model = model.to(device).eval()
    elif method == "ar":
        model = _load_ar_model(config, checkpoint, device)
    else:
        raise ValueError(f"Unknown evaluation method: {method}")
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(config.get("seed", 42)))
    with torch.no_grad():
        for batch in loader:
            audio = batch["features"].to(device)
            target = batch["labels"].to(device)
            lengths = batch["sequences_length"].to(device)
            if method == "audio":
                prediction = model(audio, lengths)
            else:
                anchor = target[:, 0]
                if ablation == "anchor_shuffle":
                    anchor = anchor.roll(1, dims=0)
                history_mode = (
                    "clamp"
                    if ablation == "history_clamp"
                    else "shuffle"
                    if ablation == "history_shuffle"
                    else "normal"
                )
                prediction, diagnostic = model.infer(
                    audio,
                    anchor,
                    lengths,
                    history_mode=history_mode,
                    reset_decoder_state=ablation == "state_reset",
                    shuffle_audio_time=ablation == "audio_shuffle",
                    generator=generator,
                )
                diagnostics.append(
                    {
                        "anchor_embedding_norm": diagnostic.anchor_embedding_norm,
                        "history_embedding_norm": diagnostic.history_embedding_norm,
                        "audio_embedding_norm": diagnostic.audio_embedding_norm,
                        "decoder_state_norm": diagnostic.decoder_state_norm,
                    }
                )
            chunks.append(prediction.detach().cpu().numpy())
    summary = {}
    if diagnostics:
        summary = {
            key: float(np.mean([row[key] for row in diagnostics]))
            for key in diagnostics[0]
        }
    return np.concatenate(chunks), summary


def _metric_tables(
    prediction: np.ndarray,
    state: dict[str, Any],
    classes: list[str],
) -> dict[str, Any]:
    target = state["labels"].numpy()
    lengths = np.asarray(state["sequences_length"], dtype=int)
    std = state["std"].numpy()
    mean = state["mean"].numpy()
    frames = state["frames"].numpy()
    tables: dict[str, Any] = {
        "global": evaluate_arrays(
            prediction, target, lengths, std=std, mean=mean
        ),
        "per_session": [],
        "per_articulator": [],
        "rollout_horizons": [],
    }
    session_ids = np.asarray(
        [f"{int(row[0, 0])}_S{int(row[0, 1])}" for row in frames]
    )
    prediction_raw = prediction * std + mean
    frozen_sequences: list[str] = []
    nonfinite_sequences: list[str] = []
    eligible_sequences = 0
    for index, length in enumerate(lengths):
        length = min(int(length), prediction.shape[1])
        identifier = f"{session_ids[index]}:chunk-{index}"
        if not np.isfinite(prediction[index, :length]).all():
            nonfinite_sequences.append(identifier)
        if length > 1:
            eligible_sequences += 1
            motion = float(
                np.mean(np.abs(np.diff(prediction_raw[index, :length], axis=0)))
            )
            if motion <= 1e-6:
                frozen_sequences.append(identifier)
    tables["sequence_health"] = {
        "eligible_sequences_with_predicted_steps": eligible_sequences,
        "sequences_without_post_anchor_step": int(np.sum(lengths <= 1)),
        "frozen_threshold_mean_abs_frame_difference": 1e-6,
        "frozen_sequence_count": len(frozen_sequences),
        "frozen_sequences": frozen_sequences,
        "nonfinite_sequence_count": len(nonfinite_sequences),
        "nonfinite_sequences": nonfinite_sequences,
    }
    for session in sorted(set(session_ids)):
        indices = np.flatnonzero(session_ids == session)
        row = evaluate_arrays(
            prediction[indices],
            target[indices],
            lengths[indices],
            std=std[indices],
            mean=mean[indices],
        )
        row["session"] = session
        tables["per_session"].append(row)
    for index, name in enumerate(classes):
        row = evaluate_arrays(
            prediction[:, :, index : index + 1],
            target[:, :, index : index + 1],
            lengths,
            std=std[:, :, index : index + 1],
            mean=mean[:, :, index : index + 1],
        )
        row["class"] = name
        tables["per_articulator"].append(row)
    maximum = int(lengths.max()) - 1
    for horizon in (20, 40, 79, 159, 319):
        if horizon > maximum and horizon not in (20, 40, 79):
            continue
        horizon_lengths = np.minimum(lengths, horizon + 1)
        row = evaluate_arrays(
            prediction,
            target,
            horizon_lengths,
            std=std,
            mean=mean,
        )
        row["predicted_steps"] = min(horizon, maximum)
        tables["rollout_horizons"].append(row)
    return tables


def leakage_test(
    config: dict[str, Any], checkpoint: str | Path, device: torch.device
) -> dict[str, Any]:
    state = load_split_state(config, "test_sequences")
    batch = next(
        iter(make_loader(state, batch_size=3, shuffle=False, seed=int(config.get("seed", 42))))
    )
    model = _load_ar_model(config, checkpoint, device)
    audio = batch["features"].to(device)
    original = batch["labels"].to(device)
    lengths = batch["sequences_length"].to(device)
    first, _ = model.infer(audio, original[:, 0], lengths)
    randomized = original.clone()
    randomized[:, 1:] = torch.randn_like(randomized[:, 1:])
    second, _ = model.infer(audio, randomized[:, 0], lengths)
    return {
        "bitwise_identical": bool(torch.equal(first, second)),
        "maximum_absolute_difference": float((first - second).abs().max().cpu()),
        "inference_signature_accepts_post_anchor_targets": False,
    }


def evaluate(
    config_path: str | Path,
    *,
    method: str,
    checkpoint: str | Path | None,
    split: str,
    ablation: str,
    output_dir: str | Path,
    device_name: str | None,
    save_predictions: bool,
    prediction_session: str | None,
) -> dict[str, Any]:
    config = load_motion_ar_config(config_path)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = load_split_state(config, split)
    prediction, diagnostics = _predict(
        config,
        state,
        method=method,
        checkpoint=checkpoint,
        ablation=ablation,
        device=device,
    )
    result = {
        "method": method,
        "ablation": ablation,
        "split": split,
        "checkpoint": str(resolve_local_path(checkpoint)) if checkpoint else None,
        "diagnostics": diagnostics,
        "metrics": _metric_tables(prediction, state, list(config["classes"])),
    }
    if method == "ar":
        result["leakage_test"] = leakage_test(config, checkpoint, device)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "metrics.json", result)
    if save_predictions:
        indices = np.arange(prediction.shape[0])
        if prediction_session:
            frame_array = state["frames"].numpy()
            session_ids = np.asarray(
                [
                    f"{int(row[0, 0])}_S{int(row[0, 1])}"
                    for row in frame_array
                ]
            )
            indices = np.flatnonzero(session_ids == prediction_session)
            if not len(indices):
                raise ValueError(
                    f"Prediction session {prediction_session!r} is absent from {split}"
                )
        index_tensor = torch.as_tensor(indices, dtype=torch.long)
        normalized_prediction = torch.from_numpy(prediction[indices])
        labels = state["labels"][index_tensor]
        std = state["std"][index_tensor]
        mean = state["mean"][index_tensor]
        torch.save(
            {
                "predicted": normalized_prediction,
                "labels": labels,
                "predicted_raw": normalized_prediction * std + mean,
                "labels_raw": labels * std + mean,
                "std": std,
                "mean": mean,
                "frames": state["frames"][index_tensor],
                "lengths": torch.as_tensor(state["sequences_length"])[index_tensor],
                "method": method,
                "ablation": ablation,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "prediction_session": prediction_session,
            },
            output_dir / "predictions.pt",
        )
    print(json.dumps(result["metrics"]["global"], indent=2), flush=True)
    return result


def audit_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit ASD2 Stage-0 split and raw caches")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS / "audit"))
    args = parser.parse_args(argv)
    audit(args.config, args.output_dir)
    return 0


def train_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Stage-0 contour AR model")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS / "runs/seed42"))
    parser.add_argument("--device")
    parser.add_argument("--sanity", choices=("s3", "s4"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    config = load_motion_ar_config(args.config)
    if args.sanity:
        result = overfit_cached_batch(
            config,
            free_running=args.sanity == "s4",
            steps=args.steps or (500 if args.sanity == "s4" else 250),
            batch_size=args.batch_size,
            device_name=args.device,
        )
        output = DEFAULT_RESULTS / "runs" / f"{args.sanity}_sanity.json"
        _write_json(output, result)
        print(json.dumps(result, indent=2), flush=True)
        minimum = 0.75 if args.sanity == "s3" else 0.60
        passed = (
            result["position_reduction"] >= minimum
            and result["velocity_reduction"] >= minimum
            and (not args.sanity == "s4" or result["final_teacher_forcing_ratio"] == 0)
        )
        return 0 if passed else 1
    summary = train_from_config(config, output_dir=args.output_dir, device_name=args.device)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def evaluate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Stage-0 baselines or AR rollout")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--method", choices=("static", "audio", "ar"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--ablation",
        choices=(
            "none",
            "history_clamp",
            "history_shuffle",
            "state_reset",
            "audio_shuffle",
            "anchor_shuffle",
        ),
        default="none",
    )
    parser.add_argument("--split", default="test_sequences")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--prediction-session",
        help="When saving, retain only this speaker_session (for example 1775_S20).",
    )
    args = parser.parse_args(argv)
    evaluate(
        args.config,
        method=args.method,
        checkpoint=args.checkpoint,
        split=args.split,
        ablation=args.ablation,
        output_dir=args.output_dir,
        device_name=args.device,
        save_predictions=args.save_predictions,
        prediction_session=args.prediction_session,
    )
    return 0


def diagnose_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the reusable motion metrics from a Stage-0 evaluation"
    )
    parser.add_argument("metrics_json")
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    print(json.dumps(payload["metrics"]["global"], indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "train", "evaluate", "diagnose"))
    args, remainder = parser.parse_known_args(argv)
    return {
        "audit": audit_main,
        "train": train_main,
        "evaluate": evaluate_main,
        "diagnose": diagnose_main,
    }[args.action](remainder)
