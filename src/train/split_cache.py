from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from preprocessing.session_cache import (
    SPLIT_FILES,
    atomic_torch_save,
    contour_pack_path,
    dataset_type_for_sequence,
    raw_session_part_path,
    recompute_mean_datas,
    write_metadata,
)
from utils.normalization import (
    TRAINING_SPLIT_CACHE_KEYS,
    apply_std_floor,
    load_validated_split_cache_state,
    normalization_floor_metadata,
    normalization_std_floors,
)


REQUIRED_RAW_KEYS = {"features", "contours", "frames", "phonemes", "length_datas"}
DEFAULT_NORMALIZATION_MODE = "train_global"


def session_cache_dir(config: Dict[str, Any]) -> Path:
    value = config.get("session_cache_dir") or config.get("dataset_cache_dir")
    if not value:
        raise KeyError("Config must define session_cache_dir or dataset_cache_dir")
    return Path(value).resolve()


def split_cache_dir(config: Dict[str, Any]) -> Path:
    value = config.get("split_cache_dir") or config.get("dataset_cache_dir")
    if not value:
        raise KeyError("Config must define split_cache_dir or dataset_cache_dir")
    return Path(value).resolve()


def split_cache_path(config: Dict[str, Any], split_key: str) -> Path:
    if split_key not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split_key}; expected one of {list(SPLIT_FILES)}")
    return split_cache_dir(config) / SPLIT_FILES[split_key]


def split_normalization_plan(
    config: Dict[str, Any],
    splits: Iterable[str],
) -> Tuple[Tuple[str, ...], str]:
    """Resolve split-cache normalization from the explicit normalization mode."""
    available_splits = tuple(splits)
    mode = str(config.get("normalization_mode", DEFAULT_NORMALIZATION_MODE)).lower()
    if mode in {"train_global", "train_only_global", "unseen_speaker"}:
        fit_split = str(config.get("normalization_fit_split", "train_sequences"))
        if fit_split not in available_splits:
            raise ValueError(
                f"normalization_fit_split must be one of {list(available_splits)}, got {fit_split}"
            )
        return (fit_split,), "train_global"
    if mode in {"all_splits_global", "all_global", "speaker_dependent"}:
        return available_splits, "all_splits_global"
    raise ValueError(
        f"Unsupported normalization_mode={config.get('normalization_mode')!r}; "
        "expected train_global or all_splits_global"
    )


def _iter_declared_sessions(config: Dict[str, Any], splits: Iterable[str]) -> Iterable[Tuple[str, str, str]]:
    for split_key in splits:
        for bucket, sessions in config.get(split_key, {}).items():
            for session in sessions:
                yield split_key, str(bucket), str(session)


def _validate_one_session(
    config: Dict[str, Any],
    cache_dir: Path,
    bucket: str,
    session: str,
) -> Tuple[bool, Dict[str, Any]]:
    dataset_type = dataset_type_for_sequence(config, bucket)
    raw_path = raw_session_part_path(cache_dir, config, bucket, session)
    npz_path = contour_pack_path(cache_dir, config, bucket, session)
    item = {
        "dataset_type": dataset_type,
        "bucket": bucket,
        "session": session,
        "raw_session_path": str(raw_path),
        "contour_npz_path": str(npz_path),
    }
    if not raw_path.exists():
        item["error"] = "missing_raw_session_pt"
        return False, item
    if not npz_path.exists():
        item["error"] = "missing_contour_npz"
        return False, item
    try:
        payload = torch.load(raw_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        item["error"] = f"raw_session_load_failed: {exc!r}"
        return False, item
    raw = payload.get("raw") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        item["error"] = "raw_payload_missing"
        return False, item
    missing_keys = sorted(REQUIRED_RAW_KEYS - set(raw))
    if missing_keys:
        item["error"] = f"raw_payload_missing_keys: {missing_keys}"
        return False, item
    item["num_chunks"] = int(payload.get("num_chunks", len(raw["features"])))
    return True, item


def validate_session_cache(
    config: Dict[str, Any],
    splits: Iterable[str] = SPLIT_FILES,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cache_dir = session_cache_dir(config)
    usable = copy.deepcopy(config)
    report: Dict[str, Any] = {
        "session_cache_dir": str(cache_dir),
        "declared_sessions": 0,
        "usable_sessions": 0,
        "failed_sessions": 0,
        "splits": {},
        "failed": [],
    }
    skip_bad = bool(config.get("skip_bad_session_cache", True))
    for split_key in splits:
        usable_split: Dict[str, List[str]] = {}
        split_report = {"declared": 0, "usable": 0, "failed": 0}
        for bucket, sessions in config.get(split_key, {}).items():
            kept = []
            for session in sessions:
                split_report["declared"] += 1
                report["declared_sessions"] += 1
                ok, item = _validate_one_session(config, cache_dir, str(bucket), str(session))
                item["split"] = split_key
                if ok:
                    kept.append(session)
                    split_report["usable"] += 1
                    report["usable_sessions"] += 1
                else:
                    split_report["failed"] += 1
                    report["failed_sessions"] += 1
                    report["failed"].append(item)
                    if not skip_bad:
                        raise RuntimeError(
                            "Bad session cache and skip_bad_session_cache=false: "
                            f"{split_key}/{bucket}/{session}: {item['error']}"
                        )
            if kept:
                usable_split[bucket] = kept
        usable[split_key] = usable_split
        report["splits"][split_key] = split_report
    return usable, report


def _load_raw_session(config: Dict[str, Any], cache_dir: Path, bucket: str, session: str) -> Dict[str, Any]:
    payload = torch.load(
        raw_session_part_path(cache_dir, config, bucket, session),
        map_location="cpu",
        weights_only=False,
    )
    return payload["raw"]


def load_raw_split(
    config: Dict[str, Any],
    split_key: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]:
    cache_dir = session_cache_dir(config)
    features: List[np.ndarray] = []
    contours: List[np.ndarray] = []
    frames: List[np.ndarray] = []
    phonemes: List[np.ndarray] = []
    length_datas: List[int] = []
    for bucket, sessions in config.get(split_key, {}).items():
        for session in sessions:
            raw = _load_raw_session(config, cache_dir, str(bucket), str(session))
            features.extend(raw["features"])
            contours.extend(raw["contours"])
            frames.extend(raw["frames"])
            phonemes.extend(raw["phonemes"])
            length_datas.extend([int(value) for value in raw["length_datas"]])
    return features, contours, frames, phonemes, length_datas


def fit_normalization(
    config: Dict[str, Any],
    fit_splits: Iterable[str],
    mode: str,
) -> Dict[str, Any]:
    contour_std_floor, mfcc_std_floor = normalization_std_floors(config)

    fit_features: List[np.ndarray] = []
    fit_contours: List[np.ndarray] = []
    for split_key in fit_splits:
        features, contours, _, _, _ = load_raw_split(config, split_key)
        fit_features.extend(features)
        fit_contours.extend(contours)
    if not fit_features or not fit_contours:
        raise ValueError(f"No raw data available to fit normalization for mode={mode}")

    contour_flat = [
        contour.reshape(contour.shape[0], contour.shape[1] * contour.shape[2])
        for contour in fit_contours
    ]
    std_contour = np.mean(np.array([np.std(item, axis=0) for item in contour_flat]), axis=0)
    mean_contour = np.mean(np.array([np.mean(item, axis=0) for item in contour_flat]), axis=0)
    std_mfcc = np.mean(np.array([np.std(item, axis=0) for item in fit_features]), axis=0)
    mean_mfcc = np.mean(np.array([np.mean(item, axis=0) for item in fit_features]), axis=0)

    return {
        "normalization_mode": mode,
        "normalization_fit_splits": list(fit_splits),
        **normalization_floor_metadata(config),
        "std_mfcc": apply_std_floor(std_mfcc, mfcc_std_floor, np.float32),
        "mean_mfcc": mean_mfcc.astype(np.float32),
        "std_contour": apply_std_floor(
            std_contour.reshape(len(config["classes"]), int(config["output_layer"])),
            contour_std_floor,
            np.float32,
        ),
        "mean_contour": mean_contour.reshape(
            len(config["classes"]),
            int(config["output_layer"]),
        ).astype(np.float32),
        "fit_num_sequences": len(fit_features),
        "fit_num_frames": int(sum(int(feature.shape[0]) for feature in fit_features)),
    }


def _pad_time_dim(tensor: torch.Tensor, target_length: int, dim: int = 1) -> torch.Tensor:
    current_length = int(tensor.shape[dim])
    if current_length == target_length:
        return tensor
    if current_length > target_length:
        index = [slice(None)] * tensor.ndim
        index[dim] = slice(0, target_length)
        return tensor[tuple(index)].contiguous()
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_length - current_length
    return torch.cat([tensor, torch.zeros(*pad_shape, dtype=tensor.dtype)], dim=dim)


def assemble_split_direct(
    config: Dict[str, Any],
    split_key: str,
    norm_stats: Dict[str, Any],
    rebuild: bool = False,
) -> Dict[str, Any]:
    output_path = split_cache_path(config, split_key)
    if output_path.exists() and not rebuild:
        state, floor_summary = load_validated_split_cache_state(
            output_path,
            config,
            required_keys=TRAINING_SPLIT_CACHE_KEYS,
        )
        return {
            "split": split_key,
            "path": str(output_path),
            "status": "existing",
            "num_samples": int(state["features"].shape[0]),
            "feature_shape": list(state["features"].shape),
            "label_shape": list(state["labels"].shape),
            **floor_summary,
        }
    features, contours, frames, phonemes, length_datas = load_raw_split(config, split_key)
    if not features:
        raise ValueError(f"No usable sessions remain for split {split_key}")

    target_length = int(config["sequence_length"])
    std_mfcc = norm_stats["std_mfcc"]
    mean_mfcc = norm_stats["mean_mfcc"]
    std_contour = norm_stats["std_contour"]
    mean_contour = norm_stats["mean_contour"]

    norm_features = [
        torch.tensor((feature - mean_mfcc) / std_mfcc, dtype=torch.float32)
        for feature in features
    ]
    norm_contours = [
        torch.tensor((contour - mean_contour[None, :, :]) / std_contour[None, :, :], dtype=torch.float32)
        for contour in contours
    ]
    frame_tensors = [torch.tensor(item, dtype=torch.float32) for item in frames]
    phoneme_tensors = [torch.tensor(item, dtype=torch.float32) for item in phonemes]
    lengths = [int(item.shape[0]) for item in features]

    padded_inputs = _pad_time_dim(pad_sequence(norm_features, batch_first=True), target_length)
    padded_labels = _pad_time_dim(pad_sequence(norm_contours, batch_first=True), target_length)
    padded_frames = _pad_time_dim(pad_sequence(frame_tensors, batch_first=True), target_length)
    padded_phonemes = _pad_time_dim(pad_sequence(phoneme_tensors, batch_first=True), target_length)
    sample_count = len(norm_features)
    labels = padded_labels.view(
        sample_count,
        target_length,
        len(config["classes"]),
        int(config["output_layer"]),
    )
    std_template = torch.tensor(std_contour[None, :, :], dtype=torch.float32)
    mean_template = torch.tensor(mean_contour[None, :, :], dtype=torch.float32)
    state = {
        "features": padded_inputs,
        "labels": labels,
        "frames": padded_frames,
        "phonemes": padded_phonemes,
        "std": torch.stack([std_template.clone() for _ in features], dim=0),
        "mean": torch.stack([mean_template.clone() for _ in features], dim=0),
        "mean_datas": recompute_mean_datas(labels, lengths),
        "length_datas": length_datas,
        "sequences_length": lengths,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(state, output_path)
    return {
        "split": split_key,
        "path": str(output_path),
        "status": "built",
        "num_samples": int(state["features"].shape[0]),
        "feature_shape": list(state["features"].shape),
        "label_shape": list(state["labels"].shape),
        "normalization_mode": norm_stats["normalization_mode"],
        "normalization_fit_splits": norm_stats["normalization_fit_splits"],
    }


def ensure_split_caches(config: Dict[str, Any]) -> Dict[str, Any]:
    splits = tuple(SPLIT_FILES)
    output_dir = split_cache_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    usable_config, validation = validate_session_cache(config, splits)
    write_metadata(output_dir / "session_cache_validation.json", validation)
    print(
        "session_cache_validation "
        f"declared={validation['declared_sessions']} "
        f"usable={validation['usable_sessions']} "
        f"failed={validation['failed_sessions']}",
        flush=True,
    )
    for split_key, split_report in validation["splits"].items():
        print(
            f"  {split_key}: declared={split_report['declared']} "
            f"usable={split_report['usable']} failed={split_report['failed']}",
            flush=True,
        )

    fit_splits, normalization_mode_name = split_normalization_plan(usable_config, splits)
    norm_stats = fit_normalization(usable_config, fit_splits, normalization_mode_name)

    rebuild = bool(usable_config.get("rebuild_split_cache", usable_config.get("rebuild_dataset_cache", False)))
    assemble_results = [
        assemble_split_direct(usable_config, split_key, norm_stats, rebuild)
        for split_key in splits
    ]

    metadata = {
        "session_cache_dir": str(session_cache_dir(usable_config)),
        "split_cache_dir": str(output_dir),
        "normalization_mode": norm_stats["normalization_mode"],
        "normalization_fit_splits": norm_stats["normalization_fit_splits"],
        "normalization_fit_split": (
            norm_stats["normalization_fit_splits"][0]
            if len(norm_stats["normalization_fit_splits"]) == 1
            else None
        ),
        "normalization_contour_std_floor": norm_stats["normalization_contour_std_floor"],
        "normalization_mfcc_std_floor": norm_stats["normalization_mfcc_std_floor"],
        "normalization_contour_std_floor_source": norm_stats["normalization_contour_std_floor_source"],
        "normalization_mfcc_std_floor_source": norm_stats["normalization_mfcc_std_floor_source"],
        "normalization_used_legacy_std_floor_key": norm_stats["normalization_used_legacy_std_floor_key"],
        "fit_num_sequences": norm_stats["fit_num_sequences"],
        "fit_num_frames": norm_stats["fit_num_frames"],
        "assemble_results": assemble_results,
        "validation_report": str(output_dir / "session_cache_validation.json"),
    }
    np.savez(
        output_dir / "normalization_stats.npz",
        std_mfcc=norm_stats["std_mfcc"],
        mean_mfcc=norm_stats["mean_mfcc"],
        std_contour=norm_stats["std_contour"],
        mean_contour=norm_stats["mean_contour"],
    )
    write_metadata(output_dir / "split_cache_metadata.json", metadata)
    return metadata
