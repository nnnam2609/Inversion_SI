from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config_validation import load_yaml_config

SPLIT_FILES = {
    "train_sequences": "train_sequences.pt",
    "valid_sequences": "valid_sequences.pt",
    "test_sequences": "test_sequences.pt",
}

SUPPORT_FILES = (
    "normalization_stats.npz",
    "normalization_train_global.npz",
    "session_cache_validation.json",
)


def load_yaml(path: Path) -> dict[str, Any]:
    return load_yaml_config(path)


def dump_yaml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    tmp_path.replace(path)


def split_cache_dir(config: dict[str, Any]) -> Path:
    value = config.get("split_cache_dir") or config.get("dataset_cache_dir")
    if not value:
        raise KeyError("Config must define split_cache_dir or dataset_cache_dir")
    return Path(value)


def split_filename(split: str) -> str:
    try:
        return SPLIT_FILES[split]
    except KeyError as exc:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(SPLIT_FILES)}") from exc


def copy_file(source: Path, target: Path, force: bool) -> None:
    if not source.exists():
        return
    if target.exists() and not force:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_support_files(base_dir: Path, output_dir: Path, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SUPPORT_FILES:
        copy_file(base_dir / name, output_dir / name, force)


def copy_split(base_dir: Path, output_dir: Path, split: str, force: bool) -> None:
    source = base_dir / split_filename(split)
    target = output_dir / split_filename(split)
    if not source.exists():
        raise FileNotFoundError(f"Missing base split cache: {source}")
    copy_file(source, target, force)


def ensure_output_split_writable(output_dir: Path, split: str, force: bool) -> Path:
    output_split_path = output_dir / split_filename(split)
    if output_split_path.exists() and not force:
        raise FileExistsError(f"Output split cache exists; pass --force to overwrite: {output_split_path}")
    return output_split_path


def prepare_override_cache_dir(
    base_dir: Path,
    output_dir: Path,
    target_split: str,
    force: bool,
) -> Path:
    output_split_path = ensure_output_split_writable(output_dir, target_split, force)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_support_files(base_dir, output_dir, force)
    for split in SPLIT_FILES:
        if split != target_split:
            copy_split(base_dir, output_dir, split, force)
    return output_split_path


def load_mfcc_norm_stats(base_dir: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    for name in ("normalization_stats.npz", "normalization_train_global.npz"):
        path = base_dir / name
        if not path.exists():
            continue
        with np.load(path) as pack:
            return pack["mean_mfcc"].astype(np.float32), pack["std_mfcc"].astype(np.float32), path
    raise FileNotFoundError(f"No MFCC normalization stats found in {base_dir}")


def write_json_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def load_json_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def update_split_metadata(
    base_dir: Path,
    output_dir: Path,
    key: str,
    override_metadata: dict[str, Any],
) -> None:
    metadata = load_json_metadata(base_dir / "split_cache_metadata.json")
    metadata["split_cache_dir"] = str(output_dir)
    metadata[key] = override_metadata
    write_json_metadata(output_dir / "split_cache_metadata.json", metadata)


def feature_change_summary(old_values: np.ndarray, new_values: np.ndarray) -> dict[str, float]:
    old_np = np.asarray(old_values, dtype=np.float32)
    new_np = np.asarray(new_values, dtype=np.float32)
    return {
        "old_feature_mean": float(old_np.mean()),
        "old_feature_std": float(old_np.std()),
        "new_feature_mean": float(new_np.mean()),
        "new_feature_std": float(new_np.std()),
        "mean_abs_feature_delta": float(np.mean(np.abs(new_np - old_np))),
    }


def feature_motion_summary(old_values: np.ndarray, new_values: np.ndarray) -> dict[str, float]:
    old_np = np.asarray(old_values, dtype=np.float32)
    new_np = np.asarray(new_values, dtype=np.float32)
    return {
        "old_feature_diff_mean_abs": float(np.abs(np.diff(old_np, axis=0)).mean()),
        "old_feature_coord_std_mean": float(np.std(old_np, axis=0).mean()),
        "new_feature_diff_mean_abs": float(np.abs(np.diff(new_np, axis=0)).mean()),
        "new_feature_coord_std_mean": float(np.std(new_np, axis=0).mean()),
    }
