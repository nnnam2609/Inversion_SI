from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_CONTOUR_STD_FLOOR = 0.1
DEFAULT_MFCC_STD_FLOOR = 1e-8
MIN_CONTOUR_STD_FLOOR = DEFAULT_CONTOUR_STD_FLOOR
NORMALIZATION_STD_POLICY_KEY = "normalization_std_policy"
DEFAULT_NORMALIZATION_STD_POLICY = "floor"
RAW_POSITIVE_NORMALIZATION_STD_POLICY = "raw_positive"
CONTOUR_STD_FLOOR_KEY = "normalization_contour_std_floor"
MFCC_STD_FLOOR_KEY = "normalization_mfcc_std_floor"
LEGACY_CONTOUR_STD_FLOOR_KEY = "contour_std_floor"
LEGACY_MFCC_STD_FLOOR_KEY = "mfcc_std_floor"
LOW_CONTOUR_STD_FLOOR_DIAGNOSTIC_KEY = "allow_low_contour_std_floor_diagnostic"
SPLIT_CACHE_METADATA_NAMES = (
    "split_cache_metadata.json",
    "metadata_blind_single_task5.json",
    "metadata_asd2_full_single_task5.json",
    "metadata.json",
)
DENORM_SPLIT_CACHE_KEYS = ("std", "mean")
INFERENCE_SPLIT_CACHE_KEYS = (
    "features",
    "labels",
    "frames",
    "phonemes",
    "std",
    "mean",
    "sequences_length",
)
TRAINING_SPLIT_CACHE_KEYS = (
    *INFERENCE_SPLIT_CACHE_KEYS,
    "mean_datas",
    "length_datas",
)
FEATURE_OVERRIDE_SPLIT_CACHE_KEYS = (
    "features",
    "frames",
    "sequences_length",
    "std",
)


def _configured_floor(config: dict[str, Any], key: str, legacy_key: str, default: float) -> tuple[float, str]:
    if key in config:
        return float(config[key]), key
    if legacy_key in config:
        return float(config[legacy_key]), legacy_key
    return float(default), "default"


def normalization_std_policy(config: dict[str, Any]) -> str:
    """Return the std scaling policy while preserving the historical default."""
    policy = str(config.get(NORMALIZATION_STD_POLICY_KEY, DEFAULT_NORMALIZATION_STD_POLICY)).lower()
    if policy not in {DEFAULT_NORMALIZATION_STD_POLICY, RAW_POSITIVE_NORMALIZATION_STD_POLICY}:
        raise ValueError(
            f"Unsupported {NORMALIZATION_STD_POLICY_KEY}={policy!r}; expected "
            f"{DEFAULT_NORMALIZATION_STD_POLICY!r} or {RAW_POSITIVE_NORMALIZATION_STD_POLICY!r}"
        )
    return policy


def validate_raw_positive_std(
    values: Any,
    value_name: str,
    *,
    classes: list[str] | tuple[str, ...] | None = None,
    output_layer: int | None = None,
) -> np.ndarray:
    """Validate an unfloored fitted std and report every bad feature/coordinate."""
    array = np.asarray(values)
    invalid = ~np.isfinite(array) | (array <= 0)
    if not np.any(invalid):
        return array

    details = []
    for index in np.argwhere(invalid)[:50]:
        index_tuple = tuple(int(value) for value in index.tolist())
        value = array[index_tuple]
        if value_name == "contour" and classes is not None and output_layer is not None:
            flat_index = int(np.ravel_multi_index(index_tuple, array.shape))
            class_index, coordinate_index = divmod(flat_index, int(output_layer))
            class_name = classes[class_index] if class_index < len(classes) else f"class_{class_index}"
            details.append(
                {
                    "class_index": class_index,
                    "class_name": class_name,
                    "coordinate_index": coordinate_index,
                    "value": float(value),
                }
            )
        else:
            details.append({"feature_index": index_tuple, "value": float(value)})
    raise ValueError(
        f"Raw fitted {value_name} std contains {int(invalid.sum())} non-finite or non-positive "
        f"values; first_bad={details}. No std floor was applied."
    )


def normalization_std_floors(config: dict[str, Any]) -> tuple[float, float]:
    """Return the project std floors, rejecting contour floors that can freeze motion."""
    policy = normalization_std_policy(config)
    if policy == RAW_POSITIVE_NORMALIZATION_STD_POLICY:
        raise ValueError(
            f"{NORMALIZATION_STD_POLICY_KEY}={policy!r} has no std floors; "
            "validate and use the raw fitted std instead"
        )
    contour_std_floor, _ = _configured_floor(
        config,
        CONTOUR_STD_FLOOR_KEY,
        LEGACY_CONTOUR_STD_FLOOR_KEY,
        DEFAULT_CONTOUR_STD_FLOOR,
    )
    mfcc_std_floor, _ = _configured_floor(
        config,
        MFCC_STD_FLOOR_KEY,
        LEGACY_MFCC_STD_FLOOR_KEY,
        DEFAULT_MFCC_STD_FLOOR,
    )
    if contour_std_floor <= 0:
        raise ValueError(f"{CONTOUR_STD_FLOOR_KEY} must be > 0, got {contour_std_floor}")
    if (
        contour_std_floor < MIN_CONTOUR_STD_FLOOR
        and not bool(config.get(LOW_CONTOUR_STD_FLOOR_DIAGNOSTIC_KEY, False))
    ):
        raise ValueError(
            f"{CONTOUR_STD_FLOOR_KEY} must be >= {MIN_CONTOUR_STD_FLOOR} for normal runs, "
            f"got {contour_std_floor}. Lower contour floors can make de-normalized "
            f"predictions look under-moving; set {LOW_CONTOUR_STD_FLOOR_DIAGNOSTIC_KEY}: true "
            "only for a diagnostic run."
        )
    if mfcc_std_floor <= 0:
        raise ValueError(f"{MFCC_STD_FLOOR_KEY} must be > 0, got {mfcc_std_floor}")
    return contour_std_floor, mfcc_std_floor


def normalization_floor_metadata(config: dict[str, Any]) -> dict[str, Any]:
    policy = normalization_std_policy(config)
    if policy == RAW_POSITIVE_NORMALIZATION_STD_POLICY:
        return {
            NORMALIZATION_STD_POLICY_KEY: policy,
            CONTOUR_STD_FLOOR_KEY: None,
            MFCC_STD_FLOOR_KEY: None,
            "normalization_contour_std_floor_source": "disabled_raw_positive",
            "normalization_mfcc_std_floor_source": "disabled_raw_positive",
            "normalization_used_legacy_std_floor_key": False,
            "normalization_low_contour_std_floor_diagnostic": False,
        }
    contour_std_floor, contour_source = _configured_floor(
        config,
        CONTOUR_STD_FLOOR_KEY,
        LEGACY_CONTOUR_STD_FLOOR_KEY,
        DEFAULT_CONTOUR_STD_FLOOR,
    )
    mfcc_std_floor, mfcc_source = _configured_floor(
        config,
        MFCC_STD_FLOOR_KEY,
        LEGACY_MFCC_STD_FLOOR_KEY,
        DEFAULT_MFCC_STD_FLOOR,
    )
    normalization_std_floors(config)
    return {
        NORMALIZATION_STD_POLICY_KEY: policy,
        CONTOUR_STD_FLOOR_KEY: contour_std_floor,
        MFCC_STD_FLOOR_KEY: mfcc_std_floor,
        "normalization_contour_std_floor_source": contour_source,
        "normalization_mfcc_std_floor_source": mfcc_source,
        "normalization_used_legacy_std_floor_key": bool(
            contour_source == LEGACY_CONTOUR_STD_FLOOR_KEY
            or mfcc_source == LEGACY_MFCC_STD_FLOOR_KEY
        ),
        "normalization_low_contour_std_floor_diagnostic": bool(
            config.get(LOW_CONTOUR_STD_FLOOR_DIAGNOSTIC_KEY, False)
        ),
    }


def apply_std_floor(values: Any, floor: float, dtype: Any | None = None) -> np.ndarray:
    floored = np.maximum(np.asarray(values), float(floor))
    if dtype is not None:
        return floored.astype(dtype, copy=False)
    return floored


def summarize_contour_std_floor(
    state: dict[str, Any],
    config: dict[str, Any],
    cache_path: str,
    cache_metadata: dict[str, Any] | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    if "std" not in state:
        raise KeyError(f"Split cache must contain contour std tensor: {cache_path}")
    std = state["std"]
    if not isinstance(std, torch.Tensor):
        std = torch.as_tensor(std)
    min_std = float(std.float().min().item())
    metadata = cache_metadata or {}
    policy = normalization_std_policy(config)
    if policy == RAW_POSITIVE_NORMALIZATION_STD_POLICY:
        finite = bool(torch.isfinite(std).all().item())
        positive = bool((std > 0).all().item())
        return {
            NORMALIZATION_STD_POLICY_KEY: policy,
            CONTOUR_STD_FLOOR_KEY: None,
            "cache_normalization_contour_std_floor": metadata.get(CONTOUR_STD_FLOOR_KEY),
            "cache_contour_std_min": min_std,
            "cache_contour_std_floor_ok": bool(finite and positive),
            "cache_contour_std_raw_positive_ok": bool(finite and positive),
            "cache_contour_std_all_finite": finite,
            "normalization_contour_std_floor_source": "disabled_raw_positive",
        }
    contour_std_floor, _ = normalization_std_floors(config)
    metadata_floor = metadata.get(CONTOUR_STD_FLOOR_KEY)
    floor_ok = min_std + tolerance >= contour_std_floor
    return {
        NORMALIZATION_STD_POLICY_KEY: policy,
        CONTOUR_STD_FLOOR_KEY: contour_std_floor,
        "cache_normalization_contour_std_floor": metadata_floor,
        "cache_contour_std_min": min_std,
        "cache_contour_std_floor_ok": bool(floor_ok),
        "normalization_contour_std_floor_source": normalization_floor_metadata(config)[
            "normalization_contour_std_floor_source"
        ],
    }


def validate_contour_std_floor(
    state: dict[str, Any],
    config: dict[str, Any],
    cache_path: str,
    cache_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = summarize_contour_std_floor(state, config, cache_path, cache_metadata)
    if not summary["cache_contour_std_floor_ok"]:
        if summary.get(NORMALIZATION_STD_POLICY_KEY) == RAW_POSITIVE_NORMALIZATION_STD_POLICY:
            raise RuntimeError(
                "Split cache contour std must be finite and strictly positive under "
                f"{NORMALIZATION_STD_POLICY_KEY}={RAW_POSITIVE_NORMALIZATION_STD_POLICY}. "
                f"cache={cache_path} min_std={summary['cache_contour_std_min']:.8g}. "
                "No std floor was applied."
            )
        raise RuntimeError(
            "Split cache contour std is below the configured normalization floor. "
            "This can make predicted contours look under-moving after denormalization. "
            f"cache={cache_path} min_std={summary['cache_contour_std_min']:.8g} "
            f"required_floor={summary['normalization_contour_std_floor']:.8g}. "
            "Rebuild the split cache with the current normalization_contour_std_floor "
            f"policy, or set {LOW_CONTOUR_STD_FLOOR_DIAGNOSTIC_KEY}: true only "
            "for a diagnostic run."
        )
    return summary


def load_cache_metadata(cache_path: Path) -> dict[str, Any]:
    for metadata_name in SPLIT_CACHE_METADATA_NAMES:
        metadata_path = cache_path.parent / metadata_name
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if isinstance(metadata, dict):
                metadata["metadata_path"] = str(metadata_path)
                return metadata
    return {}


def load_validated_split_cache_state(
    cache_path: Path | str,
    config: dict[str, Any],
    *,
    required_keys: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing split cache: {path}")
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Split cache must load to a dict: {path}")
    missing_keys = [key for key in required_keys if key not in state]
    if missing_keys:
        raise KeyError(f"Split cache is missing required keys {missing_keys}: {path}")
    cache_metadata = load_cache_metadata(path)
    floor_summary = validate_contour_std_floor(state, config, str(path), cache_metadata)
    if cache_metadata.get("metadata_path"):
        floor_summary["cache_metadata"] = cache_metadata["metadata_path"]
    return state, floor_summary


def describe_split_denorm(cache_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    cache_metadata = load_cache_metadata(cache_path)
    normalization_mode = cache_metadata.get("normalization_mode", config.get("normalization_mode"))
    normalization_fit_split = cache_metadata.get("normalization_fit_split")
    normalization_fit_splits = cache_metadata.get("normalization_fit_splits")
    train_split_stats = (
        normalization_fit_split == "train_sequences"
        or normalization_fit_splits == ["train_sequences"]
    )
    if train_split_stats:
        denorm_method = "train_global_std_mean_from_cache"
        uses_target_std_mean = False
        std_mean_source_split = "train_sequences"
    else:
        denorm_method = "split_cache_std_mean"
        uses_target_std_mean = True
        std_mean_source_split = cache_path.stem
    return {
        "denorm_cache": str(cache_path),
        "denorm_method": denorm_method,
        "normalization_mode": normalization_mode,
        "normalization_fit_split": normalization_fit_split,
        "normalization_fit_splits": normalization_fit_splits,
        "std_mean_source_split": std_mean_source_split,
        "uses_target_std_mean": uses_target_std_mean,
        "cache_metadata": cache_metadata.get("metadata_path"),
    }
