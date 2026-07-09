from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .audio_vtln import LEGACY_AUDIO_VTLN_CONFIG_KEY, reject_legacy_audio_vtln_config
from .normalization import normalization_floor_metadata


def validate_runtime_config(
    config: dict[str, Any],
    config_path: Path | str,
    *,
    allow_legacy_audio_vtln: bool = False,
) -> dict[str, Any]:
    """Validate config rules that prevent stale normalization/audio paths."""
    path = Path(config_path)
    legacy_audio_vtln_value = config.get(LEGACY_AUDIO_VTLN_CONFIG_KEY)
    if not allow_legacy_audio_vtln:
        reject_legacy_audio_vtln_config(config, path)
    floor_metadata = normalization_floor_metadata(config)
    return {
        "config_path": str(path),
        "allow_legacy_audio_vtln": bool(allow_legacy_audio_vtln),
        "legacy_audio_vtln": legacy_audio_vtln_value is not None,
        "legacy_audio_vtln_feature_npz": legacy_audio_vtln_value,
        **floor_metadata,
    }


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config did not parse to a mapping: {path}")
    return config


def load_yaml_config(
    path: Path,
    *,
    allow_legacy_audio_vtln: bool = False,
) -> dict[str, Any]:
    config = load_yaml_mapping(path)
    validate_runtime_config(
        config,
        path,
        allow_legacy_audio_vtln=allow_legacy_audio_vtln,
    )
    return config
