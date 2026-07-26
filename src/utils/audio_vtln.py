from __future__ import annotations

from pathlib import Path
from typing import Any


LEGACY_AUDIO_VTLN_CONFIG_KEY = "audio_vtln_feature_npz"
INVERSION_FRONTEND_VTLN_CONFIG_KEY = "inversion_frontend_vtln_cache_metadata"
LEGACY_AUDIO_VTLN_METHOD = "legacy_audio_vtln_npz_diagnostic"
INVERSION_FRONTEND_VTLN_METHOD = "inversion_frontend_vtln"
INVERSION_FRONTEND_VTLN_SCRIPT = (
    "scripts/inversion_si.py preprocess vtln-cache"
)

LEGACY_AUDIO_VTLN_WARNING = (
    "Legacy NPZ-feature override: exported audio-normalization NPZ features may "
    "not match the Inversion_SI MFCC frontend/chunking and can make predicted "
    f"contours under-move. Use {INVERSION_FRONTEND_VTLN_SCRIPT} for final "
    "inversion RMSE/video runs."
)


def legacy_audio_vtln_refusal() -> str:
    return (
        "Refusing to build legacy audio VTLN NPZ cache by default because it can "
        f"produce under-moving contours. Use {INVERSION_FRONTEND_VTLN_SCRIPT} "
        "for inversion RMSE/video, or pass --allow-legacy-npz-diagnostic for "
        "diagnostics only."
    )


def reject_legacy_audio_vtln_config(config: dict[str, Any], config_path: Path) -> None:
    if LEGACY_AUDIO_VTLN_CONFIG_KEY not in config:
        return
    raise RuntimeError(
        f"This config uses legacy {LEGACY_AUDIO_VTLN_CONFIG_KEY}, which is the "
        "old NPZ path that can make predicted contours under-move. Rebuild/render "
        f"with {INVERSION_FRONTEND_VTLN_SCRIPT} and a config that contains "
        f"{INVERSION_FRONTEND_VTLN_CONFIG_KEY} instead. Rejected config: {config_path}"
    )
