"""Centralized loading of internal compatibility and external dependencies."""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GRID_TRANSFORM_ROOT = REPOSITORY_ROOT / "external/grid-transform"
AUDIO_NORMALIZATION_ROOT = (
    REPOSITORY_ROOT
    / "external/audio-speaker-normalization/audio-speaker-normalization"
)


def _prepend(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


@lru_cache(maxsize=1)
def load_inversion_audio_core() -> ModuleType:
    return importlib.import_module(
        "src.adaption_pipeline.legacy.run_asd2_epoch211_selected_experiment"
    )


@lru_cache(maxsize=1)
def load_anatomy_core() -> ModuleType:
    return importlib.import_module(
        "src.adaption_pipeline.legacy."
        "run_asd2_fixedbs10_textgrid_u_grid_adaptation"
    )


def activate_audio_normalization_project() -> Path:
    _prepend(AUDIO_NORMALIZATION_ROOT)
    return AUDIO_NORMALIZATION_ROOT


def activate_grid_transform_project() -> Path:
    _prepend(GRID_TRANSFORM_ROOT)
    return GRID_TRANSFORM_ROOT
