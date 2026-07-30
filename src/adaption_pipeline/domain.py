"""Canonical strategy and condition vocabulary for the whole pipeline."""

from __future__ import annotations

from typing import Dict, Tuple


GLOBAL = "global"
MOVING_AVERAGE = "moving_average"
STRATEGIES: Tuple[str, ...] = (GLOBAL, MOVING_AVERAGE)

ORIGINAL = "original"
ANATOMICAL = "anatomical"
AUDIO = "audio"
ANATOMICAL_AUDIO = "anatomical_audio"
CONDITIONS: Tuple[str, ...] = (
    ORIGINAL,
    ANATOMICAL,
    AUDIO,
    ANATOMICAL_AUDIO,
)

PREDICTION_ARRAY_KEYS: Dict[str, str] = {
    ORIGINAL: "predicted_original",
    ANATOMICAL: "predicted_anatomical",
    AUDIO: "predicted_audio",
    ANATOMICAL_AUDIO: "predicted_anatomical_audio",
}

CONDITION_TITLES: Dict[str, str] = {
    ORIGINAL: "Original audio / no anatomy",
    ANATOMICAL: "Anatomical: affine + TPS",
    AUDIO: "Audio: RMS + VTLN",
    ANATOMICAL_AUDIO: "Anatomical + audio",
}

STRATEGY_TITLES: Dict[str, str] = {
    GLOBAL: "Global",
    MOVING_AVERAGE: "Moving average",
}


def anatomical_result_is_primary(strategy: str) -> bool:
    """Whether an anatomical transform is valid in the strategy output space."""

    return strategy == GLOBAL


def strategy_title(strategy: str) -> str:
    try:
        return STRATEGY_TITLES[strategy]
    except KeyError as error:
        raise ValueError(f"Unknown strategy {strategy!r}") from error
