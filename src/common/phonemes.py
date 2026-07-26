"""Phoneme-vector decoding shared by inference and rendering."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_phoneme_inventory(
    source: Mapping[str, Any] | str | Path,
) -> list[str]:
    """Load the ordered phoneme inventory from a config or JSON path."""

    path = Path(source["phonemesdir"]) if isinstance(source, Mapping) else Path(source)
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError(f"Expected a JSON list of phoneme strings: {path}")
    return values


def decode_phoneme(
    vector: torch.Tensor | np.ndarray, phonemes: Sequence[str]
) -> str:
    if isinstance(vector, torch.Tensor):
        values = vector.detach().cpu().numpy().reshape(-1)
    else:
        values = np.asarray(vector).reshape(-1)
    if values.size == 0 or np.allclose(values, 0):
        return "UNK"
    index = int(np.argmax(values))
    return str(phonemes[index]) if 0 <= index < len(phonemes) else f"PH{index}"
