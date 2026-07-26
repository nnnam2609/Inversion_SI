"""Dataset identity helpers shared by preprocessing and split assembly."""

from __future__ import annotations

from typing import Any, Mapping


def dataset_type_for_sequence(
    config: Mapping[str, Any], sequence: str
) -> str:
    sequence_key = str(sequence)
    dataset_types = config.get("dataset_types", {})
    if sequence_key in dataset_types:
        return str(dataset_types[sequence_key]).lower()
    dataset_type = str(config.get("dataset_type", "asd2")).lower()
    if dataset_type == "mixed":
        return "asd1" if sequence_key.upper().startswith("P") else "asd2"
    return dataset_type
