from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils.config_validation import load_yaml_config

def load_config(path: Path, *, allow_legacy_audio_vtln: bool = False) -> dict[str, Any]:
    return load_yaml_config(path, allow_legacy_audio_vtln=allow_legacy_audio_vtln)


def load_phonemes(config: dict[str, Any]) -> list[str]:
    with Path(config["phonemesdir"]).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def decode_phoneme(vector: torch.Tensor | np.ndarray, phonemes: list[str]) -> str:
    if isinstance(vector, torch.Tensor):
        arr = vector.detach().cpu().numpy()
    else:
        arr = np.asarray(vector)
    if arr.size == 0 or np.allclose(arr, 0):
        return "UNK"
    return str(phonemes[int(arr.argmax())])


def frame_token(value: float) -> str:
    rounded = int(round(value))
    if abs(value - rounded) < 1e-4:
        return f"{rounded:04d}"
    return f"{int(math.floor(value)):04d}p{int(round((value - math.floor(value)) * 10)):01d}"


def aggregate_state(state: dict[str, Any], config: dict[str, Any], speaker: int, session: int) -> list[dict[str, Any]]:
    phonemes = load_phonemes(config)
    accum: dict[float, dict[str, Any]] = {}
    predicted = state["predicted_raw"].float()
    labels = state.get("labels_raw")
    labels = labels.float() if labels is not None else None
    frames = state["frames"]
    lengths = state["lengths"]
    phoneme_vectors = state["phonemes"]

    for seq_idx in range(predicted.shape[0]):
        length = int(lengths[seq_idx])
        for offset in range(length):
            frame = frames[seq_idx, offset]
            spk = int(round(float(frame[0])))
            ses = int(round(float(frame[1])))
            if spk != speaker or ses != session:
                continue
            frame_number = float(frame[2])
            item = accum.setdefault(
                frame_number,
                {
                    "frame_number": frame_number,
                    "predicted": [],
                    "phonemes": [],
                },
            )
            item["predicted"].append(predicted[seq_idx, offset].detach().cpu().numpy())
            if labels is not None:
                item.setdefault("labels", []).append(labels[seq_idx, offset].detach().cpu().numpy())
            item["phonemes"].append(decode_phoneme(phoneme_vectors[seq_idx, offset, 0], phonemes))

    rows = []
    for frame_number, item in sorted(accum.items()):
        rows.append(
            {
                "frame_number": frame_number,
                "frame": frame_token(frame_number),
                "predicted": np.mean(np.stack(item["predicted"], axis=0), axis=0).astype(np.float32),
                "labels": (
                    np.mean(np.stack(item["labels"], axis=0), axis=0).astype(np.float32)
                    if "labels" in item
                    else None
                ),
                "phoneme": Counter(item["phonemes"]).most_common(1)[0][0],
                "held": False,
            }
        )
    if not rows:
        raise RuntimeError(f"No rows found for P{speaker}/S{session} in {state.get('predictions', '<payload>')}")
    return rows


def prediction_denorm_summary() -> dict[str, Any]:
    return {
        "prediction_denorm_mode": "payload_raw",
        "description": (
            "Using cached predicted_raw from session_inference. The payload is "
            "already denormalized with the model training split normalization."
        ),
        "uses_target_session_label_stats": False,
    }


def build_timeline(rows: list[dict[str, Any]], step: float, max_frames: int | None) -> list[dict[str, Any]]:
    by_frame = {round(float(row["frame_number"]) * 2) / 2: row for row in rows}
    start = min(by_frame)
    end = max(by_frame)
    count = int(round((end - start) / step)) + 1
    timeline = []
    previous = rows[0]
    for index in range(count):
        frame_number = round(start + index * step, 4)
        row = by_frame.get(frame_number)
        if row is None:
            row = dict(previous)
            row["frame_number"] = frame_number
            row["frame"] = frame_token(frame_number)
            row["held"] = True
        else:
            previous = row
        timeline.append(row)
        if max_frames is not None and len(timeline) >= max_frames:
            break
    return timeline
