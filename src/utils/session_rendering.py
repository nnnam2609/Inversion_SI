from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils.config_validation import load_yaml_config
from src.common.phonemes import decode_phoneme, load_phoneme_inventory as load_phonemes

def load_config(path: Path, *, allow_legacy_audio_vtln: bool = False) -> dict[str, Any]:
    return load_yaml_config(path, allow_legacy_audio_vtln=allow_legacy_audio_vtln)


def frame_token(value: float) -> str:
    rounded = int(round(value))
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"NEVER render fractional frame {value}")
    return f"{rounded:04d}"


def _load_ground_truth_contour(path: Path) -> np.ndarray:
    contour = np.load(path, allow_pickle=False)
    if contour.shape == (100,):
        contour = contour.reshape(50, 2)
    if contour.shape != (50, 2):
        raise ValueError(f"Ground-truth contour must have shape (50, 2), got {contour.shape}: {path}")
    if not np.isfinite(contour).all():
        raise ValueError(f"Ground-truth contour contains non-finite coordinates: {path}")
    return np.asarray(contour, dtype=np.float32)


def build_ground_truth_timeline(
    contour_dir: Path,
    classes: list[str],
    start_frame: float,
    end_frame: float,
    step: float,
    max_frames: int | None,
) -> list[dict[str, Any]]:
    """Build an integer-only ground-truth timeline from contour files."""
    contour_dir = contour_dir.resolve()
    if not contour_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth contour directory does not exist: {contour_dir}")
    if not classes:
        raise ValueError("Ground-truth timeline requires at least one contour class")
    if not np.isfinite([start_frame, end_frame, step]).all():
        raise ValueError("Ground-truth timeline bounds and step must be finite")
    if step <= 0:
        raise ValueError(f"Ground-truth timeline step must be positive, got {step}")
    if end_frame < start_frame:
        raise ValueError(f"Ground-truth end frame {end_frame} precedes start frame {start_frame}")
    if not math.isclose(step, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("NEVER render fractional frames; ground-truth timeline step must be 1.0")
    start_integer = int(round(start_frame))
    end_integer = int(round(end_frame))
    if not math.isclose(start_frame, start_integer, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"NEVER render fractional start frame {start_frame}")
    if not math.isclose(end_frame, end_integer, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"NEVER render fractional end frame {end_frame}")

    count = end_integer - start_integer + 1
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}")
        count = min(count, max_frames)
    frame_numbers = list(range(start_integer, start_integer + count))
    needed_integer_frames = set(frame_numbers)

    integer_cache: dict[int, list[np.ndarray | None]] = {}
    integer_errors: dict[tuple[int, str], str] = {}
    for integer_frame in sorted(needed_integer_frames):
        frame_contours: list[np.ndarray | None] = []
        for articulator in classes:
            path = contour_dir / f"{integer_frame:04d}_{articulator}.npy"
            if not path.is_file():
                frame_contours.append(None)
                integer_errors[(integer_frame, articulator)] = "file missing"
                continue
            try:
                frame_contours.append(_load_ground_truth_contour(path))
            except (OSError, ValueError) as error:
                frame_contours.append(None)
                integer_errors[(integer_frame, articulator)] = str(error)
        integer_cache[integer_frame] = frame_contours

    timeline: list[dict[str, Any]] = []
    for frame_number in frame_numbers:
        source = f"direct frame {frame_number:04d}"
        contours: list[np.ndarray] = []
        missing_contours: list[str] = []
        missing_details: dict[str, str] = {}
        for class_index, articulator in enumerate(classes):
            contour = integer_cache[frame_number][class_index]
            if contour is None:
                contours.append(np.full((50, 2), np.nan, dtype=np.float32))
                missing_contours.append(articulator)
                missing_details[articulator] = (
                    f"{frame_number:04d}: {integer_errors[(frame_number, articulator)]}"
                )
            else:
                contours.append(contour)
        flattened = np.asarray(contours, dtype=np.float32).reshape(len(classes), 100)
        timeline.append(
            {
                "frame_number": frame_number,
                "frame": frame_token(frame_number),
                "labels": flattened,
                "mode_prediction": flattened,
                "phoneme": "n/a",
                "held": False,
                "ground_truth_source": source,
                "ground_truth_lower_frame": frame_number,
                "ground_truth_upper_frame": frame_number,
                "ground_truth_interpolation_alpha": 0.0,
                "missing_ground_truth_contours": missing_contours,
                "missing_ground_truth_details": missing_details,
            }
        )
    return timeline


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
            rounded_frame = int(round(frame_number))
            if not math.isclose(frame_number, rounded_frame, rel_tol=0.0, abs_tol=1e-4):
                continue
            frame_number = float(rounded_frame)
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
    if not math.isclose(step, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("NEVER render fractional frames; timeline step must be 1.0")
    integer_rows = [
        row
        for row in rows
        if math.isclose(
            float(row["frame_number"]),
            round(float(row["frame_number"])),
            rel_tol=0.0,
            abs_tol=1e-4,
        )
    ]
    if not integer_rows:
        raise RuntimeError("No integer-numbered frames available for rendering")
    by_frame = {int(round(float(row["frame_number"]))): row for row in integer_rows}
    start = min(by_frame)
    end = max(by_frame)
    count = end - start + 1
    timeline = []
    previous = by_frame[start]
    for index in range(count):
        frame_number = start + index
        row = by_frame.get(frame_number)
        if row is None:
            row = dict(previous)
            row["frame_number"] = float(frame_number)
            row["frame"] = frame_token(float(frame_number))
            row["held"] = True
        else:
            previous = row
        timeline.append(row)
        if max_frames is not None and len(timeline) >= max_frames:
            break
    return timeline
