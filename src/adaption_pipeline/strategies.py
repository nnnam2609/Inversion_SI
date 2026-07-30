"""Inference strategies with a common session-level output contract."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .contracts import ContractError
from src.common.phonemes import decode_phoneme


EPSILON = 1e-8
MOVING_AVERAGE_PAD = 30


@dataclass
class InferredSession:
    frame_numbers: np.ndarray
    predicted_raw: np.ndarray
    ground_truth: np.ndarray
    phonemes: np.ndarray
    overlap_counts: np.ndarray
    num_input_rows: int
    num_sequences: int
    metadata: Dict[str, Any]


def aggregate_overlaps(
    *,
    predicted_by_chunk: List[np.ndarray],
    contours: List[np.ndarray],
    frames: List[np.ndarray],
    phoneme_vectors: List[np.ndarray],
    phonemes: List[str],
    metadata: Dict[str, Any],
) -> InferredSession:
    accumulator: Dict[float, Dict[str, List[Any]]] = {}
    for prediction, labels, identities, phone_rows in zip(
        predicted_by_chunk, contours, frames, phoneme_vectors
    ):
        length = min(len(prediction), len(labels), len(identities), len(phone_rows))
        for offset in range(length):
            frame_number = float(identities[offset, 2])
            item = accumulator.setdefault(
                frame_number, {"pred": [], "gt": [], "phoneme": []}
            )
            item["pred"].append(prediction[offset])
            item["gt"].append(labels[offset])
            item["phoneme"].append(decode_phoneme(phone_rows[offset], phonemes))
    if not accumulator:
        raise ContractError("Inference produced no frame rows")
    frame_numbers = np.asarray(sorted(accumulator), dtype=np.float64)
    predicted = []
    ground_truth = []
    labels = []
    overlap_counts = []
    for frame in frame_numbers:
        item = accumulator[float(frame)]
        predicted.append(np.mean(np.stack(item["pred"]), axis=0))
        ground_truth.append(np.mean(np.stack(item["gt"]), axis=0))
        labels.append(Counter(item["phoneme"]).most_common(1)[0][0])
        overlap_counts.append(len(item["pred"]))
    integer = np.isclose(frame_numbers, np.rint(frame_numbers), atol=1e-4, rtol=0.0)
    if not integer.any():
        raise ContractError("No integer MRI frames remain after inference")
    return InferredSession(
        frame_numbers=np.rint(frame_numbers[integer]).astype(np.int32),
        predicted_raw=np.asarray(predicted, dtype=np.float32)[integer].reshape(
            -1, 11, 50, 2
        ),
        ground_truth=np.asarray(ground_truth, dtype=np.float32)[integer].reshape(
            -1, 11, 50, 2
        ),
        phonemes=np.asarray(labels, dtype="U32")[integer],
        overlap_counts=np.asarray(overlap_counts, dtype=np.int16)[integer],
        num_input_rows=int(sum(len(item) for item in frames)),
        num_sequences=len(frames),
        metadata={
            **metadata,
            "discarded_fractional_frames": int((~integer).sum()),
            "saved_fractional_frame_count": 0,
        },
    )


def sofiane_moving_average(
    chunk_means: np.ndarray, pad: int = MOVING_AVERAGE_PAD
) -> np.ndarray:
    values = np.asarray(chunk_means)
    if values.ndim != 2 or not len(values):
        raise ContractError(f"Expected non-empty [N,D] chunk means, got {values.shape}")
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="symmetric")
    result = np.asarray(
        [
            np.mean(padded[index - pad : index + pad], axis=0)
            for index in range(pad, len(padded) - pad)
        ]
    )
    if result.shape != values.shape:
        raise ContractError(
            f"Moving-average shape changed: {values.shape} -> {result.shape}"
        )
    return result


def infer_moving_average(
    *,
    model: torch.nn.Module,
    device: torch.device,
    raw_cache: Path,
    phonemes: List[str],
    batch_size: int,
    feature_override: Optional[List[np.ndarray]] = None,
) -> InferredSession:
    """Moving-average inference using target-session contour statistics."""

    payload = torch.load(raw_cache, map_location="cpu", weights_only=False)["raw"]
    original_features = [np.asarray(item, dtype=np.float32) for item in payload["features"]]
    contours = [np.asarray(item, dtype=np.float32) for item in payload["contours"]]
    frames = [np.asarray(item, dtype=np.float32) for item in payload["frames"]]
    phoneme_vectors = [
        np.asarray(item, dtype=np.float32) for item in payload["phonemes"]
    ]
    features = (
        [np.asarray(item, dtype=np.float32) for item in feature_override]
        if feature_override is not None
        else original_features
    )
    counts = {len(features), len(contours), len(frames), len(phoneme_vectors)}
    if len(counts) != 1:
        raise ContractError(
            "Moving-average raw cache/feature override has unpaired chunk lists"
        )
    for index, (feature, contour) in enumerate(zip(features, contours)):
        if len(feature) != len(contour):
            raise ContractError(
                f"Moving-average chunk {index} feature/contour length mismatch"
            )

    flattened = [item.reshape(item.shape[0], -1) for item in contours]
    chunk_means = np.asarray([np.mean(item, axis=0) for item in flattened])
    moving = sofiane_moving_average(chunk_means).reshape(len(contours), 11, 100)
    contour_std_raw = np.mean(
        np.asarray([np.std(item, axis=0) for item in flattened]), axis=0
    ).reshape(11, 100)
    contour_std = np.maximum(contour_std_raw, EPSILON).astype(np.float32)
    mfcc_mean = np.mean(
        np.asarray([np.mean(item, axis=0) for item in features]), axis=0
    )
    mfcc_std_raw = np.mean(
        np.asarray([np.std(item, axis=0) for item in features]), axis=0
    )
    mfcc_std = np.maximum(mfcc_std_raw, EPSILON)
    if not all(
        np.isfinite(item).all()
        for item in (moving, contour_std, mfcc_mean, mfcc_std)
    ):
        raise ContractError("Moving-average statistics contain NaN or infinity")

    predicted_chunks: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            batch = [
                torch.as_tensor(
                    (item - mfcc_mean) / mfcc_std, dtype=torch.float32
                )
                for item in features[start:stop]
            ]
            lengths = torch.as_tensor([len(item) for item in batch], dtype=torch.long)
            padded = pad_sequence(batch, batch_first=True).to(device)
            predicted_normalized, _, _ = model(padded, lengths)
            predicted = predicted_normalized.detach().cpu().numpy()
            for local_index, source_index in enumerate(range(start, stop)):
                length = int(lengths[local_index])
                restored = (
                    predicted[local_index, :length]
                    * contour_std[None, :, :]
                    + moving[source_index][None, :, :]
                )
                predicted_chunks.append(restored.astype(np.float32))
    return aggregate_overlaps(
        predicted_by_chunk=predicted_chunks,
        contours=contours,
        frames=frames,
        phoneme_vectors=phoneme_vectors,
        phonemes=phonemes,
        metadata={
            "strategy": "moving_average",
            "uses_target_statistics": True,
            "uses_target_labels": True,
            "causal": False,
            "blind_inference_compatible": False,
            "moving_average_pad": MOVING_AVERAGE_PAD,
            "moving_average_window": MOVING_AVERAGE_PAD * 2,
            "target_contour_std_used": True,
            "target_contour_moving_center_used": True,
            "feature_override": feature_override is not None,
            "contour_epsilon_clamp_count": int((contour_std_raw < EPSILON).sum()),
            "mfcc_epsilon_clamp_count": int((mfcc_std_raw < EPSILON).sum()),
        },
    )


def from_legacy_inferred(
    inferred: Dict[str, Any], *, strategy: str, metadata: Dict[str, Any]
) -> InferredSession:
    return InferredSession(
        frame_numbers=np.asarray(inferred["frame_numbers"], dtype=np.int32),
        predicted_raw=np.asarray(inferred["predicted_raw"], dtype=np.float32),
        ground_truth=np.asarray(inferred["ground_truth"], dtype=np.float32),
        phonemes=np.asarray(inferred["phonemes"], dtype="U32"),
        overlap_counts=np.asarray(inferred["overlap_counts"], dtype=np.int16),
        num_input_rows=int(inferred["num_input_rows"]),
        num_sequences=int(inferred["num_sequences"]),
        metadata={"strategy": strategy, **metadata},
    )
