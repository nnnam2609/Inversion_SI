#!/usr/bin/env python3
"""Infer one cached session with an explicit de-normalization center."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--speaker", type=int, required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--center-key", choices=("mean", "sofiane_moving_average"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_indices(state: dict[str, Any], speaker: int, session: int) -> list[int]:
    selected = []
    for index, length in enumerate(state["sequences_length"]):
        valid = state["frames"][index, : int(length)]
        mask = (
            (torch.round(valid[:, 0]).to(torch.int64) == speaker)
            & (torch.round(valid[:, 1]).to(torch.int64) == session)
        )
        if bool(mask.any()):
            selected.append(index)
    return selected


def load_model(code_root: Path, config: dict[str, Any], checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(code_root))
    sys.path.insert(0, str(code_root / "src"))
    from src.model.baseline_5 import BaselineModel

    model = BaselineModel(
        int(config["input_layer"]),
        int(config["hidden_layer"]),
        int(config["num_layers"]),
        int(config["output_layer"]),
        len(config["classes"]),
        int(config["nbr_phonemes"]),
        config["phonemesdir"],
    )
    payload = torch.load(checkpoint, map_location="cpu")
    state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
    cleaned = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned, strict=True)
    return model.to(device).eval(), int(payload.get("epoch", -1))


def write_contours(
    output_dir: Path,
    classes: list[str],
    predicted_raw: np.ndarray,
    frames: torch.Tensor,
    lengths: np.ndarray,
    speaker: int,
    session: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    values: dict[tuple[int, str], list[np.ndarray]] = {}
    fractional = 0
    for sequence_index, length in enumerate(lengths):
        for offset in range(int(length)):
            identity = frames[sequence_index, offset].numpy()
            if int(round(identity[0])) != speaker or int(round(identity[1])) != session:
                continue
            frame_number = float(identity[2])
            if not np.isclose(frame_number, round(frame_number), atol=1e-4):
                fractional += 1
                continue
            frame = int(round(frame_number))
            for class_index, class_name in enumerate(classes):
                values.setdefault((frame, class_name), []).append(
                    predicted_raw[sequence_index, offset, class_index]
                )
    for (frame, class_name), predictions in values.items():
        contour = np.mean(np.stack(predictions), axis=0).astype(np.float32).reshape(50, 2)
        np.save(output_dir / f"{frame:04d}_{class_name}.npy", contour)
    manifest = {
        "num_unique_frames": len({frame for frame, _ in values}),
        "num_unique_frame_articulators": len(values),
        "classes": classes,
        "averaged_overlapping_predictions": True,
        "discarded_fractional_rows": fractional,
        "saved_fractional_frame_count": 0,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Inference requires an OAR GPU")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    state = torch.load(args.cache, map_location="cpu")
    if args.center_key not in state:
        raise KeyError(f"{args.cache} has no {args.center_key!r} tensor")
    indices = select_indices(state, args.speaker, args.session)
    if not indices:
        raise RuntimeError(
            f"No cache rows for {args.speaker}/S{args.session} in {args.cache}"
        )
    model, epoch = load_model(
        args.code_root.resolve(), config, args.checkpoint.resolve(), device
    )
    index = torch.as_tensor(indices, dtype=torch.long)
    features = state["features"].index_select(0, index).to(device)
    frames = state["frames"].index_select(0, index)
    lengths = np.asarray(
        [int(state["sequences_length"][item]) for item in indices], dtype=np.int64
    )
    lengths_device = torch.as_tensor(lengths, dtype=torch.long, device=device)
    std = state["std"].index_select(0, index).to(device)
    center = state[args.center_key].index_select(0, index).to(device)
    labels = state["labels"].index_select(0, index).to(device)
    with torch.no_grad():
        predicted, _, _ = model(features, lengths_device)
        labels = labels[:, : predicted.shape[1]]
        predicted_raw = predicted * std + center
        labels_raw = labels * std + center
    predicted_cpu = predicted_raw.detach().cpu().numpy()
    labels_cpu = labels_raw.detach().cpu().numpy()
    squared_sum = 0.0
    coordinate_count = 0
    integer_rows = 0
    for sequence_index, length in enumerate(lengths):
        identities = frames[sequence_index, : int(length), 2].numpy()
        integer = np.isclose(identities, np.rint(identities), atol=1e-6, rtol=0.0)
        error = predicted_cpu[sequence_index, : int(length)][integer] - labels_cpu[
            sequence_index, : int(length)
        ][integer]
        squared_sum += float(np.square(error.astype(np.float64)).sum())
        coordinate_count += int(error.size)
        integer_rows += int(integer.sum())

    output = args.output_dir.resolve()
    contours = write_contours(
        output / "predicted_contours",
        list(config["classes"]),
        predicted_cpu,
        frames,
        lengths,
        args.speaker,
        args.session,
    )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "indices": indices,
            "predicted_raw": torch.from_numpy(predicted_cpu),
            "labels_raw": torch.from_numpy(labels_cpu),
            "frames": frames,
            "lengths": torch.from_numpy(lengths),
            "center_key": args.center_key,
        },
        output / "cached_session_predictions.pt",
    )
    summary = {
        "status": "complete",
        "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "checkpoint_best_human_epoch": epoch + 1,
        "cache": str(args.cache.resolve()),
        "speaker": args.speaker,
        "session": args.session,
        "center_key": args.center_key,
        "uses_target_statistics_for_prediction_center": (
            args.center_key == "sofiane_moving_average"
        ),
        "integer_rows": integer_rows,
        "coordinate_rmse_mm": (squared_sum / coordinate_count) ** 0.5 * 1.62,
        "contours": contours,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
