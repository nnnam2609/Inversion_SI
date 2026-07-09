#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.utils.split_cache_overrides import (  # noqa: E402
    dump_yaml,
    feature_change_summary,
    load_mfcc_norm_stats,
    load_yaml,
    prepare_override_cache_dir,
    split_cache_dir,
    split_filename,
    update_split_metadata,
    write_json_metadata,
)
from src.utils.audio_vtln import (  # noqa: E402
    LEGACY_AUDIO_VTLN_CONFIG_KEY,
    LEGACY_AUDIO_VTLN_METHOD,
    LEGACY_AUDIO_VTLN_WARNING,
    INVERSION_FRONTEND_VTLN_SCRIPT,
    legacy_audio_vtln_refusal,
)
from src.utils.normalization import (  # noqa: E402
    FEATURE_OVERRIDE_SPLIT_CACHE_KEYS,
    load_validated_split_cache_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone an existing split cache and replace one target session's MFCC "
            "features with VTLN-normalized audio features, normalized by the "
            "original train-speaker MFCC stats. This is a legacy NPZ-feature "
            "override path for diagnostics; use "
            f"{INVERSION_FRONTEND_VTLN_SCRIPT} for inversion RMSE/video "
            "runs so the MFCC frontend and sentence chunking match training."
        )
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--audio-feature-npz", type=Path, required=True)
    parser.add_argument("--output-cache-dir", type=Path, required=True)
    parser.add_argument("--speaker", type=int, default=2)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--speaker-name", default="P2")
    parser.add_argument("--session-name", default="S1")
    parser.add_argument("--split", default="test_sequences", choices=("train_sequences", "valid_sequences", "test_sequences"))
    parser.add_argument("--feature-key", default="X")
    parser.add_argument("--max-nearest-seconds", type=float, default=0.03)
    parser.add_argument(
        "--allow-legacy-npz-diagnostic",
        action="store_true",
        help=(
            "Actually run this legacy NPZ override path. Do not use it for final "
            f"inversion RMSE/video; use {INVERSION_FRONTEND_VTLN_SCRIPT}."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output cache files if they already exist.")
    return parser.parse_args()


def load_audio_features(
    npz_path: Path,
    feature_key: str,
    speaker_name: str,
    session_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing audio feature npz: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as pack:
        if feature_key not in pack:
            raise KeyError(f"Feature key {feature_key!r} not found in {npz_path}; keys={pack.files}")
        speaker_id = pack["speaker_id"].astype(str)
        session_id = pack["session_id"].astype(str)
        mask = (speaker_id == speaker_name) & (session_id == session_name)
        if not np.any(mask):
            raise RuntimeError(f"No rows found for {speaker_name}/{session_name} in {npz_path}")
        times = pack["frame_time"][mask].astype(np.float64)
        features = pack[feature_key][mask].astype(np.float32)
        phones = pack["phone_label"][mask].astype(str) if "phone_label" in pack.files else np.asarray([""] * len(times))
    order = np.argsort(times)
    times = times[order]
    features = features[order]
    phones = phones[order]
    metadata = {
        "audio_feature_npz": str(npz_path),
        "feature_key": feature_key,
        "speaker_name": speaker_name,
        "session_name": session_name,
        "num_audio_feature_rows": int(features.shape[0]),
        "audio_time_min": float(times.min()),
        "audio_time_max": float(times.max()),
        "audio_feature_shape": list(features.shape),
        "num_phone_labels": int(len(set(phones.tolist()))),
    }
    return times, features, metadata


def nearest_indices(sorted_times: np.ndarray, target_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(sorted_times, target_times)
    positions = np.clip(positions, 0, len(sorted_times) - 1)
    prev_positions = np.clip(positions - 1, 0, len(sorted_times) - 1)
    use_prev = np.abs(sorted_times[prev_positions] - target_times) <= np.abs(sorted_times[positions] - target_times)
    indices = np.where(use_prev, prev_positions, positions)
    deltas = np.abs(sorted_times[indices] - target_times)
    return indices.astype(np.int64), deltas.astype(np.float64)


def write_alignment_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sequence_index",
                "frame_offset",
                "frame_number",
                "target_time_seconds",
                "nearest_audio_index",
                "nearest_audio_time_seconds",
                "abs_delta_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.allow_legacy_npz_diagnostic:
        raise SystemExit(legacy_audio_vtln_refusal())
    base_config = load_yaml(args.base_config)
    base_dir = split_cache_dir(base_config)
    output_dir = args.output_cache_dir
    output_split_path = prepare_override_cache_dir(base_dir, output_dir, args.split, args.force)

    mean_mfcc, std_mfcc, norm_stats_path = load_mfcc_norm_stats(base_dir)
    audio_times, audio_features, audio_metadata = load_audio_features(
        args.audio_feature_npz,
        args.feature_key,
        args.speaker_name,
        args.session_name,
    )
    if audio_features.shape[1] != mean_mfcc.shape[0]:
        raise ValueError(
            f"Feature dimension mismatch: audio {audio_features.shape[1]} vs mean_mfcc {mean_mfcc.shape[0]}"
        )

    state, _floor_summary = load_validated_split_cache_state(
        base_dir / split_filename(args.split),
        base_config,
        required_keys=FEATURE_OVERRIDE_SPLIT_CACHE_KEYS,
    )
    features = state["features"].clone().float()
    frames = state["frames"]
    lengths = state["sequences_length"]
    ms_image = float(base_config.get("ms_image", 19.98))
    added_frames = float(base_config.get("added_frames", 0))

    target_times: list[float] = []
    target_positions: list[tuple[int, int]] = []
    for sequence_index, length in enumerate(lengths):
        for frame_offset in range(int(length)):
            frame = frames[sequence_index, frame_offset]
            speaker = int(round(float(frame[0])))
            session = int(round(float(frame[1])))
            if speaker != args.speaker or session != args.session:
                continue
            frame_number = float(frame[2])
            target_times.append(((frame_number + added_frames) * ms_image) / 1000.0)
            target_positions.append((sequence_index, frame_offset))

    if not target_positions:
        raise RuntimeError(
            f"No frame rows found for speaker={args.speaker} session={args.session} in {base_dir / split_filename(args.split)}"
        )

    target_times_np = np.asarray(target_times, dtype=np.float64)
    audio_indices, deltas = nearest_indices(audio_times, target_times_np)
    if float(deltas.max()) > args.max_nearest_seconds:
        raise RuntimeError(
            f"Nearest audio alignment exceeds threshold: max={float(deltas.max()):.6f}s "
            f"threshold={args.max_nearest_seconds:.6f}s"
        )

    old_rows = []
    new_rows = []
    alignment_rows = []
    for row_idx, ((sequence_index, frame_offset), audio_index) in enumerate(zip(target_positions, audio_indices)):
        old_feature = features[sequence_index, frame_offset].detach().cpu().numpy().copy()
        vtln_feature = audio_features[audio_index]
        normalized = (vtln_feature - mean_mfcc) / std_mfcc
        features[sequence_index, frame_offset] = torch.from_numpy(normalized.astype(np.float32))
        old_rows.append(old_feature)
        new_rows.append(normalized)
        frame_number = float(frames[sequence_index, frame_offset, 2])
        alignment_rows.append(
            {
                "sequence_index": sequence_index,
                "frame_offset": frame_offset,
                "frame_number": frame_number,
                "target_time_seconds": float(target_times_np[row_idx]),
                "nearest_audio_index": int(audio_index),
                "nearest_audio_time_seconds": float(audio_times[audio_index]),
                "abs_delta_seconds": float(deltas[row_idx]),
            }
        )

    state["features"] = features
    torch.save(state, output_split_path)

    old_np = np.asarray(old_rows, dtype=np.float32)
    new_np = np.asarray(new_rows, dtype=np.float32)
    override_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": LEGACY_AUDIO_VTLN_METHOD,
        "deprecated_for_inversion_video": True,
        "frontend_warning": LEGACY_AUDIO_VTLN_WARNING,
        "base_config": str(args.base_config),
        "output_config": str(args.output_config),
        "base_split_cache_dir": str(base_dir),
        "output_split_cache_dir": str(output_dir),
        "split": args.split,
        "speaker": args.speaker,
        "session": args.session,
        "speaker_name": args.speaker_name,
        "session_name": args.session_name,
        "normalization_stats": str(norm_stats_path),
        "mfcc_normalization": "audio_feature_minus_base_train_mean_mfcc_div_base_train_std_mfcc",
        "frame_time_formula": "((frame_number + added_frames) * ms_image) / 1000",
        "ms_image": ms_image,
        "added_frames": added_frames,
        "max_nearest_seconds": args.max_nearest_seconds,
        "num_replaced_frame_rows": int(len(target_positions)),
        "num_unique_replaced_frames": int(len({float(row["frame_number"]) for row in alignment_rows})),
        "nearest_delta_seconds_mean": float(deltas.mean()),
        "nearest_delta_seconds_max": float(deltas.max()),
        **feature_change_summary(old_np, new_np),
        **audio_metadata,
    }
    write_alignment_csv(output_dir / "audio_vtln_alignment.csv", alignment_rows)
    write_json_metadata(output_dir / "audio_vtln_cache_metadata.json", override_metadata)
    update_split_metadata(base_dir, output_dir, "audio_vtln_override", override_metadata)

    output_config = dict(base_config)
    output_config["experiment_name"] = "experiment_asd1_p7_stdfloor01_trainstats_p2_s1_audio_vtln_eval_st5_mfcc"
    output_config["folder_save"] = "p7_stdfloor01_to_p2_s1_audio_vtln_eval"
    output_config["model"] = "single_task5_asd1_p7_stdfloor01_trainstats_p2_s1_audio_vtln_eval_st5_mfcc"
    output_config["tag"] = "p7_stdfloor01_to_p2_s1_audio_vtln_eval"
    output_config["split_cache_dir"] = str(output_dir)
    output_config["dataset_cache_dir"] = str(output_dir)
    output_config[LEGACY_AUDIO_VTLN_CONFIG_KEY] = str(args.audio_feature_npz)
    output_config["audio_vtln_cache_metadata"] = str(output_dir / "audio_vtln_cache_metadata.json")
    dump_yaml(args.output_config, output_config)

    print(json.dumps(override_metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
