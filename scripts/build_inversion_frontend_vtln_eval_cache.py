#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from scipy.fftpack import dct

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(
    0,
    str(
        REPO_ROOT
        / "external/audio-speaker-normalization/audio-speaker-normalization/notebooks"
    ),
)

from audio_norm_utils import vtln_warp_mel_filterbank  # noqa: E402
from preprocessing.session_cache import RawContourSession, load_textgrid_with_repair  # noqa: E402
from src.utils.audio_vtln import (  # noqa: E402
    INVERSION_FRONTEND_VTLN_CONFIG_KEY,
    INVERSION_FRONTEND_VTLN_METHOD,
)
from src.utils.normalization import (  # noqa: E402
    FEATURE_OVERRIDE_SPLIT_CACHE_KEYS,
    load_validated_split_cache_state,
)
from src.utils.split_cache_overrides import (  # noqa: E402
    dump_yaml,
    feature_change_summary,
    feature_motion_summary,
    load_mfcc_norm_stats,
    load_yaml,
    prepare_override_cache_dir,
    split_cache_dir,
    split_filename,
    update_split_metadata,
    write_json_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone a split cache and replace one session's MFCC features with "
            "VTLN features extracted through the same frontend/chunking path as "
            "Inversion_SI preprocessing."
        )
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-cache-dir", type=Path, required=True)
    parser.add_argument("--speaker", type=int, default=2)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--speaker-name", default="P2")
    parser.add_argument("--session-name", default="S1")
    parser.add_argument("--split", default="test_sequences", choices=("train_sequences", "valid_sequences", "test_sequences"))
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--vtln-f-low", type=float, default=60.0)
    parser.add_argument("--vtln-f-high", type=float, default=3200.0)
    parser.add_argument("--rms-target", type=float, default=None, help="Optional waveform RMS normalization before VTLN.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def rms_normalize(wav: np.ndarray, target: float | None) -> np.ndarray:
    if target is None:
        return wav.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(np.square(wav), dtype=np.float64)))
    if rms <= 1e-12:
        return wav.astype(np.float32, copy=True)
    return (wav * (float(target) / rms)).astype(np.float32)


def compute_vtln_mfcc_like_inversion(
    wav: np.ndarray,
    sample_rate: int,
    config: dict[str, Any],
    alpha: float,
    f_low: float,
    f_high: float,
) -> tuple[np.ndarray, int, int]:
    window_length_samples = int(sample_rate * float(config["window_length_ms"]) / 1000.0)
    hop_length_samples = int(sample_rate * float(config["hop_length_ratio"]) / 1000.0)
    n_fft = 2 ** int(np.ceil(np.log2(window_length_samples)))
    power = np.abs(
        librosa.stft(
            y=wav,
            n_fft=n_fft,
            hop_length=hop_length_samples,
            win_length=window_length_samples,
            window="hamming",
            center=True,
        )
    ) ** 2
    mel_basis = vtln_warp_mel_filterbank(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=128,
        alpha=float(alpha),
        f_low=float(f_low),
        f_high=float(f_high),
    )
    log_mel_db = librosa.power_to_db(np.maximum(mel_basis @ power, 1e-12))
    mfcc = dct(log_mel_db, axis=0, type=2, norm="ortho")[: int(config["n_mfcc"])].T

    # Preserve the exact historical frontend: the trained cache computed deltas
    # on arrays shaped [time, n_mfcc], so librosa differentiates the last axis.
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(delta_mfcc, order=2)
    mfcc_final = np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=1)

    context_window = int(config.get("context_window", 0))
    padding = np.zeros((context_window, mfcc_final.shape[1]), dtype=mfcc_final.dtype)
    frames = np.concatenate([padding, mfcc_final, padding])
    mfcc_context = np.array(
        [frames[i : i + 2 * context_window + 1].flatten() for i in range(len(mfcc_final))],
        dtype=np.float32,
    )
    return mfcc_context, window_length_samples, hop_length_samples


def build_helper(config: dict[str, Any], split: str, speaker_name: str, session_name: str) -> RawContourSession:
    sub_config = copy.deepcopy(config)
    sub_config[split] = {speaker_name: [session_name]}
    sub_config["cache_dataset"] = False
    return RawContourSession(sub_config, split, rank=0)


def session_chunks_from_features(
    helper: RawContourSession,
    config: dict[str, Any],
    features: np.ndarray,
    sample_rate: int,
    audio_signal_length: int,
    window_length_samples: int,
    hop_length_samples: int,
) -> tuple[list[np.ndarray], list[np.ndarray], str, str]:
    audio_path = next(iter(helper.audio_files.values()))[0]
    textgrid_path = next(iter(helper.textgrid_files.values()))[0]
    duration_seconds = float(audio_signal_length / sample_rate)
    tg = load_textgrid_with_repair(textgrid_path, duration_seconds, config)
    silence = helper.detect_silence(features, tg, sample_rate)
    chunks, frame_lists, _, _ = helper.detect_sentences(
        features,
        tg,
        helper.all_phonemes,
        sample_rate,
        window_length_samples,
        hop_length_samples,
        silence,
    )
    return chunks, frame_lists, str(audio_path), str(textgrid_path)


def select_session_indices(state: dict[str, Any], speaker: int, session: int) -> list[int]:
    indices = []
    frames = state["frames"]
    lengths = state["sequences_length"]
    for idx, length in enumerate(lengths):
        valid = frames[idx, : int(length)]
        if valid.numel() == 0:
            continue
        speaker_match = torch.round(valid[:, 0]).to(torch.int64) == int(speaker)
        session_match = torch.round(valid[:, 1]).to(torch.int64) == int(session)
        if bool(torch.any(speaker_match & session_match)):
            indices.append(idx)
    return indices


def write_chunk_check(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base_config = load_yaml(args.base_config)
    base_dir = split_cache_dir(base_config)
    output_dir = args.output_cache_dir
    output_split_path = prepare_override_cache_dir(base_dir, output_dir, args.split, args.force)

    mean_mfcc, std_mfcc, stats_path = load_mfcc_norm_stats(base_dir)

    helper = build_helper(base_config, args.split, args.speaker_name, args.session_name)
    audio_path = next(iter(helper.audio_files.values()))[0]
    wav, sample_rate = librosa.load(audio_path, sr=None)
    wav_for_vtln = rms_normalize(wav, args.rms_target)
    vtln_features, window_length_samples, hop_length_samples = compute_vtln_mfcc_like_inversion(
        wav_for_vtln,
        sample_rate,
        base_config,
        args.alpha,
        args.vtln_f_low,
        args.vtln_f_high,
    )
    vtln_chunks, _vtln_frame_lists, audio_path, textgrid_path = session_chunks_from_features(
        helper,
        base_config,
        vtln_features,
        sample_rate,
        len(wav),
        window_length_samples,
        hop_length_samples,
    )

    raw_session_path = Path(base_config.get("session_cache_dir", REPO_ROOT / "cache")) / "raw_sessions/asd1" / args.speaker_name / f"{args.session_name}.pt"
    raw_payload = torch.load(raw_session_path, map_location="cpu")["raw"]
    raw_chunks = raw_payload["features"]
    if len(vtln_chunks) != len(raw_chunks):
        raise RuntimeError(f"VTLN chunk count {len(vtln_chunks)} != raw cache chunk count {len(raw_chunks)}")

    state, _floor_summary = load_validated_split_cache_state(
        base_dir / split_filename(args.split),
        base_config,
        required_keys=FEATURE_OVERRIDE_SPLIT_CACHE_KEYS,
    )
    indices = select_session_indices(state, args.speaker, args.session)
    if len(indices) != len(vtln_chunks):
        raise RuntimeError(f"Selected split sequences {len(indices)} != VTLN chunks {len(vtln_chunks)}")

    chunk_rows = []
    old_values = []
    new_values = []
    new_features = state["features"].clone().float()
    for order, (state_idx, vtln_chunk, raw_chunk) in enumerate(zip(indices, vtln_chunks, raw_chunks)):
        if vtln_chunk.shape != raw_chunk.shape:
            raise RuntimeError(f"Chunk {order} shape mismatch: vtln={vtln_chunk.shape} raw={raw_chunk.shape}")
        length = int(state["sequences_length"][state_idx])
        if length != int(vtln_chunk.shape[0]):
            raise RuntimeError(f"Chunk {order} length mismatch: split={length} vtln={vtln_chunk.shape[0]}")
        normalized = ((vtln_chunk.astype(np.float32) - mean_mfcc) / std_mfcc).astype(np.float32)
        old = new_features[state_idx, :length].detach().cpu().numpy().copy()
        new_features[state_idx].zero_()
        new_features[state_idx, :length] = torch.from_numpy(normalized)
        old_values.append(old)
        new_values.append(normalized)
        chunk_rows.append(
            {
                "order": order,
                "state_index": state_idx,
                "length": length,
                "raw_chunk_mean_abs": float(np.mean(np.abs(raw_chunk))),
                "vtln_chunk_mean_abs": float(np.mean(np.abs(vtln_chunk))),
                "normalized_old_mean_abs": float(np.mean(np.abs(old))),
                "normalized_new_mean_abs": float(np.mean(np.abs(normalized))),
                "normalized_delta_mean_abs": float(np.mean(np.abs(normalized - old))),
            }
        )
    state["features"] = new_features
    torch.save(state, output_split_path)

    old_np = np.concatenate(old_values, axis=0)
    new_np = np.concatenate(new_values, axis=0)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": INVERSION_FRONTEND_VTLN_METHOD,
        "base_config": str(args.base_config),
        "output_config": str(args.output_config),
        "base_split_cache_dir": str(base_dir),
        "output_split_cache_dir": str(output_dir),
        "split": args.split,
        "speaker": args.speaker,
        "session": args.session,
        "speaker_name": args.speaker_name,
        "session_name": args.session_name,
        "audio_path": audio_path,
        "textgrid_path": textgrid_path,
        "normalization_stats": str(stats_path),
        "alpha": float(args.alpha),
        "vtln_f_low": float(args.vtln_f_low),
        "vtln_f_high": float(args.vtln_f_high),
        "rms_target": None if args.rms_target is None else float(args.rms_target),
        "sample_rate": int(sample_rate),
        "window_length_samples": int(window_length_samples),
        "hop_length_samples": int(hop_length_samples),
        "num_replaced_sequences": int(len(indices)),
        "num_replaced_frames": int(sum(chunk.shape[0] for chunk in vtln_chunks)),
        **feature_change_summary(old_np, new_np),
        **feature_motion_summary(old_np, new_np),
    }
    write_chunk_check(output_dir / "inversion_frontend_vtln_chunk_check.csv", chunk_rows)
    write_json_metadata(output_dir / "inversion_frontend_vtln_cache_metadata.json", metadata)
    update_split_metadata(base_dir, output_dir, "inversion_frontend_vtln_override", metadata)

    output_config = dict(base_config)
    suffix = "inversion_vtln_rms" if args.rms_target is not None else "inversion_vtln"
    output_config["experiment_name"] = f"experiment_asd1_p7_stdfloor01_trainstats_p2_s1_{suffix}_eval_st5_mfcc"
    output_config["folder_save"] = f"p7_stdfloor01_to_p2_s1_{suffix}_eval"
    output_config["model"] = f"single_task5_asd1_p7_stdfloor01_trainstats_p2_s1_{suffix}_eval_st5_mfcc"
    output_config["tag"] = f"p7_stdfloor01_to_p2_s1_{suffix}_eval"
    output_config["split_cache_dir"] = str(output_dir)
    output_config["dataset_cache_dir"] = str(output_dir)
    output_config[INVERSION_FRONTEND_VTLN_CONFIG_KEY] = str(output_dir / "inversion_frontend_vtln_cache_metadata.json")
    output_config["inversion_frontend_vtln_alpha"] = float(args.alpha)
    output_config["inversion_frontend_vtln_rms_target"] = None if args.rms_target is None else float(args.rms_target)
    dump_yaml(args.output_config, output_config)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
