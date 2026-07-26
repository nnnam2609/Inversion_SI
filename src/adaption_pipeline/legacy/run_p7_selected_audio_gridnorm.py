#!/usr/bin/env python3
"""Evaluate RMS+VTLN audio normalization together with P7 grid transfer.

The baseline contours are read from the completed grid-normalization run.  Only
the audio-normalized branch is inferred again.  Sentence/chunk selection is
matched against the raw per-session cache so missing-annotation filtering and
silence handling remain exactly the same as in the baseline experiment.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
EXTERNAL_AUDIO_ROOT = (
    REPO_ROOT / "external/audio-speaker-normalization/audio-speaker-normalization"
)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(EXTERNAL_AUDIO_ROOT))

from audio_speaker_norm.audio_normalization import AudioNormConfig, FeatureExtractor  # noqa: E402
from notebooks.audio_norm_utils import (  # noqa: E402
    apply_cmvn,
    extract_vtln_mfcc39,
    fit_cmvn,
    fit_speaker_gmms,
    sample_rows,
    score_gmm,
)
from src.inference.vtln_cache import (  # noqa: E402
    build_helper,
    compute_vtln_mfcc_like_inversion,
    rms_normalize,
)
from src.preprocessing.session_cache import load_textgrid_with_repair  # noqa: E402
from .run_p7_all_nonp7_gridnorm import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_NORM_STATS,
    DEFAULT_RAW_CACHE,
    DEFAULT_VTLN_DIR,
    EXCLUDED_CLASSES,
    INTEGER_FRAME_POLICY,
    RAW_ROOT,
    SEPARATOR,
    SOURCE,
    STAGES,
    draw_panel,
    frame_token,
    load_normalization,
    metric_payload,
    mri_for_timestamp,
    prepare_frame,
    retain_integer_arrays,
    retain_integer_inferred,
    target_spec_for_speaker,
    transform_contour_batch,
)
from grid_transform.transfer import build_two_step_transform  # noqa: E402
from src.inference.session_inference import load_model  # noqa: E402
from src.utils.config_validation import load_yaml_config  # noqa: E402
from src.utils.mri_rendering import build_filename_dicom_index, load_or_build_mri_cache  # noqa: E402
from src.common.phonemes import decode_phoneme  # noqa: E402
from src.common.artifacts import atomic_write_csv  # noqa: E402

write_alignment = atomic_write_csv
write_csv = atomic_write_csv


SELECTION = ((1, 16), (2, 9), (3, 14), (4, 4), (5, 6), (6, 8), (8, 2), (9, 5), (10, 14))
P7_TRAIN_SESSIONS = tuple(range(1, 13))
DEFAULT_BASELINE_ROOT = REPO_ROOT / "results/p7_selected_nonp7_sessions_gridnorm_20260718"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/p7_selected_nonp7_sessions_audio_gridnorm_20260718"
TARGET_RMS = 0.03
VTLN_F_LOW = 60.0
VTLN_F_HIGH = 3200.0
MATCH_MAE_LIMIT = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--normalization-stats", type=Path, default=DEFAULT_NORM_STATS)
    parser.add_argument("--raw-cache-root", type=Path, default=DEFAULT_RAW_CACHE)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vtln-dir", type=Path, default=DEFAULT_VTLN_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--selection",
        nargs="+",
        default=None,
        metavar="P#:S#",
        help="Run only these pairs; the final report still expects the complete fixed selection.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--transform-frame-batch", type=int, default=256)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--mri-workers", type=int, default=4)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--keep-mri-cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args()


def selected_pairs(args: argparse.Namespace) -> tuple[tuple[int, int], ...]:
    if not args.selection:
        return SELECTION
    parsed = []
    allowed = set(SELECTION)
    for token in args.selection:
        try:
            speaker_token, session_token = token.upper().split(":", maxsplit=1)
            pair = (int(speaker_token.removeprefix("P")), int(session_token.removeprefix("S")))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid selection token {token!r}; expected P#:S#") from error
        if pair not in allowed:
            raise ValueError(f"Pair P{pair[0]}:S{pair[1]} is outside the fixed experiment selection")
        parsed.append(pair)
    return tuple(parsed)


def validate_args(args: argparse.Namespace) -> None:
    required = (
        args.config,
        args.checkpoint,
        args.normalization_stats,
        args.raw_cache_root,
        args.baseline_root,
        args.vtln_dir,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    for speaker, session in SELECTION:
        raw = args.raw_cache_root / f"P{speaker}/S{session}.pt"
        baseline = args.baseline_root / f"P{speaker}/S{session}/contours_and_ground_truth.npz"
        if not raw.is_file():
            raise FileNotFoundError(raw)
        if not baseline.is_file():
            raise FileNotFoundError(baseline)


def exact_audio_paths(speaker: int, session: int) -> tuple[Path, Path]:
    folder = RAW_ROOT / f"P{speaker}/OTHER/S{session}"
    wav = folder / f"DENOISED_SOUND_P{speaker}_S{session}.wav"
    textgrid = folder / f"TEXT_ALIGNMENT_P{speaker}_S{session}.textgrid"
    if not wav.is_file() or not textgrid.is_file():
        raise FileNotFoundError(f"Missing WAV/TextGrid for P{speaker}/S{session}: {wav}, {textgrid}")
    return wav, textgrid


def audio_index() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for session in P7_TRAIN_SESSIONS:
        wav, textgrid = exact_audio_paths(7, session)
        rows.append(
            {
                "speaker_id": "P7",
                "session_id": f"S{session}",
                "wav_path": str(wav),
                "textgrid_path": str(textgrid),
            }
        )
    for speaker, session in SELECTION:
        wav, textgrid = exact_audio_paths(speaker, session)
        rows.append(
            {
                "speaker_id": f"P{speaker}",
                "session_id": f"S{session}",
                "wav_path": str(wav),
                "textgrid_path": str(textgrid),
            }
        )
    return pd.DataFrame(rows)


def estimate_alphas(args: argparse.Namespace) -> dict[str, float]:
    output_dir = args.output_root / "audio_normalization"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "alpha_to_p7.json"
    if summary_path.is_file() and not args.force:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        alphas = {str(key): float(value) for key, value in payload["alpha_to_p7"].items()}
        expected = {f"P{speaker}" for speaker, _ in SELECTION}
        if expected <= set(alphas):
            print(f"REUSE audio-normalization alphas: {summary_path}", flush=True)
            return alphas

    config = AudioNormConfig(
        data_root=RAW_ROOT,
        output_root=output_dir,
        mode="full",
        target_rms=TARGET_RMS,
        gmm_components=64,
        gmm_max_iter=200,
        max_frames_per_speaker=120_000,
        alpha_min=0.80,
        alpha_max=1.20,
        alpha_step=0.025,
        vtln_f_low=VTLN_F_LOW,
        vtln_f_high=VTLN_F_HIGH,
        target_mode="P7",
        save_intermediate_features=False,
        save_models=False,
        save_figures=False,
    )
    index = audio_index()
    index.to_csv(output_dir / "data_index.csv", index=False)
    print(f"AUDIO NORM: extracting external MFCC39 from {len(index)} WAV files", flush=True)
    extractor = FeatureExtractor(config)
    payloads = extractor.extract(index)
    by_speaker = extractor.stack_by_speaker(payloads, "rms_mfcc39")
    speakers = sorted(by_speaker, key=lambda value: int(value[1:]))
    all_features = np.vstack([by_speaker[speaker] for speaker in speakers]).astype(np.float32)
    global_mean, global_std = fit_cmvn(all_features)
    global_cmvn = {
        speaker: apply_cmvn(by_speaker[speaker], global_mean, global_std)
        for speaker in speakers
    }
    gmms, gmm_table = fit_speaker_gmms(global_cmvn, config.to_helper_config(), model_dir=None)
    gmm_table.to_csv(output_dir / "speaker_gmm_summary.csv", index=False)
    target_gmm = gmms["P7"]
    self_score = float(score_gmm(target_gmm, global_cmvn["P7"]))

    curve_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    alphas: dict[str, float] = {}
    for speaker, _session in SELECTION:
        speaker_name = f"P{speaker}"
        source_payloads = [item for item in payloads if item["speaker"] == speaker_name]
        best_alpha = 1.0
        best_score = -np.inf
        base_score = float("nan")
        for alpha in config.alpha_grid:
            chunks = []
            for item in source_payloads:
                features, _ = extract_vtln_mfcc39(
                    np.asarray(item["wav_rms"]),
                    int(item["sr"]),
                    alpha=float(alpha),
                    config=config.to_helper_config(),
                    f_high=VTLN_F_HIGH,
                )
                n = min(len(features), int(item["n_total_frames"]))
                mask = np.asarray(item["speech_mask"][:n], dtype=bool)
                if mask.any():
                    chunks.append(features[:n][mask])
            combined = np.vstack(chunks).astype(np.float32)
            combined, _ = sample_rows(combined, min(20_000, len(combined)), config.random_state)
            normalized = apply_cmvn(combined, global_mean, global_std)
            score = float(score_gmm(target_gmm, normalized))
            curve_rows.append(
                {
                    "source_speaker": speaker_name,
                    "target_speaker": "P7",
                    "alpha": float(alpha),
                    "score": score,
                    "n_frames": int(len(combined)),
                }
            )
            if abs(float(alpha) - 1.0) < 1e-9:
                base_score = score
            if score > best_score:
                best_score = score
                best_alpha = float(alpha)
        alphas[speaker_name] = best_alpha
        alpha_rows.append(
            {
                "source_speaker": speaker_name,
                "target_speaker": "P7",
                "alpha_best": best_alpha,
                "score_alpha_1": base_score,
                "score_best": best_score,
                "score_gain": best_score - base_score,
                "p7_self_score": self_score,
            }
        )
        print(
            f"AUDIO NORM {speaker_name}->P7: alpha={best_alpha:.3f}, "
            f"GMM gain={best_score - base_score:+.4f}",
            flush=True,
        )

    pd.DataFrame(curve_rows).to_csv(output_dir / "alpha_curves_to_p7.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(output_dir / "alpha_summary_to_p7.csv", index=False)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "external_audio_speaker_norm RMS(0.03)+GMM VTLN; inversion-compatible re-extraction",
        "target_speaker": "P7",
        "target_reference_sessions": [f"S{session}" for session in P7_TRAIN_SESSIONS],
        "target_rms": TARGET_RMS,
        "alpha_grid": [float(value) for value in config.alpha_grid],
        "vtln_f_low": VTLN_F_LOW,
        "vtln_f_high": VTLN_F_HIGH,
        "external_code": str(EXTERNAL_AUDIO_ROOT.resolve()),
        "alpha_to_p7": alphas,
        "note": (
            "External 39D MFCC/GMM is used only to estimate alpha. Model input is re-extracted "
            "with the historical 128-mel Inversion_SI frontend."
        ),
    }
    summary_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return alphas


def detected_chunks(
    helper: Any,
    config: dict[str, Any],
    features: np.ndarray,
    tg: Any,
    sample_rate: int,
    window_length: int,
    hop_length: int,
    silence: np.ndarray,
) -> list[np.ndarray]:
    chunks, _, _, _ = helper.detect_sentences(
        features,
        tg,
        helper.all_phonemes,
        sample_rate,
        window_length,
        hop_length,
        silence,
    )
    return [np.asarray(chunk, dtype=np.float32) for chunk in chunks]


def match_cached_chunks(
    raw_chunks: list[np.ndarray], detected_unwarped: list[np.ndarray]
) -> tuple[list[int], list[dict[str, Any]]]:
    position = 0
    selected: list[int] = []
    rows: list[dict[str, Any]] = []
    for raw_index, raw_chunk in enumerate(raw_chunks):
        raw_array = np.asarray(raw_chunk, dtype=np.float32)
        candidates = []
        for detected_index in range(position, len(detected_unwarped)):
            detected = detected_unwarped[detected_index]
            if detected.shape != raw_array.shape:
                continue
            mae = float(np.mean(np.abs(detected - raw_array)))
            candidates.append((mae, detected_index))
        if not candidates:
            raise RuntimeError(
                f"No ordered detected-chunk candidate for raw chunk {raw_index}, shape={raw_array.shape}"
            )
        candidates.sort()
        mae, detected_index = candidates[0]
        runner_up = candidates[1][0] if len(candidates) > 1 else float("nan")
        if mae > MATCH_MAE_LIMIT:
            raise RuntimeError(
                f"Unsafe raw/audio chunk match at chunk {raw_index}: MAE={mae:.6f}"
            )
        selected.append(detected_index)
        rows.append(
            {
                "raw_chunk_index": raw_index,
                "detected_chunk_index": detected_index,
                "length": int(len(raw_array)),
                "unwarped_raw_mae": mae,
                "runner_up_mae": runner_up,
            }
        )
        position = detected_index + 1
    return selected, rows


def build_audio_normalized_chunks(
    config: dict[str, Any],
    speaker: int,
    session: int,
    alpha: float,
    raw_chunks: list[np.ndarray],
    rms_target: float | None = TARGET_RMS,
) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    helper = build_helper(config, "test_sequences", f"P{speaker}", f"S{session}")
    audio_path = Path(next(iter(helper.audio_files.values()))[0])
    textgrid_path = Path(next(iter(helper.textgrid_files.values()))[0])
    wav, sample_rate = librosa.load(audio_path, sr=None)
    unwarped, window_length, hop_length = compute_vtln_mfcc_like_inversion(
        wav, sample_rate, config, 1.0, VTLN_F_LOW, VTLN_F_HIGH
    )
    normalized_wav = rms_normalize(wav, rms_target)
    audio_normalized, window_length_2, hop_length_2 = compute_vtln_mfcc_like_inversion(
        normalized_wav, sample_rate, config, alpha, VTLN_F_LOW, VTLN_F_HIGH
    )
    if (window_length, hop_length) != (window_length_2, hop_length_2):
        raise RuntimeError("Frontend window/hop changed between baseline and audio-normalized extraction")
    duration_seconds = float(len(wav) / sample_rate)
    tg = load_textgrid_with_repair(str(textgrid_path), duration_seconds, config)
    silence = helper.detect_silence(unwarped, tg, sample_rate)
    detected_unwarped = detected_chunks(
        helper, config, unwarped, tg, sample_rate, window_length, hop_length, silence
    )
    detected_normalized = detected_chunks(
        helper, config, audio_normalized, tg, sample_rate, window_length, hop_length, silence
    )
    if len(detected_unwarped) != len(detected_normalized):
        raise RuntimeError("Unwarped and normalized detected chunk counts differ")
    for index, (old, new) in enumerate(zip(detected_unwarped, detected_normalized)):
        if old.shape != new.shape:
            raise RuntimeError(f"Detected chunk {index} shape differs: {old.shape} vs {new.shape}")
    selected, rows = match_cached_chunks(raw_chunks, detected_unwarped)
    result = [detected_normalized[index] for index in selected]
    raw_rms = float(np.sqrt(np.mean(np.square(wav), dtype=np.float64)))
    metadata = {
        "audio_path": str(audio_path.resolve()),
        "textgrid_path": str(textgrid_path.resolve()),
        "sample_rate": int(sample_rate),
        "raw_waveform_rms": raw_rms,
        "normalized_waveform_rms": float(
            np.sqrt(np.mean(np.square(normalized_wav), dtype=np.float64))
        ),
        "target_rms": None if rms_target is None else float(rms_target),
        "vtln_alpha_to_p7": float(alpha),
        "detected_chunks_before_annotation_filter": len(detected_unwarped),
        "cached_annotated_chunks": len(raw_chunks),
        "matched_chunk_mae_mean": float(np.mean([row["unwarped_raw_mae"] for row in rows])),
        "matched_chunk_mae_max": float(np.max([row["unwarped_raw_mae"] for row in rows])),
        "chunk_policy": "baseline unwarped silence mask + exact ordered match to cached annotated chunks",
    }
    return result, rows, metadata


def infer_with_features(
    model: torch.nn.Module,
    device: torch.device,
    raw: dict[str, Any],
    feature_chunks: list[np.ndarray],
    normalization: dict[str, np.ndarray],
    phonemes: list[str],
    batch_size: int,
) -> dict[str, np.ndarray]:
    labels_list = raw["contours"]
    frames_list = raw["frames"]
    phoneme_list = raw["phonemes"]
    if not (len(feature_chunks) == len(labels_list) == len(frames_list) == len(phoneme_list)):
        raise ValueError("Audio feature chunks do not align with raw cache lists")
    mean_mfcc = normalization["mean_mfcc"]
    std_mfcc = normalization["std_mfcc"]
    mean_contour = normalization["mean_contour"]
    std_contour = normalization["std_contour"]
    accum: dict[float, dict[str, list[Any]]] = {}
    with torch.inference_mode():
        for start in range(0, len(feature_chunks), batch_size):
            stop = min(start + batch_size, len(feature_chunks))
            batch = [
                torch.from_numpy(((item.astype(np.float32) - mean_mfcc) / std_mfcc).astype(np.float32))
                for item in feature_chunks[start:stop]
            ]
            lengths = torch.tensor([len(item) for item in batch], dtype=torch.long)
            padded = pad_sequence(batch, batch_first=True).to(device, non_blocking=True)
            predicted_norm, _, _ = model(padded, lengths)
            predicted = (
                predicted_norm.detach().cpu().numpy()
                * std_contour[None, None, :, :]
                + mean_contour[None, None, :, :]
            )
            for local, source in enumerate(range(start, stop)):
                length = int(lengths[local])
                labels = np.asarray(labels_list[source], dtype=np.float32)[:length]
                frames = np.asarray(frames_list[source], dtype=np.float32)[:length]
                phones = np.asarray(phoneme_list[source])[:length]
                for offset in range(length):
                    frame_number = float(frames[offset, 2])
                    item = accum.setdefault(frame_number, {"pred": [], "gt": [], "phone": []})
                    item["pred"].append(predicted[local, offset])
                    item["gt"].append(labels[offset])
                    item["phone"].append(decode_phoneme(phones[offset], phonemes))

    frame_numbers = np.asarray(sorted(accum), dtype=np.float32)
    predicted_rows, ground_truth, decoded, overlaps = [], [], [], []
    for frame_number in frame_numbers:
        item = accum[float(frame_number)]
        predicted_rows.append(np.mean(np.stack(item["pred"]), axis=0))
        ground_truth.append(np.mean(np.stack(item["gt"]), axis=0))
        decoded.append(Counter(item["phone"]).most_common(1)[0][0])
        overlaps.append(len(item["pred"]))
    return {
        "frame_numbers": frame_numbers,
        "predicted_raw": np.asarray(predicted_rows, dtype=np.float32).reshape(-1, 11, 50, 2),
        "ground_truth": np.asarray(ground_truth, dtype=np.float32).reshape(-1, 11, 50, 2),
        "phonemes": np.asarray(decoded, dtype="U32"),
        "overlap_counts": np.asarray(overlaps, dtype=np.int16),
        "num_input_rows": np.asarray(sum(len(item) for item in feature_chunks), dtype=np.int64),
        "num_sequences": np.asarray(len(feature_chunks), dtype=np.int64),
    }


def load_baseline_pack(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "arrays": {
                "raw": np.asarray(payload["predicted_raw"], dtype=np.float32),
                "affine": np.asarray(payload["predicted_after_affine"], dtype=np.float32),
                "affine_tps": np.asarray(payload["predicted_after_affine_tps"], dtype=np.float32),
            },
            "ground_truth": np.asarray(payload["ground_truth"], dtype=np.float32),
            "frame_numbers": np.asarray(payload["frame_numbers"], dtype=np.float32),
            "phonemes": np.asarray(payload["phonemes"]),
            "classes": [str(value) for value in payload["classes"].tolist()],
        }


def save_audio_pack(
    path: Path,
    inferred: dict[str, np.ndarray],
    affine: np.ndarray,
    final: np.ndarray,
    classes: list[str],
    alpha: float,
) -> None:
    np.savez_compressed(
        path,
        frame_numbers=inferred["frame_numbers"],
        phonemes=inferred["phonemes"],
        overlap_counts=inferred["overlap_counts"],
        predicted_audio_raw=inferred["predicted_raw"],
        predicted_audio_after_affine=affine,
        predicted_audio_after_affine_tps=final,
        ground_truth=inferred["ground_truth"],
        classes=np.asarray(classes, dtype="U64"),
        excluded_classes=np.asarray(EXCLUDED_CLASSES, dtype="U64"),
        vtln_alpha_to_p7=np.asarray(alpha, dtype=np.float32),
        target_rms=np.asarray(TARGET_RMS, dtype=np.float32),
        num_input_rows=inferred["num_input_rows"],
        num_sequences=inferred["num_sequences"],
    )


def load_audio_pack(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "arrays": {
                "raw": np.asarray(payload["predicted_audio_raw"], dtype=np.float32),
                "affine": np.asarray(payload["predicted_audio_after_affine"], dtype=np.float32),
                "affine_tps": np.asarray(payload["predicted_audio_after_affine_tps"], dtype=np.float32),
            },
            "ground_truth": np.asarray(payload["ground_truth"], dtype=np.float32),
            "frame_numbers": np.asarray(payload["frame_numbers"], dtype=np.float32),
            "phonemes": np.asarray(payload["phonemes"]),
            "num_input_rows": int(payload["num_input_rows"]),
            "num_sequences": int(payload["num_sequences"]),
        }


def write_frame_comparison(
    path: Path,
    frames: np.ndarray,
    phones: np.ndarray,
    baseline_metrics: dict[str, dict[str, np.ndarray]],
    audio_metrics: dict[str, dict[str, np.ndarray]],
) -> None:
    fields = ["frame", "frame_number", "phoneme"]
    for mode in ("all_11", "without_laryngeal_3"):
        for branch in ("baseline", "audio_normalized"):
            fields.extend(f"{branch}_{stage}_{mode}_rmse_mm" for stage in STAGES)
        fields.extend(f"audio_minus_baseline_{stage}_{mode}_mm" for stage in STAGES)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, frame in enumerate(frames):
            row: dict[str, Any] = {
                "frame": frame_token(float(frame)),
                "frame_number": float(frame),
                "phoneme": str(phones[index]),
            }
            for mode in ("all_11", "without_laryngeal_3"):
                for stage in STAGES:
                    base = float(baseline_metrics[mode][stage][index])
                    audio = float(audio_metrics[mode][stage][index])
                    row[f"baseline_{stage}_{mode}_rmse_mm"] = base
                    row[f"audio_normalized_{stage}_{mode}_rmse_mm"] = audio
                    row[f"audio_minus_baseline_{stage}_{mode}_mm"] = audio - base
            writer.writerow(row)


def render_six_panel_video(
    args: argparse.Namespace,
    speaker: int,
    session: int,
    session_dir: Path,
    baseline_arrays: dict[str, np.ndarray],
    audio_arrays: dict[str, np.ndarray],
    frames: np.ndarray,
    phones: np.ndarray,
    ground_truth: np.ndarray,
    baseline_metrics: dict[str, dict[str, np.ndarray]],
    audio_metrics: dict[str, dict[str, np.ndarray]],
    classes: list[str],
) -> Path:
    if not np.isclose(frames, np.rint(frames), atol=1e-4).all():
        raise ValueError(INTEGER_FRAME_POLICY)
    dicom_dir = RAW_ROOT / f"P{speaker}/DCM_2D/S{session}"
    needed = sorted(int(round(float(value))) for value in frames)
    dicom_index, _ = build_filename_dicom_index(dicom_dir)
    cache_path = session_dir / "mri_frames_cache.npz"
    mri_cache = load_or_build_mri_cache(
        dicom_dir, dicom_index, needed, cache_path, workers=args.mri_workers
    )
    panel_width = 136 * args.scale
    panel_height = panel_width + 106
    width = panel_width * 3 + SEPARATOR * 2
    height = panel_height * 2 + SEPARATOR
    video_path = session_dir / f"p{speaker}_s{session}_baseline_vs_audio_gridnorm_50fps.mp4"
    temporary = session_dir / f".{video_path.stem}.writing.mp4"
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {temporary}")
    titles = {
        "raw": "raw prediction",
        "affine": "after affine",
        "affine_tps": "after affine + TPS",
    }
    try:
        for index, frame in enumerate(frames):
            image = mri_for_timestamp(float(frame), mri_cache)
            canvas = np.full((height, width, 3), 15, dtype=np.uint8)
            for row_index, (branch, arrays, metrics) in enumerate(
                (
                    ("Baseline audio", baseline_arrays, baseline_metrics),
                    ("RMS+VTLN audio", audio_arrays, audio_metrics),
                )
            ):
                for column, stage in enumerate(STAGES):
                    panel = draw_panel(
                        image,
                        arrays[stage][index],
                        ground_truth[index],
                        classes,
                        f"{branch}: {titles[stage]}",
                        float(frame),
                        str(phones[index]),
                        float(metrics["all_11"][stage][index]),
                        float(metrics["without_laryngeal_3"][stage][index]),
                        args.scale,
                    )
                    x = column * (panel_width + SEPARATOR)
                    y = row_index * (panel_height + SEPARATOR)
                    canvas[y : y + panel_height, x : x + panel_width] = panel
            writer.write(canvas)
    finally:
        writer.release()
    temporary.replace(video_path)
    if not args.keep_mri_cache and cache_path.exists():
        cache_path.unlink()
    return video_path


def mean_by_stage(summary: dict[str, Any], branch: str, mode: str) -> dict[str, float]:
    return {
        stage: float(summary["metrics"][branch]["modes"][mode][stage]["mean_frame_rmse_mm"])
        for stage in STAGES
    }


def process_session(
    args: argparse.Namespace,
    speaker: int,
    session: int,
    alpha: float,
    config: dict[str, Any],
    classes: list[str],
    phonemes: list[str],
    normalization: dict[str, np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
    transform: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    session_dir = args.output_root / f"P{speaker}/S{session}"
    session_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.baseline_root / f"P{speaker}/S{session}/contours_and_ground_truth.npz"
    audio_path = session_dir / "audio_normalized_contours_and_ground_truth.npz"
    alignment_path = session_dir / "audio_chunk_alignment.csv"
    extraction_path = session_dir / "audio_extraction_metadata.json"
    baseline = load_baseline_pack(baseline_path)
    if baseline["classes"] != classes:
        raise ValueError(f"Class mismatch in baseline pack {baseline_path}")

    if audio_path.is_file() and extraction_path.is_file() and not args.force:
        audio = load_audio_pack(audio_path)
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
        print(f"REUSE P{speaker}/S{session} audio-normalized contour pack", flush=True)
    else:
        raw_path = args.raw_cache_root / f"P{speaker}/S{session}.pt"
        raw = torch.load(raw_path, map_location="cpu", weights_only=False)["raw"]
        feature_chunks, alignment_rows, extraction = build_audio_normalized_chunks(
            config, speaker, session, alpha, raw["features"]
        )
        write_alignment(alignment_path, alignment_rows)
        extraction_path.write_text(json.dumps(extraction, indent=2, sort_keys=True), encoding="utf-8")
        inferred = retain_integer_inferred(
            infer_with_features(
                model, device, raw, feature_chunks, normalization, phonemes, args.batch_size
            )
        )
        affine, final = transform_contour_batch(
            inferred["predicted_raw"], transform, args.transform_frame_batch
        )
        save_audio_pack(audio_path, inferred, affine, final, classes, alpha)
        audio = {
            "arrays": {"raw": inferred["predicted_raw"], "affine": affine, "affine_tps": final},
            "ground_truth": inferred["ground_truth"],
            "frame_numbers": inferred["frame_numbers"],
            "phonemes": inferred["phonemes"],
            "num_input_rows": int(inferred["num_input_rows"]),
            "num_sequences": int(inferred["num_sequences"]),
        }

    (
        baseline["arrays"],
        baseline["ground_truth"],
        baseline["frame_numbers"],
        baseline["phonemes"],
        baseline_discarded_fractional,
    ) = retain_integer_arrays(
        baseline["arrays"],
        baseline["ground_truth"],
        baseline["frame_numbers"],
        baseline["phonemes"],
    )
    (
        audio["arrays"],
        audio["ground_truth"],
        audio["frame_numbers"],
        audio["phonemes"],
        audio_discarded_fractional,
    ) = retain_integer_arrays(
        audio["arrays"],
        audio["ground_truth"],
        audio["frame_numbers"],
        audio["phonemes"],
    )

    if not np.array_equal(audio["frame_numbers"], baseline["frame_numbers"]):
        raise RuntimeError(f"Frame timeline changed for P{speaker}/S{session}")
    gt_max_delta = float(np.max(np.abs(audio["ground_truth"] - baseline["ground_truth"])))
    if gt_max_delta > 1e-5:
        raise RuntimeError(f"Ground truth changed for P{speaker}/S{session}: max delta {gt_max_delta}")
    baseline_summary, baseline_frame_metrics = metric_payload(
        baseline["arrays"], baseline["ground_truth"], classes
    )
    audio_summary, audio_frame_metrics = metric_payload(
        audio["arrays"], baseline["ground_truth"], classes
    )
    frame_csv = session_dir / "frame_metrics_baseline_vs_audio.csv"
    write_frame_comparison(
        frame_csv,
        baseline["frame_numbers"],
        baseline["phonemes"],
        baseline_frame_metrics,
        audio_frame_metrics,
    )
    video_path = session_dir / f"p{speaker}_s{session}_baseline_vs_audio_gridnorm_50fps.mp4"
    if not args.skip_video and (args.force or not video_path.is_file()):
        render_six_panel_video(
            args,
            speaker,
            session,
            session_dir,
            baseline["arrays"],
            audio["arrays"],
            baseline["frame_numbers"],
            baseline["phonemes"],
            baseline["ground_truth"],
            baseline_frame_metrics,
            audio_frame_metrics,
            classes,
        )
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "speaker": speaker,
        "session": session,
        "num_unique_frames": int(len(baseline["frame_numbers"])),
        "baseline_fractional_frames_discarded": baseline_discarded_fractional,
        "audio_fractional_frames_discarded": audio_discarded_fractional,
        "rendered_fractional_frame_count": 0,
        "frame_policy": INTEGER_FRAME_POLICY,
        "num_sequences": int(audio["num_sequences"]),
        "num_input_rows": int(audio["num_input_rows"]),
        "vtln_alpha_to_p7": float(alpha),
        "target_rms": TARGET_RMS,
        "baseline_contour_pack": str(baseline_path.resolve()),
        "audio_normalized_contour_pack": str(audio_path.resolve()),
        "audio_chunk_alignment": str(alignment_path.resolve()),
        "audio_extraction_metadata": str(extraction_path.resolve()),
        "frame_metrics": str(frame_csv.resolve()),
        "video": None if args.skip_video else str(video_path.resolve()),
        "ground_truth_max_abs_delta_vs_baseline": gt_max_delta,
        "metrics": {"baseline": baseline_summary, "audio_normalized": audio_summary},
        "elapsed_seconds": time.monotonic() - started,
    }
    summary_path = session_dir / "session_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    b = mean_by_stage(summary, "baseline", "all_11")
    a = mean_by_stage(summary, "audio_normalized", "all_11")
    print(
        f"DONE P{speaker}/S{session} alpha={alpha:.3f}: "
        f"baseline {b['raw']:.3f}/{b['affine']:.3f}/{b['affine_tps']:.3f}, "
        f"audio {a['raw']:.3f}/{a['affine']:.3f}/{a['affine_tps']:.3f} mm, "
        f"{summary['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return summary


def weighted_values(
    summaries: list[dict[str, Any]], branch: str, mode: str
) -> dict[str, float]:
    total = sum(int(row["num_unique_frames"]) for row in summaries)
    return {
        stage: sum(
            int(row["num_unique_frames"]) * mean_by_stage(row, branch, mode)[stage]
            for row in summaries
        )
        / total
        for stage in STAGES
    }


def aggregate_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    groups = [(f"P{row['speaker']}", [row]) for row in summaries] + [("ALL", summaries)]
    for label, group in groups:
        for mode in ("all_11", "without_laryngeal_3"):
            baseline = weighted_values(group, "baseline", mode)
            audio = weighted_values(group, "audio_normalized", mode)
            rows.append(
                {
                    "speaker": label,
                    "session": f"S{group[0]['session']}" if len(group) == 1 else "selected_9",
                    "frames": sum(int(item["num_unique_frames"]) for item in group),
                    "metric_mode": mode,
                    "vtln_alpha_to_p7": float(group[0]["vtln_alpha_to_p7"]) if len(group) == 1 else "",
                    "baseline_raw_mm": baseline["raw"],
                    "baseline_affine_mm": baseline["affine"],
                    "baseline_affine_tps_mm": baseline["affine_tps"],
                    "audio_raw_mm": audio["raw"],
                    "audio_affine_mm": audio["affine"],
                    "audio_affine_tps_mm": audio["affine_tps"],
                    "audio_effect_raw_mm": audio["raw"] - baseline["raw"],
                    "audio_effect_affine_mm": audio["affine"] - baseline["affine"],
                    "audio_effect_affine_tps_mm": audio["affine_tps"] - baseline["affine_tps"],
                    "audio_grid_raw_to_affine_mm": audio["affine"] - audio["raw"],
                    "audio_grid_affine_to_tps_mm": audio["affine_tps"] - audio["affine"],
                    "audio_grid_raw_to_final_mm": audio["affine_tps"] - audio["raw"],
                }
            )
    return rows


def generate_report(args: argparse.Namespace) -> dict[str, Any]:
    summaries = []
    for speaker, session in SELECTION:
        path = args.output_root / f"P{speaker}/S{session}/session_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Cannot report before session completes: {path}")
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    rows = aggregate_rows(summaries)
    write_csv(args.output_root / "baseline_vs_audio_gridnorm_metrics.csv", rows)
    report_path = args.output_root / "audio_gridnorm_error_report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# P7 model: audio normalization + grid transform on selected unseen speakers\n\n")
        handle.write(
            "This run reuses the existing baseline contours and ground truth. Only the RMS+VTLN "
            "branch is newly inferred. VTLN alpha is estimated toward P7 with the external "
            "audio-normalization GMM, then features are re-extracted with the exact historical "
            "Inversion_SI frontend and the original cached silence/chunk selection.\n\n"
        )
        handle.write(
            "`without 3` excludes vocal-folds, thyroid-cartilage, and epiglottis. Negative audio "
            "effect means RMS+VTLN improved over the matching baseline grid stage.\n\n"
        )
        for mode, title in (
            ("all_11", "All 11 contours"),
            ("without_laryngeal_3", "Without the three laryngeal contours"),
        ):
            handle.write(f"## {title}\n\n")
            handle.write(
                "| Speaker/session | Frames | alpha | Base raw | Base aff | Base TPS | Audio raw | "
                "Audio aff | Audio TPS | Audio effect at TPS |\n"
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
            )
            mode_rows = [row for row in rows if row["metric_mode"] == mode]
            for row in mode_rows:
                alpha = "—" if row["speaker"] == "ALL" else f"{float(row['vtln_alpha_to_p7']):.3f}"
                label = (
                    "ALL" if row["speaker"] == "ALL" else f"{row['speaker']}/{row['session']}"
                )
                handle.write(
                    f"| {label} | {row['frames']} | {alpha} | {row['baseline_raw_mm']:.3f} | "
                    f"{row['baseline_affine_mm']:.3f} | {row['baseline_affine_tps_mm']:.3f} | "
                    f"{row['audio_raw_mm']:.3f} | {row['audio_affine_mm']:.3f} | "
                    f"{row['audio_affine_tps_mm']:.3f} | {row['audio_effect_affine_tps_mm']:+.3f} |\n"
                )
            overall = next(row for row in mode_rows if row["speaker"] == "ALL")
            improved_final = sum(
                row["audio_effect_affine_tps_mm"] < 0
                for row in mode_rows
                if row["speaker"] != "ALL"
            )
            handle.write("\n")
            handle.write(
                f"Frame-weighted audio effect is {overall['audio_effect_raw_mm']:+.3f} mm at raw, "
                f"{overall['audio_effect_affine_mm']:+.3f} mm after affine, and "
                f"{overall['audio_effect_affine_tps_mm']:+.3f} mm after TPS. "
                f"Audio+TPS improves {improved_final}/9 speakers relative to baseline+TPS. "
                f"Within the audio branch, affine changes error by "
                f"{overall['audio_grid_raw_to_affine_mm']:+.3f} mm and TPS adds "
                f"{overall['audio_grid_affine_to_tps_mm']:+.3f} mm.\n\n"
            )

        handle.write("## Per-speaker interpretation\n\n")
        all_rows = {
            row["speaker"]: row
            for row in rows
            if row["metric_mode"] == "all_11" and row["speaker"] != "ALL"
        }
        without_rows = {
            row["speaker"]: row
            for row in rows
            if row["metric_mode"] == "without_laryngeal_3" and row["speaker"] != "ALL"
        }
        for summary in summaries:
            name = f"P{summary['speaker']}"
            row_all = all_rows[name]
            row_without = without_rows[name]
            result_word = "improves" if row_all["audio_effect_affine_tps_mm"] < 0 else "worsens"
            laryngeal_note = (
                "The excluded laryngeal contours account for part of the remaining mismatch."
                if row_all["audio_effect_affine_tps_mm"] > row_without["audio_effect_affine_tps_mm"]
                else "The effect is not driven mainly by the three excluded laryngeal contours."
            )
            handle.write(
                f"- **{name}/S{summary['session']} (alpha {summary['vtln_alpha_to_p7']:.3f}):** "
                f"audio normalization {result_word} final all-contour RMSE by "
                f"{row_all['audio_effect_affine_tps_mm']:+.3f} mm; without three, the change is "
                f"{row_without['audio_effect_affine_tps_mm']:+.3f} mm. {laryngeal_note}\n"
            )
        handle.write("\n## Why audio normalization may or may not help\n\n")
        handle.write(
            "1. VTLN optimizes acoustic likelihood under the P7 GMM, not contour RMSE. Better P7-like "
            "formants can therefore help inversion, but the two objectives are not identical.\n"
            "2. One alpha is shared by every phoneme and frame of a speaker/session. It corrects a stable "
            "vocal-tract-length shift but cannot model phone-dependent spectral differences.\n"
            "3. RMS normalization mainly changes energy/C0. It helps when recording-level gain differs "
            "from P7, but can move features away from the distribution seen during P7 training.\n"
            "4. Audio normalization and anatomical grid transfer act in different spaces. Their gains can "
            "add when acoustic and geometric speaker differences are both stable, or conflict when the fixed "
            "grid/TPS extrapolates poorly around the laryngeal contours.\n"
            "5. Each unseen speaker uses one selected session here. Alpha and error changes should not yet be "
            "interpreted as speaker-wide estimates without checking additional sessions.\n\n"
        )
        handle.write("## Videos and saved contours\n\n")
        for summary in summaries:
            speaker = f"P{summary['speaker']}"
            session = f"S{summary['session']}"
            base = args.output_root / speaker / session
            if summary["video"]:
                video = Path(summary["video"]).relative_to(args.output_root.resolve())
                handle.write(f"- [{speaker}/{session} six-panel video]({video.as_posix()}) — ")
            else:
                handle.write(f"- {speaker}/{session} — ")
            pack = Path(summary["audio_normalized_contour_pack"]).relative_to(args.output_root.resolve())
            metrics = Path(summary["frame_metrics"]).relative_to(args.output_root.resolve())
            handle.write(f"[audio contour pack]({pack.as_posix()}), [frame metrics]({metrics.as_posix()})\n")
        handle.write("\n")
        handle.write(
            "The original baseline packs under `p7_selected_nonp7_sessions_gridnorm_20260718` were "
            "read-only inputs and were not regenerated or modified.\n"
        )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection": [f"P{speaker}/S{session}" for speaker, session in SELECTION],
        "baseline_root": str(args.baseline_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "report": str(report_path.resolve()),
        "metrics_csv": str((args.output_root / "baseline_vs_audio_gridnorm_metrics.csv").resolve()),
        "sessions": summaries,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"report": str(report_path.resolve()), "completed_sessions": len(summaries)}


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        print(json.dumps(generate_report(args), indent=2), flush=True)
        return

    alphas = estimate_alphas(args)
    config = load_yaml_config(args.config)
    classes = list(config["classes"])
    with open(config["phonemesdir"], "r", encoding="utf-8") as handle:
        phonemes = json.load(handle)
    normalization = load_normalization(args.normalization_stats)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for {device}")
    model = load_model(config, args.checkpoint, device)

    transforms: dict[int, dict[str, Any]] = {}
    source = prepare_frame(SOURCE, args.vtln_dir)
    for speaker, session in selected_pairs(args):
        if speaker not in transforms:
            target = prepare_frame(target_spec_for_speaker(speaker), args.vtln_dir)
            transforms[speaker] = build_two_step_transform(source["grid"], target["grid"])
        process_session(
            args,
            speaker,
            session,
            alphas[f"P{speaker}"],
            config,
            classes,
            phonemes,
            normalization,
            model,
            device,
            transforms[speaker],
        )
    if not args.no_report:
        print(json.dumps(generate_report(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
