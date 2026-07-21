#!/usr/bin/env python3
"""Run the epoch-211 ASD2 model on the fixed nine-session ASD1 experiment.

This is an inference-only pipeline.  It deliberately keeps the three P7
20260718 result bundles immutable and writes a new versioned result tree.
Only integer MRI frames are saved, scored, and exposed to downstream video
rendering.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
GRID_ROOT = REPO_ROOT / "external" / "grid-transform"
AUDIO_NORM_ROOT = (
    REPO_ROOT / "external" / "audio-speaker-normalization" / "audio-speaker-normalization"
)
sys.path[:0] = [
    str(REPO_ROOT),
    str(REPO_ROOT / "src"),
    str(REPO_ROOT / "scripts"),
    str(GRID_ROOT),
    str(AUDIO_NORM_ROOT),
]

from audio_speaker_norm.audio_normalization import AudioNormConfig, FeatureExtractor  # noqa: E402
from grid_transform.annotation_projection import build_resize_affine, transform_reference_contours  # noqa: E402
from grid_transform.io import _load_roi_contours_from_zip  # noqa: E402
from grid_transform.transfer import build_two_step_transform  # noqa: E402
from grid_transform.vt import build_grid  # noqa: E402
from notebooks.audio_norm_utils import (  # noqa: E402
    apply_cmvn,
    extract_vtln_mfcc39,
    fit_cmvn,
    fit_speaker_gmms,
    sample_rows,
    score_gmm,
)
from render_p7_grid_transform_selected_speakers import (  # noqa: E402
    CLASSES,
    TARGETS,
    prepare_frame,
)
from run_p7_all_nonp7_gridnorm import (  # noqa: E402
    DEFAULT_RAW_CACHE,
    DEFAULT_VTLN_DIR,
    RAW_ROOT,
    STAGES,
    infer_session,
    load_normalization,
    retain_integer_inferred,
    transform_contour_batch,
)
from run_p7_selected_audio_gridnorm import (  # noqa: E402
    MATCH_MAE_LIMIT,
    TARGET_RMS,
    VTLN_F_HIGH,
    VTLN_F_LOW,
    build_audio_normalized_chunks,
    infer_with_features,
)
from src.inference.session_inference import load_model  # noqa: E402
from src.utils.config_validation import load_yaml_config  # noqa: E402
from src.utils.mri_rendering import normalize_mri_frame  # noqa: E402
from src.utils.video_rendering import MM_PER_PIXEL  # noqa: E402


SELECTION = (
    (1, 16),
    (2, 9),
    (3, 14),
    (4, 4),
    (5, 6),
    (6, 8),
    (8, 2),
    (9, 5),
    (10, 14),
)
UNSEEN_SELECTION = tuple(pair for pair in SELECTION if pair[0] != 10)
ASD2_ROOT = Path(
    "/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/"
    "ArtSpeech_Database_2"
)
BF_ROOT = WORKSPACE_ROOT / "bf" / "inference"
DEFAULT_CONFIG = (
    REPO_ROOT
    / "repro/asd2_11contour_vtln20260719_train_global/auto_batch_configs/6784374/"
    "asd2_11contour_vtln20260719_train_global_rawstd_st5_mfcc_500epoch_"
    "auto80_bs1150_4gpu_20260719_164428.yaml"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "mlruns/610923529796522440/adbb5c9946704b568d1f7a48b445a0c0/"
    "artifacts/best_model.pth"
)
DEFAULT_NORMALIZATION = (
    REPO_ROOT
    / "repro/asd2_11contour_vtln20260719_train_global/splits/normalization_stats.npz"
)
DEFAULT_SOURCE_PACK = (
    REPO_ROOT
    / "cache_variants/asd2_11_vtln_20260719/raw_contour_npz/asd2/1791/S14.npz"
)
DEFAULT_OUTPUT = REPO_ROOT / "results/asd2_epoch211_asd1_selected_grid_audio_20260719"
OLD_GRID_ROOT = REPO_ROOT / "results/p7_selected_nonp7_sessions_gridnorm_20260718"
OLD_AUDIO_ROOT = REPO_ROOT / "results/p7_selected_nonp7_sessions_audio_gridnorm_20260718"
OLD_ABLATION_ROOT = REPO_ROOT / "results/p7_selected_nonp7_sessions_audio_ablation_20260718"
SOURCE_BUCKET = "1791"
SOURCE_SESSION = "S14"
SOURCE_FRAME = 3020
SOURCE_TOKEN = "143020"
SOURCE_CASE_ROOT = (
    GRID_ROOT / "VTLN/data/nnunet_data_80/2008-003^01-1791/test"
)
SOURCE_REFERENCE = "ASD2 1791/S14/F3020 (token 143020)"
INTEGER_FRAME_POLICY = "NEVER save, score, or render fractional MRI frames; integer frames only"
EXCLUDED_LARYNGEAL = ("vocal-folds", "thyroid-cartilage", "epiglottis")
INCISORS = ("lower-incisor", "upper-incisor")
MODE_INDICES = {
    "all_11": tuple(range(11)),
    "without_laryngeal_3": tuple(
        index for index, name in enumerate(CLASSES) if name not in EXCLUDED_LARYNGEAL
    ),
    "without_incisors_2": tuple(
        index for index, name in enumerate(CLASSES) if name not in INCISORS
    ),
    "without_laryngeal_3_and_incisors_2": tuple(
        index
        for index, name in enumerate(CLASSES)
        if name not in set(EXCLUDED_LARYNGEAL) | set(INCISORS)
    ),
}
BRANCHES = ("baseline", "rms_vtln", "rms_only", "vtln_only")
GATE_AGGREGATE_THRESHOLD_MM = 0.02
GATE_SESSION_THRESHOLD_MM = 0.10
BOOTSTRAP_SEED = 20260719


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("preflight", "baseline", "audio", "ablation", "all"),
        default="all",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--normalization-stats", type=Path, default=DEFAULT_NORMALIZATION)
    parser.add_argument("--source-pack", type=Path, default=DEFAULT_SOURCE_PACK)
    parser.add_argument("--raw-cache-root", type=Path, default=DEFAULT_RAW_CACHE)
    parser.add_argument("--vtln-dir", type=Path, default=DEFAULT_VTLN_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection", nargs="+", default=None, metavar="P#:S#")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--transform-frame-batch", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-input-tree-hash", action="store_true")
    return parser.parse_args()


def selected_pairs(args: argparse.Namespace) -> tuple[tuple[int, int], ...]:
    if not args.selection:
        return SELECTION
    allowed = set(SELECTION)
    parsed: list[tuple[int, int]] = []
    for token in args.selection:
        try:
            speaker_token, session_token = token.upper().split(":", maxsplit=1)
            pair = (
                int(speaker_token.removeprefix("P")),
                int(session_token.removeprefix("S")),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid selection {token!r}; expected P#:S#") from error
        if pair not in allowed:
            raise ValueError(f"P{pair[0]}/S{pair[1]} is outside the fixed nine-session set")
        if pair not in parsed:
            parsed.append(pair)
    return tuple(parsed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def exact_asd1_audio_paths(speaker: int, session: int) -> tuple[Path, Path]:
    folder = RAW_ROOT / f"P{speaker}" / "OTHER" / f"S{session}"
    wav = folder / f"DENOISED_SOUND_P{speaker}_S{session}.wav"
    textgrid = folder / f"TEXT_ALIGNMENT_P{speaker}_S{session}.textgrid"
    if not wav.is_file() or not textgrid.is_file():
        raise FileNotFoundError(f"Missing ASD1 audio pair: {wav}, {textgrid}")
    return wav, textgrid


def exact_asd2_audio_paths(bucket: str, session: str) -> tuple[Path, Path]:
    folder = ASD2_ROOT / str(bucket) / str(session)
    wav_candidates = sorted(
        path
        for path in folder.glob("*.wav")
        if not path.name.endswith("_mocap.wav")
    )
    if not wav_candidates:
        raise FileNotFoundError(f"Missing non-mocap ASD2 training WAV under {folder}")
    wav = wav_candidates[0]
    textgrid = folder / f"{wav.stem}_adjusted.textgrid"
    if not textgrid.is_file():
        raise FileNotFoundError(f"Missing ASD2 training audio pair: {wav}, {textgrid}")
    return wav, textgrid


def target_spec(speaker: int):
    for spec in TARGETS:
        if spec.speaker == f"P{speaker}":
            return spec
    raise KeyError(f"No target grid reference for P{speaker}")


def load_source_model_contours(source_pack: Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    with np.load(source_pack, allow_pickle=False) as payload:
        frames = np.asarray(payload["frame_numbers"], dtype=np.int32)
        indices = np.flatnonzero(frames == SOURCE_FRAME)
        if len(indices) != 1:
            raise RuntimeError(
                f"Expected exactly one F{SOURCE_FRAME} row in {source_pack}, got {len(indices)}"
            )
        articulators = [str(value) for value in payload["articulators"].tolist()]
        contours = np.asarray(payload["contours"], dtype=np.float32).reshape(-1, 11, 50, 2)
        row = contours[int(indices[0])]
        source_folder = str(payload["source_folder"].item())
        incisor_root = str(payload["incisor_source_root"].item())
    if articulators != list(CLASSES):
        raise ValueError(f"Source pack class order differs: {articulators}")
    by_class = {name: row[index].astype(np.float64) for index, name in enumerate(articulators)}
    return row, by_class, {
        "source_pack": str(source_pack.resolve()),
        "source_pack_sha256": file_sha256(source_pack),
        "source_folder_classes_0_to_8": source_folder,
        "incisor_source_root": incisor_root,
    }


def load_case_c_contours(target_shape: tuple[int, int]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    image_path = SOURCE_CASE_ROOT / "PNG_MR" / f"{SOURCE_TOKEN}.png"
    zip_path = SOURCE_CASE_ROOT / "contours" / f"{SOURCE_TOKEN}.zip"
    image = np.asarray(Image.open(image_path))
    contours_480 = _load_roi_contours_from_zip(zip_path, image_name=SOURCE_TOKEN)
    selected = {name: contours_480[name] for name in ("c1", "c2", "c3", "c4", "c5", "c6")}
    affine = build_resize_affine(image.shape[:2], target_shape)
    transformed = transform_reference_contours(selected, affine)
    return transformed, {
        "token": SOURCE_TOKEN,
        "image": str(image_path.resolve()),
        "image_sha256": file_sha256(image_path),
        "contours": str(zip_path.resolve()),
        "contours_sha256": file_sha256(zip_path),
        "source_shape": list(image.shape[:2]),
        "target_shape": list(target_shape),
    }


def build_source_grid(source_pack: Path):
    _row, annotations, source_metadata = load_source_model_contours(source_pack)
    image_path = ASD2_ROOT / SOURCE_BUCKET / SOURCE_SESSION / "NPY_MR_registered" / f"{SOURCE_FRAME}.npy"
    image = normalize_mri_frame(np.load(image_path, allow_pickle=False))
    grid_contours = {
        "incisior-hard-palate": annotations["upper-incisor"],
        "mandible-incisior": annotations["lower-incisor"],
        "lower-lip": annotations["lower-lip"],
        "pharynx": annotations["pharynx"],
        "soft-palate-midline": annotations["soft-palate-midline"],
        "tongue": annotations["tongue"],
        "upper-lip": annotations["upper-lip"],
    }
    c_contours, c_metadata = load_case_c_contours(image.shape[:2])
    grid_contours.update(c_contours)
    grid = build_grid(image, grid_contours, n_vert=9, n_points=250, frame_number=SOURCE_FRAME)
    source_metadata.update(
        {
            "reference": SOURCE_REFERENCE,
            "mri": str(image_path.resolve()),
            "mri_sha256": file_sha256(image_path),
            "c_contours": c_metadata,
            "class_sources": {
                "classes_0_to_8": "ASD2 registered contour pack used by training",
                "classes_9_to_10": "workspace VTLN-incisor overlay used by training",
            },
        }
    )
    return grid, grid_contours, source_metadata


def build_transforms(args: argparse.Namespace) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    source_grid, source_contours, source_metadata = build_source_grid(args.source_pack)
    transforms: dict[int, dict[str, Any]] = {}
    target_rows = []
    for speaker in sorted({speaker for speaker, _session in SELECTION}):
        spec = target_spec(speaker)
        target = prepare_frame(spec, args.vtln_dir)
        transform = build_two_step_transform(source_grid, target["grid"])
        probe = np.concatenate([source_contours[name] for name in sorted(source_contours)], axis=0)
        mapped = np.asarray(transform["apply_two_step"](probe), dtype=np.float64)
        if mapped.shape != probe.shape or not np.isfinite(mapped).all():
            raise RuntimeError(f"Invalid ASD2->P{speaker} grid transform")
        transforms[speaker] = transform
        target_rows.append(
            {
                "speaker": f"P{speaker}",
                "reference": spec.label,
                "dynamic_frame": int(spec.frame),
                "vtln_anchor": spec.vtln_anchor,
                "probe_points": int(len(probe)),
                "probe_mapped_finite": True,
            }
        )
    return transforms, {"source": source_metadata, "targets": target_rows}


def old_reference_pack(speaker: int, session: int) -> Path:
    return OLD_GRID_ROOT / f"P{speaker}" / f"S{session}" / "contours_and_ground_truth.npz"


def load_old_reference(speaker: int, session: int) -> tuple[np.ndarray, np.ndarray]:
    path = old_reference_pack(speaker, session)
    with np.load(path, allow_pickle=False) as payload:
        frames = np.asarray(payload["frame_numbers"])
        ground_truth = np.asarray(payload["ground_truth"], dtype=np.float32)
    if not np.issubdtype(frames.dtype, np.integer):
        raise ValueError(f"Old reference timeline is not integer typed: {path} ({frames.dtype})")
    return frames.astype(np.int32), ground_truth


def validate_inferred_against_old(
    speaker: int,
    session: int,
    inferred: dict[str, Any],
) -> dict[str, Any]:
    frames = np.asarray(inferred["frame_numbers"])
    if not np.issubdtype(frames.dtype, np.integer):
        raise AssertionError(f"New frame dtype is not integer for P{speaker}/S{session}: {frames.dtype}")
    old_frames, old_gt = load_old_reference(speaker, session)
    frame_equal = bool(np.array_equal(frames.astype(np.int32), old_frames))
    gt_shape_equal = bool(np.asarray(inferred["ground_truth"]).shape == old_gt.shape)
    gt_delta = (
        float(np.max(np.abs(np.asarray(inferred["ground_truth"], dtype=np.float32) - old_gt)))
        if gt_shape_equal
        else float("inf")
    )
    if not frame_equal or not gt_shape_equal or gt_delta > 1e-5:
        raise RuntimeError(
            f"Old/new population mismatch P{speaker}/S{session}: frames={frame_equal}, "
            f"gt_shape={gt_shape_equal}, gt_delta={gt_delta}"
        )
    return {
        "old_reference_pack": str(old_reference_pack(speaker, session).resolve()),
        "frame_timeline_equal": frame_equal,
        "ground_truth_shape_equal": gt_shape_equal,
        "ground_truth_max_abs_delta": gt_delta,
        "integer_frame_dtype": str(frames.dtype),
        "saved_fractional_frame_count": 0,
        "scored_fractional_frame_count": 0,
    }


def frame_rmse_mm(predicted: np.ndarray, labels: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    difference = predicted[:, indices].astype(np.float64) - labels[:, indices].astype(np.float64)
    return np.sqrt(np.mean(difference * difference, axis=(1, 2, 3))) * MM_PER_PIXEL


def compute_metrics(arrays: dict[str, np.ndarray], ground_truth: np.ndarray) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "modes": {},
        "per_class_mean_frame_rmse_mm": {},
    }
    for mode, indices in MODE_INDICES.items():
        stages: dict[str, Any] = {}
        raw_values = frame_rmse_mm(arrays["raw"], ground_truth, indices)
        raw_mean = float(np.mean(raw_values))
        for stage in STAGES:
            values = frame_rmse_mm(arrays[stage], ground_truth, indices)
            mean = float(np.mean(values))
            stages[stage] = {
                "mean_frame_rmse_mm": mean,
                "median_frame_rmse_mm": float(np.median(values)),
                "std_frame_rmse_mm": float(np.std(values)),
                "delta_vs_raw_mm": mean - raw_mean,
            }
        stages["affine_tps"]["delta_vs_affine_mm"] = (
            stages["affine_tps"]["mean_frame_rmse_mm"]
            - stages["affine"]["mean_frame_rmse_mm"]
        )
        payload["modes"][mode] = stages
    for stage in STAGES:
        difference = arrays[stage].astype(np.float64) - ground_truth.astype(np.float64)
        values = np.sqrt(np.mean(difference * difference, axis=(2, 3))) * MM_PER_PIXEL
        payload["per_class_mean_frame_rmse_mm"][stage] = {
            class_name: float(np.mean(values[:, index]))
            for index, class_name in enumerate(CLASSES)
        }
    return payload


def pack_path(output_root: Path, speaker: int, session: int, branch: str) -> Path:
    return output_root / f"P{speaker}" / f"S{session}" / f"{branch}.npz"


def summary_path(output_root: Path, speaker: int, session: int, branch: str) -> Path:
    return output_root / f"P{speaker}" / f"S{session}" / f"{branch}_summary.json"


def save_pack(
    path: Path,
    branch: str,
    inferred: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    alpha_to_asd2: float,
    rms_target: float | None,
) -> None:
    frames = np.asarray(inferred["frame_numbers"])
    if not np.issubdtype(frames.dtype, np.integer):
        raise AssertionError(f"Refusing non-integer frame dtype: {frames.dtype}")
    if not all(np.isfinite(arrays[stage]).all() for stage in STAGES):
        raise ValueError(f"Non-finite prediction in {branch}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing.npz")
    np.savez_compressed(
        temporary,
        branch=np.asarray(branch),
        frame_numbers=frames.astype(np.int32),
        phonemes=np.asarray(inferred["phonemes"]),
        overlap_counts=np.asarray(inferred["overlap_counts"]),
        predicted_raw=np.asarray(arrays["raw"], dtype=np.float32),
        predicted_after_affine=np.asarray(arrays["affine"], dtype=np.float32),
        predicted_after_affine_tps=np.asarray(arrays["affine_tps"], dtype=np.float32),
        ground_truth=np.asarray(inferred["ground_truth"], dtype=np.float32),
        classes=np.asarray(CLASSES, dtype="U64"),
        alpha_to_asd2_train=np.asarray(alpha_to_asd2, dtype=np.float32),
        target_rms=np.asarray(np.nan if rms_target is None else rms_target, dtype=np.float32),
        num_input_rows=np.asarray(inferred["num_input_rows"]),
        num_sequences=np.asarray(inferred["num_sequences"]),
        num_fractional_frames_discarded=np.asarray(
            int(inferred.get("num_fractional_frames_discarded", 0)), dtype=np.int64
        ),
        saved_fractional_frame_count=np.asarray(0, dtype=np.int64),
        scored_fractional_frame_count=np.asarray(0, dtype=np.int64),
        frame_policy=np.asarray(INTEGER_FRAME_POLICY),
    )
    temporary.replace(path)


def load_pack(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        frames = np.asarray(payload["frame_numbers"])
        result = {
            "branch": str(payload["branch"].item()),
            "frame_numbers": frames,
            "phonemes": np.asarray(payload["phonemes"]),
            "overlap_counts": np.asarray(payload["overlap_counts"]),
            "arrays": {
                "raw": np.asarray(payload["predicted_raw"], dtype=np.float32),
                "affine": np.asarray(payload["predicted_after_affine"], dtype=np.float32),
                "affine_tps": np.asarray(payload["predicted_after_affine_tps"], dtype=np.float32),
            },
            "ground_truth": np.asarray(payload["ground_truth"], dtype=np.float32),
            "classes": [str(value) for value in payload["classes"].tolist()],
            "num_input_rows": int(payload["num_input_rows"]),
            "num_sequences": int(payload["num_sequences"]),
            "num_fractional_frames_discarded": int(payload["num_fractional_frames_discarded"]),
        }
    if not np.issubdtype(frames.dtype, np.integer):
        raise ValueError(f"Packed frames are not integer typed: {path} ({frames.dtype})")
    if result["classes"] != list(CLASSES):
        raise ValueError(f"Class mismatch in {path}")
    return result


def inferred_from_pack(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_numbers": payload["frame_numbers"],
        "phonemes": payload["phonemes"],
        "overlap_counts": payload["overlap_counts"],
        "predicted_raw": payload["arrays"]["raw"],
        "ground_truth": payload["ground_truth"],
        "num_input_rows": payload["num_input_rows"],
        "num_sequences": payload["num_sequences"],
        "num_fractional_frames_discarded": payload["num_fractional_frames_discarded"],
    }


def write_branch_summary(
    args: argparse.Namespace,
    speaker: int,
    session: int,
    branch: str,
    inferred: dict[str, Any],
    arrays: dict[str, np.ndarray],
    population_audit: dict[str, Any],
    elapsed_seconds: float,
    *,
    alpha_to_asd2: float,
    rms_target: float | None,
    extraction_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frames = np.asarray(inferred["frame_numbers"], dtype=np.int32)
    summary = {
        "created_at": now(),
        "branch": branch,
        "speaker": speaker,
        "session": session,
        "cohort": "same_speaker_control" if speaker == 10 else "unseen_speaker",
        "num_unique_frames": int(len(frames)),
        "frame_min": int(frames.min()),
        "frame_max": int(frames.max()),
        "num_sequences": int(inferred["num_sequences"]),
        "num_input_rows": int(inferred["num_input_rows"]),
        "num_fractional_frames_discarded": int(
            inferred.get("num_fractional_frames_discarded", 0)
        ),
        "saved_fractional_frame_count": 0,
        "scored_fractional_frame_count": 0,
        "rendered_fractional_frame_count": 0,
        "frame_policy": INTEGER_FRAME_POLICY,
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "normalization_stats": str(args.normalization_stats.resolve()),
        "source_grid_reference": SOURCE_REFERENCE,
        "target_grid_reference": target_spec(speaker).label,
        "alpha_to_asd2_train": float(alpha_to_asd2),
        "target_rms": None if rms_target is None else float(rms_target),
        "contour_pack": str(pack_path(args.output_root, speaker, session, branch).resolve()),
        "population_audit": population_audit,
        "metrics": compute_metrics(arrays, np.asarray(inferred["ground_truth"])),
        "extraction_metadata": extraction_metadata,
        "elapsed_seconds": float(elapsed_seconds),
    }
    atomic_json(summary_path(args.output_root, speaker, session, branch), summary)
    return summary


def validate_args(args: argparse.Namespace) -> None:
    for path in (
        args.config,
        args.checkpoint,
        args.normalization_stats,
        args.source_pack,
        args.raw_cache_root,
        args.vtln_dir,
        OLD_GRID_ROOT,
        OLD_AUDIO_ROOT,
        OLD_ABLATION_ROOT,
        SOURCE_CASE_ROOT,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if not math.isclose(float(args.batch_size), round(float(args.batch_size)), abs_tol=0.0):
        raise ValueError("batch-size must be an integer")
    for speaker, session in SELECTION:
        raw_path = args.raw_cache_root / f"P{speaker}" / f"S{session}.pt"
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        exact_asd1_audio_paths(speaker, session)
        load_old_reference(speaker, session)


def input_tree_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    roots = (OLD_GRID_ROOT, OLD_AUDIO_ROOT, OLD_ABLATION_ROOT)
    snapshot = {
        "created_at": now(),
        "roots": {},
    }
    for root in roots:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry: dict[str, Any] = {
            "root": str(root.resolve()),
            "manifest_sha256": file_sha256(manifest_path),
            "selection": manifest.get("selection"),
            "total_integer_frames": manifest.get("total_integer_frames"),
        }
        if not args.skip_input_tree_hash:
            hashes = tree_hashes(root)
            canonical = json.dumps(hashes, sort_keys=True).encode("utf-8")
            entry["file_count"] = len(hashes)
            entry["files"] = hashes
            entry["tree_sha256"] = hashlib.sha256(canonical).hexdigest()
        snapshot["roots"][root.name] = entry
    return snapshot


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    transforms, grid_metadata = build_transforms(args)
    if set(transforms) != {speaker for speaker, _session in SELECTION}:
        raise AssertionError("Grid transform inventory mismatch")
    config = load_yaml_config(args.config)
    if list(config["classes"]) != list(CLASSES):
        raise ValueError("Runtime config class order is not the canonical 11-class order")
    if config.get("normalization_mode") != "train_global":
        raise ValueError("Expected train_global normalization")
    if config.get("normalization_std_policy") != "raw_positive":
        raise ValueError("Expected raw_positive normalization std policy")
    normalization = load_normalization(args.normalization_stats)
    if not all(np.isfinite(value).all() for value in normalization.values()):
        raise ValueError("Normalization stats contain non-finite values")
    if np.any(normalization["std_mfcc"] <= 0) or np.any(normalization["std_contour"] <= 0):
        raise ValueError("Normalization std must be strictly positive")

    train_audio_count = 0
    for bucket, sessions in config["train_sequences"].items():
        for session in sessions:
            exact_asd2_audio_paths(str(bucket), str(session))
            train_audio_count += 1
    if train_audio_count != 85:
        raise RuntimeError(f"Expected 85 ASD2 training-audio sessions, got {train_audio_count}")

    snapshot = input_tree_snapshot(args)
    atomic_json(args.output_root / "provenance" / "input_bundles_before.json", snapshot)
    source_npz = args.output_root / "provenance" / "asd2_source_grid_contours.npz"
    _grid, source_contours, _meta = build_source_grid(args.source_pack)
    source_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        source_npz,
        **{name: np.asarray(points, dtype=np.float32) for name, points in source_contours.items()},
    )
    git_state = {
        "repo_head": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "grid_transform_head": subprocess.check_output(
            ["git", "-C", str(GRID_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    payload = {
        "created_at": now(),
        "status": "passed",
        "inference_only": True,
        "training_launched": False,
        "selection": [f"P{s}/S{x}" for s, x in SELECTION],
        "unseen_selection": [f"P{s}/S{x}" for s, x in UNSEEN_SELECTION],
        "same_speaker_control": "P10/S14",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "normalization": str(args.normalization_stats.resolve()),
        "normalization_sha256": file_sha256(args.normalization_stats),
        "normalization_mode": config["normalization_mode"],
        "normalization_std_policy": config["normalization_std_policy"],
        "asd2_training_audio_sessions": train_audio_count,
        "grid": grid_metadata,
        "git": git_state,
        "old_bundle_snapshot": str(
            (args.output_root / "provenance" / "input_bundles_before.json").resolve()
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(args.output_root / "preflight.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def load_runtime(args: argparse.Namespace):
    config = load_yaml_config(args.config)
    normalization = load_normalization(args.normalization_stats)
    with Path(config["phonemesdir"]).open("r", encoding="utf-8") as handle:
        phonemes = json.load(handle)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This experiment requires OAR GPU inference; CPU fallback is forbidden")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the OAR allocation")
    model = load_model(config, args.checkpoint, device)
    return config, normalization, phonemes, device, model


def process_baseline(
    args: argparse.Namespace,
    pairs: tuple[tuple[int, int], ...],
    transforms: dict[int, dict[str, Any]],
    runtime: tuple[Any, ...],
) -> list[dict[str, Any]]:
    _config, normalization, phonemes, device, model = runtime
    summaries = []
    for speaker, session in pairs:
        output_pack = pack_path(args.output_root, speaker, session, "baseline")
        started = time.monotonic()
        if output_pack.is_file() and not args.force:
            packed = load_pack(output_pack)
            inferred = inferred_from_pack(packed)
            arrays = packed["arrays"]
            print(f"REUSE baseline P{speaker}/S{session}", flush=True)
        else:
            raw_path = args.raw_cache_root / f"P{speaker}" / f"S{session}.pt"
            inferred = retain_integer_inferred(
                infer_session(model, device, raw_path, normalization, phonemes, args.batch_size)
            )
            affine, final = transform_contour_batch(
                inferred["predicted_raw"], transforms[speaker], args.transform_frame_batch
            )
            arrays = {
                "raw": inferred["predicted_raw"],
                "affine": affine,
                "affine_tps": final,
            }
            save_pack(
                output_pack,
                "baseline",
                inferred,
                arrays,
                alpha_to_asd2=1.0,
                rms_target=None,
            )
        population = validate_inferred_against_old(speaker, session, inferred)
        summary = write_branch_summary(
            args,
            speaker,
            session,
            "baseline",
            inferred,
            arrays,
            population,
            time.monotonic() - started,
            alpha_to_asd2=1.0,
            rms_target=None,
        )
        summaries.append(summary)
        metric = summary["metrics"]["modes"]["all_11"]
        print(
            f"DONE baseline P{speaker}/S{session}: {summary['num_unique_frames']} integer frames, "
            f"raw={metric['raw']['mean_frame_rmse_mm']:.3f}, "
            f"affine={metric['affine']['mean_frame_rmse_mm']:.3f}, "
            f"TPS={metric['affine_tps']['mean_frame_rmse_mm']:.3f} mm",
            flush=True,
        )
    return summaries


def audio_reference_index(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for bucket, sessions in config["train_sequences"].items():
        for session in sessions:
            wav, textgrid = exact_asd2_audio_paths(str(bucket), str(session))
            rows.append(
                {
                    "speaker_id": "ASD2_TRAIN",
                    "session_id": f"{bucket}_{session}",
                    "wav_path": str(wav),
                    "textgrid_path": str(textgrid),
                }
            )
    for speaker, session in SELECTION:
        wav, textgrid = exact_asd1_audio_paths(speaker, session)
        rows.append(
            {
                "speaker_id": f"P{speaker}",
                "session_id": f"S{session}",
                "wav_path": str(wav),
                "textgrid_path": str(textgrid),
            }
        )
    return pd.DataFrame(rows)


def estimate_alphas(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, float]:
    output_dir = args.output_root / "audio_normalization"
    summary_path_value = output_dir / "alpha_to_asd2_train.json"
    if summary_path_value.is_file() and not args.force:
        payload = json.loads(summary_path_value.read_text(encoding="utf-8"))
        alphas = {str(key): float(value) for key, value in payload["alpha_to_asd2_train"].items()}
        if {f"P{speaker}" for speaker, _session in SELECTION} <= set(alphas):
            print(f"REUSE audio alphas: {summary_path_value}", flush=True)
            return alphas

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_config = AudioNormConfig(
        data_root=ASD2_ROOT,
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
        target_mode="ASD2_TRAIN",
        save_intermediate_features=False,
        save_models=False,
        save_figures=False,
    )
    index = audio_reference_index(config)
    if int((index["speaker_id"] == "ASD2_TRAIN").sum()) != 85:
        raise RuntimeError("ASD2 audio reference must contain exactly the 85 train sessions")
    index.to_csv(output_dir / "data_index.csv", index=False)
    print(f"AUDIO NORM: extracting {len(index)} WAV files (85 ASD2 train + 9 targets)", flush=True)
    extractor = FeatureExtractor(audio_config)
    payloads = extractor.extract(index)
    reference_payloads = [item for item in payloads if item["speaker"] == "ASD2_TRAIN"]
    reference_features = np.vstack(
        [np.asarray(item["rms_mfcc39"], dtype=np.float32) for item in reference_payloads]
    )
    global_mean, global_std = fit_cmvn(reference_features)
    reference_norm = apply_cmvn(reference_features, global_mean, global_std)
    gmms, gmm_table = fit_speaker_gmms(
        {"ASD2_TRAIN": reference_norm}, audio_config.to_helper_config(), model_dir=None
    )
    gmm_table.to_csv(output_dir / "asd2_reference_gmm_summary.csv", index=False)
    target_gmm = gmms["ASD2_TRAIN"]
    reference_sample, _ = sample_rows(
        reference_norm,
        min(20_000, len(reference_norm)),
        audio_config.random_state,
    )
    reference_self_score = float(score_gmm(target_gmm, reference_sample))
    np.savez_compressed(
        output_dir / "asd2_training_audio_reference_cmvn.npz",
        mean=global_mean,
        std=global_std,
        num_training_sessions=np.asarray(85, dtype=np.int32),
        num_reference_frames=np.asarray(len(reference_features), dtype=np.int64),
    )

    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    alphas: dict[str, float] = {}
    for speaker, session in SELECTION:
        speaker_name = f"P{speaker}"
        source_payloads = [item for item in payloads if item["speaker"] == speaker_name]
        if len(source_payloads) != 1:
            raise RuntimeError(f"Expected one target payload for {speaker_name}, got {len(source_payloads)}")
        best_alpha = 1.0
        best_score = -np.inf
        base_score = float("nan")
        for alpha in audio_config.alpha_grid:
            chunks = []
            for item in source_payloads:
                features, _ = extract_vtln_mfcc39(
                    np.asarray(item["wav_rms"]),
                    int(item["sr"]),
                    alpha=float(alpha),
                    config=audio_config.to_helper_config(),
                    f_high=VTLN_F_HIGH,
                )
                count = min(len(features), int(item["n_total_frames"]))
                mask = np.asarray(item["speech_mask"][:count], dtype=bool)
                if mask.any():
                    chunks.append(features[:count][mask])
            combined = np.vstack(chunks).astype(np.float32)
            combined, _ = sample_rows(
                combined, min(20_000, len(combined)), audio_config.random_state
            )
            normalized = apply_cmvn(combined, global_mean, global_std)
            score = float(score_gmm(target_gmm, normalized))
            curve_rows.append(
                {
                    "source_speaker": speaker_name,
                    "source_session": f"S{session}",
                    "target_reference": "ASD2_TRAIN_85_SESSIONS",
                    "alpha": float(alpha),
                    "score": score,
                    "n_frames": int(len(combined)),
                }
            )
            if math.isclose(float(alpha), 1.0, rel_tol=0.0, abs_tol=1e-9):
                base_score = score
            if score > best_score:
                best_score = score
                best_alpha = float(alpha)
        alphas[speaker_name] = best_alpha
        summary_rows.append(
            {
                "source_speaker": speaker_name,
                "source_session": f"S{session}",
                "target_reference": "ASD2_TRAIN_85_SESSIONS",
                "alpha_best": best_alpha,
                "score_alpha_1": base_score,
                "score_best": best_score,
                "score_gain": best_score - base_score,
                "reference_self_score": reference_self_score,
            }
        )
        print(
            f"AUDIO NORM {speaker_name}/S{session}->ASD2_TRAIN: alpha={best_alpha:.3f}, "
            f"gain={best_score - base_score:+.4f}",
            flush=True,
        )
    write_csv(output_dir / "alpha_curves_to_asd2_train.csv", curve_rows)
    write_csv(output_dir / "alpha_summary_to_asd2_train.csv", summary_rows)
    metadata = {
        "created_at": now(),
        "method": "RMS(0.03)+GMM VTLN alpha search with Inversion_SI frontend re-extraction",
        "target_reference": "ASD2 training audio only",
        "target_reference_session_count": 85,
        "target_contours_or_labels_used": False,
        "target_rms": TARGET_RMS,
        "alpha_grid": [float(value) for value in audio_config.alpha_grid],
        "vtln_f_low": VTLN_F_LOW,
        "vtln_f_high": VTLN_F_HIGH,
        "global_cmvn_fit_population": "ASD2 train audio only",
        "alpha_to_asd2_train": alphas,
    }
    atomic_json(summary_path_value, metadata)
    return alphas


def asd1_audio_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["dataset_type"] = "asd1"
    result["datadir"] = str(RAW_ROOT)
    result["asd1_datadir"] = str(RAW_ROOT)
    result["asd1_annotation_dir"] = str(BF_ROOT)
    return result


def process_audio_branch(
    args: argparse.Namespace,
    pairs: tuple[tuple[int, int], ...],
    transforms: dict[int, dict[str, Any]],
    runtime: tuple[Any, ...],
    branch: str,
    alphas: dict[str, float],
) -> list[dict[str, Any]]:
    config, normalization, phonemes, device, model = runtime
    audio_config = asd1_audio_config(config)
    summaries = []
    for speaker, session in pairs:
        started = time.monotonic()
        alpha = float(alphas[f"P{speaker}"])
        if branch == "rms_vtln":
            applied_alpha, rms_target = alpha, TARGET_RMS
        elif branch == "rms_only":
            applied_alpha, rms_target = 1.0, TARGET_RMS
        elif branch == "vtln_only":
            applied_alpha, rms_target = alpha, None
        else:
            raise ValueError(branch)
        output_pack = pack_path(args.output_root, speaker, session, branch)
        extraction_path = (
            args.output_root / f"P{speaker}" / f"S{session}" / f"{branch}_audio_extraction.json"
        )
        alignment_path = (
            args.output_root / f"P{speaker}" / f"S{session}" / f"{branch}_chunk_alignment.csv"
        )
        if output_pack.is_file() and extraction_path.is_file() and not args.force:
            packed = load_pack(output_pack)
            inferred = inferred_from_pack(packed)
            arrays = packed["arrays"]
            extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
            print(f"REUSE {branch} P{speaker}/S{session}", flush=True)
        else:
            raw_path = args.raw_cache_root / f"P{speaker}" / f"S{session}.pt"
            raw = torch.load(raw_path, map_location="cpu", weights_only=False)["raw"]
            feature_chunks, alignment, extraction = build_audio_normalized_chunks(
                audio_config,
                speaker,
                session,
                applied_alpha,
                raw["features"],
                rms_target=rms_target,
            )
            extraction.pop("vtln_alpha_to_p7", None)
            extraction.update(
                {
                    "branch": branch,
                    "vtln_alpha_to_asd2_train": applied_alpha,
                    "audio_reference": "ASD2 train audio only (85 sessions)",
                    "target_contours_or_labels_used_for_audio_normalization": False,
                    "playback_audio": "original target WAV; normalized waveform is model-input only",
                    "match_mae_limit": MATCH_MAE_LIMIT,
                }
            )
            write_csv(alignment_path, alignment)
            atomic_json(extraction_path, extraction)
            inferred = retain_integer_inferred(
                infer_with_features(
                    model,
                    device,
                    raw,
                    feature_chunks,
                    normalization,
                    phonemes,
                    args.batch_size,
                )
            )
            affine, final = transform_contour_batch(
                inferred["predicted_raw"], transforms[speaker], args.transform_frame_batch
            )
            arrays = {
                "raw": inferred["predicted_raw"],
                "affine": affine,
                "affine_tps": final,
            }
            save_pack(
                output_pack,
                branch,
                inferred,
                arrays,
                alpha_to_asd2=applied_alpha,
                rms_target=rms_target,
            )
        population = validate_inferred_against_old(speaker, session, inferred)
        baseline = load_pack(pack_path(args.output_root, speaker, session, "baseline"))
        if not np.array_equal(inferred["frame_numbers"], baseline["frame_numbers"]):
            raise RuntimeError(f"{branch}/baseline timeline mismatch P{speaker}/S{session}")
        gt_delta = float(np.max(np.abs(inferred["ground_truth"] - baseline["ground_truth"])))
        if gt_delta > 1e-5:
            raise RuntimeError(f"{branch}/baseline GT mismatch P{speaker}/S{session}: {gt_delta}")
        population["ground_truth_max_abs_delta_vs_new_baseline"] = gt_delta
        summary = write_branch_summary(
            args,
            speaker,
            session,
            branch,
            inferred,
            arrays,
            population,
            time.monotonic() - started,
            alpha_to_asd2=applied_alpha,
            rms_target=rms_target,
            extraction_metadata=extraction,
        )
        summaries.append(summary)
        final_metric = summary["metrics"]["modes"]["all_11"]["affine_tps"][
            "mean_frame_rmse_mm"
        ]
        print(
            f"DONE {branch} P{speaker}/S{session}: alpha={applied_alpha:.3f}, "
            f"final={final_metric:.3f} mm",
            flush=True,
        )
    return summaries


def load_summaries(
    output_root: Path,
    branch: str,
    pairs: tuple[tuple[int, int], ...] = SELECTION,
) -> list[dict[str, Any]]:
    rows = []
    for speaker, session in pairs:
        path = summary_path(output_root, speaker, session, branch)
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def weighted_final(rows: list[dict[str, Any]]) -> float:
    total = sum(int(row["num_unique_frames"]) for row in rows)
    return sum(
        int(row["num_unique_frames"])
        * float(
            row["metrics"]["modes"]["all_11"]["affine_tps"]["mean_frame_rmse_mm"]
        )
        for row in rows
    ) / total


def evaluate_ablation_gate(args: argparse.Namespace) -> dict[str, Any]:
    baseline = load_summaries(args.output_root, "baseline")
    combined = load_summaries(args.output_root, "rms_vtln")
    deltas = []
    for baseline_row, audio_row in zip(baseline, combined):
        if (baseline_row["speaker"], baseline_row["session"]) != (
            audio_row["speaker"],
            audio_row["session"],
        ):
            raise RuntimeError("Gate session order mismatch")
        base = float(
            baseline_row["metrics"]["modes"]["all_11"]["affine_tps"][
                "mean_frame_rmse_mm"
            ]
        )
        audio = float(
            audio_row["metrics"]["modes"]["all_11"]["affine_tps"][
                "mean_frame_rmse_mm"
            ]
        )
        deltas.append(
            {
                "speaker": f"P{baseline_row['speaker']}",
                "session": f"S{baseline_row['session']}",
                "frames": int(baseline_row["num_unique_frames"]),
                "baseline_affine_tps_all11_mm": base,
                "rms_vtln_affine_tps_all11_mm": audio,
                "delta_mm": audio - base,
            }
        )
    total_frames = sum(int(row["frames"]) for row in deltas)
    aggregate_delta = sum(int(row["frames"]) * float(row["delta_mm"]) for row in deltas) / total_frames
    any_large_session = any(abs(float(row["delta_mm"])) >= GATE_SESSION_THRESHOLD_MM for row in deltas)
    signs = {int(np.sign(float(row["delta_mm"]))) for row in deltas if not math.isclose(float(row["delta_mm"]), 0.0)}
    mixed_signs = -1 in signs and 1 in signs
    aggregate_trigger = abs(aggregate_delta) >= GATE_AGGREGATE_THRESHOLD_MM
    triggered = aggregate_trigger or any_large_session or mixed_signs
    payload = {
        "created_at": now(),
        "status": "triggered" if triggered else "not_triggered",
        "triggered": triggered,
        "comparison": "new ASD2-model RMS+VTLN minus new ASD2-model baseline after affine+TPS",
        "metric_mode": "all_11",
        "population": "all nine fixed sessions (P10 retained as same-speaker control)",
        "frame_weighted_aggregate_delta_mm": aggregate_delta,
        "aggregate_threshold_abs_mm": GATE_AGGREGATE_THRESHOLD_MM,
        "aggregate_trigger": aggregate_trigger,
        "session_threshold_abs_mm": GATE_SESSION_THRESHOLD_MM,
        "any_session_trigger": any_large_session,
        "mixed_effect_signs_trigger": mixed_signs,
        "session_deltas": deltas,
    }
    atomic_json(args.output_root / "ablation_gate.json", payload)
    write_csv(args.output_root / "ablation_gate_session_deltas.csv", deltas)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def run_manifest(args: argparse.Namespace, gate: dict[str, Any] | None) -> None:
    sessions = []
    for speaker, session in SELECTION:
        branches = {}
        for branch in BRANCHES:
            path = pack_path(args.output_root, speaker, session, branch)
            if path.is_file():
                packed = load_pack(path)
                branches[branch] = {
                    "pack": str(path.resolve()),
                    "frames": int(len(packed["frame_numbers"])),
                    "frame_min": int(packed["frame_numbers"].min()),
                    "frame_max": int(packed["frame_numbers"].max()),
                    "saved_fractional_frame_count": 0,
                    "scored_fractional_frame_count": 0,
                }
        sessions.append(
            {
                "speaker": speaker,
                "session": session,
                "cohort": "same_speaker_control" if speaker == 10 else "unseen_speaker",
                "branches": branches,
            }
        )
    payload = {
        "created_at": now(),
        "experiment": "asd2_epoch211_on_fixed_asd1_nine_session_grid_audio",
        "inference_only": True,
        "training_launched": False,
        "selection": [f"P{s}/S{x}" for s, x in SELECTION],
        "unseen_selection": [f"P{s}/S{x}" for s, x in UNSEEN_SELECTION],
        "same_speaker_control": "P10/S14",
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "normalization": str(args.normalization_stats.resolve()),
        "source_grid_reference": SOURCE_REFERENCE,
        "frame_policy": INTEGER_FRAME_POLICY,
        "saved_fractional_frame_count": 0,
        "scored_fractional_frame_count": 0,
        "rendered_fractional_frame_count": 0,
        "expected_scored_integer_frames_per_branch": 8585,
        "ablation_gate": gate,
        "sessions": sessions,
    }
    atomic_json(args.output_root / "inference_manifest.json", payload)


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    pairs = selected_pairs(args)
    if args.phase in ("preflight", "all"):
        run_preflight(args)
        if args.phase == "preflight":
            return

    validate_args(args)
    transforms, _grid_metadata = build_transforms(args)
    runtime = load_runtime(args)
    gate = None
    if args.phase in ("baseline", "all"):
        process_baseline(args, pairs, transforms, runtime)
        if args.phase == "baseline":
            run_manifest(args, None)
            return
    if args.phase in ("audio", "all"):
        if pairs != SELECTION:
            raise ValueError("Audio phase requires the complete fixed nine-session selection")
        alphas = estimate_alphas(args, runtime[0])
        process_audio_branch(args, pairs, transforms, runtime, "rms_vtln", alphas)
        gate = evaluate_ablation_gate(args)
        if args.phase == "audio":
            run_manifest(args, gate)
            return
    if args.phase in ("ablation", "all"):
        if pairs != SELECTION:
            raise ValueError("Ablation phase requires the complete fixed nine-session selection")
        gate_path = args.output_root / "ablation_gate.json"
        if gate is None:
            if not gate_path.is_file():
                gate = evaluate_ablation_gate(args)
            else:
                gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate["triggered"]:
            alphas = estimate_alphas(args, runtime[0])
            process_audio_branch(args, pairs, transforms, runtime, "rms_only", alphas)
            process_audio_branch(args, pairs, transforms, runtime, "vtln_only", alphas)
        else:
            print("ABLATION NOT TRIGGERED; RMS-only and VTLN-only were not run", flush=True)
    run_manifest(args, gate)


if __name__ == "__main__":
    main()
