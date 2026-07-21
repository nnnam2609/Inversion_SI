#!/usr/bin/env python3
"""Rebuild the three selected non-P7 result bundles from existing contour packs.

This entrypoint deliberately has no model, checkpoint, acoustic-feature, or
grid-transform execution path.  It filters already-saved prediction packs to
integer MRI frame numbers, recomputes every metric/report, renders from the
matching integer DICOM frame, audits the staged bundles, and can atomically
replace the three current result directories while preserving backups.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]

from src.utils.colors import COLORS  # noqa: E402
from src.utils.mri_rendering import (  # noqa: E402
    build_filename_dicom_index,
    load_or_build_mri_cache,
)
from src.utils.video_rendering import (  # noqa: E402
    MM_PER_PIXEL,
    draw_dashed_polyline,
    rgb_to_bgr255,
    scale_points,
)


FRAME_POLICY = "NEVER render fractional MRI frames; integer frames only"
INTEGER_ATOL = 1e-4
SELECTION = ((1, 16), (2, 9), (3, 14), (4, 4), (5, 6), (6, 8), (8, 2), (9, 5), (10, 14))
STAGES = ("raw", "affine", "affine_tps")
STAGE_LABELS = {"raw": "raw", "affine": "affine", "affine_tps": "affine + TPS"}
EXCLUDED_CLASSES = ("vocal-folds", "thyroid-cartilage", "epiglottis")
MODES = ("all_11", "without_laryngeal_3")
VARIANTS = ("baseline", "rms_only", "vtln_only", "rms_vtln")
VARIANT_LABELS = {
    "baseline": "Baseline",
    "rms_only": "RMS-only",
    "vtln_only": "VTLN-only",
    "rms_vtln": "RMS+VTLN",
}
INFO_HEIGHT = 106
SEPARATOR = 8
RAW_ROOT = Path(
    "/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/"
    "ArtSpeech_Database_1_raw"
)

CURRENT_BASELINE = REPO_ROOT / "results/p7_selected_nonp7_sessions_gridnorm_20260718"
CURRENT_AUDIO = REPO_ROOT / "results/p7_selected_nonp7_sessions_audio_gridnorm_20260718"
CURRENT_ABLATION = REPO_ROOT / "results/p7_selected_nonp7_sessions_audio_ablation_20260718"
STAGING_BASELINE = REPO_ROOT / "results/.staging_p7_selected_nonp7_sessions_gridnorm_integer_20260718"
STAGING_AUDIO = REPO_ROOT / "results/.staging_p7_selected_nonp7_sessions_audio_gridnorm_integer_20260718"
STAGING_ABLATION = REPO_ROOT / "results/.staging_p7_selected_nonp7_sessions_audio_ablation_integer_20260718"
BACKUP_BASELINE = REPO_ROOT / "results/p7_selected_nonp7_sessions_gridnorm_20260718_with_fractional_backup"
BACKUP_AUDIO = REPO_ROOT / "results/p7_selected_nonp7_sessions_audio_gridnorm_20260718_with_fractional_backup"
BACKUP_ABLATION = REPO_ROOT / "results/p7_selected_nonp7_sessions_audio_ablation_20260718_with_fractional_backup"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-baseline-root", type=Path, default=CURRENT_BASELINE)
    parser.add_argument("--source-audio-root", type=Path, default=CURRENT_AUDIO)
    parser.add_argument("--source-ablation-root", type=Path, default=CURRENT_ABLATION)
    parser.add_argument("--staging-baseline-root", type=Path, default=STAGING_BASELINE)
    parser.add_argument("--staging-audio-root", type=Path, default=STAGING_AUDIO)
    parser.add_argument("--staging-ablation-root", type=Path, default=STAGING_ABLATION)
    parser.add_argument("--backup-baseline-root", type=Path, default=BACKUP_BASELINE)
    parser.add_argument("--backup-audio-root", type=Path, default=BACKUP_AUDIO)
    parser.add_argument("--backup-ablation-root", type=Path, default=BACKUP_ABLATION)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--mri-workers", type=int, default=4)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def relative_session_path(speaker: int, session: int, filename: str) -> str:
    return f"P{speaker}/S{session}/{filename}"


def integer_mask(frame_numbers: np.ndarray) -> np.ndarray:
    frames = np.asarray(frame_numbers, dtype=np.float64)
    return np.isclose(frames, np.rint(frames), atol=INTEGER_ATOL)


def assert_integer_frames(frame_numbers: np.ndarray, context: str) -> np.ndarray:
    frames = np.asarray(frame_numbers)
    if frames.ndim != 1 or len(frames) == 0:
        raise ValueError(f"{context}: expected a non-empty 1D frame timeline")
    if not integer_mask(frames).all():
        bad = frames[~integer_mask(frames)][:10]
        raise ValueError(f"{context}: fractional frames remain: {bad.tolist()}")
    integer = np.rint(frames).astype(np.int32)
    if len(np.unique(integer)) != len(integer):
        raise ValueError(f"{context}: duplicate integer frame numbers after filtering")
    return integer


def arrays_for_kind(payload: dict[str, np.ndarray], kind: str) -> dict[str, np.ndarray]:
    if kind == "audio":
        keys = {
            "raw": "predicted_audio_raw",
            "affine": "predicted_audio_after_affine",
            "affine_tps": "predicted_audio_after_affine_tps",
        }
    else:
        keys = {
            "raw": "predicted_raw",
            "affine": "predicted_after_affine",
            "affine_tps": "predicted_after_affine_tps",
        }
    missing = [key for key in (*keys.values(), "ground_truth", "frame_numbers") if key not in payload]
    if missing:
        raise KeyError(f"Pack kind={kind} is missing {missing}")
    return {stage: np.asarray(payload[key], dtype=np.float32) for stage, key in keys.items()}


def frame_rmse_mm(predicted: np.ndarray, labels: np.ndarray, indices: list[int]) -> np.ndarray:
    difference = predicted[:, indices].astype(np.float64) - labels[:, indices].astype(np.float64)
    return np.sqrt(np.mean(difference * difference, axis=(1, 2, 3))) * MM_PER_PIXEL


def metric_payload(
    arrays: dict[str, np.ndarray], ground_truth: np.ndarray, classes: list[str]
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    excluded = set(EXCLUDED_CLASSES)
    mode_indices = {
        "all_11": list(range(len(classes))),
        "without_laryngeal_3": [index for index, name in enumerate(classes) if name not in excluded],
    }
    frame_metrics: dict[str, dict[str, np.ndarray]] = {}
    summary: dict[str, Any] = {
        "excluded_classes": list(EXCLUDED_CLASSES),
        "kept_classes_without_laryngeal_3": [name for name in classes if name not in excluded],
        "modes": {},
        "per_class_mean_frame_rmse_mm": {},
    }
    for mode, indices in mode_indices.items():
        frame_metrics[mode] = {
            stage: frame_rmse_mm(arrays[stage], ground_truth, indices) for stage in STAGES
        }
        stage_summary: dict[str, Any] = {}
        raw_mean = float(np.mean(frame_metrics[mode]["raw"]))
        for stage in STAGES:
            values = frame_metrics[mode][stage]
            mean_value = float(np.mean(values))
            diff = arrays[stage][:, indices].astype(np.float64) - ground_truth[:, indices].astype(np.float64)
            stage_summary[stage] = {
                "mean_frame_rmse_mm": mean_value,
                "global_point_rmse_mm": float(np.sqrt(np.mean(diff * diff)) * MM_PER_PIXEL),
                "delta_vs_raw_mm": mean_value - raw_mean,
                "percent_vs_raw": 100.0 * (mean_value - raw_mean) / raw_mean if raw_mean else None,
            }
        stage_summary["affine_tps"]["delta_vs_affine_mm"] = (
            stage_summary["affine_tps"]["mean_frame_rmse_mm"]
            - stage_summary["affine"]["mean_frame_rmse_mm"]
        )
        summary["modes"][mode] = stage_summary
    for stage in STAGES:
        diff = arrays[stage].astype(np.float64) - ground_truth.astype(np.float64)
        per_class = np.sqrt(np.mean(diff * diff, axis=(2, 3))) * MM_PER_PIXEL
        summary["per_class_mean_frame_rmse_mm"][stage] = {
            name: float(np.mean(per_class[:, index])) for index, name in enumerate(classes)
        }
    return summary, frame_metrics


def filter_and_save_pack(source: Path, destination: Path, kind: str) -> dict[str, Any]:
    with np.load(source, allow_pickle=False) as archive:
        original = {name: np.asarray(archive[name]) for name in archive.files}
    frames = np.asarray(original["frame_numbers"], dtype=np.float64)
    mask = integer_mask(frames)
    source_count = int(len(frames))
    integer_count = int(np.count_nonzero(mask))
    fractional_count = source_count - integer_count
    if integer_count == 0:
        raise RuntimeError(f"No integer frames in {source}")
    classes = [str(value) for value in original["classes"].tolist()]
    source_arrays = arrays_for_kind(original, kind)
    source_ground_truth = np.asarray(original["ground_truth"], dtype=np.float32)
    source_metrics, _ = metric_payload(source_arrays, source_ground_truth, classes)

    filtered: dict[str, np.ndarray] = {}
    for name, values in original.items():
        if name == "frame_numbers":
            filtered[name] = np.rint(frames[mask]).astype(np.int32)
        elif values.ndim > 0 and values.shape[0] == source_count:
            filtered[name] = values[mask]
        else:
            filtered[name] = values
    filtered["frame_policy"] = np.asarray(FRAME_POLICY)
    filtered["num_source_frames"] = np.asarray(source_count, dtype=np.int64)
    filtered["num_fractional_frames_discarded"] = np.asarray(fractional_count, dtype=np.int64)
    filtered["saved_fractional_frame_count"] = np.asarray(0, dtype=np.int64)
    filtered["scored_fractional_frame_count"] = np.asarray(0, dtype=np.int64)
    filtered["rendered_fractional_frame_count"] = np.asarray(0, dtype=np.int64)
    filtered["integer_mask_atol"] = np.asarray(INTEGER_ATOL, dtype=np.float64)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **filtered)

    arrays = arrays_for_kind(filtered, kind)
    ground_truth = np.asarray(filtered["ground_truth"], dtype=np.float32)
    frame_numbers = assert_integer_frames(filtered["frame_numbers"], str(destination))
    metrics, frame_metrics = metric_payload(arrays, ground_truth, classes)
    return {
        "source": str(source),
        "path": str(destination),
        "kind": kind,
        "arrays": arrays,
        "ground_truth": ground_truth,
        "frame_numbers": frame_numbers,
        "phonemes": np.asarray(filtered["phonemes"]),
        "overlap_counts": np.asarray(filtered["overlap_counts"]),
        "classes": classes,
        "source_count": source_count,
        "integer_count": integer_count,
        "fractional_count": fractional_count,
        "metrics": metrics,
        "source_metrics": source_metrics,
        "frame_metrics": frame_metrics,
    }


def assert_aligned(reference: dict[str, Any], candidate: dict[str, Any], context: str) -> None:
    if not np.array_equal(reference["frame_numbers"], candidate["frame_numbers"]):
        raise RuntimeError(f"{context}: frame timelines differ")
    if not np.array_equal(reference["phonemes"], candidate["phonemes"]):
        raise RuntimeError(f"{context}: phoneme timelines differ")
    if not np.array_equal(reference["overlap_counts"], candidate["overlap_counts"]):
        raise RuntimeError(f"{context}: overlap-count timelines differ")
    if reference["ground_truth"].shape != candidate["ground_truth"].shape:
        raise RuntimeError(f"{context}: ground-truth shapes differ")
    maximum = float(np.max(np.abs(reference["ground_truth"] - candidate["ground_truth"])))
    if maximum > 1e-5:
        raise RuntimeError(f"{context}: ground truth differs, max_abs={maximum}")
    if reference["classes"] != candidate["classes"]:
        raise RuntimeError(f"{context}: class lists differ")


def frame_token(frame_number: float | int) -> str:
    value = float(frame_number)
    rounded = int(round(value))
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=INTEGER_ATOL):
        raise ValueError(f"Fractional frame token forbidden: {value}")
    return f"{rounded:04d}"


def draw_panel(
    image: np.ndarray,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    classes: list[str],
    title: str,
    frame_number: int,
    phoneme: str,
    all_rmse: float,
    without_rmse: float,
    scale: int,
) -> np.ndarray:
    size = 136 * scale
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_bgr = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((size + INFO_HEIGHT, size, 3), 15, dtype=np.uint8)
    canvas[INFO_HEIGHT:] = image_bgr
    for index, class_name in enumerate(classes):
        color = rgb_to_bgr255(COLORS.get(class_name, "white"))
        gt_points = scale_points(ground_truth[index], scale)
        pred_points = scale_points(predicted[index], scale)
        gt_points[:, 1] += INFO_HEIGHT
        pred_points[:, 1] += INFO_HEIGHT
        cv2.polylines(canvas, [gt_points], False, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.polylines(canvas, [gt_points], False, color, 1, cv2.LINE_AA)
        draw_dashed_polyline(canvas, pred_points, (0, 0, 0), 2, dash_length=7, gap_length=9)
        draw_dashed_polyline(canvas, pred_points, color, 1, dash_length=7, gap_length=9)
    lines = (
        title,
        f"frame {frame_token(frame_number)} | {phoneme}",
        f"RMSE all 11: {all_rmse:.3f} mm",
        f"RMSE without 3: {without_rmse:.3f} mm",
        "solid GT | dashed prediction",
    )
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (8, 17 + 19 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return canvas


def render_video(
    path: Path,
    branches: list[tuple[str, dict[str, Any]]],
    mri_cache: dict[int, np.ndarray],
    fps: float,
    scale: int,
) -> None:
    reference = branches[0][1]
    frames = assert_integer_frames(reference["frame_numbers"], str(path))
    for label, branch in branches[1:]:
        assert_aligned(reference, branch, f"render {path.name} branch {label}")
    panel_width = 136 * scale
    panel_height = panel_width + INFO_HEIGHT
    width = panel_width * 3 + SEPARATOR * 2
    height = panel_height * len(branches) + SEPARATOR * (len(branches) - 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.writing.mp4"
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open {temporary}")
    try:
        for frame_index, frame_number in enumerate(frames):
            image = mri_cache[int(frame_number)]
            canvas = np.full((height, width, 3), 15, dtype=np.uint8)
            for row_index, (branch_label, branch) in enumerate(branches):
                for column, stage in enumerate(STAGES):
                    panel = draw_panel(
                        image,
                        branch["arrays"][stage][frame_index],
                        reference["ground_truth"][frame_index],
                        reference["classes"],
                        f"{branch_label}: {STAGE_LABELS[stage]}",
                        int(frame_number),
                        str(reference["phonemes"][frame_index]),
                        float(branch["frame_metrics"]["all_11"][stage][frame_index]),
                        float(branch["frame_metrics"]["without_laryngeal_3"][stage][frame_index]),
                        scale,
                    )
                    x = column * (panel_width + SEPARATOR)
                    y = row_index * (panel_height + SEPARATOR)
                    canvas[y : y + panel_height, x : x + panel_width] = panel
            writer.write(canvas)
    finally:
        writer.release()
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_baseline_frame_csv(path: Path, branch: dict[str, Any]) -> None:
    rows = []
    for index, frame in enumerate(branch["frame_numbers"]):
        row: dict[str, Any] = {
            "frame": frame_token(frame),
            "frame_number": int(frame),
            "phoneme": str(branch["phonemes"][index]),
        }
        for mode in MODES:
            for stage in STAGES:
                row[f"{stage}_{mode}_rmse_mm"] = float(branch["frame_metrics"][mode][stage][index])
        rows.append(row)
    write_csv(path, rows)


def write_audio_frame_csv(path: Path, baseline: dict[str, Any], audio: dict[str, Any]) -> None:
    rows = []
    for index, frame in enumerate(baseline["frame_numbers"]):
        row: dict[str, Any] = {
            "frame": frame_token(frame),
            "frame_number": int(frame),
            "phoneme": str(baseline["phonemes"][index]),
        }
        for mode in MODES:
            for stage in STAGES:
                base = float(baseline["frame_metrics"][mode][stage][index])
                normalized = float(audio["frame_metrics"][mode][stage][index])
                row[f"baseline_{stage}_{mode}_rmse_mm"] = base
                row[f"audio_normalized_{stage}_{mode}_rmse_mm"] = normalized
                row[f"audio_minus_baseline_{stage}_{mode}_mm"] = normalized - base
        rows.append(row)
    write_csv(path, rows)


def write_ablation_frame_csv(path: Path, branches: dict[str, dict[str, Any]]) -> None:
    reference = branches["baseline"]
    rows = []
    for index, frame in enumerate(reference["frame_numbers"]):
        row: dict[str, Any] = {
            "frame": frame_token(frame),
            "frame_number": int(frame),
            "phoneme": str(reference["phonemes"][index]),
        }
        for mode in MODES:
            for variant in VARIANTS:
                for stage in STAGES:
                    row[f"{variant}_{stage}_{mode}_rmse_mm"] = float(
                        branches[variant]["frame_metrics"][mode][stage][index]
                    )
        rows.append(row)
    write_csv(path, rows)


def copy_if_present(source: Path, destination: Path) -> None:
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def required_zero_fields(fractional_count: int) -> dict[str, Any]:
    return {
        "frame_policy": FRAME_POLICY,
        "rendered_fractional_frame_count": 0,
        "scored_fractional_frame_count": 0,
        "saved_fractional_frame_count": 0,
        "num_fractional_frames_discarded": int(fractional_count),
    }


def worker_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        name: str(getattr(args, name))
        for name in (
            "source_baseline_root",
            "source_audio_root",
            "source_ablation_root",
            "staging_baseline_root",
            "staging_audio_root",
            "staging_ablation_root",
        )
    } | {
        "fps": float(args.fps),
        "scale": int(args.scale),
        "mri_workers": int(args.mri_workers),
        "resume": bool(args.resume),
    }


def rebuild_session(config: dict[str, Any], speaker: int, session: int) -> dict[str, Any]:
    cv2.setNumThreads(1)
    source_baseline = Path(config["source_baseline_root"])
    source_audio = Path(config["source_audio_root"])
    source_ablation = Path(config["source_ablation_root"])
    stage_baseline = Path(config["staging_baseline_root"])
    stage_audio = Path(config["staging_audio_root"])
    stage_ablation = Path(config["staging_ablation_root"])
    baseline_dir = stage_baseline / f"P{speaker}/S{session}"
    audio_dir = stage_audio / f"P{speaker}/S{session}"
    ablation_dir = stage_ablation / f"P{speaker}/S{session}"
    marker = ablation_dir / ".integer_rebuild_complete.json"
    if config["resume"] and marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        expected = [
            baseline_dir / f"p{speaker}_s{session}_raw_affine_tps_compare_50fps.mp4",
            audio_dir / f"p{speaker}_s{session}_baseline_vs_audio_gridnorm_50fps.mp4",
            ablation_dir / f"p{speaker}_s{session}_audio_ablation_50fps.mp4",
        ]
        if all(path.is_file() and path.stat().st_size > 0 for path in expected):
            return payload

    started = time.monotonic()
    baseline = filter_and_save_pack(
        source_baseline / f"P{speaker}/S{session}/contours_and_ground_truth.npz",
        baseline_dir / "contours_and_ground_truth.npz",
        "baseline",
    )
    audio = filter_and_save_pack(
        source_audio / f"P{speaker}/S{session}/audio_normalized_contours_and_ground_truth.npz",
        audio_dir / "audio_normalized_contours_and_ground_truth.npz",
        "audio",
    )
    rms_only = filter_and_save_pack(
        source_ablation / f"P{speaker}/S{session}/rms_only_contours_and_ground_truth.npz",
        ablation_dir / "rms_only_contours_and_ground_truth.npz",
        "variant",
    )
    vtln_only = filter_and_save_pack(
        source_ablation / f"P{speaker}/S{session}/vtln_only_contours_and_ground_truth.npz",
        ablation_dir / "vtln_only_contours_and_ground_truth.npz",
        "variant",
    )
    for name, branch in (("audio", audio), ("rms_only", rms_only), ("vtln_only", vtln_only)):
        assert_aligned(baseline, branch, f"P{speaker}/S{session} {name}")

    shutil.copy2(
        baseline_dir / "contours_and_ground_truth.npz",
        ablation_dir / "baseline_contours_and_ground_truth.npz",
    )
    shutil.copy2(
        audio_dir / "audio_normalized_contours_and_ground_truth.npz",
        ablation_dir / "rms_vtln_contours_and_ground_truth.npz",
    )

    for filename in ("audio_chunk_alignment.csv", "audio_extraction_metadata.json"):
        copy_if_present(
            source_audio / f"P{speaker}/S{session}/{filename}",
            audio_dir / filename,
        )
    for variant in ("rms_only", "vtln_only"):
        for suffix in ("chunk_alignment.csv", "extraction_metadata.json"):
            filename = f"{variant}_{suffix}"
            copy_if_present(
                source_ablation / f"P{speaker}/S{session}/{filename}",
                ablation_dir / filename,
            )

    baseline_csv = baseline_dir / "frame_metrics.csv"
    audio_csv = audio_dir / "frame_metrics_baseline_vs_audio.csv"
    ablation_csv = ablation_dir / "ablation_frame_metrics.csv"
    write_baseline_frame_csv(baseline_csv, baseline)
    write_audio_frame_csv(audio_csv, baseline, audio)
    ablation_branches = {
        "baseline": baseline,
        "rms_only": rms_only,
        "vtln_only": vtln_only,
        "rms_vtln": audio,
    }
    write_ablation_frame_csv(ablation_csv, ablation_branches)

    frames = baseline["frame_numbers"]
    dicom_dir = RAW_ROOT / f"P{speaker}/DCM_2D/S{session}"
    if not dicom_dir.is_dir():
        raise FileNotFoundError(dicom_dir)
    dicom_index, _ = build_filename_dicom_index(dicom_dir)
    cache_path = baseline_dir / ".integer_dicom_cache.npz"
    mri_cache = load_or_build_mri_cache(
        dicom_dir,
        dicom_index,
        [int(value) for value in frames],
        cache_path,
        workers=int(config["mri_workers"]),
    )
    baseline_video = baseline_dir / f"p{speaker}_s{session}_raw_affine_tps_compare_50fps.mp4"
    audio_video = audio_dir / f"p{speaker}_s{session}_baseline_vs_audio_gridnorm_50fps.mp4"
    ablation_video = ablation_dir / f"p{speaker}_s{session}_audio_ablation_50fps.mp4"
    render_video(
        baseline_video,
        [("P7 prediction", baseline)],
        mri_cache,
        float(config["fps"]),
        int(config["scale"]),
    )
    render_video(
        audio_video,
        [("Baseline audio", baseline), ("RMS+VTLN audio", audio)],
        mri_cache,
        float(config["fps"]),
        int(config["scale"]),
    )
    render_video(
        ablation_video,
        [(VARIANT_LABELS[name], ablation_branches[name]) for name in VARIANTS],
        mri_cache,
        float(config["fps"]),
        int(config["scale"]),
    )
    cache_path.unlink(missing_ok=True)

    alpha_path = source_audio / "audio_normalization/alpha_to_p7.json"
    alpha_payload = json.loads(alpha_path.read_text(encoding="utf-8"))
    alpha = float(alpha_payload["alpha_to_p7"][f"P{speaker}"])
    baseline_summary = {
        "created_at": now(),
        "speaker": speaker,
        "session": session,
        "source_num_frames": baseline["source_count"],
        "num_unique_frames": baseline["integer_count"],
        "frame_min": int(frames.min()),
        "frame_max": int(frames.max()),
        "fps": float(config["fps"]),
        **required_zero_fields(baseline["fractional_count"]),
        "contour_pack": relative_session_path(speaker, session, "contours_and_ground_truth.npz"),
        "frame_metrics": relative_session_path(speaker, session, "frame_metrics.csv"),
        "video": relative_session_path(
            speaker, session, f"p{speaker}_s{session}_raw_affine_tps_compare_50fps.mp4"
        ),
        "source_contour_pack": str(
            source_baseline / f"P{speaker}/S{session}/contours_and_ground_truth.npz"
        ),
        "metrics": baseline["metrics"],
        "metrics_before_fractional_filter": baseline["source_metrics"],
        "model_inference_rerun": False,
    }
    (baseline_dir / "session_summary.json").write_text(
        json.dumps(baseline_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    audio_summary = {
        "created_at": now(),
        "speaker": speaker,
        "session": session,
        "source_num_frames": baseline["source_count"],
        "num_unique_frames": baseline["integer_count"],
        "fps": float(config["fps"]),
        **required_zero_fields(baseline["fractional_count"]),
        "branch_fractional_frames_discarded": {
            "baseline": baseline["fractional_count"],
            "rms_vtln": audio["fractional_count"],
        },
        "vtln_alpha_to_p7": alpha,
        "baseline_contour_pack": str(
            CURRENT_BASELINE / f"P{speaker}/S{session}/contours_and_ground_truth.npz"
        ),
        "audio_normalized_contour_pack": relative_session_path(
            speaker, session, "audio_normalized_contours_and_ground_truth.npz"
        ),
        "frame_metrics": relative_session_path(speaker, session, "frame_metrics_baseline_vs_audio.csv"),
        "video": relative_session_path(
            speaker, session, f"p{speaker}_s{session}_baseline_vs_audio_gridnorm_50fps.mp4"
        ),
        "metrics": {"baseline": baseline["metrics"], "audio_normalized": audio["metrics"]},
        "metrics_before_fractional_filter": {
            "baseline": baseline["source_metrics"],
            "audio_normalized": audio["source_metrics"],
        },
        "model_inference_rerun": False,
        "audio_feature_extraction_rerun": False,
    }
    (audio_dir / "session_summary.json").write_text(
        json.dumps(audio_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    ablation_summary = {
        "created_at": now(),
        "speaker": speaker,
        "session": session,
        "source_num_frames": baseline["source_count"],
        "num_unique_frames": baseline["integer_count"],
        "fps": float(config["fps"]),
        **required_zero_fields(baseline["fractional_count"]),
        "branch_fractional_frames_discarded": {
            "baseline": baseline["fractional_count"],
            "rms_only": rms_only["fractional_count"],
            "vtln_only": vtln_only["fractional_count"],
            "rms_vtln": audio["fractional_count"],
        },
        "vtln_alpha_to_p7": alpha,
        "paths": {
            "baseline_pack": relative_session_path(speaker, session, "baseline_contours_and_ground_truth.npz"),
            "rms_only_pack": relative_session_path(speaker, session, "rms_only_contours_and_ground_truth.npz"),
            "vtln_only_pack": relative_session_path(speaker, session, "vtln_only_contours_and_ground_truth.npz"),
            "rms_vtln_pack": relative_session_path(speaker, session, "rms_vtln_contours_and_ground_truth.npz"),
            "frame_metrics": relative_session_path(speaker, session, "ablation_frame_metrics.csv"),
            "video": relative_session_path(speaker, session, f"p{speaker}_s{session}_audio_ablation_50fps.mp4"),
        },
        "metrics": {name: ablation_branches[name]["metrics"] for name in VARIANTS},
        "metrics_before_fractional_filter": {
            name: ablation_branches[name]["source_metrics"] for name in VARIANTS
        },
        "model_inference_rerun": False,
        "audio_feature_extraction_rerun": False,
    }
    (ablation_dir / "session_ablation_summary.json").write_text(
        json.dumps(ablation_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    result = {
        "speaker": speaker,
        "session": session,
        "integer_frames": baseline["integer_count"],
        "fractional_frames_discarded": baseline["fractional_count"],
        "elapsed_seconds": time.monotonic() - started,
    }
    marker.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def load_summaries(root: Path, filename: str) -> list[dict[str, Any]]:
    summaries = []
    for speaker, session in SELECTION:
        path = root / f"P{speaker}/S{session}/{filename}"
        if not path.is_file():
            raise FileNotFoundError(path)
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def stage_value(summary: dict[str, Any], mode: str, stage: str, before: bool = False) -> float:
    key = "metrics_before_fractional_filter" if before else "metrics"
    return float(summary[key]["modes"][mode][stage]["mean_frame_rmse_mm"])


def weighted_stage(
    summaries: list[dict[str, Any]], mode: str, stage: str, before: bool = False
) -> float:
    count_key = "source_num_frames" if before else "num_unique_frames"
    total = sum(int(row[count_key]) for row in summaries)
    return sum(int(row[count_key]) * stage_value(row, mode, stage, before) for row in summaries) / total


def baseline_aggregate_row(label: str, rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    values = {stage: weighted_stage(rows, mode, stage) for stage in STAGES}
    raw = values["raw"]
    return {
        "speaker": label,
        "sessions": len(rows),
        "frames": sum(int(row["num_unique_frames"]) for row in rows),
        "fractional_frames_discarded": sum(int(row["num_fractional_frames_discarded"]) for row in rows),
        "metric_mode": mode,
        "raw_rmse_mm": raw,
        "affine_rmse_mm": values["affine"],
        "affine_tps_rmse_mm": values["affine_tps"],
        "raw_to_affine_delta_mm": values["affine"] - raw,
        "affine_to_tps_delta_mm": values["affine_tps"] - values["affine"],
        "raw_to_final_delta_mm": values["affine_tps"] - raw,
    }


def report_header(handle: Any, title: str, summaries: list[dict[str, Any]]) -> None:
    integer_frames = sum(int(row["num_unique_frames"]) for row in summaries)
    discarded = sum(int(row["num_fractional_frames_discarded"]) for row in summaries)
    handle.write(f"Frame policy: {FRAME_POLICY}.\n\n")
    handle.write(f"# {title}\n\n")
    handle.write(f"- Integer frames evaluated/rendered: **{integer_frames:,}**.\n")
    handle.write(f"- Fractional frames discarded from the source timeline: **{discarded:,}**.\n")
    handle.write("- Fractional frames saved/scored/rendered: **0 / 0 / 0**.\n")
    handle.write("- Model inference rerun: **no**; existing contour packs were filtered and reused.\n\n")


def build_baseline_report(root: Path) -> dict[str, Any]:
    summaries = load_summaries(root, "session_summary.json")
    session_rows = []
    for summary in summaries:
        for mode in MODES:
            values = summary["metrics"]["modes"][mode]
            raw = float(values["raw"]["mean_frame_rmse_mm"])
            affine = float(values["affine"]["mean_frame_rmse_mm"])
            final = float(values["affine_tps"]["mean_frame_rmse_mm"])
            session_rows.append({
                "speaker": f"P{summary['speaker']}",
                "session": f"S{summary['session']}",
                "frames": summary["num_unique_frames"],
                "fractional_frames_discarded": summary["num_fractional_frames_discarded"],
                "metric_mode": mode,
                "raw_rmse_mm": raw,
                "affine_rmse_mm": affine,
                "affine_tps_rmse_mm": final,
                "raw_to_affine_delta_mm": affine - raw,
                "affine_to_tps_delta_mm": final - affine,
                "raw_to_final_delta_mm": final - raw,
            })
    write_csv(root / "session_metrics.csv", session_rows)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        grouped[int(summary["speaker"])].append(summary)
    aggregate_rows = []
    for mode in MODES:
        for speaker in sorted(grouped):
            aggregate_rows.append(baseline_aggregate_row(f"P{speaker}", grouped[speaker], mode))
        aggregate_rows.append(baseline_aggregate_row("ALL", summaries, mode))
    write_csv(root / "speaker_and_overall_metrics.csv", aggregate_rows)

    report = root / "all_nonp7_gridnorm_error_report.md"
    with report.open("w", encoding="utf-8") as handle:
        report_header(handle, "P7 grid normalization on selected non-P7 sessions", summaries)
        for mode, title in (("all_11", "All 11 contours"), ("without_laryngeal_3", "Without laryngeal 3")):
            handle.write(f"## {title}\n\n")
            handle.write("| Speaker/session | Frames | Raw | Affine | Affine+TPS | raw→affine | affine→TPS | raw→final |\n")
            handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in [item for item in session_rows if item["metric_mode"] == mode]:
                handle.write(
                    f"| {row['speaker']}/{row['session']} | {row['frames']} | {row['raw_rmse_mm']:.3f} | "
                    f"{row['affine_rmse_mm']:.3f} | {row['affine_tps_rmse_mm']:.3f} | "
                    f"{row['raw_to_affine_delta_mm']:+.3f} | {row['affine_to_tps_delta_mm']:+.3f} | "
                    f"{row['raw_to_final_delta_mm']:+.3f} |\n"
                )
            overall = next(row for row in aggregate_rows if row["speaker"] == "ALL" and row["metric_mode"] == mode)
            handle.write(
                f"| **ALL weighted** | **{overall['frames']}** | **{overall['raw_rmse_mm']:.3f}** | "
                f"**{overall['affine_rmse_mm']:.3f}** | **{overall['affine_tps_rmse_mm']:.3f}** | "
                f"**{overall['raw_to_affine_delta_mm']:+.3f}** | **{overall['affine_to_tps_delta_mm']:+.3f}** | "
                f"**{overall['raw_to_final_delta_mm']:+.3f}** |\n\n"
            )
            direction = "decreases" if overall["raw_to_final_delta_mm"] < 0 else "increases"
            handle.write(f"Affine+TPS {direction} weighted error versus raw by {abs(overall['raw_to_final_delta_mm']):.3f} mm.\n\n")
        handle.write("## Before versus after removing fractional frames\n\n")
        handle.write("| Metric mode | Stage | All timestamps before | Integer-only after | Change |\n|---|---|---:|---:|---:|\n")
        for mode in MODES:
            for stage in STAGES:
                before = weighted_stage(summaries, mode, stage, before=True)
                after = weighted_stage(summaries, mode, stage)
                handle.write(f"| {mode} | {stage} | {before:.3f} | {after:.3f} | {after-before:+.3f} |\n")
        handle.write("\n## Session videos and integer-only contour packs\n\n")
        for summary in summaries:
            handle.write(
                f"- P{summary['speaker']}/S{summary['session']}: "
                f"[video]({summary['video']}), [contour pack]({summary['contour_pack']})\n"
            )
    manifest = base_manifest("baseline_gridnorm", summaries, report, [
        root / "session_metrics.csv", root / "speaker_and_overall_metrics.csv"
    ])
    write_manifest(root, manifest)
    return manifest


def branch_stage(summary: dict[str, Any], branch: str, mode: str, stage: str, before: bool = False) -> float:
    key = "metrics_before_fractional_filter" if before else "metrics"
    return float(summary[key][branch]["modes"][mode][stage]["mean_frame_rmse_mm"])


def weighted_branch(
    rows: list[dict[str, Any]], branch: str, mode: str, stage: str, before: bool = False
) -> float:
    key = "source_num_frames" if before else "num_unique_frames"
    total = sum(int(row[key]) for row in rows)
    return sum(int(row[key]) * branch_stage(row, branch, mode, stage, before) for row in rows) / total


def build_audio_report(root: Path) -> dict[str, Any]:
    summaries = load_summaries(root, "session_summary.json")
    rows = []
    groups = [(f"P{row['speaker']}", [row]) for row in summaries] + [("ALL", summaries)]
    for label, group in groups:
        for mode in MODES:
            base = {stage: weighted_branch(group, "baseline", mode, stage) for stage in STAGES}
            audio = {stage: weighted_branch(group, "audio_normalized", mode, stage) for stage in STAGES}
            rows.append({
                "speaker": label,
                "session": f"S{group[0]['session']}" if len(group) == 1 else "selected_9",
                "frames": sum(int(item["num_unique_frames"]) for item in group),
                "metric_mode": mode,
                "baseline_raw_mm": base["raw"],
                "baseline_affine_mm": base["affine"],
                "baseline_affine_tps_mm": base["affine_tps"],
                "audio_raw_mm": audio["raw"],
                "audio_affine_mm": audio["affine"],
                "audio_affine_tps_mm": audio["affine_tps"],
                "audio_effect_raw_mm": audio["raw"] - base["raw"],
                "audio_effect_affine_mm": audio["affine"] - base["affine"],
                "audio_effect_affine_tps_mm": audio["affine_tps"] - base["affine_tps"],
                "audio_raw_to_affine_mm": audio["affine"] - audio["raw"],
                "audio_affine_to_tps_mm": audio["affine_tps"] - audio["affine"],
                "audio_raw_to_final_mm": audio["affine_tps"] - audio["raw"],
            })
    metrics_path = root / "baseline_vs_audio_gridnorm_metrics.csv"
    write_csv(metrics_path, rows)
    report = root / "audio_gridnorm_error_report.md"
    with report.open("w", encoding="utf-8") as handle:
        report_header(handle, "Audio normalization plus grid normalization", summaries)
        for mode, title in (("all_11", "All 11 contours"), ("without_laryngeal_3", "Without laryngeal 3")):
            handle.write(f"## {title}\n\n")
            handle.write("| Speaker/session | Frames | Base raw | Base aff | Base TPS | RMS+VTLN raw | RMS+VTLN aff | RMS+VTLN TPS | Audio effect at TPS |\n")
            handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in [item for item in rows if item["metric_mode"] == mode]:
                label = "ALL weighted" if row["speaker"] == "ALL" else f"{row['speaker']}/{row['session']}"
                handle.write(
                    f"| {label} | {row['frames']} | {row['baseline_raw_mm']:.3f} | {row['baseline_affine_mm']:.3f} | "
                    f"{row['baseline_affine_tps_mm']:.3f} | {row['audio_raw_mm']:.3f} | {row['audio_affine_mm']:.3f} | "
                    f"{row['audio_affine_tps_mm']:.3f} | {row['audio_effect_affine_tps_mm']:+.3f} |\n"
                )
            overall = next(row for row in rows if row["speaker"] == "ALL" and row["metric_mode"] == mode)
            word = "reduces" if overall["audio_effect_affine_tps_mm"] < 0 else "increases"
            handle.write(
                f"\nRMS+VTLN {word} final weighted error by {abs(overall['audio_effect_affine_tps_mm']):.3f} mm. "
                f"Within the audio branch, raw→affine is {overall['audio_raw_to_affine_mm']:+.3f} mm and "
                f"affine→TPS is {overall['audio_affine_to_tps_mm']:+.3f} mm.\n\n"
            )
        handle.write("## Before versus after removing fractional frames\n\n")
        handle.write("| Mode | Branch | Stage | Before | Integer-only | Change |\n|---|---|---|---:|---:|---:|\n")
        for mode in MODES:
            for branch in ("baseline", "audio_normalized"):
                for stage in STAGES:
                    before = weighted_branch(summaries, branch, mode, stage, before=True)
                    after = weighted_branch(summaries, branch, mode, stage)
                    handle.write(f"| {mode} | {branch} | {stage} | {before:.3f} | {after:.3f} | {after-before:+.3f} |\n")
        handle.write("\n## Session videos and integer-only contour packs\n\n")
        for summary in summaries:
            handle.write(
                f"- P{summary['speaker']}/S{summary['session']}: [six-panel video]({summary['video']}), "
                f"[audio contour pack]({summary['audio_normalized_contour_pack']})\n"
            )
    manifest = base_manifest("audio_gridnorm", summaries, report, [metrics_path])
    write_manifest(root, manifest)
    return manifest


def ablation_stage(summary: dict[str, Any], variant: str, mode: str, stage: str, before: bool = False) -> float:
    key = "metrics_before_fractional_filter" if before else "metrics"
    return float(summary[key][variant]["modes"][mode][stage]["mean_frame_rmse_mm"])


def weighted_ablation(
    rows: list[dict[str, Any]], variant: str, mode: str, stage: str, before: bool = False
) -> float:
    key = "source_num_frames" if before else "num_unique_frames"
    total = sum(int(row[key]) for row in rows)
    return sum(int(row[key]) * ablation_stage(row, variant, mode, stage, before) for row in rows) / total


def build_ablation_report(root: Path) -> dict[str, Any]:
    summaries = load_summaries(root, "session_ablation_summary.json")
    long_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    groups = [(f"P{row['speaker']}", [row]) for row in summaries] + [("ALL", summaries)]
    for label, group in groups:
        session_label = f"S{group[0]['session']}" if len(group) == 1 else "selected_9"
        for mode in MODES:
            finals = {}
            for variant in VARIANTS:
                values = {stage: weighted_ablation(group, variant, mode, stage) for stage in STAGES}
                finals[variant] = values["affine_tps"]
                long_rows.append({
                    "speaker": label,
                    "session": session_label,
                    "frames": sum(int(row["num_unique_frames"]) for row in group),
                    "metric_mode": mode,
                    "variant": variant,
                    "raw_rmse_mm": values["raw"],
                    "affine_rmse_mm": values["affine"],
                    "affine_tps_rmse_mm": values["affine_tps"],
                    "raw_to_affine_mm": values["affine"] - values["raw"],
                    "affine_to_tps_mm": values["affine_tps"] - values["affine"],
                    "raw_to_final_mm": values["affine_tps"] - values["raw"],
                })
            comparison_rows.append({
                "speaker": label,
                "session": session_label,
                "frames": sum(int(row["num_unique_frames"]) for row in group),
                "metric_mode": mode,
                "baseline_final_mm": finals["baseline"],
                "rms_only_final_mm": finals["rms_only"],
                "rms_effect_mm": finals["rms_only"] - finals["baseline"],
                "vtln_only_final_mm": finals["vtln_only"],
                "vtln_effect_mm": finals["vtln_only"] - finals["baseline"],
                "rms_vtln_final_mm": finals["rms_vtln"],
                "combined_effect_mm": finals["rms_vtln"] - finals["baseline"],
                "best_variant": min(finals, key=finals.get),
            })
    class_rows = []
    classes = next(iter(summaries))["metrics"]["baseline"]["kept_classes_without_laryngeal_3"] + list(EXCLUDED_CLASSES)
    # Preserve the actual source class order from a staged pack.
    with np.load(root / "P1/S16/baseline_contours_and_ground_truth.npz", allow_pickle=False) as payload:
        classes = [str(value) for value in payload["classes"].tolist()]
    for class_name in classes:
        for stage in STAGES:
            row: dict[str, Any] = {
                "class": class_name,
                "excluded_in_without_3": class_name in EXCLUDED_CLASSES,
                "stage": stage,
            }
            total = sum(int(item["num_unique_frames"]) for item in summaries)
            for variant in VARIANTS:
                value = sum(
                    int(item["num_unique_frames"])
                    * float(item["metrics"][variant]["per_class_mean_frame_rmse_mm"][stage][class_name])
                    for item in summaries
                ) / total
                row[f"{variant}_mm"] = value
                if variant != "baseline":
                    row[f"{variant}_effect_mm"] = value
            for variant in ("rms_only", "vtln_only", "rms_vtln"):
                row[f"{variant}_effect_mm"] = row[f"{variant}_mm"] - row["baseline_mm"]
            class_rows.append(row)
    long_path = root / "ablation_metrics_long.csv"
    comparison_path = root / "ablation_final_comparison.csv"
    class_path = root / "ablation_per_class_metrics.csv"
    write_csv(long_path, long_rows)
    write_csv(comparison_path, comparison_rows)
    write_csv(class_path, class_rows)
    report = root / "rms_vtln_ablation_detailed_report.md"
    with report.open("w", encoding="utf-8") as handle:
        report_header(handle, "RMS/VTLN audio ablation", summaries)
        for mode, title in (("all_11", "All 11 contours"), ("without_laryngeal_3", "Without laryngeal 3")):
            handle.write(f"## {title}: stage-by-stage weighted metrics\n\n")
            handle.write("| Variant | Raw | Affine | Affine+TPS | raw→affine | affine→TPS | raw→final |\n|---|---:|---:|---:|---:|---:|---:|\n")
            for row in [item for item in long_rows if item["speaker"] == "ALL" and item["metric_mode"] == mode]:
                handle.write(
                    f"| {VARIANT_LABELS[row['variant']]} | {row['raw_rmse_mm']:.3f} | {row['affine_rmse_mm']:.3f} | "
                    f"{row['affine_tps_rmse_mm']:.3f} | {row['raw_to_affine_mm']:+.3f} | "
                    f"{row['affine_to_tps_mm']:+.3f} | {row['raw_to_final_mm']:+.3f} |\n"
                )
            handle.write("\n## Final affine+TPS comparison per session\n\n")
            handle.write("| Speaker/session | Baseline | RMS-only | Δ | VTLN-only | Δ | RMS+VTLN | Δ | Best |\n|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
            for row in [item for item in comparison_rows if item["metric_mode"] == mode]:
                label = "ALL weighted" if row["speaker"] == "ALL" else f"{row['speaker']}/{row['session']}"
                handle.write(
                    f"| {label} | {row['baseline_final_mm']:.3f} | {row['rms_only_final_mm']:.3f} | "
                    f"{row['rms_effect_mm']:+.3f} | {row['vtln_only_final_mm']:.3f} | {row['vtln_effect_mm']:+.3f} | "
                    f"{row['rms_vtln_final_mm']:.3f} | {row['combined_effect_mm']:+.3f} | "
                    f"{VARIANT_LABELS[row['best_variant']]} |\n"
                )
            overall = next(row for row in comparison_rows if row["speaker"] == "ALL" and row["metric_mode"] == mode)
            handle.write(f"\nBest weighted final branch: **{VARIANT_LABELS[overall['best_variant']]}**. Negative deltas improve over baseline.\n\n")
        handle.write("## Before versus after removing fractional frames\n\n")
        handle.write("| Mode | Variant | Stage | Before | Integer-only | Change |\n|---|---|---|---:|---:|---:|\n")
        for mode in MODES:
            for variant in VARIANTS:
                for stage in STAGES:
                    before = weighted_ablation(summaries, variant, mode, stage, before=True)
                    after = weighted_ablation(summaries, variant, mode, stage)
                    handle.write(f"| {mode} | {variant} | {stage} | {before:.3f} | {after:.3f} | {after-before:+.3f} |\n")
        handle.write("\n## Session videos and integer-only contour packs\n\n")
        for summary in summaries:
            paths = summary["paths"]
            handle.write(
                f"- P{summary['speaker']}/S{summary['session']}: [video]({paths['video']}), "
                f"[baseline]({paths['baseline_pack']}), [RMS-only]({paths['rms_only_pack']}), "
                f"[VTLN-only]({paths['vtln_only_pack']}), [RMS+VTLN]({paths['rms_vtln_pack']})\n"
            )
    manifest = base_manifest("audio_ablation", summaries, report, [long_path, comparison_path, class_path])
    write_manifest(root, manifest)
    return manifest


def base_manifest(
    experiment: str, summaries: list[dict[str, Any]], report: Path, metrics: list[Path]
) -> dict[str, Any]:
    return {
        "created_at": now(),
        "experiment": experiment,
        "frame_policy": FRAME_POLICY,
        "rendered_fractional_frame_count": 0,
        "scored_fractional_frame_count": 0,
        "saved_fractional_frame_count": 0,
        "num_fractional_frames_discarded": sum(
            int(row["num_fractional_frames_discarded"]) for row in summaries
        ),
        "total_integer_frames": sum(int(row["num_unique_frames"]) for row in summaries),
        "expected_sessions": 9,
        "completed_sessions": len(summaries),
        "selection": [f"P{speaker}/S{session}" for speaker, session in SELECTION],
        "report": report.name,
        "aggregate_metrics": [path.name for path in metrics],
        "model_inference_rerun": False,
        "audio_feature_extraction_rerun": False,
        "sessions": summaries,
    }


def write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def probe_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    opencv_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    opencv_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames,avg_frame_rate,width,height",
        "-of", "json", str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    stream = payload["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", maxsplit=1)
    ffprobe_fps = float(numerator) / float(denominator)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "opencv_frame_count": opencv_frames,
        "opencv_fps": opencv_fps,
        "ffprobe_frame_count": int(stream["nb_read_frames"]),
        "ffprobe_fps": ffprobe_fps,
        "width": width,
        "height": height,
    }


def extract_visual_sample(video: Path, destination: Path, frame_index: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not extract visual sample from {video}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), frame):
        raise RuntimeError(f"Could not write {destination}")
    return {
        "video": str(video),
        "sample": str(destination),
        "frame_index": frame_index,
        "shape": list(frame.shape),
        "pixel_std": float(np.std(frame)),
    }


def source_audit() -> dict[str, Any]:
    paths = [
        REPO_ROOT / "scripts/rebuild_p7_integer_only_results.py",
        REPO_ROOT / "scripts/run_p7_all_nonp7_gridnorm.py",
        REPO_ROOT / "scripts/run_p7_selected_audio_gridnorm.py",
        REPO_ROOT / "scripts/run_p7_audio_ablation.py",
        REPO_ROOT / "scripts/render_gridnorm_session_video.py",
        REPO_ROOT / "src/utils/mri_rendering.py",
        REPO_ROOT / "src/utils/gridnorm_rendering.py",
        REPO_ROOT / "src/utils/session_rendering.py",
    ]
    forbidden = {
        # Build the audit tokens from fragments so this rule table cannot
        # trigger a false positive when the audit checks its own source file.
        "timeline_step_default_half": "default=" + "0.5",
        "fractional_filename_token": "p" + "5",
        "mri_floor": "math." + "floor(frame_number)",
        "mri_ceil": "math." + "ceil(frame_number)",
    }
    findings = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for name, token in forbidden.items():
            if token in text:
                findings.append({"file": str(path), "rule": name, "token": token})
    return {"files_checked": [str(path) for path in paths], "findings": findings, "passed": not findings}


def audit_experiment(root: Path, experiment: str) -> dict[str, Any]:
    if experiment == "baseline_gridnorm":
        pack_names = ["contours_and_ground_truth.npz"]
        csv_name = "frame_metrics.csv"
        summary_name = "session_summary.json"
        video_name = lambda speaker, session: f"p{speaker}_s{session}_raw_affine_tps_compare_50fps.mp4"
    elif experiment == "audio_gridnorm":
        pack_names = ["audio_normalized_contours_and_ground_truth.npz"]
        csv_name = "frame_metrics_baseline_vs_audio.csv"
        summary_name = "session_summary.json"
        video_name = lambda speaker, session: f"p{speaker}_s{session}_baseline_vs_audio_gridnorm_50fps.mp4"
    elif experiment == "audio_ablation":
        pack_names = [
            "baseline_contours_and_ground_truth.npz",
            "rms_only_contours_and_ground_truth.npz",
            "vtln_only_contours_and_ground_truth.npz",
            "rms_vtln_contours_and_ground_truth.npz",
        ]
        csv_name = "ablation_frame_metrics.csv"
        summary_name = "session_ablation_summary.json"
        video_name = lambda speaker, session: f"p{speaker}_s{session}_audio_ablation_50fps.mp4"
    else:
        raise ValueError(experiment)

    session_audits = []
    for speaker, session in SELECTION:
        session_dir = root / f"P{speaker}/S{session}"
        pack_audits = []
        reference_frames = None
        for pack_name in pack_names:
            pack = session_dir / pack_name
            with np.load(pack, allow_pickle=False) as payload:
                frames = np.asarray(payload["frame_numbers"])
                integer = assert_integer_frames(frames, str(pack))
                if frames.dtype != np.int32:
                    raise AssertionError(f"{pack}: frame_numbers dtype is {frames.dtype}, expected int32")
                for key in (
                    "saved_fractional_frame_count",
                    "scored_fractional_frame_count",
                    "rendered_fractional_frame_count",
                ):
                    if int(payload[key]) != 0:
                        raise AssertionError(f"{pack}: {key} != 0")
                if str(payload["frame_policy"].item()) != FRAME_POLICY:
                    raise AssertionError(f"{pack}: wrong frame policy")
            if reference_frames is None:
                reference_frames = integer
            elif not np.array_equal(reference_frames, integer):
                raise AssertionError(f"{session_dir}: pack timelines differ")
            pack_audits.append({"path": str(pack), "frames": len(integer), "fractional_frames": 0})
        assert reference_frames is not None
        csv_path = session_dir / csv_name
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        csv_frames = np.asarray([float(row["frame_number"]) for row in csv_rows])
        csv_integer = assert_integer_frames(csv_frames, str(csv_path))
        if not np.array_equal(reference_frames, csv_integer):
            raise AssertionError(f"{session_dir}: NPZ and CSV timelines differ")
        summary_path = session_dir / summary_name
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for key in (
            "saved_fractional_frame_count",
            "scored_fractional_frame_count",
            "rendered_fractional_frame_count",
        ):
            if int(summary[key]) != 0:
                raise AssertionError(f"{summary_path}: {key} != 0")
        if summary["frame_policy"] != FRAME_POLICY:
            raise AssertionError(f"{summary_path}: wrong frame policy")
        video_path = session_dir / video_name(speaker, session)
        video = probe_video(video_path)
        if video["size_bytes"] <= 0:
            raise AssertionError(f"Empty video {video_path}")
        if video["opencv_frame_count"] != len(reference_frames) or video["ffprobe_frame_count"] != len(reference_frames):
            raise AssertionError(f"{video_path}: video/NPZ frame count mismatch")
        if not math.isclose(video["opencv_fps"], 50.0, abs_tol=1e-6) or not math.isclose(video["ffprobe_fps"], 50.0, abs_tol=1e-6):
            raise AssertionError(f"{video_path}: FPS is not 50")
        session_audits.append({
            "speaker": f"P{speaker}",
            "session": f"S{session}",
            "packs": pack_audits,
            "csv": {"path": str(csv_path), "rows": len(csv_rows), "fractional_frames": 0},
            "video": video,
            "summary": str(summary_path),
        })
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["completed_sessions"] != 9 or len(manifest["sessions"]) != 9:
        raise AssertionError(f"{manifest_path}: expected exactly 9 sessions")
    for key in (
        "saved_fractional_frame_count",
        "scored_fractional_frame_count",
        "rendered_fractional_frame_count",
    ):
        if int(manifest[key]) != 0:
            raise AssertionError(f"{manifest_path}: {key} != 0")
    first = session_audits[0]
    middle = first["video"]["opencv_frame_count"] // 2
    sample = extract_visual_sample(
        Path(first["video"]["path"]),
        root / "audit_visual_samples" / f"{experiment}_P1_S16_integer_frame.png",
        middle,
    )
    if sample["pixel_std"] <= 1.0:
        raise AssertionError(f"Visual sample appears blank: {sample}")
    result = {
        "created_at": now(),
        "experiment": experiment,
        "frame_policy": FRAME_POLICY,
        "passed": True,
        "sessions_checked": len(session_audits),
        "rendered_fractional_frame_count": 0,
        "scored_fractional_frame_count": 0,
        "saved_fractional_frame_count": 0,
        "session_audits": session_audits,
        "visual_sample": sample,
    }
    (root / "integer_only_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest["audit"] = "integer_only_audit.json"
    manifest["audit_passed"] = True
    write_manifest(root, manifest)
    return result


def audit_all(baseline: Path, audio: Path, ablation: Path) -> dict[str, Any]:
    audits = {
        "baseline_gridnorm": audit_experiment(baseline, "baseline_gridnorm"),
        "audio_gridnorm": audit_experiment(audio, "audio_gridnorm"),
        "audio_ablation": audit_experiment(ablation, "audio_ablation"),
    }
    code = source_audit()
    if not code["passed"]:
        raise AssertionError(f"Source audit failed: {code['findings']}")
    result = {
        "created_at": now(),
        "frame_policy": FRAME_POLICY,
        "passed": all(item["passed"] for item in audits.values()) and code["passed"],
        "experiments": audits,
        "source_audit": code,
        "model_inference_rerun": False,
        "audio_feature_extraction_rerun": False,
    }
    for root in (baseline, audio, ablation):
        (root / "cross_experiment_integer_audit.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
    return result


def prepare_staging(args: argparse.Namespace) -> None:
    for source in (args.source_baseline_root, args.source_audio_root, args.source_ablation_root):
        if not source.is_dir():
            raise FileNotFoundError(source)
    for staging in (args.staging_baseline_root, args.staging_audio_root, args.staging_ablation_root):
        if staging.exists() and not args.resume:
            raise FileExistsError(f"Staging exists; pass --resume to continue safely: {staging}")
        staging.mkdir(parents=True, exist_ok=True)
    audio_norm_source = args.source_audio_root / "audio_normalization"
    audio_norm_destination = args.staging_audio_root / "audio_normalization"
    if audio_norm_source.is_dir() and not audio_norm_destination.exists():
        shutil.copytree(audio_norm_source, audio_norm_destination)


def atomic_replace(args: argparse.Namespace) -> None:
    triples = [
        (args.source_baseline_root, args.backup_baseline_root, args.staging_baseline_root),
        (args.source_audio_root, args.backup_audio_root, args.staging_audio_root),
        (args.source_ablation_root, args.backup_ablation_root, args.staging_ablation_root),
    ]
    for current, backup, staging in triples:
        if not current.is_dir() or not staging.is_dir():
            raise FileNotFoundError(f"Replacement paths missing: {current}, {staging}")
        if backup.exists():
            raise FileExistsError(f"Backup already exists; refusing to overwrite: {backup}")
        if current.parent.resolve() != staging.parent.resolve() or current.parent.resolve() != backup.parent.resolve():
            raise RuntimeError("Atomic rename requires current, staging, and backup under the same parent")
    moved_to_backup: list[tuple[Path, Path, Path]] = []
    installed: list[tuple[Path, Path, Path]] = []
    try:
        for current, backup, staging in triples:
            current.rename(backup)
            moved_to_backup.append((current, backup, staging))
        for current, backup, staging in triples:
            staging.rename(current)
            installed.append((current, backup, staging))
    except Exception:
        for current, _backup, staging in reversed(installed):
            if current.exists() and not staging.exists():
                current.rename(staging)
        for current, backup, _staging in reversed(moved_to_backup):
            if backup.exists() and not current.exists():
                backup.rename(current)
        raise


def main() -> None:
    args = parse_args()
    if args.jobs <= 0 or args.mri_workers <= 0:
        raise ValueError("--jobs and --mri-workers must be positive")
    if not math.isclose(args.fps, 50.0, abs_tol=1e-12):
        raise ValueError("This rebuild requires exactly 50 fps")
    if args.audit_only:
        result = audit_all(
            args.staging_baseline_root, args.staging_audio_root, args.staging_ablation_root
        )
        print(json.dumps({"audit_passed": result["passed"]}, indent=2), flush=True)
        return
    prepare_staging(args)
    config = worker_config(args)
    started = time.monotonic()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(rebuild_session, config, speaker, session): (speaker, session)
            for speaker, session in SELECTION
        }
        for future in concurrent.futures.as_completed(futures):
            speaker, session = futures[future]
            result = future.result()
            results.append(result)
            print(
                f"DONE P{speaker}/S{session}: {result['integer_frames']} integer frames, "
                f"discarded {result['fractional_frames_discarded']}, {result['elapsed_seconds']:.1f}s",
                flush=True,
            )
    if len(results) != 9:
        raise RuntimeError(f"Expected 9 rebuilt sessions, got {len(results)}")
    build_baseline_report(args.staging_baseline_root)
    build_audio_report(args.staging_audio_root)
    build_ablation_report(args.staging_ablation_root)
    audit = audit_all(
        args.staging_baseline_root, args.staging_audio_root, args.staging_ablation_root
    )
    if not audit["passed"]:
        raise RuntimeError("Staging audit did not pass")
    if args.replace:
        atomic_replace(args)
        post = audit_all(
            args.source_baseline_root, args.source_audio_root, args.source_ablation_root
        )
        if not post["passed"]:
            raise RuntimeError("Post-replacement audit failed")
    print(
        json.dumps(
            {
                "completed_at": now(),
                "elapsed_seconds": time.monotonic() - started,
                "sessions": len(results),
                "integer_frames_per_experiment": sum(item["integer_frames"] for item in results),
                "fractional_frames_discarded_per_experiment": sum(
                    item["fractional_frames_discarded"] for item in results
                ),
                "saved_scored_rendered_fractional_frames": [0, 0, 0],
                "audit_passed": True,
                "replaced": bool(args.replace),
                "model_inference_rerun": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
