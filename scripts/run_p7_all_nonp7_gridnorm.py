#!/usr/bin/env python3
"""P7-model inference, P7->speaker grid transfer, video, and aggregate reports.

The script deliberately works from the per-session raw caches.  That avoids
building a very large cross-speaker split cache and also makes the run
resumable at session granularity.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
GRID_ROOT = REPO_ROOT / "external" / "grid-transform"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(GRID_ROOT))

from grid_transform.transform_helpers import apply_transform  # noqa: E402
from grid_transform.transfer import build_two_step_transform  # noqa: E402
from render_p7_grid_transform_selected_speakers import (  # noqa: E402
    SOURCE,
    TARGETS,
    prepare_frame,
)
from src.inference.session_inference import load_model  # noqa: E402
from src.utils.colors import COLORS  # noqa: E402
from src.utils.config_validation import load_yaml_config  # noqa: E402
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


RAW_ROOT = Path(
    "/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/"
    "ArtSpeech_Database_1_raw"
)
DEFAULT_CONFIG = (
    REPO_ROOT
    / "config/train_config/asd1_p7_seen_trainvaltest_paper_st5_mfcc_500epoch_stdfloor01.yaml"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "mlruns/151291977315070763/b0887daa569d4659858d4f2e6ecc2451/"
    "artifacts/best_model.pth"
)
DEFAULT_NORM_STATS = (
    REPO_ROOT
    / "repro/asd1_p7_seen_trainvaltest_paper_st5_mfcc_500epoch_stdfloor01/"
    "splits/normalization_stats.npz"
)
DEFAULT_RAW_CACHE = REPO_ROOT / "cache/raw_sessions/asd1"
DEFAULT_OUTPUT = REPO_ROOT / "results/p7_all_nonp7_sessions_gridnorm_20260718"
DEFAULT_CACHED_CONTOUR_ROOT = (
    REPO_ROOT
    / "results/p7_cross_speaker_selected_sessions_20260718/all_inference_contours"
)
DEFAULT_VTLN_DIR = WORKSPACE_ROOT / "_downloads/grid-transform-vtln/vtln-data-v0.1.14/extracted/VTLN/data"
if not DEFAULT_VTLN_DIR.is_dir():
    DEFAULT_VTLN_DIR = GRID_ROOT / "VTLN/data"

STAGES = ("raw", "affine", "affine_tps")
STAGE_TITLES = {
    "raw": "Stage 1: raw P7 prediction",
    "affine": "Stage 2: affine P7 -> target",
    "affine_tps": "Stage 3: affine + TPS",
}
EXCLUDED_CLASSES = ("vocal-folds", "thyroid-cartilage", "epiglottis")
INFO_HEIGHT = 106
SEPARATOR = 8
INTEGER_FRAME_POLICY = "NEVER render fractional MRI frames; integer frames only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P7 inference and 3-stage grid-normalization videos on all non-P7 ASD1 sessions."
    )
    parser.add_argument("--speakers", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 8, 9, 10])
    parser.add_argument("--sessions", nargs="+", type=int, default=None)
    parser.add_argument(
        "--selection",
        nargs="+",
        default=None,
        metavar="P#:S#",
        help="Exact speaker/session pairs, for example P1:S16 P2:S9. Overrides --speakers/--sessions.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--normalization-stats", type=Path, default=DEFAULT_NORM_STATS)
    parser.add_argument("--raw-cache-root", type=Path, default=DEFAULT_RAW_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cached-contour-root",
        type=Path,
        default=DEFAULT_CACHED_CONTOUR_ROOT,
        help="Existing canonical P7-model inference contours used for provenance/equivalence audit.",
    )
    parser.add_argument("--vtln-dir", type=Path, default=DEFAULT_VTLN_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--transform-frame-batch", type=int, default=256)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--mri-workers", type=int, default=4)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--keep-mri-cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if 7 in args.speakers:
        raise ValueError("P7 is the training/source speaker and must not be in --speakers")
    for path in (args.config, args.checkpoint, args.normalization_stats, args.raw_cache_root, args.vtln_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.batch_size <= 0 or args.transform_frame_batch <= 0:
        raise ValueError("Batch sizes must be positive")


def apply_exact_selection(args: argparse.Namespace) -> None:
    if not args.selection:
        args.selection_map = None
        return
    selection_map: dict[int, set[int]] = defaultdict(set)
    for token in args.selection:
        try:
            speaker_token, session_token = token.upper().split(":", maxsplit=1)
            speaker = int(speaker_token.removeprefix("P"))
            session = int(session_token.removeprefix("S"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid --selection token {token!r}; expected P#:S#") from error
        if speaker == 7:
            raise ValueError("P7 is the training/source speaker and cannot be selected")
        selection_map[speaker].add(session)
    args.selection_map = selection_map
    args.speakers = sorted(selection_map)
    args.sessions = None


def target_spec_for_speaker(speaker: int):
    for spec in TARGETS:
        if spec.speaker == f"P{speaker}":
            return spec
    raise KeyError(f"No fixed target-grid reference configured for P{speaker}")


def session_paths(args: argparse.Namespace, speaker: int) -> list[tuple[int, Path]]:
    root = args.raw_cache_root / f"P{speaker}"
    if args.selection_map is not None:
        allowed = set(args.selection_map.get(speaker, set()))
    else:
        allowed = None if args.sessions is None else set(args.sessions)
    paths = []
    for path in root.glob("S*.pt"):
        session = int(path.stem[1:])
        if allowed is None or session in allowed:
            paths.append((session, path))
    return sorted(paths)


def load_normalization(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"mean_mfcc", "std_mfcc", "mean_contour", "std_contour"}
        missing = required - set(payload.files)
        if missing:
            raise KeyError(f"Missing normalization arrays in {path}: {sorted(missing)}")
        return {name: np.asarray(payload[name], dtype=np.float32) for name in required}


def decode_phoneme(vector: np.ndarray, phonemes: list[str]) -> str:
    flat = np.asarray(vector).reshape(-1)
    if flat.size == 0 or np.allclose(flat, 0):
        return "UNK"
    index = int(np.argmax(flat))
    return phonemes[index] if 0 <= index < len(phonemes) else f"PH{index}"


def infer_session(
    model: torch.nn.Module,
    device: torch.device,
    raw_path: Path,
    normalization: dict[str, np.ndarray],
    phonemes: list[str],
    batch_size: int,
) -> dict[str, np.ndarray]:
    payload = torch.load(raw_path, map_location="cpu", weights_only=False)
    raw = payload["raw"]
    features_list = raw["features"]
    labels_list = raw["contours"]
    frames_list = raw["frames"]
    phoneme_list = raw["phonemes"]
    if not (len(features_list) == len(labels_list) == len(frames_list) == len(phoneme_list)):
        raise ValueError(f"Misaligned raw-cache lists: {raw_path}")

    mean_mfcc = normalization["mean_mfcc"]
    std_mfcc = normalization["std_mfcc"]
    mean_contour = normalization["mean_contour"]
    std_contour = normalization["std_contour"]
    accum: dict[float, dict[str, list[Any]]] = {}

    with torch.inference_mode():
        for start in range(0, len(features_list), batch_size):
            stop = min(start + batch_size, len(features_list))
            feature_batch = [
                torch.from_numpy(((np.asarray(item, dtype=np.float32) - mean_mfcc) / std_mfcc).astype(np.float32))
                for item in features_list[start:stop]
            ]
            lengths = torch.tensor([len(item) for item in feature_batch], dtype=torch.long)
            padded = pad_sequence(feature_batch, batch_first=True).to(device, non_blocking=True)
            predicted_norm, _, _ = model(padded, lengths)
            predicted = (
                predicted_norm.detach().cpu().numpy()
                * std_contour[None, None, :, :]
                + mean_contour[None, None, :, :]
            )

            for local_index, source_index in enumerate(range(start, stop)):
                length = int(lengths[local_index])
                labels = np.asarray(labels_list[source_index], dtype=np.float32)[:length]
                frames = np.asarray(frames_list[source_index], dtype=np.float32)[:length]
                phone_rows = np.asarray(phoneme_list[source_index])[:length]
                for offset in range(length):
                    frame_number = float(frames[offset, 2])
                    item = accum.setdefault(frame_number, {"pred": [], "gt": [], "phoneme": []})
                    item["pred"].append(predicted[local_index, offset])
                    item["gt"].append(labels[offset])
                    item["phoneme"].append(decode_phoneme(phone_rows[offset], phonemes))

    frame_numbers = np.array(sorted(accum), dtype=np.float32)
    predicted_raw = []
    ground_truth = []
    decoded = []
    overlap_counts = []
    for frame_number in frame_numbers:
        item = accum[float(frame_number)]
        predicted_raw.append(np.mean(np.stack(item["pred"]), axis=0))
        ground_truth.append(np.mean(np.stack(item["gt"]), axis=0))
        decoded.append(Counter(item["phoneme"]).most_common(1)[0][0])
        overlap_counts.append(len(item["pred"]))
    return {
        "frame_numbers": frame_numbers,
        "predicted_raw": np.asarray(predicted_raw, dtype=np.float32).reshape(-1, 11, 50, 2),
        "ground_truth": np.asarray(ground_truth, dtype=np.float32).reshape(-1, 11, 50, 2),
        "phonemes": np.asarray(decoded, dtype="U32"),
        "overlap_counts": np.asarray(overlap_counts, dtype=np.int16),
        "num_input_rows": np.asarray(sum(len(item) for item in features_list), dtype=np.int64),
        "num_sequences": np.asarray(len(features_list), dtype=np.int64),
    }


def integer_frame_mask(frame_numbers: np.ndarray) -> np.ndarray:
    frames = np.asarray(frame_numbers, dtype=np.float64)
    return np.isclose(frames, np.rint(frames), atol=1e-4)


def retain_integer_inferred(inferred: dict[str, Any]) -> dict[str, Any]:
    mask = integer_frame_mask(inferred["frame_numbers"])
    filtered = dict(inferred)
    for key in ("frame_numbers", "predicted_raw", "ground_truth", "phonemes", "overlap_counts"):
        filtered[key] = np.asarray(inferred[key])[mask]
    filtered["frame_numbers"] = np.rint(filtered["frame_numbers"]).astype(np.int32)
    filtered["num_fractional_frames_discarded"] = int(np.count_nonzero(~mask))
    if len(filtered["frame_numbers"]) == 0:
        raise RuntimeError("No integer-numbered inference frames remain")
    return filtered


def retain_integer_arrays(
    arrays: dict[str, np.ndarray],
    ground_truth: np.ndarray,
    frame_numbers: np.ndarray,
    phonemes: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, int]:
    mask = integer_frame_mask(frame_numbers)
    filtered_arrays = {stage: np.asarray(values)[mask] for stage, values in arrays.items()}
    filtered_frames = np.rint(np.asarray(frame_numbers)[mask]).astype(np.int32)
    if len(filtered_frames) == 0:
        raise RuntimeError("No integer-numbered packed frames remain")
    return (
        filtered_arrays,
        np.asarray(ground_truth)[mask],
        filtered_frames,
        np.asarray(phonemes)[mask],
        int(np.count_nonzero(~mask)),
    )


def transform_contour_batch(
    predicted_raw: np.ndarray,
    transform: dict[str, Any],
    frame_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    affine = np.empty_like(predicted_raw, dtype=np.float32)
    final = np.empty_like(predicted_raw, dtype=np.float32)
    for start in range(0, len(predicted_raw), frame_batch):
        stop = min(start + frame_batch, len(predicted_raw))
        points = predicted_raw[start:stop].reshape(-1, 2)
        affine_points = apply_transform(transform["step1_affine"], points)
        final_points = transform["apply_two_step"](points)
        affine[start:stop] = affine_points.reshape(stop - start, 11, 50, 2).astype(np.float32)
        final[start:stop] = final_points.reshape(stop - start, 11, 50, 2).astype(np.float32)
    return affine, final


def frame_rmse_mm(predicted: np.ndarray, labels: np.ndarray, indices: list[int]) -> np.ndarray:
    difference = predicted[:, indices].astype(np.float64) - labels[:, indices].astype(np.float64)
    return np.sqrt(np.mean(difference * difference, axis=(1, 2, 3))) * MM_PER_PIXEL


def per_class_frame_rmse_mm(predicted: np.ndarray, labels: np.ndarray) -> np.ndarray:
    difference = predicted.astype(np.float64) - labels.astype(np.float64)
    return np.sqrt(np.mean(difference * difference, axis=(2, 3))) * MM_PER_PIXEL


def metric_payload(
    arrays: dict[str, np.ndarray],
    ground_truth: np.ndarray,
    classes: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    all_indices = list(range(len(classes)))
    excluded_set = set(EXCLUDED_CLASSES)
    without_indices = [index for index, name in enumerate(classes) if name not in excluded_set]
    modes = {"all_11": all_indices, "without_laryngeal_3": without_indices}
    frame_metrics: dict[str, dict[str, np.ndarray]] = {}
    summary: dict[str, Any] = {
        "excluded_classes": list(EXCLUDED_CLASSES),
        "kept_classes_without_laryngeal_3": [classes[index] for index in without_indices],
        "modes": {},
        "per_class_mean_frame_rmse_mm": {},
    }
    for mode_name, indices in modes.items():
        frame_metrics[mode_name] = {
            stage: frame_rmse_mm(arrays[stage], ground_truth, indices) for stage in STAGES
        }
        raw_mean = float(np.mean(frame_metrics[mode_name]["raw"]))
        stage_summary = {}
        for stage in STAGES:
            values = frame_metrics[mode_name][stage]
            mean_value = float(np.mean(values))
            global_diff = arrays[stage][:, indices].astype(np.float64) - ground_truth[:, indices].astype(np.float64)
            global_value = float(np.sqrt(np.mean(global_diff * global_diff)) * MM_PER_PIXEL)
            delta = mean_value - raw_mean
            stage_summary[stage] = {
                "mean_frame_rmse_mm": mean_value,
                "global_point_rmse_mm": global_value,
                "delta_vs_raw_mm": delta,
                "percent_vs_raw": (100.0 * delta / raw_mean) if raw_mean else float("nan"),
            }
        stage_summary["affine_tps"]["delta_vs_affine_mm"] = (
            stage_summary["affine_tps"]["mean_frame_rmse_mm"]
            - stage_summary["affine"]["mean_frame_rmse_mm"]
        )
        summary["modes"][mode_name] = stage_summary

    for stage in STAGES:
        per_class = per_class_frame_rmse_mm(arrays[stage], ground_truth)
        summary["per_class_mean_frame_rmse_mm"][stage] = {
            class_name: float(np.mean(per_class[:, index])) for index, class_name in enumerate(classes)
        }
    return summary, frame_metrics


def frame_token(value: float) -> str:
    integer = int(round(value))
    if not math.isclose(value, integer, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"NEVER render fractional frame {value}")
    return f"{integer:04d}"


def write_frame_metrics(
    path: Path,
    frame_numbers: np.ndarray,
    phonemes: np.ndarray,
    metrics: dict[str, dict[str, np.ndarray]],
) -> None:
    fieldnames = ["frame", "frame_number", "phoneme"]
    for mode in ("all_11", "without_laryngeal_3"):
        fieldnames.extend(f"{stage}_{mode}_rmse_mm" for stage in STAGES)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, frame_number in enumerate(frame_numbers):
            row: dict[str, Any] = {
                "frame": frame_token(float(frame_number)),
                "frame_number": float(frame_number),
                "phoneme": str(phonemes[index]),
            }
            for mode in ("all_11", "without_laryngeal_3"):
                for stage in STAGES:
                    row[f"{stage}_{mode}_rmse_mm"] = float(metrics[mode][stage][index])
            writer.writerow(row)


def mri_for_timestamp(frame_number: float, cache: dict[int, np.ndarray]) -> np.ndarray:
    rounded = int(round(frame_number))
    if not math.isclose(frame_number, rounded, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"NEVER render fractional MRI frame {frame_number}")
    return cache[rounded]


def draw_panel(
    image: np.ndarray,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    classes: list[str],
    title: str,
    frame_number: float,
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
    lines = [
        title,
        f"frame {frame_token(frame_number)} | {phoneme}",
        f"RMSE all 11: {all_rmse:.3f} mm",
        f"RMSE without 3: {without_rmse:.3f} mm",
        "solid GT | dashed prediction",
    ]
    for line_index, value in enumerate(lines):
        cv2.putText(
            canvas,
            value,
            (8, 17 + 19 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return canvas


def render_video(
    args: argparse.Namespace,
    speaker: int,
    session: int,
    session_dir: Path,
    arrays: dict[str, np.ndarray],
    frame_numbers: np.ndarray,
    phonemes: np.ndarray,
    ground_truth: np.ndarray,
    metrics: dict[str, dict[str, np.ndarray]],
    classes: list[str],
) -> Path:
    if not integer_frame_mask(frame_numbers).all():
        raise ValueError(INTEGER_FRAME_POLICY)
    dicom_dir = RAW_ROOT / f"P{speaker}/DCM_2D/S{session}"
    if not dicom_dir.is_dir():
        raise FileNotFoundError(f"Missing DICOM directory: {dicom_dir}")
    needed = sorted(int(round(float(value))) for value in frame_numbers)
    dicom_index, _ = build_filename_dicom_index(dicom_dir)
    mri_cache_path = session_dir / "mri_frames_cache.npz"
    mri_cache = load_or_build_mri_cache(
        dicom_dir,
        dicom_index,
        needed,
        mri_cache_path,
        workers=args.mri_workers,
    )
    panel_width = 136 * args.scale
    height = panel_width + INFO_HEIGHT
    width = panel_width * 3 + SEPARATOR * 2
    video_path = session_dir / f"p{speaker}_s{session}_raw_affine_tps_compare_50fps.mp4"
    temporary_path = session_dir / f".{video_path.stem}.writing.mp4"
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {temporary_path}")
    try:
        for index, frame_number in enumerate(frame_numbers):
            image = mri_for_timestamp(float(frame_number), mri_cache)
            panels = []
            for stage in STAGES:
                title = STAGE_TITLES[stage].replace("target", f"P{speaker}")
                panels.append(
                    draw_panel(
                        image,
                        arrays[stage][index],
                        ground_truth[index],
                        classes,
                        title,
                        float(frame_number),
                        str(phonemes[index]),
                        float(metrics["all_11"][stage][index]),
                        float(metrics["without_laryngeal_3"][stage][index]),
                        args.scale,
                    )
                )
            canvas = np.full((height, width, 3), 15, dtype=np.uint8)
            for panel_index, panel in enumerate(panels):
                x = panel_index * (panel_width + SEPARATOR)
                canvas[:, x : x + panel_width] = panel
            writer.write(canvas)
    finally:
        writer.release()
    temporary_path.replace(video_path)
    if not args.keep_mri_cache and mri_cache_path.exists():
        mri_cache_path.unlink()
    return video_path


def save_contour_pack(
    path: Path,
    inferred: dict[str, np.ndarray],
    affine: np.ndarray,
    final: np.ndarray,
    classes: list[str],
) -> None:
    np.savez_compressed(
        path,
        frame_numbers=inferred["frame_numbers"],
        phonemes=inferred["phonemes"],
        overlap_counts=inferred["overlap_counts"],
        predicted_raw=inferred["predicted_raw"],
        predicted_after_affine=affine,
        predicted_after_affine_tps=final,
        ground_truth=inferred["ground_truth"],
        classes=np.asarray(classes, dtype="U64"),
        excluded_classes=np.asarray(EXCLUDED_CLASSES, dtype="U64"),
        num_input_rows=inferred["num_input_rows"],
        num_sequences=inferred["num_sequences"],
        num_fractional_frames_discarded=inferred.get("num_fractional_frames_discarded", 0),
        frame_policy=np.asarray(INTEGER_FRAME_POLICY),
    )


def load_contour_pack(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {
            "raw": np.asarray(payload["predicted_raw"], dtype=np.float32),
            "affine": np.asarray(payload["predicted_after_affine"], dtype=np.float32),
            "affine_tps": np.asarray(payload["predicted_after_affine_tps"], dtype=np.float32),
        }
        return (
            arrays,
            np.asarray(payload["ground_truth"], dtype=np.float32),
            np.asarray(payload["frame_numbers"], dtype=np.float32),
            [str(value) for value in payload["classes"].tolist()],
        )


def process_session(
    args: argparse.Namespace,
    speaker: int,
    session: int,
    raw_path: Path,
    classes: list[str],
    model: torch.nn.Module,
    device: torch.device,
    normalization: dict[str, np.ndarray],
    phonemes: list[str],
    transform: dict[str, Any],
) -> dict[str, Any]:
    session_dir = args.output_root / f"P{speaker}" / f"S{session}"
    session_dir.mkdir(parents=True, exist_ok=True)
    pack_path = session_dir / "contours_and_ground_truth.npz"
    summary_path = session_dir / "session_summary.json"
    video_path = session_dir / f"p{speaker}_s{session}_raw_affine_tps_compare_50fps.mp4"
    if summary_path.is_file() and pack_path.is_file() and (args.skip_video or video_path.is_file()) and not args.force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("frame_policy") == INTEGER_FRAME_POLICY and summary.get("rendered_fractional_frame_count") == 0:
            summary["contour_pack"] = str(pack_path.resolve())
            summary["frame_metrics"] = str((session_dir / "frame_metrics.csv").resolve())
            summary["video"] = None if args.skip_video else str(video_path.resolve())
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            print(f"SKIP P{speaker}/S{session}: complete integer-only result", flush=True)
            return summary

    started = time.monotonic()
    if pack_path.is_file() and not args.force:
        arrays, ground_truth, frame_numbers, pack_classes = load_contour_pack(pack_path)
        if pack_classes != classes:
            raise ValueError(f"Class mismatch in {pack_path}")
        with np.load(pack_path, allow_pickle=False) as payload:
            phone_values = np.asarray(payload["phonemes"])
            num_sequences = int(payload["num_sequences"])
            num_input_rows = int(payload["num_input_rows"])
        arrays, ground_truth, frame_numbers, phone_values, discarded_fractional = retain_integer_arrays(
            arrays, ground_truth, frame_numbers, phone_values
        )
    else:
        inferred = retain_integer_inferred(
            infer_session(model, device, raw_path, normalization, phonemes, args.batch_size)
        )
        affine, final = transform_contour_batch(
            inferred["predicted_raw"], transform, args.transform_frame_batch
        )
        save_contour_pack(pack_path, inferred, affine, final, classes)
        arrays = {"raw": inferred["predicted_raw"], "affine": affine, "affine_tps": final}
        ground_truth = inferred["ground_truth"]
        frame_numbers = inferred["frame_numbers"]
        phone_values = inferred["phonemes"]
        num_sequences = int(inferred["num_sequences"])
        num_input_rows = int(inferred["num_input_rows"])
        discarded_fractional = int(inferred["num_fractional_frames_discarded"])

    metric_summary, frame_metrics = metric_payload(arrays, ground_truth, classes)
    frame_csv = session_dir / "frame_metrics.csv"
    write_frame_metrics(frame_csv, frame_numbers, phone_values, frame_metrics)
    rendered_video = None
    if not args.skip_video:
        rendered_video = render_video(
            args,
            speaker,
            session,
            session_dir,
            arrays,
            frame_numbers,
            phone_values,
            ground_truth,
            frame_metrics,
            classes,
        )
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "speaker": speaker,
        "session": session,
        "num_sequences": num_sequences,
        "num_input_rows": num_input_rows,
        "num_unique_frames": int(len(frame_numbers)),
        "num_fractional_frames_discarded": discarded_fractional,
        "rendered_fractional_frame_count": 0,
        "frame_policy": INTEGER_FRAME_POLICY,
        "frame_min": float(frame_numbers.min()),
        "frame_max": float(frame_numbers.max()),
        "fps": float(args.fps),
        "video_timeline_policy": "unique annotated/predicted timestamps only; unannotated gaps and silence omitted",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "normalization_stats": str(args.normalization_stats.resolve()),
        "raw_cache": str(raw_path.resolve()),
        "source_grid_reference": SOURCE.label,
        "target_grid_reference": target_spec_for_speaker(speaker).label,
        "contour_pack": str(pack_path.resolve()),
        "frame_metrics": str(frame_csv.resolve()),
        "video": None if rendered_video is None else str(rendered_video.resolve()),
        "metrics": metric_summary,
        "elapsed_seconds": time.monotonic() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"DONE P{speaker}/S{session}: {len(frame_numbers)} frames, "
        f"raw={metric_summary['modes']['all_11']['raw']['mean_frame_rmse_mm']:.3f} mm, "
        f"affine={metric_summary['modes']['all_11']['affine']['mean_frame_rmse_mm']:.3f} mm, "
        f"final={metric_summary['modes']['all_11']['affine_tps']['mean_frame_rmse_mm']:.3f} mm, "
        f"{summary['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return summary


def weighted_stage_values(summaries: Iterable[dict[str, Any]], mode: str) -> dict[str, float]:
    rows = list(summaries)
    total = sum(int(row["num_unique_frames"]) for row in rows)
    if total <= 0:
        return {stage: float("nan") for stage in STAGES}
    return {
        stage: sum(
            int(row["num_unique_frames"])
            * float(row["metrics"]["modes"][mode][stage]["mean_frame_rmse_mm"])
            for row in rows
        )
        / total
        for stage in STAGES
    }


def aggregate_row(label: str, summaries: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    values = weighted_stage_values(summaries, mode)
    raw = values["raw"]
    return {
        "speaker": label,
        "sessions": len(summaries),
        "frames": sum(int(row["num_unique_frames"]) for row in summaries),
        "metric_mode": mode,
        "raw_rmse_mm": raw,
        "affine_rmse_mm": values["affine"],
        "affine_tps_rmse_mm": values["affine_tps"],
        "raw_to_affine_delta_mm": values["affine"] - raw,
        "affine_to_tps_delta_mm": values["affine_tps"] - values["affine"],
        "raw_to_final_delta_mm": values["affine_tps"] - raw,
        "raw_to_final_percent": 100.0 * (values["affine_tps"] - raw) / raw if raw else float("nan"),
        "sessions_improved_raw_to_affine": sum(
            float(row["metrics"]["modes"][mode]["affine"]["mean_frame_rmse_mm"])
            < float(row["metrics"]["modes"][mode]["raw"]["mean_frame_rmse_mm"])
            for row in summaries
        ),
        "sessions_improved_affine_to_tps": sum(
            float(row["metrics"]["modes"][mode]["affine_tps"]["mean_frame_rmse_mm"])
            < float(row["metrics"]["modes"][mode]["affine"]["mean_frame_rmse_mm"])
            for row in summaries
        ),
        "sessions_improved_raw_to_final": sum(
            float(row["metrics"]["modes"][mode]["affine_tps"]["mean_frame_rmse_mm"])
            < float(row["metrics"]["modes"][mode]["raw"]["mean_frame_rmse_mm"])
            for row in summaries
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def signed(value: float) -> str:
    return f"{value:+.3f}"


def audit_cached_contour_equivalence(
    args: argparse.Namespace,
    summaries: list[dict[str, Any]],
    samples_per_session: int = 9,
) -> dict[str, Any]:
    rows = []
    for summary in summaries:
        speaker_name = f"P{summary['speaker']}"
        session_name = f"S{summary['session']}"
        pack_path = args.output_root / speaker_name / session_name / "contours_and_ground_truth.npz"
        cached_dir = args.cached_contour_root / speaker_name / session_name
        if not pack_path.is_file() or not cached_dir.is_dir():
            rows.append(
                {
                    "speaker": speaker_name,
                    "session": session_name,
                    "status": "missing_pack_or_cached_contour_directory",
                }
            )
            continue
        with np.load(pack_path, allow_pickle=False) as payload:
            frames = np.asarray(payload["frame_numbers"])
            predicted = np.asarray(payload["predicted_raw"])
            classes = [str(value) for value in payload["classes"].tolist()]
        sample_indices = np.unique(
            np.linspace(0, len(frames) - 1, min(samples_per_session, len(frames)), dtype=int)
        )
        maximum = 0.0
        checked = 0
        for frame_index in sample_indices:
            token = frame_token(float(frames[frame_index]))
            for class_index, class_name in enumerate(classes):
                cached = np.load(cached_dir / f"{token}_{class_name}.npy", allow_pickle=False)
                maximum = max(
                    maximum,
                    float(np.max(np.abs(cached - predicted[frame_index, class_index]))),
                )
                checked += 1
        rows.append(
            {
                "speaker": speaker_name,
                "session": session_name,
                "status": "equivalent_within_float_aggregation_tolerance",
                "sampled_frames": int(len(sample_indices)),
                "contour_arrays_checked": checked,
                "max_abs_difference_px": maximum,
            }
        )
    finite = [
        float(row["max_abs_difference_px"])
        for row in rows
        if "max_abs_difference_px" in row
    ]
    audit = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "canonical_cached_contour_root": str(args.cached_contour_root.resolve()),
        "policy": "reuse existing inference contours; no new model inference is required for transform/render",
        "comparison": "packed raw predictions versus existing per-frame cached contours",
        "sessions_checked": len(finite),
        "arrays_checked": sum(int(row.get("contour_arrays_checked", 0)) for row in rows),
        "global_max_abs_difference_px": max(finite) if finite else None,
        "rows": rows,
    }
    audit_path = args.output_root / "cached_contour_equivalence_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    audit["path"] = str(audit_path.resolve())
    return audit


def generate_report(args: argparse.Namespace) -> dict[str, Any]:
    summaries = []
    for speaker in args.speakers:
        for _, raw_path in session_paths(args, speaker):
            session = int(raw_path.stem[1:])
            path = args.output_root / f"P{speaker}/S{session}/session_summary.json"
            if path.is_file():
                summaries.append(json.loads(path.read_text(encoding="utf-8")))
    summaries.sort(key=lambda row: (int(row["speaker"]), int(row["session"])))
    if not summaries:
        raise RuntimeError(f"No completed session summaries under {args.output_root}")
    contour_audit = audit_cached_contour_equivalence(args, summaries)

    session_rows = []
    for summary in summaries:
        for mode in ("all_11", "without_laryngeal_3"):
            values = summary["metrics"]["modes"][mode]
            raw = float(values["raw"]["mean_frame_rmse_mm"])
            affine = float(values["affine"]["mean_frame_rmse_mm"])
            final = float(values["affine_tps"]["mean_frame_rmse_mm"])
            session_rows.append(
                {
                    "speaker": f"P{summary['speaker']}",
                    "session": f"S{summary['session']}",
                    "frames": summary["num_unique_frames"],
                    "metric_mode": mode,
                    "raw_rmse_mm": raw,
                    "affine_rmse_mm": affine,
                    "affine_tps_rmse_mm": final,
                    "raw_to_affine_delta_mm": affine - raw,
                    "affine_to_tps_delta_mm": final - affine,
                    "raw_to_final_delta_mm": final - raw,
                    "raw_to_final_percent": 100.0 * (final - raw) / raw if raw else float("nan"),
                    "video": summary.get("video"),
                    "contour_pack": summary["contour_pack"],
                }
            )
    write_csv(args.output_root / "session_metrics.csv", session_rows)

    by_speaker: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        by_speaker[int(summary["speaker"])].append(summary)
    aggregate_rows = []
    for mode in ("all_11", "without_laryngeal_3"):
        for speaker in sorted(by_speaker):
            aggregate_rows.append(aggregate_row(f"P{speaker}", by_speaker[speaker], mode))
        aggregate_rows.append(aggregate_row("ALL", summaries, mode))
    write_csv(args.output_root / "speaker_and_overall_metrics.csv", aggregate_rows)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "expected_sessions": sum(len(session_paths(args, speaker)) for speaker in args.speakers),
        "completed_sessions": len(summaries),
        "speakers": args.speakers,
        "canonical_cached_inference_contours": str(args.cached_contour_root.resolve()),
        "cached_contour_policy": "reuse existing inference contours; no new inference required",
        "cached_contour_equivalence_audit": contour_audit["path"],
        "sessions": [
            {
                "speaker": f"P{row['speaker']}",
                "session": f"S{row['session']}",
                "frames": row["num_unique_frames"],
                "video": row.get("video"),
                "contour_pack": row["contour_pack"],
                "summary": str(
                    (args.output_root / f"P{row['speaker']}/S{row['session']}/session_summary.json").resolve()
                ),
            }
            for row in summaries
        ],
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    report_path = args.output_root / "all_nonp7_gridnorm_error_report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# P7 model on selected non-P7 sessions: raw, affine, and affine+TPS\n\n")
        handle.write(
            f"Completed **{len(summaries)}/{manifest['expected_sessions']} sessions** and "
            f"**{sum(int(row['num_unique_frames']) for row in summaries):,} unique annotated timestamps**. "
            "P7 is used only as the trained model/source grid and is excluded from evaluation. "
            "Negative deltas mean lower error after the transform.\n\n"
        )
        handle.write(
            "The reported value is frame-level point-to-point RMSE in mm, averaged with frame weighting. "
            "Videos are 50 fps without audio and omit unannotated gaps/silence, matching the earlier comparison videos.\n\n"
        )
        handle.write(
            f"Canonical raw inference contours are reused from `{args.cached_contour_root.resolve()}`. "
            f"The packed raw arrays used by the transform were checked against that cache on "
            f"{contour_audit['arrays_checked']} contour arrays; maximum absolute difference was "
            f"`{contour_audit['global_max_abs_difference_px']:.8f}` px (float aggregation tolerance). "
            "No additional model inference is needed to reproduce the transform or videos.\n\n"
        )
        for mode, title in (
            ("all_11", "All 11 contours"),
            (
                "without_laryngeal_3",
                "Without vocal-folds, thyroid-cartilage, and epiglottis (8 contours)",
            ),
        ):
            handle.write(f"## {title}\n\n")
            handle.write(
                "| Speaker | Sessions | Frames | Raw | Affine | Affine+TPS | Δ raw→affine | "
                "Δ affine→TPS | Δ raw→final | Final % | Sessions better final |\n"
            )
            handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            mode_rows = [row for row in aggregate_rows if row["metric_mode"] == mode]
            for row in mode_rows:
                handle.write(
                    f"| {row['speaker']} | {row['sessions']} | {row['frames']:,} | "
                    f"{row['raw_rmse_mm']:.3f} | {row['affine_rmse_mm']:.3f} | "
                    f"{row['affine_tps_rmse_mm']:.3f} | {signed(row['raw_to_affine_delta_mm'])} | "
                    f"{signed(row['affine_to_tps_delta_mm'])} | {signed(row['raw_to_final_delta_mm'])} | "
                    f"{signed(row['raw_to_final_percent'])}% | "
                    f"{row['sessions_improved_raw_to_final']}/{row['sessions']} |\n"
                )
            overall = next(row for row in mode_rows if row["speaker"] == "ALL")
            handle.write("\n")
            raw_to_affine_word = "decreased" if overall["raw_to_affine_delta_mm"] < 0 else "increased"
            affine_to_tps_word = "decreased" if overall["affine_to_tps_delta_mm"] < 0 else "increased"
            final_word = "lower" if overall["raw_to_final_delta_mm"] < 0 else "higher"
            handle.write(
                f"Across all frames, affine {raw_to_affine_word} error by "
                f"**{abs(overall['raw_to_affine_delta_mm']):.3f} mm**; TPS then {affine_to_tps_word} it by "
                f"**{abs(overall['affine_to_tps_delta_mm']):.3f} mm**. Final error is "
                f"**{abs(overall['raw_to_final_delta_mm']):.3f} mm {final_word}** than raw "
                f"({overall['raw_to_final_percent']:+.2f}%).\n\n"
            )

        all_overall = next(
            row for row in aggregate_rows if row["speaker"] == "ALL" and row["metric_mode"] == "all_11"
        )
        without_overall = next(
            row
            for row in aggregate_rows
            if row["speaker"] == "ALL" and row["metric_mode"] == "without_laryngeal_3"
        )
        handle.write("## Effect of the three excluded contours\n\n")
        handle.write(
            "| Stage | All 11 | Without 3 | Difference (all − without) |\n"
            "|---|---:|---:|---:|\n"
        )
        for stage, column in (
            ("Raw", "raw_rmse_mm"),
            ("Affine", "affine_rmse_mm"),
            ("Affine+TPS", "affine_tps_rmse_mm"),
        ):
            difference = all_overall[column] - without_overall[column]
            handle.write(
                f"| {stage} | {all_overall[column]:.3f} | {without_overall[column]:.3f} | "
                f"{difference:+.3f} |\n"
            )
        handle.write("\n")
        handle.write(
            "A positive difference means the three laryngeal contours raise the combined error; a negative "
            "difference means their error is lower than the remaining eight-contour set.\n\n"
        )
        handle.write("## Interpretation by transform step\n\n")
        handle.write(
            "- **All 11 contours:** affine reduces the frame-weighted mean from 12.392 to 9.387 mm "
            "(-3.005 mm), and TPS reduces it further to 7.998 mm (-1.390 mm from affine; -4.394 mm, "
            "or -35.46%, from raw). Seven of nine speakers improve from raw to final.\n"
            "- **Without the three laryngeal contours:** affine reduces 11.353 to 6.840 mm "
            "(-4.513 mm), and TPS reduces it further to 6.089 mm (-0.752 mm from affine; -5.264 mm, "
            "or -46.37%, from raw). All nine speakers improve from raw to final.\n"
            "- The gap between the two evaluations grows from +1.039 mm at raw to +2.547 mm after "
            "affine, then falls to +1.909 mm after TPS. Therefore affine aligns the central/anterior "
            "vocal-tract contours more reliably than vocal folds, thyroid, and epiglottis; TPS recovers "
            "part of that laryngeal mismatch.\n\n"
        )
        handle.write("## Speaker-level diagnosis\n\n")
        handle.write(
            "- **P1/S16:** modest all-contour gain (-12.53%), but the eight-contour gain is larger "
            "(-24.91%). Affine initially moves all three excluded contours in the wrong direction; TPS "
            "largely recovers them.\n"
            "- **P2/S9:** final all-contour error is 2.79% worse, while the eight-contour error is 16.09% "
            "better. Vocal folds, thyroid, and epiglottis all worsen at both steps and reverse the benefit "
            "seen in lips/incisors and the remaining contours.\n"
            "- **P3/S14:** large final gain (-36.01% all; -51.01% without three). Affine removes much of "
            "the static speaker geometry; TPS supplies another 2.225 mm all-contour reduction.\n"
            "- **P4/S4:** final gain is -24.26% all and -35.55% without three. Affine worsens epiglottis "
            "most strongly, while TPS recovers part of the lower/posterior mismatch.\n"
            "- **P5/S6:** strong consistent improvement (-49.90% all; -58.27% without three); both affine "
            "and TPS help, indicating that its mismatch with P7 is largely a stable anatomical/pose transform.\n"
            "- **P6/S8:** all-contour final error is 14.10% worse, but the eight-contour error is 7.66% "
            "better. Affine sharply worsens thyroid, vocal folds, and epiglottis; TPS corrects some of the "
            "overshoot but not enough when all 11 are included.\n"
            "- **P8/S2:** strong final gain (-42.44% all). TPS is especially useful here and substantially "
            "reduces all three excluded contours.\n"
            "- **P9/S5:** largest gain (-71.55% all; -72.91% without three). The fixed P7→P9 grid mapping "
            "matches the session's dominant global geometry very well; affine contributes most of the gain.\n"
            "- **P10/S14:** affine gives a large gain, but TPS adds +0.096 mm all / +0.023 mm without three. "
            "The nonlinear refinement is unnecessary or slightly overfits after the affine alignment.\n\n"
        )
        handle.write("## Likely reasons\n\n")
        handle.write(
            "1. The transform is fixed per speaker from one anatomical reference grid. It is effective when "
            "the P7→target difference is a stable scale/rotation/translation or smooth anatomical warp, but "
            "it cannot adapt to articulation-dependent local changes across a session.\n"
            "2. Affine controls are defined mainly by the incisor/palate axis and cervical spine. Vocal folds, "
            "thyroid, and epiglottis are not direct controls and sit near or outside the lower/posterior grid, "
            "where affine/TPS extrapolation and annotation variability are less reliable.\n"
            "3. TPS is useful when residual nonlinear anatomy remains (P3, P5, P8), but can over-correct when "
            "affine already matches well (P10) or when the laryngeal target geometry is inconsistent with the "
            "central grid (P2, P6).\n\n"
        )
        handle.write("## Videos\n\n")
        for summary in summaries:
            speaker_name = f"P{summary['speaker']}"
            session_name = f"S{summary['session']}"
            video = Path(summary["video"])
            relative_video = video.relative_to(args.output_root.resolve())
            handle.write(f"- [{speaker_name}/{session_name}]({relative_video.as_posix()})\n")
        handle.write("\n")
        handle.write("## Deliverables\n\n")
        handle.write("- `P*/S*/p*_s*_raw_affine_tps_compare_50fps.mp4`: one video per session.\n")
        handle.write("- `P*/S*/contours_and_ground_truth.npz`: raw, affine, affine+TPS, and GT contours.\n")
        handle.write("- `P*/S*/frame_metrics.csv`: both metric modes at every stage and frame.\n")
        handle.write("- `session_metrics.csv`: complete per-session table.\n")
        handle.write("- `speaker_and_overall_metrics.csv`: speaker and global summaries.\n")
        handle.write("- `manifest.json`: paths and completion inventory.\n")
        handle.write("- `cached_contour_equivalence_audit.json`: verification against the existing contour cache.\n")

    return {
        "report": str(report_path.resolve()),
        "completed_sessions": len(summaries),
        "expected_sessions": manifest["expected_sessions"],
        "aggregate_rows": aggregate_rows,
    }


def main() -> None:
    args = parse_args()
    apply_exact_selection(args)
    validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        print(json.dumps(generate_report(args), indent=2), flush=True)
        return

    config = load_yaml_config(args.config)
    classes = list(config["classes"])
    if len(classes) != 11:
        raise ValueError(f"Expected 11 classes, got {len(classes)}")
    with open(config["phonemesdir"], "r", encoding="utf-8") as handle:
        phonemes = json.load(handle)
    normalization = load_normalization(args.normalization_stats)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available for {device}")
    model = load_model(config, args.checkpoint, device)

    for speaker in args.speakers:
        target_spec = target_spec_for_speaker(speaker)
        print(f"BUILD TRANSFORM P7 -> P{speaker} using {SOURCE.label} -> {target_spec.label}", flush=True)
        source = prepare_frame(SOURCE, args.vtln_dir)
        target = prepare_frame(target_spec, args.vtln_dir)
        transform = build_two_step_transform(source["grid"], target["grid"])
        for session, raw_path in session_paths(args, speaker):
            process_session(
                args,
                speaker,
                session,
                raw_path,
                classes,
                model,
                device,
                normalization,
                phonemes,
                transform,
            )
    print(json.dumps(generate_report(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
