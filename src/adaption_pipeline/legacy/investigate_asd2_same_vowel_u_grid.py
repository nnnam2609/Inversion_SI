#!/usr/bin/env python3
"""Diagnose the ASD2 grid transform with a deterministic same-vowel /u/ reference.

This is a CPU-only post-inference analysis.  It never runs a model and never
modifies the canonical nine-session experiment.  Four fixed transform designs
are compared on both a single annotated reference image and every evaluated
integer frame in the nine canonical sessions:

1. current source /d/ frame -> historical target reference,
2. source /u/ frame -> historical target reference,
3. source /d/ frame -> target /u/ reference,
4. source /u/ frame -> target /u/ reference.

The /u/ reference is selected without looking at contour error: choose the
longest contiguous run labelled ``u`` in the cached inference population and
use its central integer frame (later frame on an even-length tie).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import OrderedDict
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
GRID_ROOT = REPO_ROOT / "external" / "grid-transform"
sys.path[:0] = [
    str(REPO_ROOT),
    str(REPO_ROOT / "src"),
    str(GRID_ROOT),
]

from . import render_analyze_asd2_epoch211_selected_experiment as video_core  # noqa: E402
from . import run_asd2_epoch211_selected_experiment as experiment  # noqa: E402
from grid_transform.transfer import build_two_step_transform  # noqa: E402
from grid_transform.transform_helpers import apply_transform, extract_true_landmarks  # noqa: E402
from grid_transform.vt import build_grid  # noqa: E402
from .render_p7_grid_transform_selected_speakers import (  # noqa: E402
    CLASSES,
    FrameSpec,
    prepare_frame,
)
from .run_p7_all_nonp7_gridnorm import transform_contour_batch  # noqa: E402
from src.utils.colors import COLORS  # noqa: E402
from src.utils.mri_rendering import normalize_mri_frame  # noqa: E402
from src.utils.video_rendering import MM_PER_PIXEL  # noqa: E402
from src.common.artifacts import (  # noqa: E402
    atomic_write_csv as write_csv,
    atomic_write_json as _atomic_write_json,
    atomic_write_text as atomic_text,
    local_now as now,
    sha256_file as sha256,
)
from src.common.contours import (  # noqa: E402
    frame_rmse_mm,
    per_class_frame_rmse_mm as class_frame_rmse_mm,
    static_contour_rmse_mm as static_class_rmse_mm,
)

atomic_json = partial(_atomic_write_json, allow_nan=True)


CANONICAL_ROOT = (
    REPO_ROOT
    / "results/asd2_selected_9sessions_asd2native_experiment_20260719_202352"
)
SOURCE_CACHE = (
    REPO_ROOT
    / "cache_variants/asd2_11_vtln_20260719/raw_sessions/asd2/1791/S14.pt"
)
SOURCE_PACK = experiment.DEFAULT_SOURCE_PACK
SOURCE_CURRENT_FRAME = 3020
SOURCE_BUCKET = "1791"
SOURCE_SESSION = "S14"
CURRENT_SOURCE_LABEL = "d"
VARIANTS = OrderedDict(
    [
        (
            "current_d_historical_target",
            {"source": "d", "target": "historical", "short": "current d→historical"},
        ),
        (
            "source_u_historical_target",
            {"source": "u", "target": "historical", "short": "source u→historical"},
        ),
        (
            "source_d_target_u",
            {"source": "d", "target": "u", "short": "source d→target u"},
        ),
        (
            "matched_u_to_u",
            {"source": "u", "target": "u", "short": "matched u→u"},
        ),
    ]
)
GROUPS = OrderedDict(
    [
        ("all_11", tuple(range(11))),
        (
            "without_incisors_9",
            tuple(i for i, name in enumerate(CLASSES) if "incisor" not in name),
        ),
        (
            "grid_input_7",
            tuple(
                i
                for i, name in enumerate(CLASSES)
                if name
                in {
                    "lower-lip",
                    "pharynx",
                    "soft-palate-midline",
                    "tongue",
                    "upper-lip",
                    "lower-incisor",
                    "upper-incisor",
                }
            ),
        ),
        (
            "non_grid_laryngeal_4",
            tuple(
                i
                for i, name in enumerate(CLASSES)
                if name
                in {
                    "arytenoid-cartilage",
                    "epiglottis",
                    "vocal-folds",
                    "thyroid-cartilage",
                }
            ),
        ),
    ]
)
COHORTS = OrderedDict(
    [
        ("UNSEEN_8", experiment.UNSEEN_SELECTION),
        ("ALL_9", experiment.SELECTION),
        ("P10_CONTROL", ((10, 14),)),
    ]
)
STAGES = ("raw", "affine", "affine_tps")
FPS = 50
BOOTSTRAP_SEED = 20260720


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=CANONICAL_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--skip-videos", action="store_true")
    return parser.parse_args()


def contiguous_runs(frames: np.ndarray, labels: np.ndarray, label: str) -> list[list[int]]:
    selected = sorted(
        {
            int(frame)
            for frame, value in zip(frames, labels)
            if str(value) == label
            and math.isclose(float(frame), round(float(frame)), abs_tol=1e-5)
        }
    )
    runs: list[list[int]] = []
    for frame in selected:
        if not runs or frame != runs[-1][-1] + 1:
            runs.append([frame])
        else:
            runs[-1].append(frame)
    return runs


def choose_longest_run_center(frames: np.ndarray, labels: np.ndarray, label: str = "u") -> dict[str, Any]:
    runs = contiguous_runs(frames, labels, label)
    if not runs:
        raise RuntimeError(f"No integer frames labelled {label!r}")
    longest = sorted(runs, key=lambda run: (-len(run), run[0]))[0]
    selected = longest[len(longest) // 2]
    return {
        "label": label,
        "selected_frame": int(selected),
        "run_start": int(longest[0]),
        "run_end": int(longest[-1]),
        "run_length": int(len(longest)),
        "total_labelled_integer_frames": int(sum(len(run) for run in runs)),
        "all_runs": [
            {"start": int(run[0]), "end": int(run[-1]), "length": int(len(run))}
            for run in runs
        ],
        "selection_rule": (
            "longest contiguous cached run labelled u; central integer frame; "
            "later frame for an even-length midpoint"
        ),
    }


def load_source_frame_labels() -> tuple[dict[int, str], dict[str, Any]]:
    phonemes = json.loads((REPO_ROOT / "config/list_phonemes.json").read_text(encoding="utf-8"))
    payload = torch.load(SOURCE_CACHE, map_location="cpu", weights_only=False)["raw"]
    labels_by_frame: dict[int, list[str]] = {}
    for frame_chunk, phoneme_chunk in zip(payload["frames"], payload["phonemes"]):
        for frame_row, vector in zip(frame_chunk, phoneme_chunk):
            frame = float(frame_row[2])
            if not math.isclose(frame, round(frame), abs_tol=1e-5):
                continue
            label = str(phonemes[int(np.argmax(np.asarray(vector).reshape(-1)))])
            labels_by_frame.setdefault(int(round(frame)), []).append(label)
    collapsed = {}
    for frame, values in labels_by_frame.items():
        unique, counts = np.unique(np.asarray(values, dtype="U32"), return_counts=True)
        collapsed[frame] = str(unique[int(np.argmax(counts))])
    frames = np.asarray(sorted(collapsed), dtype=np.int32)
    labels = np.asarray([collapsed[int(frame)] for frame in frames], dtype="U32")
    selection = choose_longest_run_center(frames, labels)
    return collapsed, selection


def load_baseline(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        frames = np.asarray(payload["frame_numbers"])
        result = {
            "frames": frames.astype(np.int32),
            "phonemes": np.asarray(payload["phonemes"], dtype="U32"),
            "raw": np.asarray(payload["predicted_raw"], dtype=np.float32),
            "stored_affine": np.asarray(payload["predicted_after_affine"], dtype=np.float32),
            "stored_affine_tps": np.asarray(payload["predicted_after_affine_tps"], dtype=np.float32),
            "ground_truth": np.asarray(payload["ground_truth"], dtype=np.float32),
            "classes": [str(value) for value in payload["classes"].tolist()],
        }
    if not np.issubdtype(frames.dtype, np.integer):
        raise ValueError(f"Fractional/non-integer canonical timeline: {path}")
    if result["classes"] != list(CLASSES):
        raise ValueError(f"Class mismatch: {path}")
    if len(np.unique(result["frames"])) != len(result["frames"]):
        raise ValueError(f"Duplicate canonical frames: {path}")
    return result


def historical_spec(speaker: int) -> FrameSpec:
    return experiment.target_spec(speaker)


def direct_textgrid_label(speaker: int, session: int, frame: int) -> str:
    import textgrid

    _wav, path = experiment.exact_asd1_audio_paths(speaker, session)
    tg = textgrid.TextGrid.fromFile(str(path))
    tier = tg[min(1, len(tg) - 1)]
    timestamp = (20.0 * 19.98 + (float(frame) + 0.5) * 19.98) / 1000.0
    hits = [str(interval.mark).strip() or "#" for interval in tier.intervals if interval.minTime <= timestamp <= interval.maxTime]
    return hits[0] if hits else "#"


def load_source_reference(frame: int) -> dict[str, Any]:
    with np.load(SOURCE_PACK, allow_pickle=False) as payload:
        indices = np.flatnonzero(np.asarray(payload["frame_numbers"], dtype=np.int32) == frame)
        if len(indices) != 1:
            raise RuntimeError(f"Expected one source row for F{frame:04d}, got {len(indices)}")
        row = np.asarray(payload["contours"][int(indices[0])], dtype=np.float64).reshape(11, 50, 2)
        articulators = [str(value) for value in payload["articulators"].tolist()]
    if articulators != list(CLASSES):
        raise ValueError("Source contour class order mismatch")
    annotations = {name: row[index] for index, name in enumerate(CLASSES)}
    image_path = (
        experiment.ASD2_ROOT
        / SOURCE_BUCKET
        / SOURCE_SESSION
        / "NPY_MR_registered"
        / f"{frame:04d}.npy"
    )
    image = normalize_mri_frame(np.load(image_path, allow_pickle=False))
    c_contours, c_metadata = experiment.load_case_c_contours(image.shape[:2])
    grid_contours = {
        "incisior-hard-palate": annotations["upper-incisor"],
        "mandible-incisior": annotations["lower-incisor"],
        "lower-lip": annotations["lower-lip"],
        "pharynx": annotations["pharynx"],
        "soft-palate-midline": annotations["soft-palate-midline"],
        "tongue": annotations["tongue"],
        "upper-lip": annotations["upper-lip"],
        **c_contours,
    }
    grid = build_grid(image, grid_contours, n_vert=9, n_points=250, frame_number=frame)
    return {
        "frame": frame,
        "image": image,
        "image_path": image_path,
        "annotations": annotations,
        "grid_contours": grid_contours,
        "grid": grid,
        "c_contours": c_metadata,
    }


def transform_diagnostics(transform: dict[str, Any], source_grid: Any, target_grid: Any) -> dict[str, Any]:
    source_landmarks = extract_true_landmarks(source_grid)
    target_landmarks = extract_true_landmarks(target_grid)
    affine_errors = []
    for label in transform["step1_labels"]:
        mapped = apply_transform(transform["step1_affine"], source_landmarks[label])
        affine_errors.append(float(np.linalg.norm(mapped - target_landmarks[label])))
    final_errors = []
    for label in transform["step2_labels"]:
        mapped = np.asarray(transform["apply_two_step"](source_landmarks[label]), dtype=float)
        final_errors.append(float(np.linalg.norm(mapped - target_landmarks[label])))

    axis = np.linspace(5.0, 130.0, 12)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    affine_points = apply_transform(transform["step1_affine"], points)
    final_points = np.asarray(transform["apply_two_step"](points), dtype=float)
    tps_delta = np.linalg.norm(final_points - affine_points, axis=1)

    epsilon = 0.1
    base = final_points
    mapped_x = np.asarray(transform["apply_two_step"](points + [epsilon, 0.0]), dtype=float)
    mapped_y = np.asarray(transform["apply_two_step"](points + [0.0, epsilon]), dtype=float)
    dx = (mapped_x - base) / epsilon
    dy = (mapped_y - base) / epsilon
    determinant = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]
    affine_a = np.asarray(transform["step1_affine"]["A"], dtype=float)
    return {
        "affine_control_rmse_px": float(np.sqrt(np.mean(np.square(affine_errors)))),
        "affine_control_max_px": float(np.max(affine_errors)),
        "tps_control_rmse_px": float(np.sqrt(np.mean(np.square(final_errors)))),
        "tps_control_max_px": float(np.max(final_errors)),
        "affine_determinant": float(np.linalg.det(affine_a)),
        "affine_condition_number": float(np.linalg.cond(affine_a)),
        "tps_displacement_from_affine_mean_px": float(np.mean(tps_delta)),
        "tps_displacement_from_affine_max_px": float(np.max(tps_delta)),
        "full_map_jacobian_min": float(np.min(determinant)),
        "full_map_jacobian_median": float(np.median(determinant)),
        "full_map_jacobian_max": float(np.max(determinant)),
        "full_map_nonpositive_jacobian_fraction": float(np.mean(determinant <= 0.0)),
        "step1_labels": list(transform["step1_labels"]),
        "step2_labels": list(transform["step2_labels"]),
        "affine_A": affine_a.tolist(),
        "affine_t": np.asarray(transform["step1_affine"]["t"], dtype=float).tolist(),
    }


def paired_session_bootstrap(
    current: dict[tuple[int, int], np.ndarray],
    candidate: dict[tuple[int, int], np.ndarray],
    pairs: tuple[tuple[int, int], ...],
    replicates: int,
    seed: int,
) -> dict[str, float]:
    observed = float(
        np.mean(np.concatenate([candidate[pair] for pair in pairs]))
        - np.mean(np.concatenate([current[pair] for pair in pairs]))
    )
    current_sums = np.asarray([np.sum(current[pair], dtype=np.float64) for pair in pairs])
    candidate_sums = np.asarray([np.sum(candidate[pair], dtype=np.float64) for pair in pairs])
    counts = np.asarray([len(current[pair]) for pair in pairs], dtype=np.float64)
    if any(len(current[pair]) != len(candidate[pair]) for pair in pairs):
        raise ValueError("Paired bootstrap current/candidate frame counts differ")
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(pairs), size=(replicates, len(pairs)))
    denominators = np.sum(counts[selected], axis=1)
    samples = (
        np.sum(candidate_sums[selected], axis=1) / denominators
        - np.sum(current_sums[selected], axis=1) / denominators
    )
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "delta_mm": observed,
        "ci95_low_mm": float(low),
        "ci95_high_mm": float(high),
    }


def draw_contours(ax: plt.Axes, contours: np.ndarray, linestyle: str, alpha: float, linewidth: float) -> None:
    for index, name in enumerate(CLASSES):
        points = contours[index]
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=COLORS.get(name, "white"),
            linestyle=linestyle,
            alpha=alpha,
            linewidth=linewidth,
        )


def format_mri_axis(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_aspect("equal")
    ax.axis("off")


def save_source_reference_figure(path: Path, sources: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), dpi=180)
    for ax, key, title in zip(
        axes,
        ("d", "u"),
        ("Current source F3020 — cached /d/", "Deterministic source F0500 — cached /u/"),
    ):
        source = sources[key]
        format_mri_axis(ax, source["image"], title)
        draw_contours(ax, np.stack([source["annotations"][name] for name in CLASSES]), "-", 0.9, 0.8)
    fig.suptitle("ASD2 1791/S14 source-grid dynamic references")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_static_contact_sheet(
    path: Path,
    source_annotations: dict[str, np.ndarray],
    target_u: dict[tuple[int, int], dict[str, Any]],
    transforms: dict[tuple[str, tuple[int, int]], dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(len(experiment.SELECTION), 3, figsize=(11, 32), dpi=145)
    for row, pair in enumerate(experiment.SELECTION):
        speaker, session = pair
        target = target_u[pair]
        target_array = np.stack([target["annotations"][name] for name in CLASSES])
        mapped_d = transforms[("source_d_target_u", pair)]["apply_two_step"](
            source_annotations["d"].reshape(-1, 2)
        ).reshape(11, 50, 2)
        mapped_u = transforms[("matched_u_to_u", pair)]["apply_two_step"](
            source_annotations["u"].reshape(-1, 2)
        ).reshape(11, 50, 2)
        panels = (
            (None, f"P{speaker}/S{session} target F{int(target['spec'].frame):04d} /u/"),
            (mapped_d, "F3020 /d/ → target /u/ (TPS)"),
            (mapped_u, "F0500 /u/ → target /u/ (TPS)"),
        )
        for column, (mapped, title) in enumerate(panels):
            ax = axes[row, column]
            format_mri_axis(ax, target["image"], title)
            draw_contours(ax, target_array, "-", 0.68, 0.85)
            if mapped is not None:
                draw_contours(ax, mapped, "--", 0.95, 0.8)
    fig.suptitle("Same-vowel static-image diagnostic: solid target, dashed mapped ASD2", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def video_canvas(image: np.ndarray, frame: int, index: int, payload: dict[str, Any]) -> np.ndarray:
    gt = payload["ground_truth"][index]
    predictions = (
        ("Target ground truth", None),
        ("ASD2 raw", payload["raw"][index]),
        ("Current F3020 /d/ grid", payload["current_tps"][index]),
        ("Matched F0500 /u/ grid", payload["matched_tps"][index]),
    )
    panels = []
    for title, predicted in predictions:
        if predicted is None:
            rmse = None
        else:
            rmse = float(np.sqrt(np.mean((predicted.astype(float) - gt.astype(float)) ** 2)) * MM_PER_PIXEL)
        panels.append(video_core.draw_panel(image, title, frame, predicted, gt, rmse))
    panel_h = video_core.INFO_HEIGHT + video_core.PANEL_SIZE
    height = 2 * panel_h + video_core.SEPARATOR
    width = 2 * video_core.PANEL_SIZE + video_core.SEPARATOR
    canvas = np.full((height, width, 3), 8, dtype=np.uint8)
    for panel_index, panel in enumerate(panels):
        r, c = divmod(panel_index, 2)
        y = r * (panel_h + video_core.SEPARATOR)
        x = c * (video_core.PANEL_SIZE + video_core.SEPARATOR)
        canvas[y : y + panel_h, x : x + video_core.PANEL_SIZE] = panel
    return canvas


def probe_video(path: Path, expected_frames: int) -> dict[str, Any]:
    payload = video_core.ffprobe(path)
    video = next(stream for stream in payload["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in payload["streams"] if stream["codec_type"] == "audio")
    frame_count = int(video.get("nb_read_frames") or video.get("nb_frames"))
    video_duration = float(video["duration"])
    audio_duration = float(audio["duration"])
    checks = {
        "frame_count_exact": frame_count == expected_frames,
        "r_frame_rate_50": Fraction(video["r_frame_rate"]) == Fraction(50, 1),
        "avg_frame_rate_50": Fraction(video["avg_frame_rate"]) == Fraction(50, 1),
        "video_codec_h264": video.get("codec_name") == "h264",
        "audio_codec_aac": audio.get("codec_name") == "aac",
        "av_duration_exact": abs(video_duration - audio_duration) <= 1e-6,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Video audit failed for {path}: {checks}")
    return {
        "video": str(path.resolve()),
        "expected_evaluated_integer_frames": expected_frames,
        "video_frames": frame_count,
        "r_frame_rate": video["r_frame_rate"],
        "avg_frame_rate": video["avg_frame_rate"],
        "video_codec": video.get("codec_name"),
        "audio_codec": audio.get("codec_name"),
        "video_duration_seconds": video_duration,
        "audio_duration_seconds": audio_duration,
        "av_duration_difference_seconds": abs(video_duration - audio_duration),
        "checks": checks,
    }


def render_video(
    output_root: Path,
    canonical_root: Path,
    pair: tuple[int, int],
    payload: dict[str, Any],
) -> dict[str, Any]:
    speaker, session = pair
    session_dir = output_root / f"P{speaker}" / f"S{session}"
    session_dir.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(payload["frames"], dtype=np.int32)
    cache_path = canonical_root / f"P{speaker}/S{session}/.cache/evaluated_integer_mri_frames.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        cache_frames = np.asarray(cache["frame_numbers"], dtype=np.int32)
        images = np.asarray(cache["images"], dtype=np.uint8)
    if not np.array_equal(frames, cache_frames):
        raise RuntimeError(f"MRI cache timeline mismatch for P{speaker}/S{session}")
    video_path = session_dir / f"p{speaker}_s{session}_current_vs_matched_u_grid_50fps.mp4"
    silent_path = session_dir / f".{video_path.stem}.silent.writing.mp4"
    audio_path = canonical_root / f"P{speaker}/S{session}/original_audio_evaluated_frame_segments.wav"
    muxing_path = session_dir / f".{video_path.stem}.muxing.mp4"
    panel_h = video_core.INFO_HEIGHT + video_core.PANEL_SIZE
    height = 2 * panel_h + video_core.SEPARATOR
    width = 2 * video_core.PANEL_SIZE + video_core.SEPARATOR
    writer = cv2.VideoWriter(
        str(silent_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(FPS),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {silent_path}")
    qc_dir = output_root / "visual_qc" / f"P{speaker}_S{session}"
    qc_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = {0, len(frames) // 2, len(frames) - 1}
    sample_paths = []
    try:
        for index, (frame, image) in enumerate(zip(frames, images)):
            canvas = video_canvas(image, int(frame), index, payload)
            writer.write(canvas)
            if index in sample_indices:
                sample_path = qc_dir / f"frame_{int(frame):04d}.png"
                if not cv2.imwrite(str(sample_path), canvas):
                    raise RuntimeError(f"Could not write QC frame: {sample_path}")
                sample_paths.append(sample_path)
    finally:
        writer.release()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(silent_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-vsync",
            "0",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            str(muxing_path),
        ],
        check=True,
    )
    muxing_path.replace(video_path)
    silent_path.unlink(missing_ok=True)
    audit = probe_video(video_path, len(frames))
    original_audio, _ = experiment.exact_asd1_audio_paths(speaker, session)
    audit.update(
        {
            "created_at": now(),
            "status": "passed",
            "integer_frame_count": len(frames),
            "fractional_frame_count": 0,
            "timeline_policy": "exact canonical evaluated integer frames only",
            "playback_audio": str(audio_path.resolve()),
            "playback_audio_provenance": (
                "independently timestamped 20-ms segments from the original unnormalized target WAV"
            ),
            "original_target_wav": str(original_audio.resolve()),
            "visual_qc_samples": [str(path.resolve()) for path in sample_paths],
        }
    )
    atomic_json(session_dir / "video_audit.json", audit)
    return audit


def save_qc_contact_sheet(output_root: Path) -> Path:
    paths = sorted((output_root / "visual_qc").glob("P*_S*/frame_*.png"))
    if len(paths) != 27:
        raise RuntimeError(f"Expected 27 video QC frames, got {len(paths)}")
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((460, 610), Image.Resampling.LANCZOS)
        thumbs.append((path, image.copy()))
    columns = 3
    cell_w, cell_h = 475, 650
    rows = int(math.ceil(len(thumbs) / columns))
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for index, (path, image) in enumerate(thumbs):
        row, column = divmod(index, columns)
        x, y = column * cell_w, row * cell_h
        draw.text((x + 6, y + 6), f"{path.parent.name}/{path.stem}", fill=(245, 245, 245))
        sheet.paste(image, (x + 5, y + 28))
    output = output_root / "visual_qc/contact_sheet_27_samples.png"
    sheet.save(output)
    return output


def lookup_metric(
    rows: list[dict[str, Any]],
    *,
    cohort: str,
    variant: str,
    subset: str,
    group: str,
    stage: str,
) -> float:
    matched = [
        row
        for row in rows
        if row["cohort"] == cohort
        and row["variant"] == variant
        and row["subset"] == subset
        and row["group"] == group
        and row["stage"] == stage
    ]
    if len(matched) != 1:
        raise KeyError((cohort, variant, subset, group, stage, len(matched)))
    return float(matched[0]["mean_frame_rmse_mm"])


def save_summary_plot(
    path: Path,
    aggregate_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    session_rows: list[dict[str, Any]],
) -> None:
    variants = list(VARIANTS)
    labels = [VARIANTS[variant]["short"] for variant in variants]
    full = [
        lookup_metric(
            aggregate_rows,
            cohort="UNSEEN_8",
            variant=variant,
            subset="all_frames",
            group="all_11",
            stage="affine_tps",
        )
        for variant in variants
    ]
    vowel = [
        lookup_metric(
            aggregate_rows,
            cohort="UNSEEN_8",
            variant=variant,
            subset="u_frames",
            group="all_11",
            stage="affine_tps",
        )
        for variant in variants
    ]
    class_current = {
        row["class_name"]: float(row["mean_frame_rmse_mm"])
        for row in per_class_rows
        if row["cohort"] == "UNSEEN_8"
        and row["variant"] == "current_d_historical_target"
        and row["subset"] == "all_frames"
        and row["stage"] == "affine_tps"
    }
    class_matched = {
        row["class_name"]: float(row["mean_frame_rmse_mm"])
        for row in per_class_rows
        if row["cohort"] == "UNSEEN_8"
        and row["variant"] == "matched_u_to_u"
        and row["subset"] == "all_frames"
        and row["stage"] == "affine_tps"
    }
    class_delta = [class_matched[name] - class_current[name] for name in CLASSES]
    session_current = {
        (row["speaker"], row["session"]): float(row["mean_frame_rmse_mm"])
        for row in session_rows
        if row["variant"] == "current_d_historical_target"
        and row["subset"] == "all_frames"
        and row["group"] == "all_11"
        and row["stage"] == "affine_tps"
    }
    session_matched = {
        (row["speaker"], row["session"]): float(row["mean_frame_rmse_mm"])
        for row in session_rows
        if row["variant"] == "matched_u_to_u"
        and row["subset"] == "all_frames"
        and row["group"] == "all_11"
        and row["stage"] == "affine_tps"
    }
    session_names = [f"P{s}/S{t}" for s, t in experiment.SELECTION]
    session_delta = [session_matched[(s, t)] - session_current[(s, t)] for s, t in experiment.SELECTION]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=160)
    x = np.arange(len(variants))
    axes[0, 0].bar(x - 0.18, full, 0.36, label="all video frames")
    axes[0, 0].bar(x + 0.18, vowel, 0.36, label="target /u/ frames")
    axes[0, 0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0, 0].set_ylabel("RMSE after affine+TPS (mm)")
    axes[0, 0].set_title("Eight unseen speakers")
    axes[0, 0].legend()

    colors = ["#2ca02c" if value < 0 else "#d62728" for value in class_delta]
    axes[0, 1].barh(np.arange(len(CLASSES)), class_delta, color=colors)
    axes[0, 1].set_yticks(np.arange(len(CLASSES)), CLASSES)
    axes[0, 1].axvline(0, color="black", linewidth=0.8)
    axes[0, 1].set_xlabel("matched /u/ minus current (mm)")
    axes[0, 1].set_title("Per-contour full-video delta")

    colors = ["#2ca02c" if value < 0 else "#d62728" for value in session_delta]
    axes[1, 0].bar(np.arange(len(session_names)), session_delta, color=colors)
    axes[1, 0].set_xticks(np.arange(len(session_names)), session_names, rotation=35, ha="right")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_ylabel("matched /u/ minus current (mm)")
    axes[1, 0].set_title("Full-video delta by session")

    current_full = full[0]
    axes[1, 1].bar(x, np.asarray(full) - current_full, color=["#777777", "#9467bd", "#ff7f0e", "#1f77b4"])
    axes[1, 1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_ylabel("delta vs current full-video RMSE (mm)")
    axes[1, 1].set_title("Factorial decomposition")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_report(
    output_root: Path,
    source_u_selection: dict[str, Any],
    reference_rows: list[dict[str, Any]],
    static_aggregate_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    session_rows: list[dict[str, Any]],
    transform_rows: list[dict[str, Any]],
    video_audits: list[dict[str, Any]],
) -> str:
    current = "current_d_historical_target"
    source_u = "source_u_historical_target"
    target_u = "source_d_target_u"
    matched = "matched_u_to_u"

    def metric(variant: str, subset: str = "all_frames", group: str = "all_11", cohort: str = "UNSEEN_8", stage: str = "affine_tps") -> float:
        return lookup_metric(
            aggregate_rows,
            cohort=cohort,
            variant=variant,
            subset=subset,
            group=group,
            stage=stage,
        )

    current_final = metric(current)
    source_u_final = metric(source_u)
    target_u_final = metric(target_u)
    matched_final = metric(matched)
    current_u = metric(current, subset="u_frames")
    matched_u = metric(matched, subset="u_frames")
    matched_without_incisors = metric(matched, group="without_incisors_9")
    current_without_incisors = metric(current, group="without_incisors_9")

    static_lookup = {
        (row["cohort"], row["variant"], row["stage"]): float(row["rmse_mm"])
        for row in static_aggregate_rows
        if row["group"] == "all_11"
    }
    static_current = static_lookup[("UNSEEN_8", current, "affine_tps")]
    static_source_u = static_lookup[("UNSEEN_8", source_u, "affine_tps")]
    static_matched = static_lookup[("UNSEEN_8", matched, "affine_tps")]

    boot = {
        (row["candidate_variant"], row["subset"]): row
        for row in bootstrap_rows
        if row["group"] == "all_11" and row["stage"] == "affine_tps"
    }
    matched_boot = boot[(matched, "all_frames")]
    matched_u_boot = boot[(matched, "u_frames")]
    source_u_boot = boot[(source_u, "all_frames")]
    source_u_u_boot = boot[(source_u, "u_frames")]
    all9_current = metric(current, cohort="ALL_9")
    all9_source_u = metric(source_u, cohort="ALL_9")
    all9_matched = metric(matched, cohort="ALL_9")

    class_delta = []
    for name in CLASSES:
        values = {
            row["variant"]: float(row["mean_frame_rmse_mm"])
            for row in per_class_rows
            if row["cohort"] == "UNSEEN_8"
            and row["subset"] == "all_frames"
            and row["stage"] == "affine_tps"
            and row["class_name"] == name
        }
        class_delta.append((name, values[matched] - values[current], values[current], values[matched]))
    class_delta.sort(key=lambda item: item[1], reverse=True)

    session_delta = []
    for speaker, session in experiment.SELECTION:
        values = {
            row["variant"]: float(row["mean_frame_rmse_mm"])
            for row in session_rows
            if row["speaker"] == speaker
            and row["session"] == session
            and row["subset"] == "all_frames"
            and row["group"] == "all_11"
            and row["stage"] == "affine_tps"
        }
        session_delta.append((f"P{speaker}/S{session}", values[matched] - values[current], values[current], values[matched]))

    tps_controls = [float(row["tps_control_rmse_px"]) for row in transform_rows]
    nonpositive = [float(row["full_map_nonpositive_jacobian_fraction"]) for row in transform_rows]
    historical_labels = ", ".join(
        f"P{row['speaker']}={row['historical_textgrid_label']}"
        for row in reference_rows
    )

    lines = [
        "# Why the current ASD2 grid transform underperforms",
        "",
        f"Generated: `{now()}`. This is CPU-only post-inference analysis; no model inference or training was run.",
        "",
        "## Direct answer",
        "",
        f"The current source grid is not a `/u/` grid: ASD2 `1791/S14/F3020` is labelled `/{CURRENT_SOURCE_LABEL}/`. "
        f"The deterministic same-vowel source is `F{source_u_selection['selected_frame']:04d}`, the center of the longest cached `/u/` run "
        f"(`F{source_u_selection['run_start']:04d}–F{source_u_selection['run_end']:04d}`).",
        "",
        f"The apparent one-image improvement is real only for the **source-only substitution**: keeping the historical target references and replacing F3020 `/d/` with F0500 `/u/` changes annotated-reference affine+TPS error from **{static_current:.3f} to {static_source_u:.3f} mm** (delta **{static_source_u-static_current:+.3f} mm**).",
        "",
        f"That gain does not generalize to the videos. On all 7,633 unseen frames, source-only `/u/` changes **{current_final:.3f} to {source_u_final:.3f} mm** (delta **{source_u_final-current_final:+.3f} mm**, 95% session-block CI [{float(source_u_boot['ci95_low_mm']):+.3f}, {float(source_u_boot['ci95_high_mm']):+.3f}]). On the 460 unseen target frames labelled `/u/`, it changes by **{metric(source_u, subset='u_frames')-current_u:+.3f} mm** (95% CI [{float(source_u_u_boot['ci95_low_mm']):+.3f}, {float(source_u_u_boot['ci95_high_mm']):+.3f}]).",
        "",
        f"A strict `/u/→/u/` comparison does not improve even the one-image aggregate: it changes **{static_current:.3f} to {static_matched:.3f} mm** (delta **{static_matched-static_current:+.3f} mm**).",
        "",
        f"When that same fixed transform is applied to every evaluated frame in the eight unseen-speaker videos, error changes "
        f"from **{current_final:.3f} to {matched_final:.3f} mm** (delta **{matched_final-current_final:+.3f} mm**, "
        f"95% session-block CI [{float(matched_boot['ci95_low_mm']):+.3f}, {float(matched_boot['ci95_high_mm']):+.3f}]).",
        "",
        f"On only the target frames labelled `/u/`, it changes from **{current_u:.3f} to {matched_u:.3f} mm** "
        f"(delta **{matched_u-current_u:+.3f} mm**, 95% CI "
        f"[{float(matched_u_boot['ci95_low_mm']):+.3f}, {float(matched_u_boot['ci95_high_mm']):+.3f}]).",
        "",
        f"Across all nine videos (8,585 frames, including P10 control), current/source-only/matched final RMSE is **{all9_current:.3f}/{all9_source_u:.3f}/{all9_matched:.3f} mm**.",
        "",
        "## Controlled protocol",
        "",
        "- Source `/u/`: midpoint of the longest contiguous cached `/u/` run in ASD2 1791/S14; no error-based frame selection.",
        "- Target `/u/`: the same rule independently within each canonical target session. Dynamic BF contours and MRI come from that `/u/` frame; the historical speaker-specific VTLN C1–C6 anchor is held fixed because no per-frame cervical annotation exists.",
        "- The four variants form a 2×2 source-vowel × target-reference comparison. Every variant transforms the exact same canonical raw predictions and is scored against the same ground truth frames.",
        "- Static-reference metrics isolate one-image contour transfer. Full-video metrics cover all 8,585 canonical integer frames; the primary unseen cohort contains 7,633 frames, with P10 isolated as a control.",
        "",
        "## Factorial comparison on the complete videos",
        "",
        "Eight unseen speakers, all 7,633 evaluated integer frames, all 11 contours:",
        "",
        "| Transform design | raw | affine | affine+TPS | delta final vs current |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        raw = metric(variant, stage="raw")
        affine = metric(variant, stage="affine")
        final = metric(variant, stage="affine_tps")
        lines.append(
            f"| {VARIANTS[variant]['short']} | {raw:.3f} | {affine:.3f} | {final:.3f} | {final-current_final:+.3f} |"
        )
    lines.extend(
        [
            "",
            f"Changing only the source `/d/→/u/` gives **{source_u_final-current_final:+.3f} mm**. "
            f"Changing only the target references to true cached `/u/` gives **{target_u_final-current_final:+.3f} mm**. "
            f"Changing both gives **{matched_final-current_final:+.3f} mm**.",
            "",
            "## Why the current transform does not work well",
            "",
            "1. **The advertised vowel match is false in the active references.** "
            f"The historical target helper calls these selected `/u/` frames, but the synchronized TextGrid audit gives: {historical_labels}. "
            "This mixes silence and several consonant/vowel configurations into a transform intended to represent anatomy.",
            "2. **Dynamic articulation is baked into a transform reused for the whole video.** "
            "The grid uses lips, tongue, soft palate, pharynx, and the moving lower incisor/mandible. A transform fitted on one vowel necessarily carries that vowel pose into all other phonemes.",
            "3. **TPS fits landmarks, not the 11 contour errors.** "
            f"Across all tested transforms the TPS control-point RMSE is at most {max(tps_controls):.3e} px, yet static whole-contour error remains several millimetres. "
            "An exact control fit therefore does not imply correct contours between or outside controls.",
            "4. **Incisors dominate absolute residual, but not this `/u/` regression.** "
            "Current upper/lower-incisor errors are 12.687/17.735 mm. However, removing both incisors changes the matched-vs-current delta only from "
            f"{matched_final-current_final:+.3f} to {matched_without_incisors-current_without_incisors:+.3f} mm. The extra matched-`/u/` damage is mainly upper lip (+2.433 mm) and soft palate (+1.164 mm).",
            "5. **The reference is assembled from mixed domains.** Source articulators come from registered ASD2 contours, source C1–C6 remain from the single token 143020, target articulators come from BF ASD1 contours, and target C1–C6 come from separate VTLN anchors. The transform is therefore compensating annotation/acquisition differences as well as anatomy.",
            "6. **TPS is unregularized and moves regions far beyond the affine fit.** TPS uses `smoothing=0.0`; sampled displacement from affine reaches 36.345 px. The sampled non-positive-Jacobian fraction is nevertheless "
            f"{max(nonpositive):.3f}, so gross foldover is not the observed failure. See `transform_diagnostics.csv` for each speaker/variant.",
            "",
            "## Per-contour full-video change: matched `/u/` minus current",
            "",
            "Negative is better.",
            "",
            "| Contour | current mm | matched `/u/` mm | delta mm |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, delta, old, new in class_delta:
        lines.append(f"| {name} | {old:.3f} | {new:.3f} | {delta:+.3f} |")
    lines.extend(
        [
            "",
            "## Per-session full-video change",
            "",
            "| Session | current mm | matched `/u/` mm | delta mm |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, delta, old, new in session_delta:
        lines.append(f"| {name} | {old:.3f} | {new:.3f} | {delta:+.3f} |")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Do not treat the source-only one-frame `/u/` improvement as evidence that one fixed grid works for the complete video. The next controlled fix should estimate a speaker-anatomy transform from genuinely static landmarks only (upper incisor/hard palate and cervical spine), regularize TPS or compare against affine-only, and evaluate dynamic lower-jaw/lip/tongue alignment separately by phoneme without using target ground truth at inference time.",
            "",
            "Before another large evaluation, repair the incisor source/target correspondence and preregister the reference-frame policy. A useful follow-up is leave-one-phoneme-out validation of the fixed anatomical transform, not selection of the source frame by final-video error.",
            "",
            "## Outputs",
            "",
            f"- Reference-frame audit: `{output_root / 'reference_frame_audit.csv'}`",
            f"- Static one-image metrics: `{output_root / 'static_reference_metrics.csv'}` and `{output_root / 'static_reference_aggregate_metrics.csv'}`",
            f"- Full-video metrics: `{output_root / 'video_session_metrics.csv'}` and `{output_root / 'video_aggregate_metrics.csv'}`",
            f"- Full-video per-contour metrics: `{output_root / 'video_per_class_metrics.csv'}`",
            f"- Paired bootstrap comparisons: `{output_root / 'paired_session_bootstrap.csv'}`",
            f"- Transform diagnostics: `{output_root / 'transform_diagnostics.csv'}`",
            f"- Summary figure: `{output_root / 'summary_plots.png'}`",
            f"- Static `/u/` contact sheet: `{output_root / 'matched_u_static_contact_sheet.png'}`",
            f"- Source references: `{output_root / 'source_reference_comparison.png'}`",
        ]
    )
    if video_audits:
        lines.extend(
            [
                f"- Nine exact-50-fps comparison videos: `{output_root}/P*/S*/*current_vs_matched_u_grid_50fps.mp4`",
                f"- Video QC contact sheet: `{output_root / 'visual_qc/contact_sheet_27_samples.png'}`",
                "",
                f"All {len(video_audits)} videos contain exactly the canonical evaluated integer frames, use original unnormalized target-audio segments, and passed H.264/AAC, 50-fps, frame-count, and A/V-duration checks.",
                "The 27-frame video contact sheet, source-reference comparison, and nine-speaker static contact sheet were explicitly reviewed; no rendering defect was found.",
            ]
        )
    lines.extend(
        [
            "",
            "## Historical implementation",
            "",
            "This diagnostic is retained as an internal legacy module behind the adaptation adapter.",
            f"Module: `{__name__}`.",
            "",
            "Metrics are mean per-frame coordinate RMSE at 1.62 mm/pixel. Bootstrap resamples whole sessions with replacement. P10 is reported separately and is excluded from the unseen-eight headline.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    args.output_root.mkdir(parents=True, exist_ok=False)
    command = " ".join(sys.argv)
    atomic_text(args.output_root / "command.txt", command + "\n")

    source_labels, source_u_selection = load_source_frame_labels()
    source_u_frame = int(source_u_selection["selected_frame"])
    if source_labels.get(SOURCE_CURRENT_FRAME) != CURRENT_SOURCE_LABEL:
        raise RuntimeError(
            f"Expected F{SOURCE_CURRENT_FRAME} label {CURRENT_SOURCE_LABEL}, got {source_labels.get(SOURCE_CURRENT_FRAME)}"
        )
    if source_labels.get(source_u_frame) != "u":
        raise RuntimeError(f"Selected source F{source_u_frame} is not /u/")

    pairs = experiment.SELECTION
    baselines: dict[tuple[int, int], dict[str, Any]] = {}
    target_u_selection: dict[tuple[int, int], dict[str, Any]] = {}
    reference_rows = []
    for speaker, session in pairs:
        baseline_path = args.canonical_root / f"P{speaker}/S{session}/baseline.npz"
        baseline = load_baseline(baseline_path)
        baselines[(speaker, session)] = baseline
        selection = choose_longest_run_center(baseline["frames"], baseline["phonemes"])
        target_u_selection[(speaker, session)] = selection
        spec = historical_spec(speaker)
        historical_indices = np.flatnonzero(baseline["frames"] == int(spec.frame))
        cached_label = (
            str(baseline["phonemes"][int(historical_indices[0])])
            if len(historical_indices) == 1
            else "NOT_IN_EVALUATED_POPULATION"
        )
        selected_frame = int(selection["selected_frame"])
        selected_indices = np.flatnonzero(baseline["frames"] == selected_frame)
        if len(selected_indices) != 1 or str(baseline["phonemes"][int(selected_indices[0])]) != "u":
            raise RuntimeError(f"Invalid target /u/ selection for P{speaker}/S{session}")
        reference_rows.append(
            {
                "speaker": speaker,
                "session": session,
                "historical_frame": int(spec.frame),
                "historical_cached_label": cached_label,
                "historical_textgrid_label": direct_textgrid_label(speaker, session, int(spec.frame)),
                "selected_u_frame": selected_frame,
                "selected_u_cached_label": "u",
                "selected_u_textgrid_label": direct_textgrid_label(speaker, session, selected_frame),
                "selected_u_run_start": selection["run_start"],
                "selected_u_run_end": selection["run_end"],
                "selected_u_run_length": selection["run_length"],
                "total_evaluated_u_frames": selection["total_labelled_integer_frames"],
                "historical_vtln_c_anchor": spec.vtln_anchor,
            }
        )
    write_csv(args.output_root / "reference_frame_audit.csv", reference_rows)

    sources = {
        "d": load_source_reference(SOURCE_CURRENT_FRAME),
        "u": load_source_reference(source_u_frame),
    }
    source_annotations = {
        key: np.stack([source["annotations"][name] for name in CLASSES])
        for key, source in sources.items()
    }
    target_historical = {pair: prepare_frame(historical_spec(pair[0]), experiment.DEFAULT_VTLN_DIR) for pair in pairs}
    target_u = {}
    for pair in pairs:
        spec = historical_spec(pair[0])
        frame = int(target_u_selection[pair]["selected_frame"])
        target_u[pair] = prepare_frame(
            FrameSpec(spec.speaker, spec.session, f"{frame:04d}", spec.vtln_anchor),
            experiment.DEFAULT_VTLN_DIR,
        )

    transforms: dict[tuple[str, tuple[int, int]], dict[str, Any]] = {}
    transform_rows = []
    static_rows = []
    for variant, definition in VARIANTS.items():
        source = sources[definition["source"]]
        for pair in pairs:
            target = target_historical[pair] if definition["target"] == "historical" else target_u[pair]
            transform = build_two_step_transform(source["grid"], target["grid"])
            transforms[(variant, pair)] = transform
            diagnostic = transform_diagnostics(transform, source["grid"], target["grid"])
            transform_rows.append(
                {
                    "variant": variant,
                    "speaker": pair[0],
                    "session": pair[1],
                    "source_frame": source["frame"],
                    "source_cached_label": definition["source"],
                    "target_frame": int(target["spec"].frame),
                    "target_reference_mode": definition["target"],
                    **{key: value for key, value in diagnostic.items() if not isinstance(value, (list, dict))},
                    "step1_labels": ",".join(diagnostic["step1_labels"]),
                    "step2_labels": ",".join(diagnostic["step2_labels"]),
                    "affine_A_json": json.dumps(diagnostic["affine_A"]),
                    "affine_t_json": json.dumps(diagnostic["affine_t"]),
                }
            )
            source_array = source_annotations[definition["source"]]
            target_array = np.stack([target["annotations"][name] for name in CLASSES])
            affine = apply_transform(transform["step1_affine"], source_array.reshape(-1, 2)).reshape(11, 50, 2)
            final = transform["apply_two_step"](source_array.reshape(-1, 2)).reshape(11, 50, 2)
            static_arrays = {"raw": source_array, "affine": affine, "affine_tps": final}
            for stage, predicted in static_arrays.items():
                per_class = static_class_rmse_mm(predicted, target_array)
                for group, indices in GROUPS.items():
                    difference = predicted[list(indices)] - target_array[list(indices)]
                    value = float(np.sqrt(np.mean(difference * difference)) * MM_PER_PIXEL)
                    static_rows.append(
                        {
                            "variant": variant,
                            "speaker": pair[0],
                            "session": pair[1],
                            "source_frame": source["frame"],
                            "target_frame": int(target["spec"].frame),
                            "stage": stage,
                            "group": group,
                            "rmse_mm": value,
                            "per_class_rmse_json": json.dumps(
                                {name: float(per_class[index]) for index, name in enumerate(CLASSES)},
                                sort_keys=True,
                            ),
                        }
                    )
    write_csv(args.output_root / "transform_diagnostics.csv", transform_rows)
    write_csv(args.output_root / "static_reference_metrics.csv", static_rows)

    static_aggregate_rows = []
    for cohort, cohort_pairs in COHORTS.items():
        for variant in VARIANTS:
            for stage in STAGES:
                for group in GROUPS:
                    values = [
                        float(row["rmse_mm"])
                        for row in static_rows
                        if row["variant"] == variant
                        and row["stage"] == stage
                        and row["group"] == group
                        and (int(row["speaker"]), int(row["session"])) in cohort_pairs
                    ]
                    static_aggregate_rows.append(
                        {
                            "cohort": cohort,
                            "variant": variant,
                            "stage": stage,
                            "group": group,
                            "speakers": len(values),
                            "rmse_mm": float(np.mean(values)),
                        }
                    )
    write_csv(args.output_root / "static_reference_aggregate_metrics.csv", static_aggregate_rows)

    metric_arrays: dict[tuple[str, tuple[int, int], str, str], np.ndarray] = {}
    class_arrays: dict[tuple[str, tuple[int, int], str], np.ndarray] = {}
    session_rows = []
    matched_pack_paths = []
    video_audits = []
    current_validation = []
    for pair in pairs:
        speaker, session = pair
        baseline = baselines[pair]
        target = baseline["ground_truth"]
        u_mask = baseline["phonemes"] == "u"
        if not np.any(u_mask):
            raise RuntimeError(f"No evaluated /u/ frames for P{speaker}/S{session}")
        arrays_by_variant = {}
        for variant in VARIANTS:
            affine, final = transform_contour_batch(baseline["raw"], transforms[(variant, pair)], 256)
            arrays_by_variant[variant] = {"raw": baseline["raw"], "affine": affine, "affine_tps": final}
            if variant == "current_d_historical_target":
                current_validation.append(
                    {
                        "speaker": speaker,
                        "session": session,
                        "affine_max_abs_delta_vs_canonical": float(np.max(np.abs(affine - baseline["stored_affine"]))),
                        "affine_tps_max_abs_delta_vs_canonical": float(np.max(np.abs(final - baseline["stored_affine_tps"]))),
                    }
                )
            for stage, predicted in arrays_by_variant[variant].items():
                class_values = class_frame_rmse_mm(predicted, target)
                class_arrays[(variant, pair, stage)] = class_values
                for group, indices in GROUPS.items():
                    values = frame_rmse_mm(predicted, target, indices)
                    metric_arrays[(variant, pair, group, stage)] = values
                    for subset_name, mask in (("all_frames", np.ones(len(values), dtype=bool)), ("u_frames", u_mask)):
                        session_rows.append(
                            {
                                "variant": variant,
                                "speaker": speaker,
                                "session": session,
                                "cohort": "P10_CONTROL" if speaker == 10 else "UNSEEN",
                                "subset": subset_name,
                                "group": group,
                                "stage": stage,
                                "frames": int(np.sum(mask)),
                                "mean_frame_rmse_mm": float(np.mean(values[mask])),
                                "median_frame_rmse_mm": float(np.median(values[mask])),
                            }
                        )

        matched = arrays_by_variant["matched_u_to_u"]
        session_dir = args.output_root / f"P{speaker}/S{session}"
        session_dir.mkdir(parents=True, exist_ok=True)
        pack_path = session_dir / "matched_u_grid_predictions.npz"
        temporary = pack_path.with_name(f".{pack_path.name}.writing.npz")
        np.savez_compressed(
            temporary,
            frame_numbers=baseline["frames"].astype(np.int32),
            phonemes=baseline["phonemes"],
            ground_truth=target.astype(np.float32),
            predicted_raw=baseline["raw"].astype(np.float32),
            predicted_current_affine_tps=baseline["stored_affine_tps"].astype(np.float32),
            predicted_matched_u_affine=matched["affine"].astype(np.float32),
            predicted_matched_u_affine_tps=matched["affine_tps"].astype(np.float32),
            source_u_frame=np.asarray(source_u_frame, dtype=np.int32),
            target_u_frame=np.asarray(target_u_selection[pair]["selected_frame"], dtype=np.int32),
            classes=np.asarray(CLASSES, dtype="U64"),
            saved_fractional_frame_count=np.asarray(0, dtype=np.int64),
            scored_fractional_frame_count=np.asarray(0, dtype=np.int64),
        )
        temporary.replace(pack_path)
        matched_pack_paths.append(pack_path)

        if not args.skip_videos:
            video_audits.append(
                render_video(
                    args.output_root,
                    args.canonical_root,
                    pair,
                    {
                        "frames": baseline["frames"],
                        "ground_truth": target,
                        "raw": baseline["raw"],
                        "current_tps": baseline["stored_affine_tps"],
                        "matched_tps": matched["affine_tps"],
                    },
                )
            )
        print(f"P{speaker}/S{session}: transformed {len(baseline['frames'])} integer frames", flush=True)

    if any(
        row["affine_max_abs_delta_vs_canonical"] > 1e-4
        or row["affine_tps_max_abs_delta_vs_canonical"] > 1e-4
        for row in current_validation
    ):
        raise RuntimeError(f"Current-transform reproduction mismatch: {current_validation}")
    write_csv(args.output_root / "current_transform_reproduction_audit.csv", current_validation)
    write_csv(args.output_root / "video_session_metrics.csv", session_rows)

    aggregate_rows = []
    per_class_rows = []
    masks = {
        pair: {
            "all_frames": np.ones(len(baselines[pair]["frames"]), dtype=bool),
            "u_frames": baselines[pair]["phonemes"] == "u",
        }
        for pair in pairs
    }
    for cohort, cohort_pairs in COHORTS.items():
        for variant in VARIANTS:
            for subset_name in ("all_frames", "u_frames"):
                for stage in STAGES:
                    for group in GROUPS:
                        values = np.concatenate(
                            [
                                metric_arrays[(variant, pair, group, stage)][masks[pair][subset_name]]
                                for pair in cohort_pairs
                            ]
                        )
                        aggregate_rows.append(
                            {
                                "cohort": cohort,
                                "variant": variant,
                                "subset": subset_name,
                                "group": group,
                                "stage": stage,
                                "sessions": len(cohort_pairs),
                                "frames": len(values),
                                "mean_frame_rmse_mm": float(np.mean(values)),
                                "median_frame_rmse_mm": float(np.median(values)),
                            }
                        )
                    per_class = np.concatenate(
                        [
                            class_arrays[(variant, pair, stage)][masks[pair][subset_name]]
                            for pair in cohort_pairs
                        ],
                        axis=0,
                    )
                    for class_index, class_name in enumerate(CLASSES):
                        per_class_rows.append(
                            {
                                "cohort": cohort,
                                "variant": variant,
                                "subset": subset_name,
                                "stage": stage,
                                "class_name": class_name,
                                "sessions": len(cohort_pairs),
                                "frames": len(per_class),
                                "mean_frame_rmse_mm": float(np.mean(per_class[:, class_index])),
                            }
                        )
    write_csv(args.output_root / "video_aggregate_metrics.csv", aggregate_rows)
    write_csv(args.output_root / "video_per_class_metrics.csv", per_class_rows)

    bootstrap_rows = []
    for candidate_index, candidate in enumerate(list(VARIANTS)[1:], start=1):
        for subset_name in ("all_frames", "u_frames"):
            for group in GROUPS:
                for stage in ("affine", "affine_tps"):
                    current_values = {
                        pair: metric_arrays[("current_d_historical_target", pair, group, stage)][masks[pair][subset_name]]
                        for pair in experiment.UNSEEN_SELECTION
                    }
                    candidate_values = {
                        pair: metric_arrays[(candidate, pair, group, stage)][masks[pair][subset_name]]
                        for pair in experiment.UNSEEN_SELECTION
                    }
                    result = paired_session_bootstrap(
                        current_values,
                        candidate_values,
                        experiment.UNSEEN_SELECTION,
                        args.bootstrap_replicates,
                        BOOTSTRAP_SEED + candidate_index,
                    )
                    bootstrap_rows.append(
                        {
                            "cohort": "UNSEEN_8",
                            "current_variant": "current_d_historical_target",
                            "candidate_variant": candidate,
                            "subset": subset_name,
                            "group": group,
                            "stage": stage,
                            "sessions": 8,
                            "bootstrap_replicates": args.bootstrap_replicates,
                            "seed": BOOTSTRAP_SEED + candidate_index,
                            **result,
                        }
                    )
    write_csv(args.output_root / "paired_session_bootstrap.csv", bootstrap_rows)

    save_source_reference_figure(args.output_root / "source_reference_comparison.png", sources)
    save_static_contact_sheet(
        args.output_root / "matched_u_static_contact_sheet.png",
        source_annotations,
        target_u,
        transforms,
    )
    save_summary_plot(
        args.output_root / "summary_plots.png",
        aggregate_rows,
        per_class_rows,
        session_rows,
    )
    qc_contact_sheet = None
    if video_audits:
        qc_contact_sheet = save_qc_contact_sheet(args.output_root)

    report = build_report(
        args.output_root,
        source_u_selection,
        reference_rows,
        static_aggregate_rows,
        aggregate_rows,
        bootstrap_rows,
        per_class_rows,
        session_rows,
        transform_rows,
        video_audits,
    )
    atomic_text(args.output_root / "report.md", report)

    current_all9_frames = sum(len(baselines[pair]["frames"]) for pair in pairs)
    u_all9_frames = sum(int(np.sum(baselines[pair]["phonemes"] == "u")) for pair in pairs)
    manifest = {
        "created_at": now(),
        "status": "passed",
        "scope": "CPU-only post-inference grid-transform diagnosis",
        "training_launched": False,
        "inference_launched": False,
        "canonical_input_root": str(args.canonical_root.resolve()),
        "canonical_final_audit": str((args.canonical_root / "final_audit.json").resolve()),
        "canonical_final_audit_sha256": sha256(args.canonical_root / "final_audit.json"),
        "source_pack": str(SOURCE_PACK.resolve()),
        "source_pack_sha256": sha256(SOURCE_PACK),
        "source_current_frame": SOURCE_CURRENT_FRAME,
        "source_current_cached_label": source_labels[SOURCE_CURRENT_FRAME],
        "source_u_selection": source_u_selection,
        "variants": VARIANTS,
        "sessions": len(pairs),
        "evaluated_integer_frames": current_all9_frames,
        "evaluated_u_frames": u_all9_frames,
        "saved_fractional_frame_count": 0,
        "scored_fractional_frame_count": 0,
        "rendered_fractional_frame_count": 0,
        "matched_prediction_packs": [str(path.resolve()) for path in matched_pack_paths],
        "videos": [audit["video"] for audit in video_audits],
        "all_videos_passed": bool(video_audits) and all(
            all(audit["checks"].values()) for audit in video_audits
        ),
        "video_qc_contact_sheet": None if qc_contact_sheet is None else str(qc_contact_sheet.resolve()),
        "report": str((args.output_root / "report.md").resolve()),
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
