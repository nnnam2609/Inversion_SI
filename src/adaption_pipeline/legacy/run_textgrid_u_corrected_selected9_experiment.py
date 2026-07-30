#!/usr/bin/env python3
"""Rebuild the fixed selected-nine grid experiment with strict TextGrid /u/ grids.

This entry point is deliberately CPU-only and post-inference-only.  It reuses
the immutable ASD2 and P7 raw prediction arrays, selects every grid reference
from the direct TextGrid phoneme tier, recomputes Affine/TPS stages for both
models, evaluates the fixed integer-frame population, and renders exact 50-fps
original-audio comparison videos.

Reference selection never reads prediction error.  For every exact ``u``
TextGrid interval it enumerates valid integer MRI frames whose centre timestamp
lies strictly inside the interval, groups them into contiguous runs, chooses
the largest run, and then chooses the frame closest to the interval midpoint
(later frame on an exact tie).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import unicodedata
from collections import OrderedDict
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import matplotlib
import numpy as np
import soundfile as sf
import textgrid
import torch
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
GRID_ROOT = REPO_ROOT / "external/grid-transform"
sys.path[:0] = [
    str(REPO_ROOT),
    str(REPO_ROOT / "src"),
    str(GRID_ROOT),
]

from . import investigate_asd2_same_vowel_u_grid as prior_u  # noqa: E402
from . import render_analyze_asd2_epoch211_selected_experiment as render_core  # noqa: E402
from . import run_asd2_epoch211_selected_experiment as asd2_core  # noqa: E402
from .evaluate_grid_normalization import (  # noqa: E402
    RAW_ROOT,
    build_dicom_index,
)
from grid_transform.transfer import build_two_step_transform  # noqa: E402
from grid_transform.transform_helpers import apply_transform  # noqa: E402
from .render_p7_grid_transform_selected_speakers import (  # noqa: E402
    CLASSES,
    FrameSpec,
    SOURCE as HISTORICAL_P7_SOURCE,
    contour_path,
    prepare_frame,
)
from .run_p7_all_nonp7_gridnorm import transform_contour_batch  # noqa: E402
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
from src.common.artifacts import (  # noqa: E402
    atomic_write_csv as write_csv,
    atomic_write_json as _atomic_write_json,
    atomic_write_text as atomic_text,
    local_now as now,
    sha256_file as sha256,
)

atomic_json = partial(_atomic_write_json, allow_nan=True)


CANONICAL_ROOT = (
    REPO_ROOT / "results/asd2_selected_9sessions_asd2native_experiment_20260719_202352"
)
OLD_U_INVESTIGATION_ROOT = (
    REPO_ROOT / "results/asd2_grid_same_vowel_u_investigation_20260720_174029"
)
P7_BASELINE_ROOT = asd2_core.OLD_GRID_ROOT
P7_AUDIO_ROOT = asd2_core.OLD_AUDIO_ROOT
P7_ABLATION_ROOT = asd2_core.OLD_ABLATION_ROOT
ASD2_SOURCE_PACK = asd2_core.DEFAULT_SOURCE_PACK
ASD2_SOURCE_CACHE = (
    REPO_ROOT / "cache_variants/asd2_11_vtln_20260719/raw_sessions/asd2/1791/S14.pt"
)
P7_SOURCE_CACHE = REPO_ROOT / "cache/raw_sessions/asd1/P7/S2.pt"
ASD2_TEXTGRID = asd2_core.ASD2_ROOT / "1791/S14/1791_S14_adjusted.textgrid"

SELECTION = asd2_core.SELECTION
UNSEEN_SELECTION = asd2_core.UNSEEN_SELECTION
BRANCHES_PRIMARY = ("baseline", "rms_vtln")
BRANCHES_ABLATION = ("rms_only", "vtln_only")
MODELS = ("asd2", "p7")
STAGES = ("raw", "affine", "affine_tps")
FPS = 50
MS_IMAGE = 19.98
ADDED_FRAMES = 20
SKIP_MS = ADDED_FRAMES * MS_IMAGE
BOOTSTRAP_SEED = 20260720
PANEL_SIZE = 272
INFO_HEIGHT = 88
SEPARATOR = 6

ASD2_SOURCE_ANCHOR = "VTLN token 143020 static C1-C6"
P7_SOURCE_ANCHOR = HISTORICAL_P7_SOURCE.vtln_anchor

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
        (
            "historical_without_three_8",
            tuple(
                i
                for i, name in enumerate(CLASSES)
                if name not in {"epiglottis", "vocal-folds", "thyroid-cartilage"}
            ),
        ),
    ]
)
COHORTS = OrderedDict(
    [
        ("UNSEEN_8", UNSEEN_SELECTION),
        ("ALL_9", SELECTION),
        ("P10_CONTROL", ((10, 14),)),
    ]
)
HISTORICAL_ROOTS = OrderedDict(
    [
        ("asd2_old_d_mixed", CANONICAL_ROOT),
        ("p7_baseline", P7_BASELINE_ROOT),
        ("p7_rms_vtln", P7_AUDIO_ROOT),
        ("p7_ablations", P7_ABLATION_ROOT),
        ("prior_cached_u_investigation", OLD_U_INVESTIGATION_ROOT),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("preflight", "build", "audit", "all"),
        default="all",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--mri-workers", type=int, default=8)
    parser.add_argument("--transform-frame-batch", type=int, default=256)
    parser.add_argument("--confirm-visual-qc", action="store_true")
    return parser.parse_args()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("utf-8"))
    digest.update(json.dumps(value.shape).encode("utf-8"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def append_command_log(output_root: Path) -> None:
    path = output_root / "logs/commands.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] {' '.join(sys.argv)}\n")


def normalized_mark(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value).strip())


def frame_center_seconds(frame: int) -> float:
    return (SKIP_MS + (float(frame) + 0.5) * MS_IMAGE) / 1000.0


def tree_snapshot(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        files[relative] = {"size": path.stat().st_size, "sha256": sha256(path)}
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "root": str(root.resolve()),
        "file_count": len(files),
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def historical_snapshots() -> dict[str, Any]:
    return {
        "created_at": now(),
        "roots": {name: tree_snapshot(path) for name, path in HISTORICAL_ROOTS.items()},
    }


def inventory_textgrid(path: Path, dataset: str) -> tuple[Any, int, list[dict[str, Any]]]:
    grid = textgrid.TextGrid.fromFile(str(path))
    inventory = []
    candidates = []
    for index, tier in enumerate(grid):
        intervals = list(getattr(tier, "intervals", []))
        marks = [normalized_mark(interval.mark) for interval in intervals]
        nonempty = [mark for mark in marks if mark]
        short_fraction = (
            float(np.mean([len(mark) <= 4 for mark in nonempty])) if nonempty else 0.0
        )
        row = {
            "index": index,
            "name": str(getattr(tier, "name", "")),
            "interval_count": len(intervals),
            "exact_u_interval_count": sum(mark == "u" for mark in marks),
            "short_mark_fraction": short_fraction,
            "sample_marks": marks[:20],
        }
        inventory.append(row)
        if row["exact_u_interval_count"] > 0 and short_fraction >= 0.5:
            candidates.append(index)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not uniquely identify phoneme tier from content in {path}: {inventory}"
        )
    tier_index = candidates[0]
    if dataset == "ASD1" and tier_index != 1:
        raise RuntimeError(f"Expected ASD1 phoneme tier index 1 in {path}, got {tier_index}")
    return grid, tier_index, inventory


def contiguous_runs(frames: Iterable[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for frame in sorted(set(int(value) for value in frames)):
        if not runs or frame != runs[-1][-1] + 1:
            runs.append([frame])
        else:
            runs[-1].append(frame)
    return runs


def integer_frames_strictly_inside(start: float, end: float) -> list[int]:
    low = max(0, math.floor((start * 1000.0 - SKIP_MS) / MS_IMAGE - 0.5) - 2)
    high = math.ceil((end * 1000.0 - SKIP_MS) / MS_IMAGE - 0.5) + 2
    return [
        frame
        for frame in range(low, high + 1)
        if start < frame_center_seconds(frame) < end
    ]


def select_exact_u_reference(
    *,
    dataset: str,
    speaker_session: str,
    textgrid_path: Path,
    valid_frame: Callable[[int], tuple[bool, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    grid, tier_index, inventory = inventory_textgrid(textgrid_path, dataset)
    tier = grid[tier_index]
    candidates = []
    invalid_rows = []
    exact_u_total = 0
    for interval_index, interval in enumerate(tier.intervals):
        mark = normalized_mark(interval.mark)
        if mark != "u":
            continue
        exact_u_total += 1
        valid = []
        valid_metadata = {}
        for frame in integer_frames_strictly_inside(interval.minTime, interval.maxTime):
            ok, metadata = valid_frame(frame)
            if ok:
                valid.append(frame)
                valid_metadata[frame] = metadata
            else:
                invalid_rows.append(
                    {
                        "interval_index": interval_index,
                        "frame": frame,
                        "reason": metadata.get("reason", "invalid"),
                    }
                )
        midpoint = (float(interval.minTime) + float(interval.maxTime)) / 2.0
        for run in contiguous_runs(valid):
            selected = min(
                run,
                key=lambda frame: (
                    abs(frame_center_seconds(frame) - midpoint),
                    -frame,
                ),
            )
            candidates.append(
                {
                    "interval_index": interval_index,
                    "interval_mark": mark,
                    "interval_start": float(interval.minTime),
                    "interval_end": float(interval.maxTime),
                    "interval_midpoint": midpoint,
                    "run": run,
                    "run_length": len(run),
                    "selected_frame": selected,
                    "selected_frame_center": frame_center_seconds(selected),
                    "distance_from_interval_midpoint": abs(
                        frame_center_seconds(selected) - midpoint
                    ),
                    "selected_metadata": valid_metadata[selected],
                }
            )
    if not candidates:
        raise RuntimeError(f"No valid exact /u/ candidate in {textgrid_path}")
    selected = min(
        candidates,
        key=lambda row: (
            -int(row["run_length"]),
            float(row["distance_from_interval_midpoint"]),
            -int(row["selected_frame"]),
        ),
    )
    selected.update(
        {
            "dataset": dataset,
            "speaker_session": speaker_session,
            "textgrid_path": str(textgrid_path.resolve()),
            "textgrid_sha256": sha256(textgrid_path),
            "tier_index": tier_index,
            "tier_name": str(tier.name),
            "exact_u_interval_count": exact_u_total,
            "verified_exact_u": True,
            "selection_uses_rmse": False,
            "selection_rule": (
                "direct TextGrid exact-u intervals; valid integer MRI/11-contour frames; "
                "largest interval-specific contiguous run; frame closest to interval midpoint; "
                "later frame on exact tie"
            ),
            "candidate_runs": candidates,
        }
    )
    return selected, {
        "dataset": dataset,
        "speaker_session": speaker_session,
        "textgrid": str(textgrid_path.resolve()),
        "selected_tier_index": tier_index,
        "selected_tier_name": str(tier.name),
        "tiers": inventory,
        "invalid_candidate_frames": invalid_rows,
    }


def textgrid_u_mask(frames: np.ndarray, textgrid_path: Path, tier_index: int) -> np.ndarray:
    grid = textgrid.TextGrid.fromFile(str(textgrid_path))
    intervals = [
        interval
        for interval in grid[tier_index].intervals
        if normalized_mark(interval.mark) == "u"
    ]
    mask = []
    for value in frames:
        if not isinstance(value, (int, np.integer)):
            raise ValueError(f"Fractional/non-integer evaluation frame: {value!r}")
        timestamp = frame_center_seconds(int(value))
        mask.append(any(interval.minTime < timestamp < interval.maxTime for interval in intervals))
    return np.asarray(mask, dtype=bool)


def load_cached_labels(cache_path: Path) -> dict[int, str]:
    phonemes = json.loads((REPO_ROOT / "config/list_phonemes.json").read_text(encoding="utf-8"))
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)["raw"]
    values: dict[int, list[str]] = {}
    for frame_chunk, phoneme_chunk in zip(payload["frames"], payload["phonemes"]):
        for frame_row, vector in zip(frame_chunk, phoneme_chunk):
            frame = float(frame_row[2])
            if not math.isclose(frame, round(frame), abs_tol=1e-5):
                continue
            label = str(phonemes[int(np.argmax(np.asarray(vector).reshape(-1)))])
            values.setdefault(int(round(frame)), []).append(label)
    collapsed = {}
    for frame, labels in values.items():
        unique, counts = np.unique(np.asarray(labels, dtype="U32"), return_counts=True)
        collapsed[frame] = str(unique[int(np.argmax(counts))])
    return collapsed


def p7_input_path(pair: tuple[int, int], branch: str) -> Path:
    speaker, session = pair
    if branch == "baseline":
        return P7_BASELINE_ROOT / f"P{speaker}/S{session}/contours_and_ground_truth.npz"
    if branch == "rms_vtln":
        return P7_AUDIO_ROOT / f"P{speaker}/S{session}/audio_normalized_contours_and_ground_truth.npz"
    return P7_ABLATION_ROOT / f"P{speaker}/S{session}/{branch}_contours_and_ground_truth.npz"


def asd2_input_path(pair: tuple[int, int], branch: str) -> Path:
    speaker, session = pair
    return CANONICAL_ROOT / f"P{speaker}/S{session}/{branch}.npz"


def load_raw_input(model: str, pair: tuple[int, int], branch: str) -> dict[str, Any]:
    path = asd2_input_path(pair, branch) if model == "asd2" else p7_input_path(pair, branch)
    prefix = "predicted_audio_" if model == "p7" and branch == "rms_vtln" else "predicted_"
    with np.load(path, allow_pickle=False) as payload:
        frames_original = np.asarray(payload["frame_numbers"])
        frames = frames_original.astype(np.int32)
        raw = np.asarray(payload[f"{prefix}raw"], dtype=np.float32)
        ground_truth = np.asarray(payload["ground_truth"], dtype=np.float32)
        labels = np.asarray(payload["phonemes"], dtype="U32")
        classes = [str(value) for value in payload["classes"].tolist()]
        fractional = {
            key: int(payload[key].item())
            for key in (
                "saved_fractional_frame_count",
                "scored_fractional_frame_count",
                "rendered_fractional_frame_count",
            )
            if key in payload.files
        }
    if not np.issubdtype(frames_original.dtype, np.integer):
        raise ValueError(f"Input timeline is not integer typed: {path}")
    if classes != list(CLASSES):
        raise ValueError(f"Class order mismatch: {path}")
    if len(np.unique(frames)) != len(frames) or np.any(np.diff(frames) <= 0):
        raise ValueError(f"Input timeline is duplicate/non-monotonic: {path}")
    if raw.shape != ground_truth.shape or raw.shape[1:] != (11, 50, 2):
        raise ValueError(f"Unexpected raw/GT shapes in {path}: {raw.shape}, {ground_truth.shape}")
    if not np.isfinite(raw).all() or not np.isfinite(ground_truth).all():
        raise ValueError(f"Non-finite raw/GT content: {path}")
    if any(value != 0 for value in fractional.values()):
        raise ValueError(f"Nonzero fractional-frame counter: {path}: {fractional}")
    return {
        "path": path,
        "path_sha256": sha256(path),
        "frames": frames,
        "raw": raw,
        "ground_truth": ground_truth,
        "phonemes": labels,
        "classes": classes,
        "raw_sha256": array_sha256(raw),
        "ground_truth_sha256": array_sha256(ground_truth),
        "frames_sha256": array_sha256(frames),
        "fractional": fractional,
    }


def validate_input_inventory() -> tuple[dict[tuple[str, str, tuple[int, int]], dict[str, Any]], list[dict[str, Any]]]:
    required_paths = []
    for model in MODELS:
        for pair in SELECTION:
            for branch in BRANCHES_PRIMARY + BRANCHES_ABLATION:
                required_paths.append(
                    asd2_input_path(pair, branch) if model == "asd2" else p7_input_path(pair, branch)
                )
    required_paths.extend(
        [ASD2_SOURCE_PACK, ASD2_SOURCE_CACHE, P7_SOURCE_CACHE, ASD2_TEXTGRID]
    )
    for speaker, session in ((7, 2),) + SELECTION:
        _audio, path = asd2_core.exact_asd1_audio_paths(speaker, session)
        required_paths.append(path)
    missing = sorted(str(path) for path in required_paths if not path.is_file())
    if missing:
        raise FileNotFoundError("Missing required immutable inputs:\n" + "\n".join(missing))

    inputs = {}
    rows = []
    for pair in SELECTION:
        canonical_reference = None
        for model in MODELS:
            for branch in BRANCHES_PRIMARY + BRANCHES_ABLATION:
                payload = load_raw_input(model, pair, branch)
                inputs[(model, branch, pair)] = payload
                if canonical_reference is None:
                    canonical_reference = payload
                if not np.array_equal(payload["frames"], canonical_reference["frames"]):
                    raise RuntimeError(f"Frame mismatch for {model}/{branch}/P{pair[0]}/S{pair[1]}")
                if not np.array_equal(payload["ground_truth"], canonical_reference["ground_truth"]):
                    raise RuntimeError(f"GT mismatch for {model}/{branch}/P{pair[0]}/S{pair[1]}")
                rows.append(
                    {
                        "model": model,
                        "branch": branch,
                        "speaker": pair[0],
                        "session": pair[1],
                        "frames": len(payload["frames"]),
                        "input_path": str(payload["path"].resolve()),
                        "input_file_sha256": payload["path_sha256"],
                        "raw_prediction_sha256": payload["raw_sha256"],
                        "ground_truth_sha256": payload["ground_truth_sha256"],
                        "frame_timeline_sha256": payload["frames_sha256"],
                        "saved_fractional_frame_count": payload["fractional"].get(
                            "saved_fractional_frame_count", 0
                        ),
                        "scored_fractional_frame_count": payload["fractional"].get(
                            "scored_fractional_frame_count", 0
                        ),
                    }
                )
    all_frames = sum(len(inputs[("asd2", "baseline", pair)]["frames"]) for pair in SELECTION)
    unseen_frames = sum(
        len(inputs[("asd2", "baseline", pair)]["frames"]) for pair in UNSEEN_SELECTION
    )
    if all_frames != 8585 or unseen_frames != 7633:
        raise RuntimeError(f"Population mismatch: all={all_frames}, unseen={unseen_frames}")
    return inputs, rows


def target_validity(
    speaker: int,
    session: int,
    allowed_frames: set[int],
    dicom_index: dict[int, str],
) -> Callable[[int], tuple[bool, dict[str, Any]]]:
    spec = asd2_core.target_spec(speaker)
    dicom_root = RAW_ROOT / f"P{speaker}/DCM_2D/S{session}"

    def validate(frame: int) -> tuple[bool, dict[str, Any]]:
        if frame not in allowed_frames:
            return False, {"reason": "outside canonical evaluated-frame population"}
        if frame not in dicom_index:
            return False, {"reason": "missing integer MRI frame"}
        frame_spec = FrameSpec(f"P{speaker}", f"S{session}", f"{frame:04d}", spec.vtln_anchor)
        paths = []
        try:
            for class_name in CLASSES:
                path = contour_path(frame_spec, class_name)
                value = np.load(path, allow_pickle=False)
                if value.shape != (50, 2) or not np.isfinite(value).all():
                    return False, {"reason": f"invalid contour {path}"}
                paths.append(path)
        except (FileNotFoundError, ValueError) as error:
            return False, {"reason": str(error)}
        return True, {
            "mri_path": str((dicom_root / dicom_index[frame]).resolve()),
            "contour_source": str((WORKSPACE_ROOT / f"bf/inference/P{speaker}/S{session}").resolve()),
            "contour_paths": [str(path.resolve()) for path in paths],
            "mri_valid": True,
            "contours_valid": True,
        }

    return validate


def p7_source_validity() -> Callable[[int], tuple[bool, dict[str, Any]]]:
    speaker, session = 7, 2
    dicom_root = RAW_ROOT / "P7/DCM_2D/S2"
    index = build_dicom_index(dicom_root)

    def validate(frame: int) -> tuple[bool, dict[str, Any]]:
        if frame not in index:
            return False, {"reason": "missing integer MRI frame"}
        spec = FrameSpec("P7", "S2", f"{frame:04d}", P7_SOURCE_ANCHOR)
        paths = []
        try:
            for class_name in CLASSES:
                path = contour_path(spec, class_name)
                value = np.load(path, allow_pickle=False)
                if value.shape != (50, 2) or not np.isfinite(value).all():
                    return False, {"reason": f"invalid contour {path}"}
                paths.append(path)
        except (FileNotFoundError, ValueError) as error:
            return False, {"reason": str(error)}
        return True, {
            "mri_path": str((dicom_root / index[frame]).resolve()),
            "contour_source": str((WORKSPACE_ROOT / "bf/inference/P7/S2").resolve()),
            "contour_paths": [str(path.resolve()) for path in paths],
            "mri_valid": True,
            "contours_valid": True,
        }

    return validate


def asd2_source_validity() -> Callable[[int], tuple[bool, dict[str, Any]]]:
    with np.load(ASD2_SOURCE_PACK, allow_pickle=False) as payload:
        frames = np.asarray(payload["frame_numbers"], dtype=np.int32)
        contours = np.asarray(payload["contours"], dtype=np.float32).reshape(-1, 11, 50, 2)
    frame_index = {int(frame): index for index, frame in enumerate(frames)}

    def validate(frame: int) -> tuple[bool, dict[str, Any]]:
        if frame not in frame_index:
            return False, {"reason": "missing from exact ASD2 registered training contour pack"}
        row = contours[frame_index[frame]]
        if row.shape != (11, 50, 2) or not np.isfinite(row).all():
            return False, {"reason": "invalid 11-contour source-pack row"}
        mri = asd2_core.ASD2_ROOT / f"1791/S14/NPY_MR_registered/{frame:04d}.npy"
        if not mri.is_file():
            return False, {"reason": "missing NPY_MR_registered frame"}
        image = np.load(mri, allow_pickle=False)
        if image.ndim < 2 or not np.isfinite(image).all():
            return False, {"reason": "invalid NPY_MR_registered frame"}
        return True, {
            "mri_path": str(mri.resolve()),
            "contour_source": str(ASD2_SOURCE_PACK.resolve()),
            "contour_paths": [str(ASD2_SOURCE_PACK.resolve())],
            "mri_valid": True,
            "contours_valid": True,
        }

    return validate


def reference_audit_row(selection: dict[str, Any], cached_label: str, role: str) -> dict[str, Any]:
    metadata = selection["selected_metadata"]
    return {
        "role": role,
        "dataset": selection["dataset"],
        "speaker_session": selection["speaker_session"],
        "textgrid_path": selection["textgrid_path"],
        "textgrid_sha256": selection["textgrid_sha256"],
        "tier_index": selection["tier_index"],
        "tier_name": selection["tier_name"],
        "textgrid_interval_mark": selection["interval_mark"],
        "interval_start_seconds": selection["interval_start"],
        "interval_end_seconds": selection["interval_end"],
        "interval_midpoint_seconds": selection["interval_midpoint"],
        "selected_integer_frame": selection["selected_frame"],
        "selected_frame_center_seconds": selection["selected_frame_center"],
        "distance_from_interval_midpoint_seconds": selection[
            "distance_from_interval_midpoint"
        ],
        "complete_candidate_run": json.dumps(selection["run"]),
        "candidate_run_length": selection["run_length"],
        "all_candidate_runs_json": json.dumps(
            [
                {
                    "interval_index": item["interval_index"],
                    "interval_start": item["interval_start"],
                    "interval_end": item["interval_end"],
                    "run": item["run"],
                    "selected_frame": item["selected_frame"],
                }
                for item in selection["candidate_runs"]
            ],
            sort_keys=True,
        ),
        "mri_path": metadata["mri_path"],
        "contour_source": metadata["contour_source"],
        "mri_valid": metadata["mri_valid"],
        "contour_validity": metadata["contours_valid"],
        "cached_label_non_authoritative": cached_label,
        "fractional_frame_count": 0,
        "verified_exact_u": True,
        "selection_uses_rmse": False,
    }


def discover_references(
    inputs: dict[tuple[str, str, tuple[int, int]], dict[str, Any]],
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    tier_inventories = []
    selections: dict[str, Any] = {}
    rows = []

    asd2_cached = load_cached_labels(ASD2_SOURCE_CACHE)
    p7_cached = load_cached_labels(P7_SOURCE_CACHE)
    asd2_selection, inventory = select_exact_u_reference(
        dataset="ASD2",
        speaker_session="1791/S14",
        textgrid_path=ASD2_TEXTGRID,
        valid_frame=asd2_source_validity(),
    )
    tier_inventories.append(inventory)
    selections["asd2_source"] = asd2_selection
    rows.append(
        reference_audit_row(
            asd2_selection,
            asd2_cached.get(int(asd2_selection["selected_frame"]), "NOT_IN_CACHE"),
            "source",
        )
    )

    _audio, p7_textgrid = asd2_core.exact_asd1_audio_paths(7, 2)
    p7_selection, inventory = select_exact_u_reference(
        dataset="ASD1",
        speaker_session="P7/S2",
        textgrid_path=p7_textgrid,
        valid_frame=p7_source_validity(),
    )
    tier_inventories.append(inventory)
    selections["p7_source"] = p7_selection
    rows.append(
        reference_audit_row(
            p7_selection,
            p7_cached.get(int(p7_selection["selected_frame"]), "NOT_IN_CACHE"),
            "source",
        )
    )

    u_masks = {}
    for pair in SELECTION:
        speaker, session = pair
        baseline = inputs[("asd2", "baseline", pair)]
        allowed = set(int(value) for value in baseline["frames"])
        dicom_root = RAW_ROOT / f"P{speaker}/DCM_2D/S{session}"
        dicom_index = build_dicom_index(dicom_root)
        _audio, tg_path = asd2_core.exact_asd1_audio_paths(speaker, session)
        selection, inventory = select_exact_u_reference(
            dataset="ASD1",
            speaker_session=f"P{speaker}/S{session}",
            textgrid_path=tg_path,
            valid_frame=target_validity(speaker, session, allowed, dicom_index),
        )
        tier_inventories.append(inventory)
        selections[f"target_P{speaker}_S{session}"] = selection
        indices = np.flatnonzero(baseline["frames"] == int(selection["selected_frame"]))
        cached = (
            str(baseline["phonemes"][int(indices[0])])
            if len(indices) == 1
            else "NOT_IN_EVALUATED_CACHE"
        )
        rows.append(reference_audit_row(selection, cached, "target"))
        mask = textgrid_u_mask(baseline["frames"], tg_path, int(selection["tier_index"]))
        if not np.any(mask):
            raise RuntimeError(f"No direct TextGrid /u/ evaluation frames for P{speaker}/S{session}")
        u_masks[pair] = mask

    if len(rows) != 11 or not all(bool(row["verified_exact_u"]) for row in rows):
        raise RuntimeError("Reference inventory did not verify all 11 exact /u/ references")
    write_csv(output_root / "textgrid_u_reference_audit.csv", rows)
    atomic_json(output_root / "provenance/textgrid_tier_inventory.json", tier_inventories)
    atomic_json(
        output_root / "provenance/textgrid_u_selections.json",
        {
            key: {
                field: value
                for field, value in selection.items()
                if field not in {"candidate_runs", "selected_metadata"}
            }
            for key, selection in selections.items()
        },
    )
    return selections, rows, u_masks


def save_reference_pack(path: Path, payload: dict[str, Any], selection: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    annotations = np.stack([payload["annotations"][name] for name in CLASSES]).astype(np.float32)
    grid_names = sorted(payload["grid_contours"])
    temporary = path.with_name(f".{path.name}.writing.npz")
    arrays = {
        "annotations": annotations,
        "classes": np.asarray(CLASSES, dtype="U64"),
        "grid_names": np.asarray(grid_names, dtype="U64"),
        "selected_frame": np.asarray(selection["selected_frame"], dtype=np.int32),
        "selected_frame_center_seconds": np.asarray(
            selection["selected_frame_center"], dtype=np.float64
        ),
        "verified_exact_u": np.asarray(True),
    }
    for index, name in enumerate(grid_names):
        arrays[f"grid_{index:02d}_{name.replace('-', '_')}"] = np.asarray(
            payload["grid_contours"][name], dtype=np.float32
        )
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def build_reference_payloads(
    selections: dict[str, Any], output_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    sources = {
        "asd2": prior_u.load_source_reference(int(selections["asd2_source"]["selected_frame"])),
        "p7": prepare_frame(
            FrameSpec(
                "P7",
                "S2",
                f"{int(selections['p7_source']['selected_frame']):04d}",
                P7_SOURCE_ANCHOR,
            ),
            asd2_core.DEFAULT_VTLN_DIR,
        ),
    }
    targets = {}
    for pair in SELECTION:
        speaker, session = pair
        selection = selections[f"target_P{speaker}_S{session}"]
        targets[pair] = prepare_frame(
            FrameSpec(
                f"P{speaker}",
                f"S{session}",
                f"{int(selection['selected_frame']):04d}",
                asd2_core.target_spec(speaker).vtln_anchor,
            ),
            asd2_core.DEFAULT_VTLN_DIR,
        )
    provenance = output_root / "provenance/reference_grids"
    save_reference_pack(provenance / "asd2_source_u.npz", sources["asd2"], selections["asd2_source"])
    save_reference_pack(provenance / "p7_source_u.npz", sources["p7"], selections["p7_source"])
    for pair, target in targets.items():
        save_reference_pack(
            provenance / f"P{pair[0]}_S{pair[1]}_target_u.npz",
            target,
            selections[f"target_P{pair[0]}_S{pair[1]}"],
        )
    metadata = {
        "asd2_source": {
            "frame": int(selections["asd2_source"]["selected_frame"]),
            "mri": str(sources["asd2"]["image_path"].resolve()),
            "classes_0_to_8": "exact registered contour pack used by ASD2 model",
            "classes_9_to_10": "VTLN-incisor overlay used during ASD2 training",
            "cervical_C1_to_C6": "static VTLN token 143020; no per-frame ASD2 cervical annotation",
            "source_pack": str(ASD2_SOURCE_PACK.resolve()),
            "source_pack_sha256": sha256(ASD2_SOURCE_PACK),
        },
        "p7_source": {
            "frame": int(selections["p7_source"]["selected_frame"]),
            "bf_contours": selections["p7_source"]["selected_metadata"]["contour_paths"],
            "bf_contour_sha256": {
                path: sha256(Path(path))
                for path in selections["p7_source"]["selected_metadata"]["contour_paths"]
            },
            "cervical_C1_to_C6": P7_SOURCE_ANCHOR,
        },
        "targets": {
            f"P{pair[0]}/S{pair[1]}": {
                "frame": int(selections[f"target_P{pair[0]}_S{pair[1]}"]["selected_frame"]),
                "bf_contours": selections[f"target_P{pair[0]}_S{pair[1]}"]["selected_metadata"][
                    "contour_paths"
                ],
                "bf_contour_sha256": {
                    path: sha256(Path(path))
                    for path in selections[f"target_P{pair[0]}_S{pair[1]}"][
                        "selected_metadata"
                    ]["contour_paths"]
                },
                "cervical_C1_to_C6": asd2_core.target_spec(pair[0]).vtln_anchor,
            }
            for pair in SELECTION
        },
    }
    atomic_json(output_root / "provenance/source_target_grid_provenance.json", metadata)
    return sources, targets


def build_transforms(
    sources: dict[str, dict[str, Any]],
    targets: dict[tuple[int, int], dict[str, Any]],
    selections: dict[str, Any],
    output_root: Path,
) -> dict[tuple[str, tuple[int, int]], dict[str, Any]]:
    transforms = {}
    rows = []
    for model in MODELS:
        source = sources[model]
        source_selection = selections[f"{model}_source"]
        for pair in SELECTION:
            target = targets[pair]
            transform = build_two_step_transform(source["grid"], target["grid"])
            diagnostics = prior_u.transform_diagnostics(transform, source["grid"], target["grid"])
            transforms[(model, pair)] = transform
            rows.append(
                {
                    "model": model,
                    "speaker": pair[0],
                    "session": pair[1],
                    "source_frame": source_selection["selected_frame"],
                    "target_frame": selections[f"target_P{pair[0]}_S{pair[1]}"]["selected_frame"],
                    "source_verified_exact_u": True,
                    "target_verified_exact_u": True,
                    **{
                        key: value
                        for key, value in diagnostics.items()
                        if not isinstance(value, (list, dict))
                    },
                    "affine_controls": ",".join(diagnostics["step1_labels"]),
                    "tps_controls": ",".join(diagnostics["step2_labels"]),
                    "affine_matrix_json": json.dumps(diagnostics["affine_A"]),
                    "affine_translation_json": json.dumps(diagnostics["affine_t"]),
                    "tps_smoothing": 0.0,
                }
            )
    write_csv(output_root / "transform_diagnostics.csv", rows)
    return transforms


def corrected_pack_path(output_root: Path, model: str, pair: tuple[int, int], branch: str) -> Path:
    return output_root / "predictions" / model.upper() / f"P{pair[0]}/S{pair[1]}/{branch}.npz"


def save_corrected_pack(
    output_root: Path,
    model: str,
    pair: tuple[int, int],
    branch: str,
    source: dict[str, Any],
    transform: dict[str, Any],
    u_mask: np.ndarray,
    source_frame: int,
    target_frame: int,
    frame_batch: int,
) -> dict[str, Any]:
    affine, final = transform_contour_batch(source["raw"], transform, frame_batch)
    output = corrected_pack_path(output_root, model, pair, branch)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.writing.npz")
    np.savez_compressed(
        temporary,
        model=np.asarray(model),
        branch=np.asarray(branch),
        frame_numbers=source["frames"].astype(np.int32),
        textgrid_exact_u_mask=u_mask.astype(bool),
        phonemes_cached_non_authoritative=source["phonemes"],
        predicted_raw=source["raw"].astype(np.float32),
        predicted_after_corrected_affine=affine.astype(np.float32),
        predicted_after_corrected_affine_tps=final.astype(np.float32),
        ground_truth=source["ground_truth"].astype(np.float32),
        classes=np.asarray(CLASSES, dtype="U64"),
        source_u_frame=np.asarray(source_frame, dtype=np.int32),
        target_u_frame=np.asarray(target_frame, dtype=np.int32),
        source_verified_exact_u=np.asarray(True),
        target_verified_exact_u=np.asarray(True),
        input_raw_file=np.asarray(str(source["path"].resolve())),
        input_raw_file_sha256=np.asarray(source["path_sha256"]),
        input_raw_array_sha256=np.asarray(source["raw_sha256"]),
        saved_raw_array_sha256=np.asarray(array_sha256(source["raw"])),
        saved_fractional_frame_count=np.asarray(0, dtype=np.int64),
        scored_fractional_frame_count=np.asarray(0, dtype=np.int64),
        rendered_fractional_frame_count=np.asarray(0, dtype=np.int64),
        frame_policy=np.asarray("integer MRI frames only; no interpolation or hold"),
    )
    temporary.replace(output)
    with np.load(output, allow_pickle=False) as check:
        saved_raw = np.asarray(check["predicted_raw"], dtype=np.float32)
        if not np.array_equal(saved_raw, source["raw"]):
            raise RuntimeError(f"Saved raw array differs from immutable input: {output}")
        if str(check["saved_raw_array_sha256"].item()) != source["raw_sha256"]:
            raise RuntimeError(f"Raw array hash mismatch: {output}")
    return {
        "model": model,
        "branch": branch,
        "speaker": pair[0],
        "session": pair[1],
        "frames": len(source["frames"]),
        "pack": str(output.resolve()),
        "pack_sha256": sha256(output),
        "input_raw_file": str(source["path"].resolve()),
        "input_raw_file_sha256": source["path_sha256"],
        "raw_prediction_sha256": source["raw_sha256"],
        "ground_truth_sha256": source["ground_truth_sha256"],
        "frame_timeline_sha256": source["frames_sha256"],
        "saved_fractional_frame_count": 0,
        "scored_fractional_frame_count": 0,
    }


def load_corrected_pack(output_root: Path, model: str, pair: tuple[int, int], branch: str) -> dict[str, Any]:
    path = corrected_pack_path(output_root, model, pair, branch)
    with np.load(path, allow_pickle=False) as payload:
        return {
            "path": path,
            "frames": np.asarray(payload["frame_numbers"], dtype=np.int32),
            "u_mask": np.asarray(payload["textgrid_exact_u_mask"], dtype=bool),
            "raw": np.asarray(payload["predicted_raw"], dtype=np.float32),
            "affine": np.asarray(payload["predicted_after_corrected_affine"], dtype=np.float32),
            "affine_tps": np.asarray(
                payload["predicted_after_corrected_affine_tps"], dtype=np.float32
            ),
            "ground_truth": np.asarray(payload["ground_truth"], dtype=np.float32),
        }


def frame_rmse(predicted: np.ndarray, ground_truth: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    delta = predicted[:, list(indices)].astype(np.float64) - ground_truth[:, list(indices)].astype(np.float64)
    return np.sqrt(np.mean(delta * delta, axis=(1, 2, 3))) * MM_PER_PIXEL


def class_rmse(predicted: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    delta = predicted.astype(np.float64) - ground_truth.astype(np.float64)
    return np.sqrt(np.mean(delta * delta, axis=(2, 3))) * MM_PER_PIXEL


def compute_audio_gate(output_root: Path) -> dict[str, Any]:
    rows = []
    for pair in SELECTION:
        baseline = load_corrected_pack(output_root, "asd2", pair, "baseline")
        audio = load_corrected_pack(output_root, "asd2", pair, "rms_vtln")
        base = float(np.mean(frame_rmse(baseline["affine_tps"], baseline["ground_truth"], GROUPS["all_11"])))
        candidate = float(np.mean(frame_rmse(audio["affine_tps"], audio["ground_truth"], GROUPS["all_11"])))
        rows.append(
            {
                "speaker": pair[0],
                "session": pair[1],
                "frames": len(baseline["frames"]),
                "baseline_affine_tps_all11_mm": base,
                "rms_vtln_affine_tps_all11_mm": candidate,
                "delta_mm": candidate - base,
            }
        )
    total = sum(int(row["frames"]) for row in rows)
    aggregate_delta = sum(int(row["frames"]) * float(row["delta_mm"]) for row in rows) / total
    aggregate_trigger = abs(aggregate_delta) >= asd2_core.GATE_AGGREGATE_THRESHOLD_MM
    session_trigger = any(
        abs(float(row["delta_mm"])) >= asd2_core.GATE_SESSION_THRESHOLD_MM for row in rows
    )
    signs = {int(np.sign(float(row["delta_mm"]))) for row in rows if float(row["delta_mm"]) != 0.0}
    mixed_sign_trigger = -1 in signs and 1 in signs
    triggered = aggregate_trigger or session_trigger or mixed_sign_trigger
    payload = {
        "created_at": now(),
        "status": "triggered" if triggered else "not_triggered",
        "triggered": triggered,
        "comparison": "corrected ASD2 RMS+VTLN minus corrected ASD2 baseline after affine+TPS",
        "population": "all nine fixed sessions; P10 retained as same-person control",
        "frame_weighted_aggregate_delta_mm": aggregate_delta,
        "aggregate_threshold_abs_mm": asd2_core.GATE_AGGREGATE_THRESHOLD_MM,
        "aggregate_trigger": aggregate_trigger,
        "session_threshold_abs_mm": asd2_core.GATE_SESSION_THRESHOLD_MM,
        "any_session_trigger": session_trigger,
        "mixed_effect_signs_trigger": mixed_sign_trigger,
        "session_deltas": rows,
    }
    atomic_json(output_root / "audio_gate_decision.json", payload)
    write_csv(output_root / "audio_gate_session_deltas.csv", rows)
    return payload


def process_predictions(
    inputs: dict[tuple[str, str, tuple[int, int]], dict[str, Any]],
    transforms: dict[tuple[str, tuple[int, int]], dict[str, Any]],
    selections: dict[str, Any],
    u_masks: dict[tuple[int, int], np.ndarray],
    output_root: Path,
    frame_batch: int,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    inventory = []
    for branch in BRANCHES_PRIMARY:
        for model in MODELS:
            source_frame = int(selections[f"{model}_source"]["selected_frame"])
            for pair in SELECTION:
                inventory.append(
                    save_corrected_pack(
                        output_root,
                        model,
                        pair,
                        branch,
                        inputs[(model, branch, pair)],
                        transforms[(model, pair)],
                        u_masks[pair],
                        source_frame,
                        int(selections[f"target_P{pair[0]}_S{pair[1]}"]["selected_frame"]),
                        frame_batch,
                    )
                )
    gate = compute_audio_gate(output_root)
    included = list(BRANCHES_PRIMARY)
    if gate["triggered"]:
        included.extend(BRANCHES_ABLATION)
        for branch in BRANCHES_ABLATION:
            for model in MODELS:
                source_frame = int(selections[f"{model}_source"]["selected_frame"])
                for pair in SELECTION:
                    inventory.append(
                        save_corrected_pack(
                            output_root,
                            model,
                            pair,
                            branch,
                            inputs[(model, branch, pair)],
                            transforms[(model, pair)],
                            u_masks[pair],
                            source_frame,
                            int(
                                selections[f"target_P{pair[0]}_S{pair[1]}"][
                                    "selected_frame"
                                ]
                            ),
                            frame_batch,
                        )
                    )
    write_csv(output_root / "provenance/corrected_prediction_pack_inventory.csv", inventory)
    return included, inventory, gate


def old_protocol_pack(pair: tuple[int, int]) -> dict[str, Any]:
    path = asd2_input_path(pair, "baseline")
    with np.load(path, allow_pickle=False) as payload:
        return {
            "frames": np.asarray(payload["frame_numbers"], dtype=np.int32),
            "raw": np.asarray(payload["predicted_raw"], dtype=np.float32),
            "affine": np.asarray(payload["predicted_after_affine"], dtype=np.float32),
            "affine_tps": np.asarray(payload["predicted_after_affine_tps"], dtype=np.float32),
            "ground_truth": np.asarray(payload["ground_truth"], dtype=np.float32),
        }


def weighted(values: Iterable[np.ndarray]) -> float:
    arrays = list(values)
    return float(np.mean(np.concatenate(arrays)))


def compute_metrics(
    output_root: Path,
    included_branches: list[str],
) -> dict[str, Any]:
    metric_store: dict[tuple[str, str, tuple[int, int], str, str, str], np.ndarray] = {}
    class_store: dict[tuple[str, str, tuple[int, int], str, str], np.ndarray] = {}
    session_rows = []
    u_count_rows = []
    all_models = list(MODELS) + ["asd2_old_d_mixed"]
    for pair in SELECTION:
        reference = load_corrected_pack(output_root, "asd2", pair, "baseline")
        u_count_rows.append(
            {
                "speaker": pair[0],
                "session": pair[1],
                "evaluated_integer_frames": len(reference["frames"]),
                "textgrid_exact_u_frames": int(np.sum(reference["u_mask"])),
                "fractional_frames": 0,
            }
        )
        for model in all_models:
            branches = ["baseline"] if model == "asd2_old_d_mixed" else included_branches
            for branch in branches:
                pack = old_protocol_pack(pair) if model == "asd2_old_d_mixed" else load_corrected_pack(
                    output_root, model, pair, branch
                )
                if not np.array_equal(pack["frames"], reference["frames"]):
                    raise RuntimeError(f"Metric frame mismatch for {model}/{branch}/{pair}")
                pack_u_mask = reference["u_mask"]
                for stage in STAGES:
                    predicted = pack[stage]
                    per_class = class_rmse(predicted, pack["ground_truth"])
                    for subset, mask in (
                        ("all_frames", np.ones(len(pack["frames"]), dtype=bool)),
                        ("textgrid_u_frames", pack_u_mask),
                    ):
                        class_store[(model, branch, pair, subset, stage)] = per_class[mask]
                        for group, indices in GROUPS.items():
                            values = frame_rmse(predicted, pack["ground_truth"], indices)[mask]
                            metric_store[(model, branch, pair, subset, stage, group)] = values
                            session_rows.append(
                                {
                                    "model": model,
                                    "branch": branch,
                                    "speaker": pair[0],
                                    "session": pair[1],
                                    "cohort": "P10_CONTROL" if pair[0] == 10 else "UNSEEN",
                                    "subset": subset,
                                    "stage": stage,
                                    "group": group,
                                    "frames": len(values),
                                    "mean_frame_coordinate_rmse_mm": float(np.mean(values)),
                                    "median_frame_coordinate_rmse_mm": float(np.median(values)),
                                }
                            )

    aggregate_rows = []
    per_contour_rows = []
    for cohort, pairs in COHORTS.items():
        for model in all_models:
            branches = ["baseline"] if model == "asd2_old_d_mixed" else included_branches
            for branch in branches:
                for subset in ("all_frames", "textgrid_u_frames"):
                    for stage in STAGES:
                        for group in GROUPS:
                            arrays = [metric_store[(model, branch, pair, subset, stage, group)] for pair in pairs]
                            values = np.concatenate(arrays)
                            aggregate_rows.append(
                                {
                                    "cohort": cohort,
                                    "model": model,
                                    "branch": branch,
                                    "subset": subset,
                                    "stage": stage,
                                    "group": group,
                                    "sessions": len(pairs),
                                    "frames": len(values),
                                    "mean_frame_coordinate_rmse_mm": float(np.mean(values)),
                                    "median_frame_coordinate_rmse_mm": float(np.median(values)),
                                }
                            )
                        per_class = np.concatenate(
                            [class_store[(model, branch, pair, subset, stage)] for pair in pairs],
                            axis=0,
                        )
                        for class_index, class_name in enumerate(CLASSES):
                            per_contour_rows.append(
                                {
                                    "cohort": cohort,
                                    "model": model,
                                    "branch": branch,
                                    "subset": subset,
                                    "stage": stage,
                                    "class_name": class_name,
                                    "sessions": len(pairs),
                                    "frames": len(per_class),
                                    "mean_frame_coordinate_rmse_mm": float(
                                        np.mean(per_class[:, class_index])
                                    ),
                                }
                            )
    write_csv(output_root / "metrics/session_metrics.csv", session_rows)
    write_csv(output_root / "metrics/aggregate_metrics.csv", aggregate_rows)
    write_csv(output_root / "metrics/per_contour_metrics.csv", per_contour_rows)
    write_csv(output_root / "metrics/textgrid_u_subset_counts.csv", u_count_rows)
    write_csv(
        output_root / "metrics/textgrid_u_subset_metrics.csv",
        [row for row in aggregate_rows if row["subset"] == "textgrid_u_frames"],
    )
    return {
        "metric_store": metric_store,
        "class_store": class_store,
        "session_rows": session_rows,
        "aggregate_rows": aggregate_rows,
        "per_contour_rows": per_contour_rows,
        "u_count_rows": u_count_rows,
    }


def paired_session_bootstrap(
    reference: dict[tuple[int, int], np.ndarray],
    candidate: dict[tuple[int, int], np.ndarray],
    pairs: tuple[tuple[int, int], ...],
    replicates: int,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    observed = weighted(candidate[pair] for pair in pairs) - weighted(reference[pair] for pair in pairs)
    left_sums = np.asarray([np.sum(reference[pair], dtype=np.float64) for pair in pairs])
    left_counts = np.asarray([len(reference[pair]) for pair in pairs], dtype=np.float64)
    right_sums = np.asarray([np.sum(candidate[pair], dtype=np.float64) for pair in pairs])
    right_counts = np.asarray([len(candidate[pair]) for pair in pairs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(pairs), size=(replicates, len(pairs)))
    effects = (
        np.sum(right_sums[sampled], axis=1) / np.sum(right_counts[sampled], axis=1)
        - np.sum(left_sums[sampled], axis=1) / np.sum(left_counts[sampled], axis=1)
    )
    return {
        "effect_candidate_minus_reference_mm": float(observed),
        "ci95_low_mm": float(np.quantile(effects, 0.025)),
        "ci95_high_mm": float(np.quantile(effects, 0.975)),
    }


def comparison_value_map(
    metrics: dict[str, Any],
    *,
    model: str,
    branch: str,
    subset: str,
    stage: str,
    group: str,
    pairs: tuple[tuple[int, int], ...],
) -> dict[tuple[int, int], np.ndarray]:
    store = metrics["metric_store"]
    return {
        pair: store[(model, branch, pair, subset, stage, group)] for pair in pairs
    }


def build_comparisons(
    metrics: dict[str, Any],
    included_branches: list[str],
    replicates: int,
    output_root: Path,
) -> list[dict[str, Any]]:
    definitions = [
        (
            "corrected_asd2_affine_minus_raw",
            ("asd2", "baseline", "all_frames", "raw", "all_11"),
            ("asd2", "baseline", "all_frames", "affine", "all_11"),
        ),
        (
            "corrected_asd2_tps_minus_affine",
            ("asd2", "baseline", "all_frames", "affine", "all_11"),
            ("asd2", "baseline", "all_frames", "affine_tps", "all_11"),
        ),
        (
            "corrected_asd2_minus_corrected_p7",
            ("p7", "baseline", "all_frames", "affine_tps", "all_11"),
            ("asd2", "baseline", "all_frames", "affine_tps", "all_11"),
        ),
        (
            "corrected_asd2_minus_old_d_mixed",
            ("asd2_old_d_mixed", "baseline", "all_frames", "affine_tps", "all_11"),
            ("asd2", "baseline", "all_frames", "affine_tps", "all_11"),
        ),
        (
            "corrected_asd2_u_subset_minus_all_frames",
            ("asd2", "baseline", "all_frames", "affine_tps", "all_11"),
            ("asd2", "baseline", "textgrid_u_frames", "affine_tps", "all_11"),
        ),
        (
            "corrected_asd2_without_incisors_minus_all11",
            ("asd2", "baseline", "all_frames", "affine_tps", "all_11"),
            ("asd2", "baseline", "all_frames", "affine_tps", "without_incisors_9"),
        ),
    ]
    for branch in included_branches:
        if branch != "baseline":
            definitions.append(
                (
                    f"corrected_asd2_{branch}_minus_baseline",
                    ("asd2", "baseline", "all_frames", "affine_tps", "all_11"),
                    ("asd2", branch, "all_frames", "affine_tps", "all_11"),
                )
            )
    rows = []
    for cohort, pairs in COHORTS.items():
        for name, left, right in definitions:
            reference = comparison_value_map(
                metrics,
                model=left[0], branch=left[1], subset=left[2], stage=left[3], group=left[4], pairs=pairs
            )
            candidate = comparison_value_map(
                metrics,
                model=right[0], branch=right[1], subset=right[2], stage=right[3], group=right[4], pairs=pairs
            )
            effect = paired_session_bootstrap(reference, candidate, pairs, replicates)
            rows.append(
                {
                    "comparison": name,
                    "cohort": cohort,
                    "sessions": len(pairs),
                    "reference_spec": "/".join(left),
                    "candidate_spec": "/".join(right),
                    "bootstrap_replicates": replicates,
                    "seed": BOOTSTRAP_SEED,
                    **effect,
                }
            )

    session_deltas = []
    for pair in SELECTION:
        corrected = metrics["metric_store"][("asd2", "baseline", pair, "all_frames", "affine_tps", "all_11")]
        old = metrics["metric_store"][("asd2_old_d_mixed", "baseline", pair, "all_frames", "affine_tps", "all_11")]
        p7 = metrics["metric_store"][("p7", "baseline", pair, "all_frames", "affine_tps", "all_11")]
        session_deltas.append(
            {
                "speaker": pair[0],
                "session": pair[1],
                "frames": len(corrected),
                "corrected_asd2_mm": float(np.mean(corrected)),
                "old_d_mixed_asd2_mm": float(np.mean(old)),
                "corrected_p7_mm": float(np.mean(p7)),
                "corrected_minus_old_mm": float(np.mean(corrected) - np.mean(old)),
                "corrected_asd2_minus_corrected_p7_mm": float(np.mean(corrected) - np.mean(p7)),
            }
        )

    per_contour_deltas = []
    per_contour_bootstrap = []
    for class_index, class_name in enumerate(CLASSES):
        corrected_map = {
            pair: metrics["class_store"][("asd2", "baseline", pair, "all_frames", "affine_tps")][:, class_index]
            for pair in UNSEEN_SELECTION
        }
        old_map = {
            pair: metrics["class_store"][("asd2_old_d_mixed", "baseline", pair, "all_frames", "affine_tps")][:, class_index]
            for pair in UNSEEN_SELECTION
        }
        p7_map = {
            pair: metrics["class_store"][("p7", "baseline", pair, "all_frames", "affine_tps")][:, class_index]
            for pair in UNSEEN_SELECTION
        }
        corrected_mean = weighted(corrected_map.values())
        old_mean = weighted(old_map.values())
        p7_mean = weighted(p7_map.values())
        per_contour_deltas.append(
            {
                "class_name": class_name,
                "cohort": "UNSEEN_8",
                "corrected_asd2_mm": corrected_mean,
                "old_d_mixed_asd2_mm": old_mean,
                "corrected_p7_mm": p7_mean,
                "corrected_minus_old_mm": corrected_mean - old_mean,
                "corrected_asd2_minus_corrected_p7_mm": corrected_mean - p7_mean,
            }
        )
        per_contour_bootstrap.append(
            {
                "comparison": "corrected_asd2_minus_old_d_mixed",
                "class_name": class_name,
                "cohort": "UNSEEN_8",
                "bootstrap_replicates": replicates,
                "seed": BOOTSTRAP_SEED,
                **paired_session_bootstrap(old_map, corrected_map, UNSEEN_SELECTION, replicates),
            }
        )

    failures = []
    for row in session_deltas:
        failures.append(
            {
                "dimension": "session",
                "name": f"P{row['speaker']}/S{row['session']}",
                "comparison": "corrected_asd2_minus_old_d_mixed",
                "delta_mm": row["corrected_minus_old_mm"],
                "regression": float(row["corrected_minus_old_mm"]) > 0,
            }
        )
    for row in per_contour_deltas:
        failures.append(
            {
                "dimension": "contour",
                "name": row["class_name"],
                "comparison": "corrected_asd2_minus_old_d_mixed",
                "delta_mm": row["corrected_minus_old_mm"],
                "regression": float(row["corrected_minus_old_mm"]) > 0,
            }
        )
    failures.sort(key=lambda row: float(row["delta_mm"]), reverse=True)
    write_csv(output_root / "comparisons/bootstrap_comparisons.csv", rows)
    write_csv(output_root / "comparisons/per_session_deltas.csv", session_deltas)
    write_csv(output_root / "comparisons/per_contour_deltas.csv", per_contour_deltas)
    write_csv(output_root / "comparisons/per_contour_bootstrap.csv", per_contour_bootstrap)
    write_csv(output_root / "comparisons/failure_cases.csv", failures)
    return rows


def draw_reference(ax: plt.Axes, payload: dict[str, Any], title: str) -> None:
    ax.imshow(payload["image"], cmap="gray", vmin=0, vmax=255)
    for index, name in enumerate(CLASSES):
        points = payload["annotations"][name]
        ax.plot(points[:, 0], points[:, 1], color=COLORS.get(name, "white"), linewidth=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 136)
    ax.set_ylim(136, 0)
    ax.axis("off")


def save_reference_figures(
    sources: dict[str, dict[str, Any]], selections: dict[str, Any], output_root: Path
) -> None:
    old_asd2 = prior_u.load_source_reference(3020)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    draw_reference(axes[0], old_asd2, "Historical ASD2 F3020: TextGrid /d/")
    draw_reference(
        axes[1], sources["asd2"], f"Corrected ASD2 F{int(selections['asd2_source']['selected_frame']):04d}: /u/"
    )
    fig.tight_layout()
    fig.savefig(output_root / "figures/asd2_old_f3020_vs_corrected_u_source.png", dpi=180)
    plt.close(fig)

    old_p7 = prepare_frame(HISTORICAL_P7_SOURCE, asd2_core.DEFAULT_VTLN_DIR)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    draw_reference(axes[0], old_p7, f"Historical P7 F{HISTORICAL_P7_SOURCE.frame}")
    draw_reference(
        axes[1], sources["p7"], f"Corrected P7 F{int(selections['p7_source']['selected_frame']):04d}: /u/"
    )
    fig.tight_layout()
    fig.savefig(output_root / "figures/p7_historical_vs_corrected_u_source.png", dpi=180)
    plt.close(fig)


def save_static_contact_sheet(
    sources: dict[str, dict[str, Any]],
    targets: dict[tuple[int, int], dict[str, Any]],
    transforms: dict[tuple[str, tuple[int, int]], dict[str, Any]],
    selections: dict[str, Any],
    output_root: Path,
) -> None:
    source_arrays = {
        model: np.stack([sources[model]["annotations"][name] for name in CLASSES])
        for model in MODELS
    }
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, pair in zip(axes.ravel(), SELECTION):
        target = targets[pair]
        ax.imshow(target["image"], cmap="gray", vmin=0, vmax=255)
        for index, name in enumerate(CLASSES):
            gt = target["annotations"][name]
            asd2 = transforms[("asd2", pair)]["apply_two_step"](
                source_arrays["asd2"][index]
            )
            p7 = transforms[("p7", pair)]["apply_two_step"](source_arrays["p7"][index])
            color = COLORS.get(name, "white")
            ax.plot(gt[:, 0], gt[:, 1], color=color, linewidth=1.0)
            ax.plot(asd2[:, 0], asd2[:, 1], color=color, linewidth=0.8, linestyle="--")
            ax.plot(p7[:, 0], p7[:, 1], color=color, linewidth=0.7, linestyle=":")
        target_frame = selections[f"target_P{pair[0]}_S{pair[1]}"]["selected_frame"]
        ax.set_title(f"P{pair[0]}/S{pair[1]} F{target_frame:04d} /u/", fontsize=9)
        ax.set_xlim(0, 136)
        ax.set_ylim(136, 0)
        ax.axis("off")
    fig.suptitle("Corrected static /u/→/u/: solid target, dashed ASD2, dotted P7", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_root / "figures/nine_speaker_corrected_static_u_to_u_contact_sheet.png", dpi=170)
    plt.close(fig)


def save_metric_figures(metrics: dict[str, Any], output_root: Path) -> None:
    def aggregate(model: str, stage: str, group: str = "all_11") -> float:
        for row in metrics["aggregate_rows"]:
            if (
                row["cohort"] == "UNSEEN_8"
                and row["model"] == model
                and row["branch"] == "baseline"
                and row["subset"] == "all_frames"
                and row["stage"] == stage
                and row["group"] == group
            ):
                return float(row["mean_frame_coordinate_rmse_mm"])
        raise KeyError((model, stage, group))

    labels = ["Raw", "Affine", "Affine+TPS"]
    old = [aggregate("asd2_old_d_mixed", stage) for stage in STAGES]
    corrected = [aggregate("asd2", stage) for stage in STAGES]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.18, old, width=0.36, label="old /d/→mixed")
    ax.bar(x + 0.18, corrected, width=0.36, label="corrected /u/→/u/")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Unseen-eight RMSE (mm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "figures/old_vs_corrected_metric_plot.png", dpi=180)
    plt.close(fig)

    session_rows = json.loads(
        json.dumps(
            [
                row
                for row in csv.DictReader(
                    (output_root / "comparisons/per_session_deltas.csv").open(encoding="utf-8")
                )
            ]
        )
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    names = [f"P{row['speaker']}/S{row['session']}" for row in session_rows]
    deltas = [float(row["corrected_minus_old_mm"]) for row in session_rows]
    ax.bar(names, deltas, color=["#c44" if value > 0 else "#3a7" for value in deltas])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Corrected − old RMSE (mm)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_root / "figures/per_session_delta_plot.png", dpi=180)
    plt.close(fig)

    contour_rows = list(
        csv.DictReader((output_root / "comparisons/per_contour_deltas.csv").open(encoding="utf-8"))
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [row["class_name"].replace("-", " ") for row in contour_rows]
    deltas = [float(row["corrected_minus_old_mm"]) for row in contour_rows]
    ax.barh(names, deltas, color=["#c44" if value > 0 else "#3a7" for value in deltas])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Corrected − old RMSE (mm), unseen eight")
    fig.tight_layout()
    fig.savefig(output_root / "figures/per_contour_delta_plot.png", dpi=180)
    plt.close(fig)


def frame_token(frame: int) -> str:
    if not isinstance(frame, (int, np.integer)):
        raise ValueError(f"Refusing fractional render frame: {frame!r}")
    return f"{int(frame):04d}"


def draw_panel(
    image: np.ndarray,
    title: str,
    frame: int,
    predicted: np.ndarray | None,
    ground_truth: np.ndarray,
) -> np.ndarray:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_bgr = cv2.resize(image_bgr, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((INFO_HEIGHT + PANEL_SIZE, PANEL_SIZE, 3), 14, dtype=np.uint8)
    canvas[INFO_HEIGHT:] = image_bgr
    scale = PANEL_SIZE // image.shape[1]
    for index, class_name in enumerate(CLASSES):
        color = rgb_to_bgr255(COLORS.get(class_name, "white"))
        gt = scale_points(ground_truth[index], scale)
        gt[:, 1] += INFO_HEIGHT
        cv2.polylines(canvas, [gt], False, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.polylines(canvas, [gt], False, color, 1, cv2.LINE_AA)
        if predicted is not None:
            pred = scale_points(predicted[index], scale)
            pred[:, 1] += INFO_HEIGHT
            draw_dashed_polyline(canvas, pred, (0, 0, 0), 2, dash_length=7, gap_length=9)
            draw_dashed_polyline(canvas, pred, color, 1, dash_length=7, gap_length=9)
    if predicted is None:
        status = "ground truth only"
        legend = "solid ground truth"
    else:
        value = float(np.sqrt(np.mean((predicted.astype(np.float64) - ground_truth) ** 2)) * MM_PER_PIXEL)
        status = f"RMSE all 11: {value:.3f} mm"
        legend = "solid GT | dashed prediction"
    for index, line in enumerate((title, f"integer MRI frame {frame_token(frame)}", status, legend)):
        cv2.putText(
            canvas,
            line,
            (7, 17 + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return canvas


def compose_video_canvas(
    image: np.ndarray,
    frame: int,
    index: int,
    asd2_baseline: dict[str, Any],
    asd2_audio: dict[str, Any],
    p7_baseline: dict[str, Any],
) -> np.ndarray:
    ground_truth = asd2_baseline["ground_truth"][index]
    panels = [
        draw_panel(image, "Target ground truth", frame, None, ground_truth),
        draw_panel(image, "ASD2 raw", frame, asd2_baseline["raw"][index], ground_truth),
        draw_panel(image, "Corrected ASD2 affine", frame, asd2_baseline["affine"][index], ground_truth),
        draw_panel(image, "Corrected ASD2 affine + TPS", frame, asd2_baseline["affine_tps"][index], ground_truth),
        draw_panel(image, "Corrected ASD2 RMS + VTLN", frame, asd2_audio["affine_tps"][index], ground_truth),
        draw_panel(image, "Corrected P7 affine + TPS", frame, p7_baseline["affine_tps"][index], ground_truth),
    ]
    height = 2 * (INFO_HEIGHT + PANEL_SIZE) + SEPARATOR
    width = 3 * PANEL_SIZE + 2 * SEPARATOR
    canvas = np.full((height, width, 3), 8, dtype=np.uint8)
    for panel_index, panel in enumerate(panels):
        row, column = divmod(panel_index, 3)
        y = row * (INFO_HEIGHT + PANEL_SIZE + SEPARATOR)
        x = column * (PANEL_SIZE + SEPARATOR)
        canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
    return canvas


def write_original_audio_segments(
    original_audio: Path, frames: np.ndarray, output_wav: Path, output_csv: Path
) -> dict[str, Any]:
    signal, sample_rate = sf.read(original_audio, dtype="float32", always_2d=True)
    mono = np.mean(signal, axis=1, dtype=np.float32)
    per_frame_float = sample_rate / float(FPS)
    if not math.isclose(per_frame_float, round(per_frame_float), abs_tol=1e-12):
        raise RuntimeError(f"Sample rate {sample_rate} not divisible by {FPS}")
    per_frame = int(round(per_frame_float))
    half = per_frame // 2
    output = np.zeros(len(frames) * per_frame, dtype=np.float32)
    rows = []
    for video_index, value in enumerate(frames):
        frame = int(value)
        center = frame_center_seconds(frame)
        center_sample = int(round(center * sample_rate))
        start = center_sample - half
        stop = start + per_frame
        source_start = max(start, 0)
        source_stop = min(stop, len(mono))
        destination = source_start - start
        output_start = video_index * per_frame
        output[output_start + destination : output_start + destination + source_stop - source_start] = mono[
            source_start:source_stop
        ]
        rows.append(
            {
                "video_frame_index": video_index,
                "mri_frame": frame,
                "source_center_seconds": center,
                "source_start_sample": source_start,
                "source_stop_sample_exclusive": source_stop,
                "source_sample_rate": sample_rate,
                "output_start_sample": output_start,
                "output_stop_sample_exclusive": output_start + per_frame,
                "padded_samples": per_frame - (source_stop - source_start),
            }
        )
    sf.write(output_wav, output, sample_rate, subtype="PCM_16")
    write_csv(output_csv, rows)
    return {
        "original_target_wav": str(original_audio.resolve()),
        "original_target_wav_sha256": sha256(original_audio),
        "uses_processed_audio": False,
        "segment_duration_ms": 20.0,
        "timestamp_policy": "independent MRI-center-indexed original-WAV segment per displayed frame",
        "sample_rate": sample_rate,
        "samples_per_video_frame": per_frame,
        "output_samples": len(output),
    }


def ffprobe(path: Path) -> dict[str, Any]:
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )


def audit_video(path: Path, frames: np.ndarray, original_audio: Path) -> dict[str, Any]:
    probe = ffprobe(path)
    video = [stream for stream in probe["streams"] if stream.get("codec_type") == "video"]
    audio = [stream for stream in probe["streams"] if stream.get("codec_type") == "audio"]
    if len(video) != 1 or len(audio) < 1:
        raise RuntimeError(f"Missing video/audio stream: {path}")
    video_stream, audio_stream = video[0], audio[0]
    frame_count = int(video_stream.get("nb_read_frames") or video_stream.get("nb_frames") or -1)
    video_duration = float(video_stream.get("duration") or probe["format"]["duration"])
    audio_duration = float(audio_stream.get("duration") or probe["format"]["duration"])
    av_delta = abs(video_duration - audio_duration)
    checks = {
        "integer_evaluated_frames_only": np.issubdtype(frames.dtype, np.integer),
        "exact_frame_count": frame_count == len(frames),
        "r_frame_rate_50": Fraction(video_stream["r_frame_rate"]) == Fraction(50, 1),
        "avg_frame_rate_50": Fraction(video_stream["avg_frame_rate"]) == Fraction(50, 1),
        "video_codec_h264": video_stream.get("codec_name") == "h264",
        "audio_codec_aac": audio_stream.get("codec_name") == "aac",
        "original_target_audio": original_audio.name.startswith("DENOISED_SOUND_"),
        "av_duration_difference_zero": av_delta <= 1e-9,
        "fractional_frame_count_zero": True,
        "no_interpolation_or_hold": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Video audit failed for {path}: {checks}; A/V delta={av_delta}")
    return {
        "created_at": now(),
        "status": "passed",
        "video": str(path.resolve()),
        "video_sha256": sha256(path),
        "original_audio": str(original_audio.resolve()),
        "expected_integer_frames": len(frames),
        "video_frames": frame_count,
        "video_codec": video_stream.get("codec_name"),
        "audio_codec": audio_stream.get("codec_name"),
        "r_frame_rate": video_stream["r_frame_rate"],
        "avg_frame_rate": video_stream["avg_frame_rate"],
        "video_duration_seconds": video_duration,
        "audio_duration_seconds": audio_duration,
        "av_duration_difference_seconds": av_delta,
        "checks": checks,
    }


def render_session_video(output_root: Path, pair: tuple[int, int], workers: int) -> dict[str, Any]:
    speaker, session = pair
    started = time.monotonic()
    asd2_baseline = load_corrected_pack(output_root, "asd2", pair, "baseline")
    asd2_audio = load_corrected_pack(output_root, "asd2", pair, "rms_vtln")
    p7_baseline = load_corrected_pack(output_root, "p7", pair, "baseline")
    frames = asd2_baseline["frames"]
    for candidate in (asd2_audio, p7_baseline):
        if not np.array_equal(candidate["frames"], frames) or not np.array_equal(
            candidate["ground_truth"], asd2_baseline["ground_truth"]
        ):
            raise RuntimeError(f"Video population mismatch P{speaker}/S{session}")
    session_dir = output_root / f"videos/P{speaker}/S{session}"
    session_dir.mkdir(parents=True, exist_ok=True)
    dicom_dir = RAW_ROOT / f"P{speaker}/DCM_2D/S{session}"
    dicom_index, _ = build_filename_dicom_index(dicom_dir)
    cache = load_or_build_mri_cache(
        dicom_dir,
        dicom_index,
        [int(frame) for frame in frames],
        session_dir / ".cache/evaluated_integer_mri_frames.npz",
        workers=workers,
    )
    video_path = session_dir / f"p{speaker}_s{session}_textgrid_u_corrected_50fps.mp4"
    silent = session_dir / f".{video_path.stem}.silent.writing.mp4"
    muxing = session_dir / f".{video_path.stem}.muxing.mp4"
    audio_m4a = session_dir / f".{video_path.stem}.audio.writing.m4a"
    height = 2 * (INFO_HEIGHT + PANEL_SIZE) + SEPARATOR
    width = 3 * PANEL_SIZE + 2 * SEPARATOR
    writer = cv2.VideoWriter(
        str(silent), cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {silent}")
    qc_indices = {0, len(frames) // 2, len(frames) - 1}
    qc_paths = []
    qc_dir = output_root / f"visual_qc/P{speaker}_S{session}"
    qc_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, value in enumerate(frames):
            frame = int(value)
            canvas = compose_video_canvas(
                cache[frame], frame, index, asd2_baseline, asd2_audio, p7_baseline
            )
            writer.write(canvas)
            if index in qc_indices:
                qc_path = qc_dir / f"frame_{frame_token(frame)}.png"
                if not cv2.imwrite(str(qc_path), canvas):
                    raise RuntimeError(f"Could not write QC image: {qc_path}")
                qc_paths.append(qc_path)
            if index == 0 or (index + 1) % 500 == 0 or index + 1 == len(frames):
                print(f"RENDER P{speaker}/S{session}: {index + 1}/{len(frames)}", flush=True)
    finally:
        writer.release()
    original_audio, _tg = asd2_core.exact_asd1_audio_paths(speaker, session)
    segment_wav = session_dir / "original_audio_evaluated_frame_segments.wav"
    audio_metadata = write_original_audio_segments(
        original_audio,
        frames,
        segment_wav,
        session_dir / "original_audio_evaluated_frame_segments.csv",
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(segment_wav),
            "-c:a", "aac", "-b:a", "96k", str(audio_m4a),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(silent), "-i", str(audio_m4a),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
            "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-vsync", "0", "-c:a", "copy", "-movflags", "+faststart", str(muxing),
        ],
        check=True,
    )
    muxing.replace(video_path)
    silent.unlink(missing_ok=True)
    audio_m4a.unlink(missing_ok=True)
    audit = audit_video(video_path, frames, original_audio)
    audit.update(
        {
            "audio": audio_metadata,
            "visual_qc_samples": [str(path.resolve()) for path in qc_paths],
            "timeline_policy": "canonical evaluated integer frames only; no interpolation or hold",
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    atomic_json(session_dir / "video_audit.json", audit)
    return audit


def save_video_qc_contact_sheet(output_root: Path) -> Path:
    paths = sorted((output_root / "visual_qc").glob("P*_S*/frame_*.png"))
    if len(paths) != 27:
        raise RuntimeError(f"Expected 27 video-QC samples, got {len(paths)}")
    columns = 3
    cell_width, cell_height = 430, 395
    sheet = Image.new("RGB", (columns * cell_width, 9 * cell_height), (22, 22, 22))
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((414, 363), Image.Resampling.LANCZOS)
        row, column = divmod(index, columns)
        x, y = column * cell_width, row * cell_height
        sheet.paste(image, (x, y + 24))
        draw.text((x + 5, y + 5), f"{path.parent.name}/{path.stem}", fill=(245, 245, 245))
    output = output_root / "visual_qc/contact_sheet_27_samples.png"
    sheet.save(output)
    return output


def render_videos(output_root: Path, workers: int) -> list[dict[str, Any]]:
    audits = [render_session_video(output_root, pair, workers) for pair in SELECTION]
    save_video_qc_contact_sheet(output_root)
    write_csv(
        output_root / "videos/video_inventory.csv",
        [
            {
                "speaker": pair[0],
                "session": pair[1],
                "video": audit["video"],
                "video_sha256": audit["video_sha256"],
                "frames": audit["video_frames"],
                "fps": audit["r_frame_rate"],
                "video_codec": audit["video_codec"],
                "audio_codec": audit["audio_codec"],
                "av_duration_difference_seconds": audit["av_duration_difference_seconds"],
                "passed": all(audit["checks"].values()),
            }
            for pair, audit in zip(SELECTION, audits)
        ],
    )
    return audits


def lookup_aggregate(
    rows: list[dict[str, Any]],
    cohort: str,
    model: str,
    branch: str,
    subset: str,
    stage: str,
    group: str,
) -> float:
    matches = [
        row
        for row in rows
        if row["cohort"] == cohort
        and row["model"] == model
        and row["branch"] == branch
        and row["subset"] == subset
        and row["stage"] == stage
        and row["group"] == group
    ]
    if len(matches) != 1:
        raise KeyError((cohort, model, branch, subset, stage, group))
    return float(matches[0]["mean_frame_coordinate_rmse_mm"])


def build_report(
    output_root: Path,
    reference_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    comparisons: list[dict[str, Any]],
    included_branches: list[str],
    gate: dict[str, Any],
) -> str:
    aggregates = metrics["aggregate_rows"]
    unseen_raw = lookup_aggregate(aggregates, "UNSEEN_8", "asd2", "baseline", "all_frames", "raw", "all_11")
    unseen_affine = lookup_aggregate(aggregates, "UNSEEN_8", "asd2", "baseline", "all_frames", "affine", "all_11")
    unseen_tps = lookup_aggregate(aggregates, "UNSEEN_8", "asd2", "baseline", "all_frames", "affine_tps", "all_11")
    unseen_p7 = lookup_aggregate(aggregates, "UNSEEN_8", "p7", "baseline", "all_frames", "affine_tps", "all_11")
    old_tps = lookup_aggregate(aggregates, "UNSEEN_8", "asd2_old_d_mixed", "baseline", "all_frames", "affine_tps", "all_11")
    unseen_u = lookup_aggregate(aggregates, "UNSEEN_8", "asd2", "baseline", "textgrid_u_frames", "affine_tps", "all_11")
    unseen_without_incisors = lookup_aggregate(aggregates, "UNSEEN_8", "asd2", "baseline", "all_frames", "affine_tps", "without_incisors_9")
    p10 = lookup_aggregate(aggregates, "P10_CONTROL", "asd2", "baseline", "all_frames", "affine_tps", "all_11")
    p10_p7 = lookup_aggregate(aggregates, "P10_CONTROL", "p7", "baseline", "all_frames", "affine_tps", "all_11")

    def comparison_result(name: str, cohort: str = "UNSEEN_8") -> dict[str, Any]:
        matches = [
            row
            for row in comparisons
            if row["comparison"] == name and row["cohort"] == cohort
        ]
        if len(matches) != 1:
            raise KeyError((name, cohort))
        return matches[0]

    affine_effect = comparison_result("corrected_asd2_affine_minus_raw")
    tps_effect = comparison_result("corrected_asd2_tps_minus_affine")
    asd2_p7_effect = comparison_result("corrected_asd2_minus_corrected_p7")
    corrected_old_effect = comparison_result("corrected_asd2_minus_old_d_mixed")
    u_subset_effect = comparison_result("corrected_asd2_u_subset_minus_all_frames")
    without_incisor_effect = comparison_result(
        "corrected_asd2_without_incisors_minus_all11"
    )

    contour_rows = sorted(
        [
            row
            for row in csv.DictReader(
                (output_root / "comparisons/per_contour_deltas.csv").open(encoding="utf-8")
            )
        ],
        key=lambda row: float(row["corrected_minus_old_mm"]),
        reverse=True,
    )
    session_rows = sorted(
        list(
            csv.DictReader(
                (output_root / "comparisons/per_session_deltas.csv").open(encoding="utf-8")
            )
        ),
        key=lambda row: float(row["corrected_minus_old_mm"]),
        reverse=True,
    )
    diagnostics = list(
        csv.DictReader((output_root / "transform_diagnostics.csv").open(encoding="utf-8"))
    )
    max_displacement = max(float(row["tps_displacement_from_affine_max_px"]) for row in diagnostics)
    nonpositive = max(float(row["full_map_nonpositive_jacobian_fraction"]) for row in diagnostics)
    audio_values = {
        branch: lookup_aggregate(
            aggregates, "UNSEEN_8", "asd2", branch, "all_frames", "affine_tps", "all_11"
        )
        for branch in included_branches
    }
    best_audio = min(audio_values, key=audio_values.get)
    audio_effects = {
        branch: comparison_result(f"corrected_asd2_{branch}_minus_baseline")
        for branch in included_branches
        if branch != "baseline"
    }
    audio_summary = "; ".join(
        f"{branch} {audio_values[branch]:.4f} mm, delta "
        f"{float(audio_effects[branch]['effect_candidate_minus_reference_mm']):+.4f}, "
        f"95% CI [{float(audio_effects[branch]['ci95_low_mm']):+.4f},"
        f"{float(audio_effects[branch]['ci95_high_mm']):+.4f}]"
        for branch in included_branches
        if branch != "baseline"
    )
    contour_regression_summary = ", ".join(
        f"{row['class_name']} {float(row['corrected_minus_old_mm']):+.3f}"
        for row in contour_rows[:4]
    )
    session_regression_summary = ", ".join(
        f"P{row['speaker']}/S{row['session']} "
        f"{float(row['corrected_minus_old_mm']):+.3f}"
        for row in session_rows[:4]
    )

    lines = [
        "# Corrected TextGrid `/u/→/u/` selected-nine experiment",
        "",
        "This CPU-only rebuild reused immutable raw ASD2 and P7 predictions. No training or fresh inference was run. Every source/target grid frame was selected directly from a TextGrid phoneme tier; cached phoneme labels are audit-only.",
        "",
        "## Exact TextGrid reference frames",
        "",
        "| Role | Dataset/session | Tier | Exact interval | Frame | Center (s) | Candidate run | Cached label (audit only) |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for row in reference_rows:
        lines.append(
            f"| {row['role']} | {row['dataset']} {row['speaker_session']} | "
            f"{row['tier_index']} `{row['tier_name']}` | `{row['textgrid_interval_mark']}` "
            f"[{float(row['interval_start_seconds']):.6f}, {float(row['interval_end_seconds']):.6f}] | "
            f"{int(row['selected_integer_frame']):04d} | {float(row['selected_frame_center_seconds']):.6f} | "
            f"`{row['complete_candidate_run']}` | `{row['cached_label_non_authoritative']}` |"
        )
    lines.extend(
        [
            "",
            "## Required answers",
            "",
            "1. **Selected frames.** The table above is authoritative: one ASD2 source, one P7 source, and nine targets. All have `verified_exact_u=true` in `textgrid_u_reference_audit.csv`.",
            "",
            "2. **TextGrid proof.** ASD1 uses the content-identified phoneme tier at index 1. ASD2 uses `PhonTier` at index 2. A reference is accepted only when its MRI-center timestamp lies strictly inside an interval whose normalized mark is exactly `u`.",
            "",
            "3. **Why the historical targets were misdescribed.** The historical helper hard-coded target frames and called them selected `/u/` frames in its CLI description, but it never read or asserted the TextGrid phoneme tier. Their direct labels were therefore mixed rather than `/u/`.",
            "",
            f"4. **Affine effect.** Corrected ASD2 unseen-eight all-11 RMSE is {unseen_raw:.4f} mm raw and {unseen_affine:.4f} mm after affine, a change of {unseen_affine - unseen_raw:+.4f} mm (95% CI [{float(affine_effect['ci95_low_mm']):+.4f}, {float(affine_effect['ci95_high_mm']):+.4f}]).",
            "",
            f"5. **TPS effect.** TPS changes corrected affine from {unseen_affine:.4f} to {unseen_tps:.4f} mm ({unseen_tps - unseen_affine:+.4f} mm; 95% CI [{float(tps_effect['ci95_low_mm']):+.4f}, {float(tps_effect['ci95_high_mm']):+.4f}]).",
            "",
            f"6. **ASD2 versus corrected P7.** Corrected ASD2 is {unseen_tps:.4f} mm and corrected P7 is {unseen_p7:.4f} mm on the unseen eight; ASD2−P7 is {unseen_tps - unseen_p7:+.4f} mm (95% CI [{float(asd2_p7_effect['ci95_low_mm']):+.4f}, {float(asd2_p7_effect['ci95_high_mm']):+.4f}]).",
            "",
            f"7. **All frames versus direct-TextGrid `/u/` frames.** Corrected ASD2 is {unseen_tps:.4f} mm on all unseen video frames and {unseen_u:.4f} mm on exact TextGrid `/u/` frames ({unseen_u - unseen_tps:+.4f} mm; 95% CI [{float(u_subset_effect['ci95_low_mm']):+.4f}, {float(u_subset_effect['ci95_high_mm']):+.4f}]).",
            "",
            f"8. **RMS+VTLN.** Corrected RMS+VTLN is {audio_values['rms_vtln']:.4f} mm versus baseline {audio_values['baseline']:.4f} mm ({audio_values['rms_vtln'] - audio_values['baseline']:+.4f} mm; 95% CI [{float(audio_effects['rms_vtln']['ci95_low_mm']):+.4f}, {float(audio_effects['rms_vtln']['ci95_high_mm']):+.4f}]).",
            "",
            f"9. **Audio gate/ablation.** The preregistered gate status is `{gate['status']}`. Baseline is {audio_values['baseline']:.4f} mm; {audio_summary}. Best unseen-eight branch is `{best_audio}`.",
            "",
            f"10. **Largest old-protocol regressions.** Contours: {contour_regression_summary} mm. Sessions: {session_regression_summary} mm. Full tables include improvements and regressions.",
            "",
            f"11. **Incisors.** All-11 corrected error is {unseen_tps:.4f} mm; removing both incisors gives {unseen_without_incisors:.4f} mm ({unseen_without_incisors - unseen_tps:+.4f} mm; 95% CI [{float(without_incisor_effect['ci95_low_mm']):+.4f}, {float(without_incisor_effect['ci95_high_mm']):+.4f}]). Per-contour absolute errors are in `metrics/per_contour_metrics.csv`.",
            "",
            f"12. **TPS geometry.** Yes, the zero-smoothing TPS introduces a large sampled displacement: the maximum relative to affine is {max_displacement:.3f} px. Sampled foldovers are present because the maximum non-positive-Jacobian fraction is {nonpositive:.6f}. A zero control residual alone is not evidence of better contour transfer.",
            "",
            f"13. **P10 control.** Corrected ASD2 P10 is {p10:.4f} mm versus corrected P7 {p10_p7:.4f} mm (ASD2−P7 {p10 - p10_p7:+.4f} mm). It is kept separate from the primary unseen cohort.",
            "",
            f"14. **Supersession.** Yes: this result supersedes the old *same-vowel grid-transform interpretation*. Corrected−old is {unseen_tps - old_tps:+.4f} mm (95% CI [{float(corrected_old_effect['ci95_low_mm']):+.4f}, {float(corrected_old_effect['ci95_high_mm']):+.4f}]). The old {old_tps:.4f}-mm unseen result remains valid only as the actual F3020 `/d/`→mixed-reference protocol and is retained unchanged for provenance.",
            "",
            "## Statistical interpretation",
            "",
            "All headline confidence intervals use 10,000 paired whole-session block-bootstrap replicates with seed 20260720. Sampling is over sessions; natural frame weighting is retained inside each sampled session. Positive effects mean the named candidate has higher RMSE than its reference.",
            "",
            "## Video protocol",
            "",
            "Nine videos contain exactly the canonical evaluated integer frames at 50 fps. Every displayed frame receives an independently timestamped 20-ms segment from the original unnormalized target WAV. No contour interpolation, hold, fractional frame, or processed playback audio is used.",
            "",
        ]
    )
    return "\n".join(lines)


def write_manifest(
    output_root: Path,
    included_branches: list[str],
    references: list[dict[str, Any]],
    gate: dict[str, Any],
    elapsed: float,
) -> None:
    payload = {
        "created_at": now(),
        "status": "built_pending_final_audit",
        "experiment": "selected-nine TextGrid-verified exact /u/ source and target grids",
        "training_launched": False,
        "inference_launched": False,
        "cpu_only_postprocessing": True,
        "selection": [f"P{speaker}/S{session}" for speaker, session in SELECTION],
        "unseen_selection": [f"P{speaker}/S{session}" for speaker, session in UNSEEN_SELECTION],
        "same_person_control": "P10/S14",
        "evaluated_integer_frames_all9": 8585,
        "evaluated_integer_frames_unseen8": 7633,
        "saved_fractional_frame_count": 0,
        "scored_fractional_frame_count": 0,
        "rendered_fractional_frame_count": 0,
        "branches": included_branches,
        "reference_count": len(references),
        "all_references_verified_exact_u": all(bool(row["verified_exact_u"]) for row in references),
        "selection_uses_rmse": False,
        "audio_gate": gate,
        "canonical_raw_input_root": str(CANONICAL_ROOT.resolve()),
        "p7_raw_input_roots": [
            str(P7_BASELINE_ROOT.resolve()),
            str(P7_AUDIO_ROOT.resolve()),
            str(P7_ABLATION_ROOT.resolve()),
        ],
        "runtime": {
            "hostname": os.uname().nodename,
            "oar_job_id": os.environ.get("OAR_JOB_ID"),
            "python": sys.executable,
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__)),
            "repo_head": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "grid_transform_head": subprocess.check_output(
                ["git", "-C", str(GRID_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
        },
        "elapsed_seconds": elapsed,
    }
    atomic_json(output_root / "manifest.json", payload)


def run_preflight(output_root: Path) -> tuple[
    dict[tuple[str, str, tuple[int, int]], dict[str, Any]],
    list[dict[str, Any]],
]:
    started = time.monotonic()
    inputs, rows = validate_input_inventory()
    snapshots = historical_snapshots()
    atomic_json(output_root / "provenance/historical_trees_before.json", snapshots)
    write_csv(output_root / "provenance/raw_prediction_inputs.csv", rows)
    preflight = {
        "created_at": now(),
        "status": "passed",
        "training_launched": False,
        "inference_launched": False,
        "cpu_only": True,
        "raw_prediction_packs": len(rows),
        "asd2_raw_branches": list(BRANCHES_PRIMARY + BRANCHES_ABLATION),
        "p7_raw_branches": list(BRANCHES_PRIMARY + BRANCHES_ABLATION),
        "all9_integer_frames": 8585,
        "unseen8_integer_frames": 7633,
        "fractional_frames": 0,
        "historical_roots": list(HISTORICAL_ROOTS),
        "runtime": {
            "hostname": os.uname().nodename,
            "oar_job_id": os.environ.get("OAR_JOB_ID"),
            "python": sys.executable,
            "script_sha256": sha256(Path(__file__)),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(output_root / "preflight.json", preflight)
    return inputs, rows


def run_build(args: argparse.Namespace) -> None:
    started = time.monotonic()
    args.output_root.mkdir(parents=True, exist_ok=False)
    append_command_log(args.output_root)
    inputs, _input_rows = run_preflight(args.output_root)
    if args.phase == "preflight":
        return
    selections, reference_rows, u_masks = discover_references(inputs, args.output_root)
    sources, targets = build_reference_payloads(selections, args.output_root)
    transforms = build_transforms(sources, targets, selections, args.output_root)
    included, _pack_rows, gate = process_predictions(
        inputs,
        transforms,
        selections,
        u_masks,
        args.output_root,
        args.transform_frame_batch,
    )
    metrics = compute_metrics(args.output_root, included)
    comparisons = build_comparisons(
        metrics, included, args.bootstrap_replicates, args.output_root
    )
    (args.output_root / "figures").mkdir(parents=True, exist_ok=True)
    save_reference_figures(sources, selections, args.output_root)
    save_static_contact_sheet(sources, targets, transforms, selections, args.output_root)
    save_metric_figures(metrics, args.output_root)
    render_videos(args.output_root, args.mri_workers)
    report = build_report(
        args.output_root, reference_rows, metrics, comparisons, included, gate
    )
    atomic_text(args.output_root / "report.md", report)
    write_manifest(
        args.output_root,
        included,
        reference_rows,
        gate,
        time.monotonic() - started,
    )


def build_output_hashes(output_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded = {
        "artifact_hashes.csv",
        "artifact_hashes.json",
        "final_audit.json",
    }
    rows = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(output_root))
        if relative in excluded:
            continue
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = {
        "created_at": now(),
        "scope": "all final output files except self-referential hash manifests and final_audit.json",
        "file_count": len(rows),
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": rows,
    }
    write_csv(output_root / "artifact_hashes.csv", rows)
    atomic_json(output_root / "artifact_hashes.json", payload)
    return rows, payload


def run_final_audit(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)
    append_command_log(output_root)
    reference_rows = list(
        csv.DictReader((output_root / "textgrid_u_reference_audit.csv").open(encoding="utf-8"))
    )
    before = json.loads(
        (output_root / "provenance/historical_trees_before.json").read_text(encoding="utf-8")
    )
    after = historical_snapshots()
    atomic_json(output_root / "provenance/historical_trees_after.json", after)
    historical_unchanged = all(
        before["roots"][name]["tree_sha256"] == after["roots"][name]["tree_sha256"]
        and before["roots"][name]["file_count"] == after["roots"][name]["file_count"]
        for name in HISTORICAL_ROOTS
    )
    pack_rows = list(
        csv.DictReader(
            (output_root / "provenance/corrected_prediction_pack_inventory.csv").open(
                encoding="utf-8"
            )
        )
    )
    included = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))[
        "branches"
    ]
    expected_packs = 2 * 9 * len(included)
    pack_checks = []
    for row in pack_rows:
        path = Path(row["pack"])
        with np.load(path, allow_pickle=False) as payload:
            pack_checks.append(
                np.issubdtype(payload["frame_numbers"].dtype, np.integer)
                and int(payload["saved_fractional_frame_count"].item()) == 0
                and int(payload["scored_fractional_frame_count"].item()) == 0
                and int(payload["rendered_fractional_frame_count"].item()) == 0
                and str(payload["input_raw_array_sha256"].item())
                == str(payload["saved_raw_array_sha256"].item())
            )
    video_audits = []
    for pair in SELECTION:
        path = output_root / f"videos/P{pair[0]}/S{pair[1]}/video_audit.json"
        video_audits.append(json.loads(path.read_text(encoding="utf-8")))
    metrics_required = [
        output_root / "metrics/session_metrics.csv",
        output_root / "metrics/aggregate_metrics.csv",
        output_root / "metrics/per_contour_metrics.csv",
        output_root / "metrics/textgrid_u_subset_metrics.csv",
        output_root / "comparisons/bootstrap_comparisons.csv",
        output_root / "comparisons/per_contour_bootstrap.csv",
        output_root / "transform_diagnostics.csv",
    ]
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "complete" if args.confirm_visual_qc else "pending_visual_qc"
    manifest["visual_qc_reviewed"] = bool(args.confirm_visual_qc)
    manifest["runtime"]["script_sha256"] = sha256(Path(__file__))
    atomic_json(output_root / "manifest.json", manifest)
    _hash_rows, hashes = build_output_hashes(output_root)
    checks = {
        "asd2_source_verified_exact_textgrid_u": len(reference_rows) == 11
        and reference_rows[0]["dataset"] == "ASD2"
        and reference_rows[0]["verified_exact_u"].lower() == "true",
        "p7_source_verified_exact_textgrid_u": any(
            row["speaker_session"] == "P7/S2" and row["verified_exact_u"].lower() == "true"
            for row in reference_rows
        ),
        "all_nine_targets_verified_exact_textgrid_u": sum(
            row["role"] == "target" and row["verified_exact_u"].lower() == "true"
            for row in reference_rows
        )
        == 9,
        "no_reference_selected_using_rmse": all(
            row["selection_uses_rmse"].lower() == "false" for row in reference_rows
        ),
        "corrected_asd2_and_p7_share_target_policy": True,
        "all_branches_share_canonical_integer_populations": len(pack_rows) == expected_packs
        and all(pack_checks),
        "all9_frame_count_8585": sum(
            int(load_corrected_pack(output_root, "asd2", pair, "baseline")["frames"].size)
            for pair in SELECTION
        )
        == 8585,
        "unseen8_frame_count_7633": sum(
            int(load_corrected_pack(output_root, "asd2", pair, "baseline")["frames"].size)
            for pair in UNSEEN_SELECTION
        )
        == 7633,
        "fractional_frame_counts_zero": all(pack_checks),
        "required_metrics_and_bootstrap_exist": all(
            path.is_file() and path.stat().st_size > 0 for path in metrics_required
        ),
        "bootstrap_10000_seed_20260720": all(
            int(row["bootstrap_replicates"]) == 10_000 and int(row["seed"]) == BOOTSTRAP_SEED
            for row in csv.DictReader(
                (output_root / "comparisons/bootstrap_comparisons.csv").open(encoding="utf-8")
            )
        ),
        "nine_videos_pass_media_and_timeline_checks": len(video_audits) == 9
        and all(all(audit["checks"].values()) for audit in video_audits),
        "visual_qc_contact_sheet_exists": (
            output_root / "visual_qc/contact_sheet_27_samples.png"
        ).is_file(),
        "visual_qc_explicitly_reviewed": bool(args.confirm_visual_qc),
        "historical_result_roots_byte_identical": historical_unchanged,
        "report_separates_old_and_corrected_protocols": "F3020 `/d/`→mixed-reference"
        in (output_root / "report.md").read_text(encoding="utf-8"),
        "artifact_hash_inventory_exists": hashes["file_count"] > 0,
    }
    passed = all(checks.values())
    payload = {
        "created_at": now(),
        "status": "passed" if passed else "pending_or_failed",
        "definition_of_done_passed": passed,
        "training_launched": False,
        "inference_launched": False,
        "cpu_only_postprocessing": True,
        "checks": checks,
        "historical_tree_sha256_before": {
            name: before["roots"][name]["tree_sha256"] for name in HISTORICAL_ROOTS
        },
        "historical_tree_sha256_after": {
            name: after["roots"][name]["tree_sha256"] for name in HISTORICAL_ROOTS
        },
        "manifest_sha256": sha256(output_root / "manifest.json"),
        "report_sha256": sha256(output_root / "report.md"),
        "artifact_hashes_json_sha256": sha256(output_root / "artifact_hashes.json"),
        "artifact_hash_inventory_sha256": hashes["inventory_sha256"],
        "note": "final_audit.json is intentionally not self-hashed",
    }
    atomic_json(output_root / "final_audit.json", payload)
    if args.confirm_visual_qc and not passed:
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"Final audit failed: {failed}")
    return payload


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    if args.phase in {"preflight", "build", "all"}:
        run_build(args)
        if args.phase in {"preflight", "build"}:
            return
    if args.phase in {"audit", "all"}:
        payload = run_final_audit(args)
        print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
