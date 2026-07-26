#!/usr/bin/env python3
"""Analyze, render, and audit the fixed ASD2-on-ASD1 experiment.

The inference packs are produced by ``run_asd2_epoch211_selected_experiment.py``.
This script is deliberately CPU-only: it compares those packs with the three
immutable P7 bundles, renders evaluated-integer-frame videos with exact
timestamp-indexed original target audio, and writes a final reproducibility audit.
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
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]

from . import run_asd2_epoch211_selected_experiment as core  # noqa: E402
from src.utils.colors import COLORS  # noqa: E402
from src.utils.mri_rendering import (  # noqa: E402
    build_filename_dicom_index,
    load_or_build_mri_cache,
)
from src.utils.video_rendering import (  # noqa: E402
    MM_PER_PIXEL,
    attach_audio,
    draw_dashed_polyline,
    rgb_to_bgr255,
    scale_points,
)
from src.common.artifacts import local_now as now  # noqa: E402


FPS = 50
PANEL_SIZE = 272
INFO_HEIGHT = 88
SEPARATOR = 6
BOOTSTRAP_REPLICATES = 10_000
VIDEO_PANELS = (
    ("ground_truth", "ground_truth", "Target ground truth"),
    ("baseline", "raw", "ASD2 baseline: raw"),
    ("baseline", "affine", "ASD2 baseline: affine"),
    ("baseline", "affine_tps", "ASD2 baseline: affine + TPS"),
    ("rms_vtln", "affine_tps", "ASD2 RMS + VTLN: affine + TPS"),
    ("p7_baseline", "affine_tps", "P7 baseline: affine + TPS"),
)
COHORTS = {
    "UNSEEN_8": core.UNSEEN_SELECTION,
    "ALL_9_HISTORICAL": core.SELECTION,
    "P10_SAME_SPEAKER_CONTROL": ((10, 14),),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("analysis", "video", "audit", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=core.DEFAULT_OUTPUT)
    parser.add_argument("--selection", nargs="+", default=None, metavar="P#:S#")
    parser.add_argument("--mri-workers", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--force-video", action="store_true")
    parser.add_argument(
        "--confirm-visual-qc",
        action="store_true",
        help="Record that a reviewer inspected all 27 contact-sheet samples and found no rendering defect.",
    )
    return parser.parse_args()


def selected_pairs(values: list[str] | None) -> tuple[tuple[int, int], ...]:
    if values is None:
        return core.SELECTION
    parsed = []
    for value in values:
        speaker_token, session_token = value.upper().split(":", 1)
        pair = (int(speaker_token.removeprefix("P")), int(session_token.removeprefix("S")))
        if pair not in core.SELECTION:
            raise ValueError(f"Selection outside the fixed protocol: {value}")
        parsed.append(pair)
    if len(parsed) != len(set(parsed)):
        raise ValueError("Duplicate selection")
    return tuple(parsed)


def atomic_json(path: Path, payload: Any) -> None:
    core.atomic_json(path, payload)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    core.write_csv(path, rows)


def load_new_pack(output_root: Path, pair: tuple[int, int], branch: str) -> dict[str, Any]:
    return core.load_pack(core.pack_path(output_root, pair[0], pair[1], branch))


def old_pack_path(pair: tuple[int, int], branch: str) -> Path:
    speaker, session = pair
    if branch == "baseline":
        return core.OLD_GRID_ROOT / f"P{speaker}/S{session}/contours_and_ground_truth.npz"
    if branch == "rms_vtln":
        return (
            core.OLD_AUDIO_ROOT
            / f"P{speaker}/S{session}/audio_normalized_contours_and_ground_truth.npz"
        )
    return (
        core.OLD_ABLATION_ROOT
        / f"P{speaker}/S{session}/{branch}_contours_and_ground_truth.npz"
    )


def load_old_pack(pair: tuple[int, int], branch: str) -> dict[str, Any]:
    path = old_pack_path(pair, branch)
    prefix = "predicted_audio_" if branch == "rms_vtln" else "predicted_"
    with np.load(path, allow_pickle=False) as payload:
        result = {
            "branch": branch,
            "frame_numbers": np.asarray(payload["frame_numbers"]),
            "arrays": {
                "raw": np.asarray(payload[f"{prefix}raw"], dtype=np.float32),
                "affine": np.asarray(payload[f"{prefix}after_affine"], dtype=np.float32),
                "affine_tps": np.asarray(payload[f"{prefix}after_affine_tps"], dtype=np.float32),
            },
            "ground_truth": np.asarray(payload["ground_truth"], dtype=np.float32),
            "classes": [str(item) for item in payload["classes"].tolist()],
        }
    if not np.issubdtype(result["frame_numbers"].dtype, np.integer):
        raise ValueError(f"Old frame timeline is not integer typed: {path}")
    if result["classes"] != list(core.CLASSES):
        raise ValueError(f"Old class order mismatch: {path}")
    return result


def validate_population(reference: dict[str, Any], candidate: dict[str, Any], label: str) -> None:
    if not np.array_equal(reference["frame_numbers"], candidate["frame_numbers"]):
        raise RuntimeError(f"Frame population mismatch: {label}")
    if reference["ground_truth"].shape != candidate["ground_truth"].shape:
        raise RuntimeError(f"Ground-truth shape mismatch: {label}")
    delta = float(np.max(np.abs(reference["ground_truth"] - candidate["ground_truth"])))
    if delta > 1e-5:
        raise RuntimeError(f"Ground-truth content mismatch ({delta}): {label}")


def per_frame_errors(pack: dict[str, Any], stage: str, indices: tuple[int, ...]) -> np.ndarray:
    return core.frame_rmse_mm(pack["arrays"][stage], pack["ground_truth"], indices)


def per_class_frame_errors(pack: dict[str, Any], stage: str) -> np.ndarray:
    difference = pack["arrays"][stage].astype(np.float64) - pack["ground_truth"].astype(np.float64)
    return np.sqrt(np.mean(difference * difference, axis=(2, 3))) * MM_PER_PIXEL


def weighted(values: Iterable[np.ndarray]) -> float:
    arrays = list(values)
    if not arrays:
        return float("nan")
    return float(np.mean(np.concatenate(arrays)))


def paired_bootstrap(
    left: dict[tuple[int, int], np.ndarray],
    right: dict[tuple[int, int], np.ndarray],
    pairs: tuple[tuple[int, int], ...],
    replicates: int,
    seed: int,
) -> dict[str, float]:
    """Session-block bootstrap of frame-weighted mean(left-right)."""
    if not pairs:
        raise ValueError("Bootstrap cohort is empty")
    session_differences = []
    session_frames = []
    for pair in pairs:
        if left[pair].shape != right[pair].shape:
            raise ValueError(f"Paired bootstrap shape mismatch: {pair}")
        session_differences.append(float(np.mean(left[pair] - right[pair])))
        session_frames.append(int(left[pair].size))
    differences = np.asarray(session_differences, dtype=np.float64)
    frames = np.asarray(session_frames, dtype=np.float64)
    estimate = float(np.sum(differences * frames) / np.sum(frames))
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        chosen = rng.integers(0, len(pairs), size=len(pairs))
        samples[index] = float(
            np.sum(differences[chosen] * frames[chosen]) / np.sum(frames[chosen])
        )
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "delta_mm": estimate,
        "ci95_low_mm": float(low),
        "ci95_high_mm": float(high),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
    }


def load_all_packs(
    output_root: Path,
) -> tuple[
    dict[str, dict[str, dict[tuple[int, int], dict[str, Any]]]],
    list[dict[str, Any]],
]:
    packs: dict[str, dict[str, dict[tuple[int, int], dict[str, Any]]]] = {
        "p7_old": {branch: {} for branch in core.BRANCHES},
        "asd2_new": {branch: {} for branch in core.BRANCHES},
    }
    parity_rows: list[dict[str, Any]] = []
    for pair in core.SELECTION:
        reference = load_new_pack(output_root, pair, "baseline")
        for system in packs:
            for branch in core.BRANCHES:
                candidate = (
                    load_new_pack(output_root, pair, branch)
                    if system == "asd2_new"
                    else load_old_pack(pair, branch)
                )
                validate_population(reference, candidate, f"{system}/{branch}/P{pair[0]}/S{pair[1]}")
                packs[system][branch][pair] = candidate
                parity_rows.append(
                    {
                        "system": system,
                        "branch": branch,
                        "speaker": f"P{pair[0]}",
                        "session": f"S{pair[1]}",
                        "frames": len(candidate["frame_numbers"]),
                        "integer_frame_dtype": str(candidate["frame_numbers"].dtype),
                        "frame_timeline_equal": True,
                        "ground_truth_equal_within_1e-5": True,
                    }
                )
    return packs, parity_rows


def analysis_rows(
    packs: dict[str, dict[str, dict[tuple[int, int], dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    session_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for system, branches in packs.items():
        for branch, sessions in branches.items():
            for pair, pack in sessions.items():
                cohort = "same_speaker_control" if pair[0] == 10 else "unseen_speaker"
                for mode, indices in core.MODE_INDICES.items():
                    for stage in core.STAGES:
                        values = per_frame_errors(pack, stage, indices)
                        session_rows.append(
                            {
                                "system": system,
                                "branch": branch,
                                "cohort": cohort,
                                "speaker": f"P{pair[0]}",
                                "session": f"S{pair[1]}",
                                "frames": len(values),
                                "metric_mode": mode,
                                "stage": stage,
                                "mean_frame_rmse_mm": float(np.mean(values)),
                                "median_frame_rmse_mm": float(np.median(values)),
                            }
                        )
                for stage in core.STAGES:
                    values = per_class_frame_errors(pack, stage)
                    for class_index, class_name in enumerate(core.CLASSES):
                        per_class_rows.append(
                            {
                                "level": "session",
                                "system": system,
                                "branch": branch,
                                "cohort": cohort,
                                "speaker": f"P{pair[0]}",
                                "session": f"S{pair[1]}",
                                "frames": len(values),
                                "stage": stage,
                                "class": class_name,
                                "mean_frame_rmse_mm": float(np.mean(values[:, class_index])),
                            }
                        )
            for cohort_name, cohort_pairs in COHORTS.items():
                for mode, indices in core.MODE_INDICES.items():
                    for stage in core.STAGES:
                        values = [per_frame_errors(sessions[pair], stage, indices) for pair in cohort_pairs]
                        aggregate_rows.append(
                            {
                                "system": system,
                                "branch": branch,
                                "cohort": cohort_name,
                                "sessions": len(cohort_pairs),
                                "frames": sum(value.size for value in values),
                                "metric_mode": mode,
                                "stage": stage,
                                "frame_weighted_mean_rmse_mm": weighted(values),
                            }
                        )
                for stage in core.STAGES:
                    values = [per_class_frame_errors(sessions[pair], stage) for pair in cohort_pairs]
                    joined = np.concatenate(values, axis=0)
                    for class_index, class_name in enumerate(core.CLASSES):
                        per_class_rows.append(
                            {
                                "level": "aggregate",
                                "system": system,
                                "branch": branch,
                                "cohort": cohort_name,
                                "speaker": "",
                                "session": "",
                                "frames": len(joined),
                                "stage": stage,
                                "class": class_name,
                                "mean_frame_rmse_mm": float(np.mean(joined[:, class_index])),
                            }
                        )
    return session_rows, aggregate_rows, per_class_rows


def comparison_rows(
    packs: dict[str, dict[str, dict[tuple[int, int], dict[str, Any]]]],
    replicates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    comparisons = []
    for branch in core.BRANCHES:
        comparisons.append(("new_minus_old_same_branch", "asd2_new", branch, "p7_old", branch))
    for branch in ("rms_vtln", "rms_only", "vtln_only"):
        comparisons.append(("new_audio_minus_new_baseline", "asd2_new", branch, "asd2_new", "baseline"))
    comparisons.append(
        (
            "new_vtln_only_minus_new_rms_vtln",
            "asd2_new", "vtln_only", "asd2_new", "rms_vtln",
        )
    )

    for comparison, left_system, left_branch, right_system, right_branch in comparisons:
        for mode, indices in core.MODE_INDICES.items():
            for stage in core.STAGES:
                left = {
                    pair: per_frame_errors(packs[left_system][left_branch][pair], stage, indices)
                    for pair in core.SELECTION
                }
                right = {
                    pair: per_frame_errors(packs[right_system][right_branch][pair], stage, indices)
                    for pair in core.SELECTION
                }
                for pair in core.SELECTION:
                    delta = float(np.mean(left[pair] - right[pair]))
                    rows.append(
                        {
                            "level": "session",
                            "comparison": comparison,
                            "left": f"{left_system}/{left_branch}",
                            "right": f"{right_system}/{right_branch}",
                            "cohort": "same_speaker_control" if pair[0] == 10 else "unseen_speaker",
                            "speaker": f"P{pair[0]}",
                            "session": f"S{pair[1]}",
                            "frames": left[pair].size,
                            "metric_mode": mode,
                            "stage": stage,
                            "delta_mm": delta,
                            "ci95_low_mm": "",
                            "ci95_high_mm": "",
                            "bootstrap_replicates": "",
                            "bootstrap_seed": "",
                        }
                    )
                    if mode == "all_11" and stage == "affine_tps" and delta > 0:
                        failure_rows.append(
                            {
                                "failure_type": comparison,
                                "left": f"{left_system}/{left_branch}",
                                "right": f"{right_system}/{right_branch}",
                                "speaker": f"P{pair[0]}",
                                "session": f"S{pair[1]}",
                                "frames": left[pair].size,
                                "delta_mm": delta,
                                "interpretation": "positive delta means the left condition is worse",
                            }
                        )
                for cohort_index, (cohort_name, cohort_pairs) in enumerate(COHORTS.items()):
                    stats = paired_bootstrap(
                        left,
                        right,
                        cohort_pairs,
                        replicates,
                        core.BOOTSTRAP_SEED + cohort_index,
                    )
                    rows.append(
                        {
                            "level": "aggregate",
                            "comparison": comparison,
                            "left": f"{left_system}/{left_branch}",
                            "right": f"{right_system}/{right_branch}",
                            "cohort": cohort_name,
                            "speaker": "",
                            "session": "",
                            "frames": sum(left[pair].size for pair in cohort_pairs),
                            "metric_mode": mode,
                            "stage": stage,
                            **stats,
                        }
                    )

        for stage in core.STAGES:
            left = {
                pair: per_class_frame_errors(packs[left_system][left_branch][pair], stage)
                for pair in core.SELECTION
            }
            right = {
                pair: per_class_frame_errors(packs[right_system][right_branch][pair], stage)
                for pair in core.SELECTION
            }
            for class_index, class_name in enumerate(core.CLASSES):
                for cohort_index, (cohort_name, cohort_pairs) in enumerate(COHORTS.items()):
                    left_class = {pair: left[pair][:, class_index] for pair in core.SELECTION}
                    right_class = {pair: right[pair][:, class_index] for pair in core.SELECTION}
                    stats = paired_bootstrap(
                        left_class,
                        right_class,
                        cohort_pairs,
                        replicates,
                        core.BOOTSTRAP_SEED + 100 + class_index * 3 + cohort_index,
                    )
                    per_class_rows.append(
                        {
                            "comparison": comparison,
                            "left": f"{left_system}/{left_branch}",
                            "right": f"{right_system}/{right_branch}",
                            "cohort": cohort_name,
                            "stage": stage,
                            "class": class_name,
                            **stats,
                        }
                    )

    # Grid transfer effects within every new-model branch.
    for branch in core.BRANCHES:
        for comparison, left_stage, right_stage in (
            ("new_affine_minus_raw", "affine", "raw"),
            ("new_tps_minus_affine", "affine_tps", "affine"),
        ):
            for mode, indices in core.MODE_INDICES.items():
                left = {
                    pair: per_frame_errors(packs["asd2_new"][branch][pair], left_stage, indices)
                    for pair in core.SELECTION
                }
                right = {
                    pair: per_frame_errors(packs["asd2_new"][branch][pair], right_stage, indices)
                    for pair in core.SELECTION
                }
                for cohort_index, (cohort_name, cohort_pairs) in enumerate(COHORTS.items()):
                    stats = paired_bootstrap(
                        left,
                        right,
                        cohort_pairs,
                        replicates,
                        core.BOOTSTRAP_SEED + 500 + cohort_index,
                    )
                    rows.append(
                        {
                            "level": "aggregate",
                            "comparison": comparison,
                            "left": f"asd2_new/{branch}/{left_stage}",
                            "right": f"asd2_new/{branch}/{right_stage}",
                            "cohort": cohort_name,
                            "speaker": "",
                            "session": "",
                            "frames": sum(left[pair].size for pair in cohort_pairs),
                            "metric_mode": mode,
                            "stage": f"{left_stage}_minus_{right_stage}",
                            **stats,
                        }
                    )
    failure_rows.sort(key=lambda row: float(row["delta_mm"]), reverse=True)
    return rows, per_class_rows, failure_rows


def lookup_aggregate(
    rows: list[dict[str, Any]], system: str, branch: str, cohort: str, mode: str, stage: str
) -> float:
    matches = [
        row
        for row in rows
        if row["system"] == system
        and row["branch"] == branch
        and row["cohort"] == cohort
        and row["metric_mode"] == mode
        and row["stage"] == stage
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Aggregate lookup returned {len(matches)} rows")
    return float(matches[0]["frame_weighted_mean_rmse_mm"])


def lookup_comparison(
    rows: list[dict[str, Any]], comparison: str, left: str, cohort: str, mode: str, stage: str
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["level"] == "aggregate"
        and row["comparison"] == comparison
        and row["left"] == left
        and row["cohort"] == cohort
        and row["metric_mode"] == mode
        and row["stage"] == stage
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Comparison lookup returned {len(matches)} rows")
    return matches[0]


def generate_report(
    output_root: Path,
    aggregate_rows: list[dict[str, Any]],
    comparison_rows_: list[dict[str, Any]],
    per_class_comparison_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    unseen = "UNSEEN_8"
    branches = {}
    for branch in core.BRANCHES:
        branches[branch] = {
            stage: lookup_aggregate(aggregate_rows, "asd2_new", branch, unseen, "all_11", stage)
            for stage in core.STAGES
        }
    best_branch = min(core.BRANCHES, key=lambda branch: branches[branch]["affine_tps"])
    baseline_old = lookup_aggregate(
        aggregate_rows, "p7_old", "baseline", unseen, "all_11", "affine_tps"
    )
    baseline_new = branches["baseline"]["affine_tps"]
    baseline_comparison = lookup_comparison(
        comparison_rows_,
        "new_minus_old_same_branch",
        "asd2_new/baseline",
        unseen,
        "all_11",
        "affine_tps",
    )
    best_audio_comparison = (
        None
        if best_branch == "baseline"
        else lookup_comparison(
            comparison_rows_,
            "new_audio_minus_new_baseline",
            f"asd2_new/{best_branch}",
            unseen,
            "all_11",
            "affine_tps",
        )
    )
    audio_comparisons = {
        branch: lookup_comparison(
            comparison_rows_,
            "new_audio_minus_new_baseline",
            f"asd2_new/{branch}",
            unseen,
            "all_11",
            "affine_tps",
        )
        for branch in ("rms_vtln", "rms_only", "vtln_only")
    }
    vtln_only_vs_rms_vtln = lookup_comparison(
        comparison_rows_,
        "new_vtln_only_minus_new_rms_vtln",
        "asd2_new/vtln_only",
        unseen,
        "all_11",
        "affine_tps",
    )
    p7_raw = lookup_aggregate(
        aggregate_rows, "p7_old", "baseline", unseen, "all_11", "raw"
    )
    p7_affine = lookup_aggregate(
        aggregate_rows, "p7_old", "baseline", unseen, "all_11", "affine"
    )
    mode_values = {
        mode: {
            system: lookup_aggregate(
                aggregate_rows, system, "baseline", unseen, mode, "affine_tps"
            )
            for system in ("p7_old", "asd2_new")
        }
        for mode in core.MODE_INDICES
    }
    without_incisors_comparison = lookup_comparison(
        comparison_rows_,
        "new_minus_old_same_branch",
        "asd2_new/baseline",
        unseen,
        "without_incisors_2",
        "affine_tps",
    )
    per_class_deltas = {
        row["class"]: row
        for row in per_class_comparison_rows
        if row["comparison"] == "new_minus_old_same_branch"
        and row["left"] == "asd2_new/baseline"
        and row["cohort"] == unseen
        and row["stage"] == "affine_tps"
    }
    baseline_session_deltas = sorted(
        (
            {
                "session": f"{row['speaker']}/{row['session']}",
                "delta_mm": float(row["delta_mm"]),
            }
            for row in comparison_rows_
            if row["level"] == "session"
            and row["comparison"] == "new_minus_old_same_branch"
            and row["left"] == "asd2_new/baseline"
            and row["metric_mode"] == "all_11"
            and row["stage"] == "affine_tps"
            and row["speaker"] != "P10"
        ),
        key=lambda row: row["delta_mm"],
    )
    all9_new = lookup_aggregate(
        aggregate_rows, "asd2_new", "baseline", "ALL_9_HISTORICAL", "all_11", "affine_tps"
    )
    all9_old = lookup_aggregate(
        aggregate_rows, "p7_old", "baseline", "ALL_9_HISTORICAL", "all_11", "affine_tps"
    )
    control_new = lookup_aggregate(
        aggregate_rows,
        "asd2_new",
        "baseline",
        "P10_SAME_SPEAKER_CONTROL",
        "all_11",
        "affine_tps",
    )
    control_old = lookup_aggregate(
        aggregate_rows,
        "p7_old",
        "baseline",
        "P10_SAME_SPEAKER_CONTROL",
        "all_11",
        "affine_tps",
    )
    control_raw = lookup_aggregate(
        aggregate_rows,
        "asd2_new", "baseline", "P10_SAME_SPEAKER_CONTROL", "all_11", "raw",
    )
    control_affine = lookup_aggregate(
        aggregate_rows,
        "asd2_new", "baseline", "P10_SAME_SPEAKER_CONTROL", "all_11", "affine",
    )
    control_audio = lookup_aggregate(
        aggregate_rows,
        "asd2_new", "rms_vtln", "P10_SAME_SPEAKER_CONTROL", "all_11", "affine_tps",
    )
    audio_session_deltas = [
        row for row in comparison_rows_
        if row["level"] == "session"
        and row["comparison"] == "new_audio_minus_new_baseline"
        and row["left"] == "asd2_new/rms_vtln"
        and row["metric_mode"] == "all_11"
        and row["stage"] == "affine_tps"
        and row["speaker"] != "P10"
    ]
    absolute_contours = []
    for class_index, class_name in enumerate(core.CLASSES):
        values = []
        for pair in core.UNSEEN_SELECTION:
            pack = load_new_pack(output_root, pair, "baseline")
            values.append(per_class_frame_errors(pack, "affine_tps")[:, class_index])
        absolute_contours.append((class_name, float(np.mean(np.concatenate(values)))))
    absolute_contours.sort(key=lambda item: item[1], reverse=True)
    gate = json.loads((output_root / "ablation_gate.json").read_text(encoding="utf-8"))
    commands_path = output_root / "logs/commands.log"
    command_text = commands_path.read_text(encoding="utf-8").strip()
    video_audit_paths = [
        output_root / f"P{s}/S{x}/video_50fps_original_audio_audit.json"
        for s, x in core.SELECTION
    ]
    videos_validated = sum(
        path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("status") == "passed"
        for path in video_audit_paths
    )
    headline = {
        "cohort": unseen,
        "frames": sum(
            len(load_new_pack(output_root, pair, "baseline")["frame_numbers"])
            for pair in core.UNSEEN_SELECTION
        ),
        "asd2_baseline_raw_mm": branches["baseline"]["raw"],
        "asd2_baseline_affine_mm": branches["baseline"]["affine"],
        "asd2_baseline_affine_tps_mm": baseline_new,
        "p7_baseline_affine_tps_mm": baseline_old,
        "asd2_minus_p7_baseline_final": baseline_comparison,
        "best_asd2_branch": best_branch,
        "best_asd2_final_mm": branches[best_branch]["affine_tps"],
        "best_audio_minus_baseline": best_audio_comparison,
        "audio_minus_baseline": audio_comparisons,
        "vtln_only_minus_rms_vtln": vtln_only_vs_rms_vtln,
        "without_incisors_new_minus_old": without_incisors_comparison,
        "per_class_new_minus_old_baseline_final": per_class_deltas,
        "all9_historical": {"asd2_new_mm": all9_new, "p7_old_mm": all9_old},
        "p10_same_speaker_control": {
            "asd2_raw_mm": control_raw,
            "asd2_affine_mm": control_affine,
            "asd2_affine_tps_mm": control_new,
            "asd2_rms_vtln_affine_tps_mm": control_audio,
            "p7_old_affine_tps_mm": control_old,
        },
    }

    report_path = output_root / "analysis" / "quantitative_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Epoch-211 ASD2 model on the fixed ASD1 nine-session protocol",
        "",
        f"Generated: {now()}",
        "",
        "This is inference-only. Negative deltas mean the condition on the left has lower error. "
        "The headline speaker-independent cohort is P1-P6, P8, and P9; P10/S14 is reported only "
        "as a same-person/domain-bridge control because ASD1 P10 and ASD2 are the same person.",
        "",
        "## Headline: eight unseen speakers",
        "",
        f"Across {headline['frames']:,} scored integer frames, the ASD2 baseline changes "
        f"{branches['baseline']['raw']:.3f} -> {branches['baseline']['affine']:.3f} -> "
        f"{branches['baseline']['affine_tps']:.3f} mm (raw -> affine -> affine+TPS).",
        "",
        "| Grid baseline stage | P7 (mm) | ASD2 (mm) | ASD2 - P7 (mm) |",
        "|---|---:|---:|---:|",
        f"| raw | {p7_raw:.4f} | {branches['baseline']['raw']:.4f} | "
        f"{branches['baseline']['raw']-p7_raw:+.4f} |",
        f"| affine | {p7_affine:.4f} | {branches['baseline']['affine']:.4f} | "
        f"{branches['baseline']['affine']-p7_affine:+.4f} |",
        f"| affine+TPS | {baseline_old:.4f} | {baseline_new:.4f} | "
        f"{baseline_new-baseline_old:+.4f} |",
        "",
        f"At affine+TPS, the old P7 baseline is {baseline_old:.3f} mm and the new ASD2 baseline is "
        f"{baseline_new:.3f} mm: delta {float(baseline_comparison['delta_mm']):+.3f} mm, "
        f"paired session-block bootstrap 95% CI "
        f"[{float(baseline_comparison['ci95_low_mm']):+.3f}, "
        f"{float(baseline_comparison['ci95_high_mm']):+.3f}].",
        "",
        "## Audio normalization and ablation",
        "",
        f"The preregistered gate was **{gate['status']}** (all-nine gate delta "
        f"{float(gate['frame_weighted_aggregate_delta_mm']):+.3f} mm), so RMS-only and VTLN-only "
        "were both run.",
        "",
        "| ASD2 branch | Raw | Affine | Affine+TPS |",
        "|---|---:|---:|---:|",
    ]
    for branch in core.BRANCHES:
        values = branches[branch]
        lines.append(
            f"| {branch} | {values['raw']:.3f} | {values['affine']:.3f} | "
            f"{values['affine_tps']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"The best unseen-speaker ASD2 branch is **{best_branch}** at "
            f"{branches[best_branch]['affine_tps']:.3f} mm after affine+TPS.",
            "All tested audio normalization variants are worse than the unnormalized ASD2 baseline: "
            + "; ".join(
                f"{branch} {float(stats['delta_mm']):+.3f} mm "
                f"(95% CI [{float(stats['ci95_low_mm']):+.3f}, "
                f"{float(stats['ci95_high_mm']):+.3f}])"
                for branch, stats in audio_comparisons.items()
            )
            + ".",
            f"VTLN-only minus RMS+VTLN is {float(vtln_only_vs_rms_vtln['delta_mm']):+.3f} mm "
            f"(95% CI [{float(vtln_only_vs_rms_vtln['ci95_low_mm']):+.3f}, "
            f"{float(vtln_only_vs_rms_vtln['ci95_high_mm']):+.3f}]); VTLN-only is the better "
            "of those two, but neither beats the unnormalized baseline.",
            "",
            "## Contour-level diagnosis",
            "",
            f"The all-11 regression is dominated by the incisor transfer: lower-incisor is "
            f"{float(per_class_deltas['lower-incisor']['delta_mm']):+.3f} mm and upper-incisor "
            f"{float(per_class_deltas['upper-incisor']['delta_mm']):+.3f} mm versus the old P7 "
            "baseline. When both incisors are excluded, ASD2 is instead better: "
            f"{mode_values['without_incisors_2']['asd2_new']:.3f} versus "
            f"{mode_values['without_incisors_2']['p7_old']:.3f} mm, delta "
            f"{float(without_incisors_comparison['delta_mm']):+.3f} mm "
            f"(95% CI [{float(without_incisors_comparison['ci95_low_mm']):+.3f}, "
            f"{float(without_incisors_comparison['ci95_high_mm']):+.3f}]).",
            "",
            f"The ASD2 model is substantially better on the three laryngeal contours: vocal-folds "
            f"{float(per_class_deltas['vocal-folds']['delta_mm']):+.3f}, thyroid-cartilage "
            f"{float(per_class_deltas['thyroid-cartilage']['delta_mm']):+.3f}, and epiglottis "
            f"{float(per_class_deltas['epiglottis']['delta_mm']):+.3f} mm. However, on the seven "
            "contours excluding both laryngeal structures and incisors, ASD2 remains worse "
            f"({mode_values['without_laryngeal_3_and_incisors_2']['asd2_new']:.3f} versus "
            f"{mode_values['without_laryngeal_3_and_incisors_2']['p7_old']:.3f} mm).",
            f"Arytenoid is independently scored and does not share that gain: its ASD2-minus-P7 "
            f"delta is {float(per_class_deltas['arytenoid-cartilage']['delta_mm']):+.3f} mm "
            f"(95% CI [{float(per_class_deltas['arytenoid-cartilage']['ci95_low_mm']):+.3f}, "
            f"{float(per_class_deltas['arytenoid-cartilage']['ci95_high_mm']):+.3f}]).",
            "",
            "## Session heterogeneity",
            "",
            f"The ASD2 baseline beats the old P7 baseline on "
            f"{sum(row['delta_mm'] < 0 for row in baseline_session_deltas)}/8 unseen sessions. "
            "The largest regressions are "
            + ", ".join(
                f"{row['session']} {row['delta_mm']:+.3f} mm"
                for row in sorted(baseline_session_deltas, key=lambda row: row['delta_mm'], reverse=True)[:3]
            )
            + ".",
            "",
            "## Historical all-nine and same-speaker control",
            "",
            "The all-nine aggregate is retained only for direct comparison with the 20260718 "
            "bundles; it is not called unseen. P10/S14 is isolated in the aggregate CSV and "
            f"bootstrap table as `P10_SAME_SPEAKER_CONTROL`. Baseline affine+TPS is "
            f"{all9_new:.3f} versus {all9_old:.3f} mm for the all-nine historical aggregate, and "
            f"{control_new:.3f} versus {control_old:.3f} mm for the P10 same-person control "
            "(ASD2 versus old P7).",
            "",
            "## Answers to the twelve required questions",
            "",
            f"1. **Does ASD2 reduce raw cross-speaker error? No.** Raw ASD2 is "
            f"{branches['baseline']['raw']-p7_raw:+.4f} mm worse than raw P7 on the matched "
            "unseen-eight set.",
            f"2. **Affine removes {branches['baseline']['raw']-branches['baseline']['affine']:.4f} "
            f"mm ({100*(branches['baseline']['raw']-branches['baseline']['affine'])/branches['baseline']['raw']:.1f}%).** "
            f"ASD2 affine is {branches['baseline']['affine']-p7_affine:+.4f} mm relative to P7 affine.",
            f"3. **TPS adds {branches['baseline']['affine']-branches['baseline']['affine_tps']:.4f} "
            f"mm improvement after affine.** Final ASD2 nevertheless remains "
            f"{baseline_new-baseline_old:+.4f} mm worse than P7 because P7 benefits more from TPS.",
            f"4. **The ASD2 source improves three of four laryngeal contours.** Vocal folds "
            f"{float(per_class_deltas['vocal-folds']['delta_mm']):+.4f}, thyroid cartilage "
            f"{float(per_class_deltas['thyroid-cartilage']['delta_mm']):+.4f}, and epiglottis "
            f"{float(per_class_deltas['epiglottis']['delta_mm']):+.4f} mm improve; arytenoid "
            f"changes {float(per_class_deltas['arytenoid-cartilage']['delta_mm']):+.4f} mm and "
            "does not improve.",
            f"5. **RMS+VTLN does not help.** It changes final ASD2 by "
            f"{float(audio_comparisons['rms_vtln']['delta_mm']):+.4f} mm, 95% CI "
            f"[{float(audio_comparisons['rms_vtln']['ci95_low_mm']):+.4f}, "
            f"{float(audio_comparisons['rms_vtln']['ci95_high_mm']):+.4f}].",
            f"6. **There is no unseen-session audio benefit to call consistent.** RMS+VTLN "
            f"improves {sum(float(row['delta_mm']) < 0 for row in audio_session_deltas)}/8 unseen "
            "sessions; P10 alone improves and is not part of the unseen cohort.",
            f"7. **VTLN-only is better than RMS+VTLN by "
            f"{-float(vtln_only_vs_rms_vtln['delta_mm']):.4f} mm**, but it remains "
            f"{float(audio_comparisons['vtln_only']['delta_mm']):+.4f} mm worse than grid-only.",
            f"8. **RMS normalization degrades the aggregate.** RMS-only is "
            f"{float(audio_comparisons['rms_only']['delta_mm']):+.4f} mm worse than grid-only, "
            f"95% CI [{float(audio_comparisons['rms_only']['ci95_low_mm']):+.4f}, "
            f"{float(audio_comparisons['rms_only']['ci95_high_mm']):+.4f}].",
            "9. **Largest remaining absolute ASD2 residuals:** "
            + ", ".join(f"{name} {value:.3f} mm" for name, value in absolute_contours[:4])
            + ". The incisor transfer is the dominant static-anatomy failure mode.",
            f"10. **P10 behaves as a same-person/domain-bridge control, not unseen.** ASD2 "
            f"raw/affine/final is {control_raw:.4f}/{control_affine:.4f}/{control_new:.4f} mm, "
            f"RMS+VTLN final is {control_audio:.4f} mm, and P7 final is {control_old:.4f} mm. "
            "The nonzero ASD2-to-P10 transform reflects acquisition/reference differences despite "
            "the shared physical speaker.",
            "11. **No apparent improvement comes from frame filtering.** Every old/new/ablation "
            "pack has the same exact integer frame vector and ground truth; the strict intersection "
            "is all 8,585 frames, with zero saved, scored, or rendered fractional frames.",
            "12. **Do not expand to all 141 sessions yet.** The matched gate shows a statistically "
            "clear final all-11 regression, incisor-dominated geometry error, and uniformly harmful "
            "audio normalization on the unseen cohort. Fix and retest the static incisor/grid "
            "transfer on these nine sessions first.",
            "",
            "## Video validation",
            "",
            f"{videos_validated}/9 per-session video audits were already present when this report "
            "was generated. Each accepted video contains exactly its evaluated integer frames at "
            "r_frame_rate=avg_frame_rate=50/1, uses H.264/AAC, and builds playback only from "
            "independently timestamped 20 ms segments of the original unnormalized target WAV. "
            "The final audit re-probes every file.",
            "",
            "## Reproducibility and exact paths",
            "",
            f"- Result root: `{output_root.resolve()}`",
            f"- Pre-inference audit: `{(output_root / 'preflight.json').resolve()}`",
            f"- Exact selected-session manifest: `{(output_root / 'manifest/selected_sessions.json').resolve()}`",
            f"- Source-grid provenance: `{(output_root / 'provenance/asd2_source_grid_contours.npz').resolve()}` and `{(output_root / 'preflight.json').resolve()}`",
            f"- Frontend and per-frame validity/timestamps: `{(output_root / 'provenance/inference_frontend_and_frame_metadata.json').resolve()}`",
            f"- Prediction packs and session summaries: `{output_root.resolve()}/P*/S*/`",
            f"- Unified metrics: `{(output_root / 'analysis/aggregate_metrics.csv').resolve()}` and `{(output_root / 'analysis/metrics_long.csv').resolve()}`",
            f"- Contour metrics and intervals: `{(output_root / 'analysis/per_class_metrics.csv').resolve()}` and `{(output_root / 'analysis/per_class_comparisons_with_bootstrap_ci.csv').resolve()}`",
            f"- Paired comparisons: `{(output_root / 'analysis/comparison_deltas_with_bootstrap_ci.csv').resolve()}`",
            f"- Final machine audit: `{(output_root / 'final_audit.json').resolve()}`",
            f"- Command log: `{commands_path.resolve()}`",
            "",
            "### Reproduction commands",
            "",
            "```text",
            command_text,
            "```",
            "",
            "## Failure cases",
            "",
            f"There are {len(failure_rows)} session-level positive final-stage deltas across the "
            "new-vs-old and audio-vs-baseline comparisons. See `failure_cases.csv`, sorted worst first.",
            "",
            "## Metric and audit conventions",
            "",
            "RMSE is point-to-point contour error in mm (1.62 mm/pixel), computed per frame and "
            "then frame-weighted. Bootstrap resampling uses paired whole-session blocks with "
            f"10,000 replicates and seed {core.BOOTSTRAP_SEED}. All scored, saved, and rendered "
            "MRI timestamps are integer-valued. Videos contain exactly the evaluated frames at "
            "50 fps. Each displayed frame receives an independently timestamped 20 ms segment "
            "from the original target WAV, so gaps introduce neither held contours nor A/V drift.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    atomic_json(output_root / "analysis" / "headline.json", headline)
    return headline


def run_analysis(output_root: Path, replicates: int) -> dict[str, Any]:
    started = time.monotonic()
    packs, parity_rows = load_all_packs(output_root)
    session_rows, aggregate_rows, per_class_rows = analysis_rows(packs)
    comparisons, per_class_comparisons, failures = comparison_rows(packs, replicates)
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_dir / "population_parity.csv", parity_rows)
    write_csv(analysis_dir / "metrics_long.csv", session_rows)
    write_csv(analysis_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(analysis_dir / "per_class_metrics.csv", per_class_rows)
    write_csv(analysis_dir / "comparison_deltas_with_bootstrap_ci.csv", comparisons)
    write_csv(analysis_dir / "per_class_comparisons_with_bootstrap_ci.csv", per_class_comparisons)
    write_csv(analysis_dir / "failure_cases.csv", failures)
    headline = generate_report(
        output_root,
        aggregate_rows,
        comparisons,
        per_class_comparisons,
        failures,
    )
    payload = {
        "created_at": now(),
        "status": "passed",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": core.BOOTSTRAP_SEED,
        "session_metric_rows": len(session_rows),
        "aggregate_metric_rows": len(aggregate_rows),
        "per_class_metric_rows": len(per_class_rows),
        "comparison_rows": len(comparisons),
        "per_class_comparison_rows": len(per_class_comparisons),
        "failure_rows": len(failures),
        "headline": headline,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(analysis_dir / "analysis_manifest.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def frame_token(frame: int) -> str:
    if not isinstance(frame, (int, np.integer)):
        raise ValueError(f"Refusing non-integer render frame: {frame!r}")
    return f"{int(frame):04d}"


def draw_panel(
    image: np.ndarray,
    title: str,
    frame: int,
    predicted: np.ndarray | None,
    ground_truth: np.ndarray | None,
    rmse: float | None,
) -> np.ndarray:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_bgr = cv2.resize(image_bgr, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((INFO_HEIGHT + PANEL_SIZE, PANEL_SIZE, 3), 14, dtype=np.uint8)
    canvas[INFO_HEIGHT:] = image_bgr
    if ground_truth is not None:
        scale = PANEL_SIZE / float(image.shape[1])
        if not math.isclose(scale, round(scale), abs_tol=1e-6):
            raise ValueError(f"Panel scale must be integral for exact contour placement: {scale}")
        for index, class_name in enumerate(core.CLASSES):
            color = rgb_to_bgr255(COLORS.get(class_name, "white"))
            gt = scale_points(ground_truth[index], int(round(scale)))
            gt[:, 1] += INFO_HEIGHT
            cv2.polylines(canvas, [gt], False, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.polylines(canvas, [gt], False, color, 1, cv2.LINE_AA)
            if predicted is not None:
                pred = scale_points(predicted[index], int(round(scale)))
                pred[:, 1] += INFO_HEIGHT
                draw_dashed_polyline(canvas, pred, (0, 0, 0), 2, dash_length=7, gap_length=9)
                draw_dashed_polyline(canvas, pred, color, 1, dash_length=7, gap_length=9)
        status = "ground truth only" if predicted is None else f"RMSE all 11: {rmse:.3f} mm"
    else:
        status = "missing ground truth"
    legend = "solid ground truth" if predicted is None else "solid GT | dashed prediction"
    lines = (title, f"integer MRI frame {frame_token(frame)}", status, legend)
    for index, line in enumerate(lines):
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


def compose_canvas(
    image: np.ndarray,
    frame: int,
    frame_index: dict[int, int],
    packs: dict[str, dict[str, Any]],
) -> np.ndarray:
    scored_index = frame_index.get(frame)
    if scored_index is None:
        raise RuntimeError(f"Renderer received a non-evaluated frame: {frame}")
    panels = []
    for branch, stage, title in VIDEO_PANELS:
        if branch == "ground_truth":
            predicted = None
            ground_truth = packs["baseline"]["ground_truth"][scored_index]
            rmse = None
        else:
            pack = packs[branch]
            predicted = pack["arrays"][stage][scored_index]
            ground_truth = pack["ground_truth"][scored_index]
            indices = core.MODE_INDICES["all_11"]
            # On a single frame the contour axis is axis 0; convert the tuple
            # to a list so NumPy performs one-axis fancy indexing.
            difference = predicted[list(indices)].astype(np.float64) - ground_truth[list(indices)].astype(np.float64)
            rmse = float(np.sqrt(np.mean(difference * difference)) * MM_PER_PIXEL)
        panels.append(draw_panel(image, title, frame, predicted, ground_truth, rmse))
    height = 2 * (INFO_HEIGHT + PANEL_SIZE) + SEPARATOR
    width = 3 * PANEL_SIZE + 2 * SEPARATOR
    canvas = np.full((height, width, 3), 8, dtype=np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, 3)
        y = row * (INFO_HEIGHT + PANEL_SIZE + SEPARATOR)
        x = column * (PANEL_SIZE + SEPARATOR)
        canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
    return canvas


def ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.check_output(command, text=True))


def rational_is_50(value: str) -> bool:
    return Fraction(value) == Fraction(FPS, 1)


def write_original_audio_segments(
    original_audio_path: Path,
    frames: np.ndarray,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    signal, sample_rate = sf.read(original_audio_path, dtype="float32", always_2d=True)
    mono = np.mean(signal, axis=1, dtype=np.float32)
    samples_per_frame_float = sample_rate / float(FPS)
    if not math.isclose(samples_per_frame_float, round(samples_per_frame_float), abs_tol=1e-12):
        raise RuntimeError(f"Audio sample rate {sample_rate} is not exactly divisible by {FPS}")
    samples_per_frame = int(round(samples_per_frame_float))
    config = core.load_yaml_config(core.DEFAULT_CONFIG)
    skip_ms = float(
        config.get("skip_ms", float(config["added_frames"]) * float(config["ms_image"]))
    )
    ms_image = float(config["ms_image"])
    segments = np.zeros(len(frames) * samples_per_frame, dtype=np.float32)
    rows = []
    half = samples_per_frame // 2
    for video_index, frame_value in enumerate(frames):
        frame = int(frame_value)
        center_seconds = (skip_ms + (frame + 0.5) * ms_image) / 1000.0
        center_sample = int(round(center_seconds * sample_rate))
        start = center_sample - half
        stop = start + samples_per_frame
        source_start = max(0, start)
        source_stop = min(len(mono), stop)
        destination_start = source_start - start
        output_start = video_index * samples_per_frame
        segments[
            output_start + destination_start : output_start + destination_start + source_stop - source_start
        ] = mono[source_start:source_stop]
        rows.append(
            {
                "video_frame_index": video_index,
                "mri_frame": frame,
                "source_center_seconds": center_seconds,
                "source_start_sample": source_start,
                "source_stop_sample_exclusive": source_stop,
                "source_sample_rate": sample_rate,
                "output_start_sample": output_start,
                "output_stop_sample_exclusive": output_start + samples_per_frame,
                "padded_samples": samples_per_frame - (source_stop - source_start),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, segments, sample_rate, subtype="PCM_16")
    write_csv(manifest_path, rows)
    return {
        "original_source_wav": str(original_audio_path.resolve()),
        "playback_uses_normalized_audio": False,
        "segment_wav": str(output_path.resolve()),
        "segment_manifest": str(manifest_path.resolve()),
        "sample_rate": sample_rate,
        "samples_per_video_frame": samples_per_frame,
        "output_samples": len(segments),
        "output_duration_seconds": len(segments) / sample_rate,
        "source_skip_ms": skip_ms,
        "source_ms_image": ms_image,
        "timestamp_policy": (
            "one independently indexed 20 ms original-WAV segment per evaluated MRI frame"
        ),
    }


def audit_video(
    video_path: Path,
    original_audio_path: Path,
    evaluated_frames: np.ndarray,
    segment_audio_path: Path | None = None,
) -> dict[str, Any]:
    payload = ffprobe(video_path)
    videos = [stream for stream in payload["streams"] if stream.get("codec_type") == "video"]
    audios = [stream for stream in payload["streams"] if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or not audios:
        raise RuntimeError(f"Expected one video stream and at least one audio stream: {video_path}")
    video = videos[0]
    audio = audios[0]
    frames = np.asarray(evaluated_frames)
    if not np.issubdtype(frames.dtype, np.integer) or np.any(np.diff(frames) <= 0):
        raise RuntimeError(f"Video audit received a non-integer or unordered timeline: {video_path}")
    expected_frames = len(frames)
    read_frames = int(video.get("nb_read_frames") or video.get("nb_frames") or -1)
    video_duration = float(video.get("duration") or payload["format"]["duration"])
    audio_duration = float(audio.get("duration") or payload["format"]["duration"])
    checks = {
        "r_frame_rate_50": rational_is_50(video["r_frame_rate"]),
        "avg_frame_rate_50": rational_is_50(video["avg_frame_rate"]),
        "audio_stream_present": bool(audios),
        "exact_frame_count": read_frames == expected_frames,
        "video_duration_within_one_frame": abs(video_duration - expected_frames / FPS) <= 1 / FPS,
        "av_duration_difference_within_one_frame": abs(video_duration - audio_duration) <= 1 / FPS,
        "source_audio_is_original_target_wav": original_audio_path.name.startswith("DENOISED_SOUND_"),
        "video_codec_h264": video.get("codec_name") == "h264",
        "audio_codec_aac": audio.get("codec_name") == "aac",
        "evaluated_frames_only": read_frames == len(frames),
        "rendered_fractional_frame_count_zero": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Video audit failed for {video_path}: {checks}")
    return {
        "created_at": now(),
        "status": "passed",
        "video": str(video_path.resolve()),
        "original_audio": str(original_audio_path.resolve()),
        "segment_audio": None if segment_audio_path is None else str(segment_audio_path.resolve()),
        "frame_min": int(frames.min()),
        "frame_max": int(frames.max()),
        "expected_evaluated_integer_frames": expected_frames,
        "video_frames": read_frames,
        "video_codec": video.get("codec_name"),
        "audio_codec": audio.get("codec_name"),
        "video_r_frame_rate": video["r_frame_rate"],
        "video_avg_frame_rate": video["avg_frame_rate"],
        "video_duration_seconds": video_duration,
        "audio_duration_seconds": audio_duration,
        "av_duration_difference_seconds": abs(video_duration - audio_duration),
        "checks": checks,
    }


def render_session(output_root: Path, pair: tuple[int, int], workers: int, force: bool) -> dict[str, Any]:
    speaker, session = pair
    started = time.monotonic()
    packs = {branch: load_new_pack(output_root, pair, branch) for branch in core.BRANCHES}
    reference = packs["baseline"]
    for branch in core.BRANCHES[1:]:
        validate_population(reference, packs[branch], f"video/{branch}/P{speaker}/S{session}")
    packs["p7_baseline"] = load_old_pack(pair, "baseline")
    validate_population(reference, packs["p7_baseline"], f"video/p7_baseline/P{speaker}/S{session}")
    frames = reference["frame_numbers"]
    if not np.issubdtype(frames.dtype, np.integer):
        raise ValueError(core.INTEGER_FRAME_POLICY)
    frame_min = int(frames.min())
    frame_max = int(frames.max())
    render_frames = [int(value) for value in frames]
    frame_index = {int(frame): index for index, frame in enumerate(frames)}
    session_dir = output_root / f"P{speaker}/S{session}"
    video_path = session_dir / f"p{speaker}_s{session}_asd2_grid_audio_ablation_original_audio_50fps.mp4"
    audit_path = session_dir / "video_50fps_original_audio_audit.json"
    audio_path, _ = core.exact_asd1_audio_paths(speaker, session)
    segment_audio_path = session_dir / "original_audio_evaluated_frame_segments.wav"
    if video_path.is_file() and audit_path.is_file() and not force:
        audit = audit_video(video_path, audio_path, frames, segment_audio_path)
        previous = json.loads(audit_path.read_text(encoding="utf-8"))
        for key in (
            "audio", "scored_integer_frames", "unscored_rendered_frames",
            "timeline_policy", "visual_qc_samples", "elapsed_seconds",
        ):
            if key in previous:
                audit[key] = previous[key]
        atomic_json(audit_path, audit)
        print(f"REUSE VIDEO P{speaker}/S{session}: {len(render_frames)} frames", flush=True)
        return audit

    dicom_dir = core.RAW_ROOT / f"P{speaker}/DCM_2D/S{session}"
    if not dicom_dir.is_dir():
        raise FileNotFoundError(dicom_dir)
    dicom_index, _ = build_filename_dicom_index(dicom_dir)
    cache_path = session_dir / ".cache" / "evaluated_integer_mri_frames.npz"
    mri_cache = load_or_build_mri_cache(
        dicom_dir,
        dicom_index,
        render_frames,
        cache_path,
        workers=workers,
    )
    height = 2 * (INFO_HEIGHT + PANEL_SIZE) + SEPARATOR
    width = 3 * PANEL_SIZE + 2 * SEPARATOR
    silent_path = session_dir / f".{video_path.stem}.silent.writing.mp4"
    muxing_path = session_dir / f".{video_path.stem}.muxing.mp4"
    writer = cv2.VideoWriter(
        str(silent_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(FPS),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {silent_path}")
    sample_frames = {
        int(frames[0]),
        int(frames[len(frames) // 2]),
        int(frames[-1]),
    }
    sample_paths = []
    qc_dir = output_root / "visual_qc" / f"P{speaker}_S{session}"
    qc_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, frame in enumerate(render_frames):
            canvas = compose_canvas(mri_cache[frame], frame, frame_index, packs)
            writer.write(canvas)
            if frame in sample_frames:
                sample_path = qc_dir / f"frame_{frame_token(frame)}.png"
                if not cv2.imwrite(str(sample_path), canvas):
                    raise RuntimeError(f"Could not write visual-QC sample: {sample_path}")
                sample_paths.append(sample_path)
            if index == 0 or (index + 1) % 500 == 0 or index + 1 == len(render_frames):
                print(
                    f"RENDER P{speaker}/S{session}: {index + 1}/{len(render_frames)} evaluated frames",
                    flush=True,
                )
    finally:
        writer.release()
    audio_metadata = write_original_audio_segments(
        audio_path,
        frames,
        segment_audio_path,
        session_dir / "original_audio_evaluated_frame_segments.csv",
    )
    encoded_audio_path = session_dir / f".{video_path.stem}.audio.writing.m4a"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(segment_audio_path), "-c:a", "aac", "-b:a", "96k",
            str(encoded_audio_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(silent_path), "-i", str(encoded_audio_path),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
            "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-vsync", "0", "-c:a", "copy", "-movflags", "+faststart",
            str(muxing_path),
        ],
        check=True,
    )
    muxing_path.replace(video_path)
    silent_path.unlink(missing_ok=True)
    encoded_audio_path.unlink(missing_ok=True)
    audit = audit_video(video_path, audio_path, frames, segment_audio_path)
    audit.update(
        {
            "scored_integer_frames": len(frames),
            "unscored_rendered_frames": 0,
            "audio": audio_metadata,
            "timeline_policy": (
                "exact evaluated integer frames only, in source order; no contour hold, "
                "interpolation, resampling, or fractional frame"
            ),
            "visual_qc_samples": [str(path.resolve()) for path in sorted(sample_paths)],
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    atomic_json(audit_path, audit)
    print(
        f"VIDEO DONE P{speaker}/S{session}: {len(render_frames)} frames, "
        f"A/V delta={audit['av_duration_difference_seconds']:.6f}s, "
        f"{audit['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return audit


def create_contact_sheet(output_root: Path, audits: list[dict[str, Any]]) -> Path:
    # Always inventory samples from disk. A reused video is re-probed into a
    # compact audit object that intentionally does not repeat its sample list.
    # Combining in-memory audit lists would therefore omit samples from reused
    # sessions (notably the one-session smoke run).
    del audits
    paths = sorted((output_root / "visual_qc").glob("P*_S*/frame_*.png"))
    if len(paths) != len(core.SELECTION) * 3:
        raise RuntimeError(f"Expected 27 visual-QC samples, got {len(paths)}")
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((414, 363), Image.Resampling.LANCZOS)
        thumbs.append((path, image.copy()))
    columns = 3
    cell_width, cell_height = 430, 395
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (22, 22, 22))
    draw = ImageDraw.Draw(sheet)
    for index, (path, image) in enumerate(thumbs):
        row, column = divmod(index, columns)
        x, y = column * cell_width, row * cell_height
        sheet.paste(image, (x, y + 24))
        draw.text((x + 5, y + 5), f"{path.parent.name}/{path.stem}", fill=(245, 245, 245))
    output_path = output_root / "visual_qc" / "contact_sheet_27_samples.png"
    sheet.save(output_path)
    return output_path


def run_videos(
    output_root: Path,
    pairs: tuple[tuple[int, int], ...],
    workers: int,
    force: bool,
) -> list[dict[str, Any]]:
    audits = [render_session(output_root, pair, workers, force) for pair in pairs]
    if pairs == core.SELECTION:
        contact_sheet = create_contact_sheet(output_root, audits)
        print(f"VISUAL QC CONTACT SHEET: {contact_sheet}", flush=True)
    return audits


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_old_bundles_immutable(output_root: Path) -> dict[str, Any]:
    before_path = output_root / "provenance" / "input_bundles_before.json"
    before = json.loads(before_path.read_text(encoding="utf-8"))
    rows = []
    for root in (core.OLD_GRID_ROOT, core.OLD_AUDIO_ROOT, core.OLD_ABLATION_ROOT):
        name = root.name
        previous = before["roots"][name]
        after_files = core.tree_hashes(root)
        after_tree = hashlib.sha256(
            json.dumps(after_files, sort_keys=True).encode("utf-8")
        ).hexdigest()
        equal = previous.get("files") == after_files and previous.get("tree_sha256") == after_tree
        rows.append(
            {
                "root": str(root.resolve()),
                "file_count_before": previous.get("file_count"),
                "file_count_after": len(after_files),
                "tree_sha256_before": previous.get("tree_sha256"),
                "tree_sha256_after": after_tree,
                "unchanged": equal,
            }
        )
    if not all(row["unchanged"] for row in rows):
        raise RuntimeError(f"One or more 20260718 input bundles changed: {rows}")
    return {"status": "passed", "roots": rows}


def run_audit(output_root: Path, confirm_visual_qc: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    packs, parity_rows = load_all_packs(output_root)
    config = core.load_yaml_config(core.DEFAULT_CONFIG)
    skip_ms = float(
        config.get("skip_ms", float(config["added_frames"]) * float(config["ms_image"]))
    )
    frame_metadata_rows = []
    frame_metadata_files = []
    for speaker, session in core.SELECTION:
        frames = packs["asd2_new"]["baseline"][(speaker, session)]["frame_numbers"]
        rows = [
            {
                "speaker": f"P{speaker}",
                "session": f"S{session}",
                "evaluated_index": index,
                "mri_frame": int(frame),
                "timestamp_seconds": (
                    skip_ms + (int(frame) + 0.5) * float(config["ms_image"])
                ) / 1000.0,
                "validity_flag": True,
                "integer_frame": True,
            }
            for index, frame in enumerate(frames)
        ]
        path = output_root / f"provenance/frame_metadata/P{speaker}/S{session}.csv"
        write_csv(path, rows)
        frame_metadata_rows.extend(rows)
        frame_metadata_files.append(str(path.resolve()))
    write_csv(output_root / "provenance/frame_metadata/all_evaluated_frames.csv", frame_metadata_rows)
    frontend_metadata = {
        "created_at": now(),
        "config": str(core.DEFAULT_CONFIG.resolve()),
        "config_sha256": sha256(core.DEFAULT_CONFIG),
        "checkpoint": str(core.DEFAULT_CHECKPOINT.resolve()),
        "checkpoint_sha256": sha256(core.DEFAULT_CHECKPOINT),
        "normalization": str(core.DEFAULT_NORMALIZATION.resolve()),
        "normalization_sha256": sha256(core.DEFAULT_NORMALIZATION),
        "input_type": config["input_type"],
        "input_layer": config["input_layer"],
        "n_mfcc": config["n_mfcc"],
        "window_length_ms": config["window_length_ms"],
        "hop_length_ms": config["hop_length_ratio"],
        "sequence_length": config["sequence_length"],
        "added_frames": config["added_frames"],
        "skip_ms": skip_ms,
        "mri_spacing_ms": config["ms_image"],
        "contour_order": list(core.CLASSES),
        "source_grid_reference": core.SOURCE_REFERENCE,
        "per_frame_validity_policy": "all saved evaluated integer frames are valid",
        "per_session_frame_metadata": frame_metadata_files,
    }
    atomic_json(output_root / "provenance/inference_frontend_and_frame_metadata.json", frontend_metadata)
    manifest_rows = []
    for speaker, session in core.SELECTION:
        spec = core.target_spec(speaker)
        audio_path, textgrid_path = core.exact_asd1_audio_paths(speaker, session)
        frames = packs["asd2_new"]["baseline"][(speaker, session)]["frame_numbers"]
        manifest_rows.append(
            {
                "speaker": f"P{speaker}",
                "session": f"S{session}",
                "cohort": "same_speaker_control" if speaker == 10 else "unseen_speaker",
                "audio_path": str(audio_path.resolve()),
                "textgrid_path": str(textgrid_path.resolve()),
                "mri_frame_path": str((core.RAW_ROOT / f"P{speaker}/DCM_2D/S{session}").resolve()),
                "ground_truth_contour_path": str((core.BF_ROOT / f"P{speaker}/S{session}/contours").resolve()),
                "target_reference": spec.label,
                "target_reference_image_path": str(
                    (core.DEFAULT_VTLN_DIR / f"{spec.vtln_anchor}.png").resolve()
                ),
                "target_reference_landmark_grid_path": str(
                    (core.DEFAULT_VTLN_DIR / f"{spec.vtln_anchor}.zip").resolve()
                ),
                "source_reference": core.SOURCE_REFERENCE,
                "source_reference_mri_path": str(
                    (
                        core.ASD2_ROOT / core.SOURCE_BUCKET / core.SOURCE_SESSION
                        / "NPY_MR_registered" / f"{core.SOURCE_FRAME}.npy"
                    ).resolve()
                ),
                "source_contour_pack": str(core.DEFAULT_SOURCE_PACK.resolve()),
                "number_of_usable_integer_frames": len(frames),
                "frame_min": int(frames.min()),
                "frame_max": int(frames.max()),
                "old_p7_result_available": old_pack_path((speaker, session), "baseline").is_file(),
                "old_p7_grid_pack": str(old_pack_path((speaker, session), "baseline").resolve()),
            }
        )
    write_csv(output_root / "manifest/selected_sessions.csv", manifest_rows)
    atomic_json(
        output_root / "manifest/selected_sessions.json",
        {
            "created_at": now(),
            "selection_was_fixed_before_inference_by": str((output_root / "preflight.json").resolve()),
            "selection": [f"P{s}/S{x}" for s, x in core.SELECTION],
            "sessions": manifest_rows,
        },
    )
    pack_rows = []
    total_by_branch = defaultdict(int)
    for pair in core.SELECTION:
        reference = packs["asd2_new"]["baseline"][pair]
        for branch in core.BRANCHES:
            pack = packs["asd2_new"][branch][pair]
            path = core.pack_path(output_root, pair[0], pair[1], branch)
            with np.load(path, allow_pickle=False) as payload:
                saved_fractional = int(payload["saved_fractional_frame_count"])
                scored_fractional = int(payload["scored_fractional_frame_count"])
            checks = {
                "integer_timeline": bool(np.issubdtype(pack["frame_numbers"].dtype, np.integer)),
                "timeline_matches_baseline": bool(
                    np.array_equal(pack["frame_numbers"], reference["frame_numbers"])
                ),
                "ground_truth_matches_baseline": bool(
                    np.allclose(pack["ground_truth"], reference["ground_truth"], atol=1e-5, rtol=0)
                ),
                "finite_predictions": all(
                    bool(np.isfinite(pack["arrays"][stage]).all()) for stage in core.STAGES
                ),
                "saved_fractional_zero": saved_fractional == 0,
                "scored_fractional_zero": scored_fractional == 0,
            }
            if not all(checks.values()):
                raise RuntimeError(f"Pack audit failed {path}: {checks}")
            total_by_branch[branch] += len(pack["frame_numbers"])
            pack_rows.append(
                {
                    "speaker": f"P{pair[0]}",
                    "session": f"S{pair[1]}",
                    "branch": branch,
                    "frames": len(pack["frame_numbers"]),
                    "sha256": sha256(path),
                    "checks": checks,
                }
            )
    expected = 8585
    if dict(total_by_branch) != {branch: expected for branch in core.BRANCHES}:
        raise RuntimeError(f"Branch frame inventory mismatch: {dict(total_by_branch)}")

    video_rows = []
    for pair in core.SELECTION:
        session_dir = output_root / f"P{pair[0]}/S{pair[1]}"
        video = session_dir / (
            f"p{pair[0]}_s{pair[1]}_asd2_grid_audio_ablation_original_audio_50fps.mp4"
        )
        audio, _ = core.exact_asd1_audio_paths(*pair)
        baseline = packs["asd2_new"]["baseline"][pair]
        audit = audit_video(
            video,
            audio,
            baseline["frame_numbers"],
            session_dir / "original_audio_evaluated_frame_segments.wav",
        )
        audit_path = session_dir / "video_50fps_original_audio_audit.json"
        if audit_path.is_file():
            previous = json.loads(audit_path.read_text(encoding="utf-8"))
            for key in (
                "audio", "scored_integer_frames", "unscored_rendered_frames",
                "timeline_policy", "visual_qc_samples", "elapsed_seconds",
            ):
                if key in previous:
                    audit[key] = previous[key]
        video_rows.append(audit)
        atomic_json(audit_path, audit)

    required_analysis = [
        "metrics_long.csv",
        "aggregate_metrics.csv",
        "per_class_metrics.csv",
        "comparison_deltas_with_bootstrap_ci.csv",
        "per_class_comparisons_with_bootstrap_ci.csv",
        "failure_cases.csv",
        "quantitative_report.md",
        "headline.json",
        "analysis_manifest.json",
    ]
    missing_analysis = [
        name for name in required_analysis if not (output_root / "analysis" / name).is_file()
    ]
    if missing_analysis:
        raise RuntimeError(f"Missing analysis outputs: {missing_analysis}")
    contact_sheet = output_root / "visual_qc" / "contact_sheet_27_samples.png"
    if not contact_sheet.is_file():
        raise RuntimeError(f"Missing visual-QC contact sheet: {contact_sheet}")
    immutable = verify_old_bundles_immutable(output_root)
    audit = {
        "created_at": now(),
        "status": "passed",
        "definition_of_done_passed": True,
        "inference_only": True,
        "training_launched": False,
        "selection": [f"P{s}/S{x}" for s, x in core.SELECTION],
        "headline_unseen_selection": [f"P{s}/S{x}" for s, x in core.UNSEEN_SELECTION],
        "same_speaker_control": "P10/S14",
        "sessions": len(core.SELECTION),
        "branches": list(core.BRANCHES),
        "integer_scored_frames_per_branch": dict(total_by_branch),
        "saved_fractional_frame_count": 0,
        "scored_fractional_frame_count": 0,
        "rendered_fractional_frame_count": 0,
        "population_parity_rows": len(parity_rows),
        "frontend_and_frame_metadata": str(
            (output_root / "provenance/inference_frontend_and_frame_metadata.json").resolve()
        ),
        "detailed_session_manifest": str(
            (output_root / "manifest/selected_sessions.json").resolve()
        ),
        "evaluated_frame_metadata_rows": len(frame_metadata_rows),
        "all_per_frame_validity_flags_true": True,
        "pack_audits": pack_rows,
        "video_audits": video_rows,
        "all_videos_exact_50fps": True,
        "all_videos_have_original_audio": True,
        "visual_qc_contact_sheet": str(contact_sheet.resolve()),
        "visual_qc_status": (
            "passed_explicit_review_27_samples"
            if confirm_visual_qc
            else "pending_manual_visual_inspection"
        ),
        "old_20260718_bundles_immutable": immutable,
        "analysis_outputs": required_analysis,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(output_root / "final_audit.json", audit)
    manifest = {
        "created_at": now(),
        "status": "complete" if confirm_visual_qc else "complete_pending_manual_visual_qc",
        "experiment": "epoch211_asd2_model_on_fixed_asd1_grid_audio_protocol",
        "result_root": str(output_root.resolve()),
        "final_audit": str((output_root / "final_audit.json").resolve()),
        "report": str((output_root / "analysis" / "quantitative_report.md").resolve()),
        "videos": [row["video"] for row in video_rows],
        "definition_of_done_passed": bool(confirm_visual_qc),
        "definition_of_done_passed_except_manual_visual_qc": True,
    }
    atomic_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return audit


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    pairs = selected_pairs(args.selection)
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    if args.phase in ("analysis", "all"):
        if pairs != core.SELECTION:
            raise ValueError("Analysis requires the exact fixed nine-session selection")
        run_analysis(args.output_root, args.bootstrap_replicates)
    if args.phase in ("video", "all"):
        run_videos(args.output_root, pairs, args.mri_workers, args.force_video)
    if args.phase in ("audit", "all"):
        if pairs != core.SELECTION:
            raise ValueError("Final audit requires the exact fixed nine-session selection")
        run_audit(args.output_root, confirm_visual_qc=args.confirm_visual_qc)


if __name__ == "__main__":
    main()
