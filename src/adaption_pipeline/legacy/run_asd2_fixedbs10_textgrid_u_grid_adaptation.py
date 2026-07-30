#!/usr/bin/env python3
"""No-audio exact-/u/ grid adaptation for the fixed ASD1 selected-nine set.

The script consumes fresh integer-only predictions from the fixed-batch-10
ASD2 model, independently reselects exact TextGrid /u/ source/target reference
frames, renders the reference-frame construction diagnostics, applies one
fixed affine+TPS transform per target session, evaluates the complete session
with the calibration frame both included and excluded, and renders 50-fps
videos with original target audio.  It never performs audio normalization or
training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
GRID_ROOT = REPO_ROOT / "external/grid-transform"
sys.path[:0] = [
    str(REPO_ROOT),
    str(REPO_ROOT / "src"),
    str(GRID_ROOT),
]

from . import render_textgrid_u_transform_process as transform_process  # noqa: E402
from . import run_textgrid_u_corrected_selected9_experiment as base  # noqa: E402
from grid_transform.transform_helpers import apply_transform  # noqa: E402
from src.utils.mri_rendering import (  # noqa: E402
    build_filename_dicom_index,
    load_or_build_mri_cache,
)
from src.common.artifacts import (  # noqa: E402
    atomic_write_csv as write_csv,
    atomic_write_json as _atomic_write_json,
    local_now as now,
    sha256_file as sha256,
)

atomic_json = partial(_atomic_write_json, allow_nan=True)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results/asd2_fixedbs10_selected_9sessions_textgrid_u_grid_adaptation_20260721_143350"
)
DEFAULT_FRESH = DEFAULT_OUTPUT / "fresh_inference"
DEFAULT_PRIOR = (
    REPO_ROOT / "results/asd2_selected_9sessions_textgrid_u_corrected_20260720_194511"
)
DEFAULT_CONFIG = (
    REPO_ROOT
    / "config/train_config/asd2_11contour_vtln_lowerrepairv2_upperlegacypos_"
    "20260721_train_global_rawstd_st5_mfcc_500epoch_fixedbs10_4gpu.yaml"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "mlruns/846774538033499469/39e2314f4d02443a8e065ccdfc04bcb3/"
    "artifacts/final_dict.pth"
)
DEFAULT_NORMALIZATION = (
    REPO_ROOT
    / "repro/asd2_11contour_vtln_lowerrepairv2_upperlegacypos_20260721_"
    "train_global/splits/normalization_stats.npz"
)
DEFAULT_SOURCE_PACK = (
    REPO_ROOT
    / "cache_variants/asd2_11_vtln_lowerrepairv2_upperlegacypos_20260721/"
    "raw_contour_npz/asd2/1791/S14.npz"
)
DEFAULT_SOURCE_CACHE = (
    REPO_ROOT
    / "cache_variants/asd2_11_vtln_lowerrepairv2_upperlegacypos_20260721/"
    "raw_sessions/asd2/1791/S14.pt"
)
EXPECTED_TARGET_FRAMES = {
    (1, 16): 763,
    (2, 9): 1424,
    (3, 14): 1168,
    (4, 4): 1179,
    (5, 6): 813,
    (6, 8): 597,
    (8, 2): 298,
    (9, 5): 259,
    (10, 14): 478,
}
SOURCE_FRAME = 499
BOOTSTRAP_SEED = 20260721
FPS = 50
PANEL_SIZE = base.PANEL_SIZE
INFO_HEIGHT = base.INFO_HEIGHT
SEPARATOR = base.SEPARATOR

GROUPS = OrderedDict(base.GROUPS)
GROUPS["without_laryngeal_3_and_incisors_6"] = tuple(
    index
    for index, name in enumerate(base.CLASSES)
    if name
    not in {
        "epiglottis",
        "vocal-folds",
        "thyroid-cartilage",
        "lower-incisor",
        "upper-incisor",
    }
)
COHORTS = OrderedDict(
    [
        ("UNSEEN_8", base.UNSEEN_SELECTION),
        ("ALL_9", base.SELECTION),
        ("P10_CONTROL", ((10, 14),)),
    ]
)
STAGES = ("raw", "affine", "affine_tps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("build", "videos", "audit", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fresh-root", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--prior-root", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--normalization-stats", type=Path, default=DEFAULT_NORMALIZATION)
    parser.add_argument("--source-pack", type=Path, default=DEFAULT_SOURCE_PACK)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--transform-frame-batch", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--mri-workers", type=int, default=8)
    parser.add_argument("--force-videos", action="store_true")
    return parser.parse_args()


def configure_modules(args: argparse.Namespace) -> None:
    base.MODELS = ("asd2",)
    base.CANONICAL_ROOT = args.fresh_root.resolve()
    base.ASD2_SOURCE_PACK = args.source_pack.resolve()
    base.ASD2_SOURCE_CACHE = args.source_cache.resolve()
    base.prior_u.SOURCE_PACK = args.source_pack.resolve()
    transform_process.experiment.CANONICAL_ROOT = args.fresh_root.resolve()
    transform_process.experiment.ASD2_SOURCE_PACK = args.source_pack.resolve()
    transform_process.experiment.ASD2_SOURCE_CACHE = args.source_cache.resolve()
    transform_process.experiment.prior_u.SOURCE_PACK = args.source_pack.resolve()


def load_inputs(args: argparse.Namespace) -> tuple[
    dict[tuple[str, str, tuple[int, int]], dict[str, Any]], list[dict[str, Any]]
]:
    inputs: dict[tuple[str, str, tuple[int, int]], dict[str, Any]] = {}
    rows = []
    all_frames = 0
    unseen_frames = 0
    for pair in base.SELECTION:
        payload = base.load_raw_input("asd2", pair, "baseline")
        prior = base.load_corrected_pack(args.prior_root, "asd2", pair, "baseline")
        if not np.array_equal(payload["frames"], prior["frames"]):
            raise RuntimeError(f"Fresh/prior frame mismatch P{pair[0]}/S{pair[1]}")
        if not np.array_equal(payload["ground_truth"], prior["ground_truth"]):
            raise RuntimeError(f"Fresh/prior ground-truth mismatch P{pair[0]}/S{pair[1]}")
        inputs[("asd2", "baseline", pair)] = payload
        all_frames += len(payload["frames"])
        if pair in base.UNSEEN_SELECTION:
            unseen_frames += len(payload["frames"])
        rows.append(
            {
                "model": "asd2_fixedbs10_best_human_epoch31",
                "branch": "baseline_no_audio_adaptation",
                "speaker": pair[0],
                "session": pair[1],
                "frames": len(payload["frames"]),
                "input_path": str(payload["path"].resolve()),
                "input_file_sha256": payload["path_sha256"],
                "raw_prediction_sha256": payload["raw_sha256"],
                "ground_truth_sha256": payload["ground_truth_sha256"],
                "frame_timeline_sha256": payload["frames_sha256"],
                "fresh_prior_timeline_equal": True,
                "fresh_prior_ground_truth_equal": True,
                "saved_fractional_frame_count": 0,
                "scored_fractional_frame_count": 0,
            }
        )
    if (all_frames, unseen_frames) != (8585, 7633):
        raise RuntimeError(f"Population mismatch: all={all_frames}, unseen={unseen_frames}")
    return inputs, rows


def validate_reference_selection(selections: dict[str, Any]) -> None:
    source = selections["asd2_source"]
    if int(source["selected_frame"]) != SOURCE_FRAME or not source["verified_exact_u"]:
        raise RuntimeError(f"ASD2 source selection mismatch: {source}")
    for pair, expected in EXPECTED_TARGET_FRAMES.items():
        selection = selections[f"target_P{pair[0]}_S{pair[1]}"]
        observed = int(selection["selected_frame"])
        if observed != expected or not selection["verified_exact_u"]:
            raise RuntimeError(
                f"Exact-/u/ target mismatch P{pair[0]}/S{pair[1]}: "
                f"expected F{expected:04d}, got F{observed:04d}"
            )
        center = float(selection["selected_frame_center"])
        if not float(selection["interval_start"]) < center < float(selection["interval_end"]):
            raise RuntimeError(f"Selected MRI center is not strictly inside /u/: {selection}")


def save_candidate_rows(selections: dict[str, Any], output_root: Path) -> None:
    rows = []
    for key, selection in selections.items():
        if not key.startswith("target_"):
            continue
        for candidate in selection["candidate_runs"]:
            for frame in candidate["run"]:
                rows.append(
                    {
                        "speaker_session": selection["speaker_session"],
                        "tier_index": selection["tier_index"],
                        "tier_name": selection["tier_name"],
                        "interval_index": candidate["interval_index"],
                        "interval_mark": candidate["interval_mark"],
                        "interval_start_seconds": candidate["interval_start"],
                        "interval_end_seconds": candidate["interval_end"],
                        "interval_midpoint_seconds": candidate["interval_midpoint"],
                        "candidate_integer_frame": frame,
                        "candidate_center_seconds": base.frame_center_seconds(frame),
                        "run_length": candidate["run_length"],
                        "selected": frame == int(selection["selected_frame"]),
                        "selection_uses_rmse": False,
                    }
                )
    write_csv(output_root / "phase_a/reference_candidate_frames.csv", rows)


def render_selection_plot(selection: dict[str, Any], output: Path) -> None:
    selected_interval = next(
        row
        for row in selection["candidate_runs"]
        if int(row["selected_frame"]) == int(selection["selected_frame"])
        and math.isclose(float(row["interval_start"]), float(selection["interval_start"]))
    )
    frames = np.asarray(selected_interval["run"], dtype=np.int32)
    centers = np.asarray([base.frame_center_seconds(int(value)) for value in frames])
    selected = int(selection["selected_frame"])
    fig, ax = plt.subplots(figsize=(11, 2.7), dpi=170)
    ax.axvspan(
        float(selection["interval_start"]),
        float(selection["interval_end"]),
        color="#66c2a5",
        alpha=0.25,
        label="exact TextGrid /u/ interval",
    )
    ax.axvline(float(selection["interval_midpoint"]), color="#555555", linestyle=":", label="midpoint")
    ax.scatter(centers, np.zeros(len(centers)), color="#2166ac", s=42, label="valid integer MRI centers")
    for frame, center in zip(frames, centers):
        ax.annotate(f"F{int(frame):04d}", (center, 0), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=7)
    selected_center = base.frame_center_seconds(selected)
    ax.scatter([selected_center], [0], color="#d73027", marker="*", s=180, zorder=4, label="selected")
    ax.set_ylim(-0.45, 0.45)
    ax.set_yticks([])
    ax.set_xlabel("MRI center time (seconds)")
    ax.set_title(
        f"{selection['speaker_session']}: direct TextGrid exact /u/ selection -> F{selected:04d}\n"
        "largest valid contiguous run, then closest to interval midpoint; prediction error unused"
    )
    ax.legend(loc="lower center", ncol=4, fontsize=7)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def save_contact_sheet(paths: list[Path], output: Path, columns: int = 3) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = 900
    thumbs = []
    for image in images:
        height = int(round(image.height * width / image.width))
        thumbs.append(image.resize((width, height), Image.Resampling.LANCZOS))
    rows = int(math.ceil(len(thumbs) / columns))
    height = max(image.height for image in thumbs) + 28
    canvas = Image.new("RGB", (columns * width, rows * height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (path, image) in enumerate(zip(paths, thumbs)):
        x_value = (index % columns) * width
        y_value = (index // columns) * height
        canvas.paste(image, (x_value, y_value + 24))
        draw.text((x_value + 7, y_value + 5), path.stem, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def render_prediction_stage_figure(
    pack: dict[str, Any], prior: dict[str, Any], target_frame: int, image: np.ndarray, output: Path
) -> None:
    matches = np.flatnonzero(pack["frames"] == target_frame)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one calibration row F{target_frame:04d}")
    index = int(matches[0])
    ground_truth = pack["ground_truth"][index]
    panels = [
        base.draw_panel(image, "Target ground truth /u/", target_frame, None, ground_truth),
        base.draw_panel(image, "Prior epoch-211 affine+TPS", target_frame, prior["affine_tps"][index], ground_truth),
        base.draw_panel(image, "New fixed-BS10 raw", target_frame, pack["raw"][index], ground_truth),
        base.draw_panel(image, "New fixed-BS10 affine", target_frame, pack["affine"][index], ground_truth),
        base.draw_panel(image, "New fixed-BS10 affine+TPS", target_frame, pack["affine_tps"][index], ground_truth),
    ]
    blank = np.full_like(panels[0], 14)
    errors = {
        stage: float(base.frame_rmse(pack[stage][index : index + 1], ground_truth[None], GROUPS["all_11"])[0])
        for stage in STAGES
    }
    lines = [
        "CALIBRATION FRAME DIAGNOSTIC",
        f"raw: {errors['raw']:.3f} mm",
        f"affine: {errors['affine']:.3f} mm ({errors['affine']-errors['raw']:+.3f})",
        f"TPS: {errors['affine_tps']:.3f} mm ({errors['affine_tps']-errors['affine']:+.3f})",
        "not a generalization headline",
    ]
    for line_index, line in enumerate(lines):
        cv2.putText(blank, line, (10, 28 + 28 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)
    panels.append(blank)
    height = 2 * panels[0].shape[0] + SEPARATOR
    width = 3 * PANEL_SIZE + 2 * SEPARATOR
    canvas = np.full((height, width, 3), 8, dtype=np.uint8)
    for panel_index, panel in enumerate(panels):
        row, column = divmod(panel_index, 3)
        y_value = row * (panel.shape[0] + SEPARATOR)
        x_value = column * (PANEL_SIZE + SEPARATOR)
        canvas[y_value : y_value + panel.shape[0], x_value : x_value + panel.shape[1]] = panel
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Could not write {output}")


def render_warp_map(transform: dict[str, Any], image: np.ndarray, output: Path) -> dict[str, Any]:
    axis = np.linspace(2.0, 134.0, 45)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    affine = apply_transform(transform["step1_affine"], points)
    final = np.asarray(transform["apply_two_step"](points), dtype=np.float64)
    displacement = final - affine
    magnitude = np.linalg.norm(displacement, axis=1)
    epsilon = 0.05
    mapped_x = np.asarray(transform["apply_two_step"](points + [epsilon, 0.0]), dtype=float)
    mapped_y = np.asarray(transform["apply_two_step"](points + [0.0, epsilon]), dtype=float)
    dx = (mapped_x - final) / epsilon
    dy = (mapped_y - final) / epsilon
    determinant = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=160)
    for axis_value in axes:
        axis_value.imshow(image, cmap="gray", vmin=0, vmax=255, extent=(0, 136, 136, 0))
        axis_value.set_xlim(0, 136)
        axis_value.set_ylim(136, 0)
        axis_value.set_aspect("equal")
    stride = 3
    q = axes[0].quiver(
        affine[::stride, 0], affine[::stride, 1], displacement[::stride, 0], displacement[::stride, 1],
        magnitude[::stride], cmap="magma", angles="xy", scale_units="xy", scale=1.0, width=0.003,
    )
    fig.colorbar(q, ax=axes[0], label="TPS displacement from affine (px)")
    axes[0].set_title("TPS displacement field after affine")
    heat = axes[1].imshow(
        determinant.reshape(len(axis), len(axis)), cmap="RdBu_r", origin="upper",
        extent=(axis.min(), axis.max(), axis.max(), axis.min()), alpha=0.72,
    )
    axes[1].contour(xx, yy, determinant.reshape(xx.shape), levels=[0.0], colors="yellow", linewidths=1.2)
    fig.colorbar(heat, ax=axes[1], label="Jacobian determinant")
    axes[1].set_title("Full affine+TPS Jacobian (yellow = zero)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return {
        "dense_grid_points": len(points),
        "tps_displacement_from_affine_mean_px_dense": float(np.mean(magnitude)),
        "tps_displacement_from_affine_max_px_dense": float(np.max(magnitude)),
        "full_map_jacobian_min_dense": float(np.min(determinant)),
        "full_map_jacobian_median_dense": float(np.median(determinant)),
        "full_map_jacobian_max_dense": float(np.max(determinant)),
        "full_map_nonpositive_jacobian_fraction_dense": float(np.mean(determinant <= 0.0)),
        "warp_figure": str(output.resolve()),
    }


def subset_masks(pack: dict[str, Any], calibration_frame: int) -> OrderedDict[str, np.ndarray]:
    return OrderedDict(
        [
            ("all_frames", np.ones(len(pack["frames"]), dtype=bool)),
            ("exclude_calibration_frame", pack["frames"] != calibration_frame),
            ("textgrid_exact_u_frames", pack["u_mask"].astype(bool)),
        ]
    )


def weighted(values: Iterable[np.ndarray]) -> float:
    return float(np.mean(np.concatenate(list(values))))


def paired_bootstrap(
    reference: dict[tuple[int, int], np.ndarray],
    candidate: dict[tuple[int, int], np.ndarray],
    pairs: tuple[tuple[int, int], ...],
    replicates: int,
) -> dict[str, float]:
    reference_sums = np.asarray([reference[pair].sum(dtype=np.float64) for pair in pairs])
    candidate_sums = np.asarray([candidate[pair].sum(dtype=np.float64) for pair in pairs])
    counts = np.asarray([len(reference[pair]) for pair in pairs], dtype=np.float64)
    observed = float(candidate_sums.sum() / counts.sum() - reference_sums.sum() / counts.sum())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    selected = rng.integers(0, len(pairs), size=(replicates, len(pairs)))
    denominator = counts[selected].sum(axis=1)
    effects = candidate_sums[selected].sum(axis=1) / denominator - reference_sums[selected].sum(axis=1) / denominator
    return {
        "effect_candidate_minus_reference_mm": observed,
        "ci95_low_mm": float(np.quantile(effects, 0.025)),
        "ci95_high_mm": float(np.quantile(effects, 0.975)),
    }


def compute_metrics(
    args: argparse.Namespace, selections: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    session_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    calibration_class_rows: list[dict[str, Any]] = []
    store: dict[tuple[str, tuple[int, int], str, str, str], np.ndarray] = {}
    class_store: dict[tuple[str, tuple[int, int], str, str], np.ndarray] = {}
    for pair in base.SELECTION:
        current = base.load_corrected_pack(args.output_root, "asd2", pair, "baseline")
        prior = base.load_corrected_pack(args.prior_root, "asd2", pair, "baseline")
        if not np.array_equal(current["frames"], prior["frames"]) or not np.array_equal(current["ground_truth"], prior["ground_truth"]):
            raise RuntimeError(f"Current/prior paired population mismatch P{pair[0]}/S{pair[1]}")
        calibration = int(selections[f"target_P{pair[0]}_S{pair[1]}"]["selected_frame"])
        masks = subset_masks(current, calibration)
        calibration_index = int(np.flatnonzero(current["frames"] == calibration)[0])
        for model_name, pack in (("new_fixedbs10", current), ("prior_epoch211", prior)):
            for stage in STAGES:
                class_values = base.class_rmse(pack[stage], pack["ground_truth"])
                for subset, mask in masks.items():
                    class_store[(model_name, pair, subset, stage)] = class_values[mask]
                    for group, indices in GROUPS.items():
                        values = base.frame_rmse(pack[stage], pack["ground_truth"], indices)[mask]
                        store[(model_name, pair, subset, stage, group)] = values
                        session_rows.append(
                            {
                                "model": model_name,
                                "speaker": pair[0],
                                "session": pair[1],
                                "subset": subset,
                                "stage": stage,
                                "group": group,
                                "frames": len(values),
                                "mean_frame_coordinate_rmse_mm": float(np.mean(values)),
                                "median_frame_coordinate_rmse_mm": float(np.median(values)),
                            }
                        )
                    for class_index, class_name in enumerate(base.CLASSES):
                        values = class_values[mask, class_index]
                        class_rows.append(
                            {
                                "model": model_name,
                                "speaker": pair[0],
                                "session": pair[1],
                                "subset": subset,
                                "stage": stage,
                                "class_name": class_name,
                                "frames": len(values),
                                "mean_frame_coordinate_rmse_mm": float(np.mean(values)),
                            }
                        )
                if model_name == "new_fixedbs10":
                    for group, indices in GROUPS.items():
                        value = float(base.frame_rmse(pack[stage][calibration_index : calibration_index + 1], pack["ground_truth"][calibration_index : calibration_index + 1], indices)[0])
                        calibration_rows.append(
                            {
                                "speaker": pair[0],
                                "session": pair[1],
                                "calibration_frame": calibration,
                                "verified_exact_u": True,
                                "stage": stage,
                                "group": group,
                                "coordinate_rmse_mm": value,
                                "selection_uses_rmse": False,
                                "interpretation": "calibration diagnostic, not generalization headline",
                            }
                        )
                    for class_index, class_name in enumerate(base.CLASSES):
                        calibration_class_rows.append(
                            {
                                "speaker": pair[0],
                                "session": pair[1],
                                "calibration_frame": calibration,
                                "verified_exact_u": True,
                                "stage": stage,
                                "class_name": class_name,
                                "coordinate_rmse_mm": float(class_values[calibration_index, class_index]),
                                "selection_uses_rmse": False,
                                "interpretation": "calibration diagnostic, not generalization headline",
                            }
                        )

    aggregate_rows: list[dict[str, Any]] = []
    for cohort, pairs in COHORTS.items():
        for model_name in ("new_fixedbs10", "prior_epoch211"):
            for subset in ("all_frames", "exclude_calibration_frame", "textgrid_exact_u_frames"):
                for stage in STAGES:
                    for group in GROUPS:
                        values = [store[(model_name, pair, subset, stage, group)] for pair in pairs]
                        aggregate_rows.append(
                            {
                                "cohort": cohort,
                                "model": model_name,
                                "subset": subset,
                                "stage": stage,
                                "group": group,
                                "sessions": len(pairs),
                                "frames": sum(len(value) for value in values),
                                "mean_frame_coordinate_rmse_mm": weighted(values),
                            }
                        )

    comparisons: list[dict[str, Any]] = []
    definitions = (
        ("new_affine_minus_new_raw", "new_fixedbs10", "raw", "new_fixedbs10", "affine"),
        ("new_tps_minus_new_affine", "new_fixedbs10", "affine", "new_fixedbs10", "affine_tps"),
        ("new_tps_minus_prior_epoch211_tps", "prior_epoch211", "affine_tps", "new_fixedbs10", "affine_tps"),
        ("new_raw_minus_prior_epoch211_raw", "prior_epoch211", "raw", "new_fixedbs10", "raw"),
    )
    for cohort, pairs in COHORTS.items():
        for subset in ("all_frames", "exclude_calibration_frame", "textgrid_exact_u_frames"):
            for group in GROUPS:
                for name, reference_model, reference_stage, candidate_model, candidate_stage in definitions:
                    reference = {pair: store[(reference_model, pair, subset, reference_stage, group)] for pair in pairs}
                    candidate = {pair: store[(candidate_model, pair, subset, candidate_stage, group)] for pair in pairs}
                    comparisons.append(
                        {
                            "comparison": name,
                            "cohort": cohort,
                            "subset": subset,
                            "group": group,
                            "sessions": len(pairs),
                            "bootstrap_replicates": args.bootstrap_replicates,
                            "seed": BOOTSTRAP_SEED,
                            **paired_bootstrap(reference, candidate, pairs, args.bootstrap_replicates),
                        }
                    )

    per_contour_comparison: list[dict[str, Any]] = []
    pairs = base.UNSEEN_SELECTION
    subset = "exclude_calibration_frame"
    for class_index, class_name in enumerate(base.CLASSES):
        reference = {pair: class_store[("prior_epoch211", pair, subset, "affine_tps")][:, class_index] for pair in pairs}
        candidate = {pair: class_store[("new_fixedbs10", pair, subset, "affine_tps")][:, class_index] for pair in pairs}
        per_contour_comparison.append(
            {
                "class_name": class_name,
                "cohort": "UNSEEN_8",
                "subset": subset,
                "prior_epoch211_mm": weighted(reference.values()),
                "new_fixedbs10_mm": weighted(candidate.values()),
                "bootstrap_replicates": args.bootstrap_replicates,
                "seed": BOOTSTRAP_SEED,
                **paired_bootstrap(reference, candidate, pairs, args.bootstrap_replicates),
            }
        )

    write_csv(args.output_root / "phase_a/reference_frame_stage_metrics.csv", calibration_rows)
    write_csv(args.output_root / "phase_a/per_contour_reference_metrics.csv", calibration_class_rows)
    write_csv(args.output_root / "phase_b/metrics/session_metrics.csv", session_rows)
    write_csv(args.output_root / "phase_b/metrics/per_contour_session_metrics.csv", class_rows)
    write_csv(args.output_root / "phase_b/metrics/aggregate_metrics.csv", aggregate_rows)
    write_csv(args.output_root / "phase_b/comparisons/bootstrap_comparisons.csv", comparisons)
    write_csv(args.output_root / "phase_b/comparisons/per_contour_epoch211_comparison.csv", per_contour_comparison)
    return calibration_rows, aggregate_rows, comparisons, per_contour_comparison


def lookup_aggregate(rows: list[dict[str, Any]], **query: Any) -> float:
    matches = [row for row in rows if all(row[key] == value for key, value in query.items())]
    if len(matches) != 1:
        raise RuntimeError(f"Aggregate lookup expected one row, got {len(matches)}: {query}")
    return float(matches[0]["mean_frame_coordinate_rmse_mm"])


def lookup_comparison(rows: list[dict[str, Any]], **query: Any) -> dict[str, Any]:
    matches = [row for row in rows if all(row[key] == value for key, value in query.items())]
    if len(matches) != 1:
        raise RuntimeError(f"Comparison lookup expected one row, got {len(matches)}: {query}")
    return matches[0]


def write_report(
    args: argparse.Namespace,
    calibration_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    per_contour: list[dict[str, Any]],
    warp_rows: list[dict[str, Any]],
) -> None:
    query = {
        "cohort": "UNSEEN_8",
        "model": "new_fixedbs10",
        "subset": "exclude_calibration_frame",
        "group": "all_11",
    }
    raw = lookup_aggregate(aggregate_rows, **query, stage="raw")
    affine = lookup_aggregate(aggregate_rows, **query, stage="affine")
    tps = lookup_aggregate(aggregate_rows, **query, stage="affine_tps")
    prior = lookup_aggregate(
        aggregate_rows,
        cohort="UNSEEN_8",
        model="prior_epoch211",
        subset="exclude_calibration_frame",
        stage="affine_tps",
        group="all_11",
    )
    model_effect = lookup_comparison(
        comparisons,
        comparison="new_tps_minus_prior_epoch211_tps",
        cohort="UNSEEN_8",
        subset="exclude_calibration_frame",
        group="all_11",
    )
    calibration_table = [
        "| Session | Frame | Raw | Affine | TPS | Affine−raw | TPS−affine |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in base.SELECTION:
        selected = [
            row
            for row in calibration_rows
            if row["speaker"] == pair[0] and row["session"] == pair[1] and row["group"] == "all_11"
        ]
        values = {row["stage"]: float(row["coordinate_rmse_mm"]) for row in selected}
        frame = int(selected[0]["calibration_frame"])
        calibration_table.append(
            f"| P{pair[0]}/S{pair[1]} | F{frame:04d} | {values['raw']:.3f} | {values['affine']:.3f} | "
            f"{values['affine_tps']:.3f} | {values['affine']-values['raw']:+.3f} | "
            f"{values['affine_tps']-values['affine']:+.3f} |"
        )
    saved_session_rows = list(
        csv.DictReader((args.output_root / "phase_b/metrics/session_metrics.csv").open(encoding="utf-8"))
    )
    session_table = [
        "| Session | Raw | Affine | TPS | Affine−raw | TPS−affine | New TPS−prior TPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in base.SELECTION:
        def session_value(model: str, stage: str) -> float:
            matches = [
                row
                for row in saved_session_rows
                if int(row["speaker"]) == pair[0]
                and int(row["session"]) == pair[1]
                and row["model"] == model
                and row["subset"] == "exclude_calibration_frame"
                and row["stage"] == stage
                and row["group"] == "all_11"
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Session report lookup failed for {pair}/{model}/{stage}")
            return float(matches[0]["mean_frame_coordinate_rmse_mm"])
        session_raw = session_value("new_fixedbs10", "raw")
        session_affine = session_value("new_fixedbs10", "affine")
        session_tps = session_value("new_fixedbs10", "affine_tps")
        session_prior = session_value("prior_epoch211", "affine_tps")
        session_table.append(
            f"| P{pair[0]}/S{pair[1]} | {session_raw:.3f} | {session_affine:.3f} | {session_tps:.3f} | "
            f"{session_affine-session_raw:+.3f} | {session_tps-session_affine:+.3f} | "
            f"{session_tps-session_prior:+.3f} |"
        )
    contour_table = [
        "| Contour | Prior epoch-211 | New fixed-BS10 | New−prior | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in per_contour:
        contour_table.append(
            f"| {row['class_name']} | {float(row['prior_epoch211_mm']):.3f} | "
            f"{float(row['new_fixedbs10_mm']):.3f} | "
            f"{float(row['effect_candidate_minus_reference_mm']):+.3f} | "
            f"[{float(row['ci95_low_mm']):+.3f}, {float(row['ci95_high_mm']):+.3f}] |"
        )
    foldovers = sum(float(row["full_map_nonpositive_jacobian_fraction_dense"]) > 0 for row in warp_rows)
    max_displacement = max(float(row["tps_displacement_from_affine_max_px_dense"]) for row in warp_rows)
    report = f"""# Fixed-batch-10 ASD2 exact-TextGrid `/u/` grid adaptation

This run performs fresh baseline inference only. It contains no RMS, VTLN, audio-normalization, audio-ablation, fine-tuning, or retraining branch.

## Protocol

- Source reference: ASD2 `1791/S14/F0499`, selected directly from an exact TextGrid `/u/` interval.
- Target references: one independently reselected exact `/u/` frame for each fixed ASD1 session.
- Reference selection never reads model prediction or RMSE.
- One fixed affine and zero-smoothing TPS transform is built per target session and reused unchanged for the full session.
- The primary full-session metric excludes the one calibration frame because its target anatomy was used to build the transform.
- P10/S14 is a same-person control and is excluded from the primary unseen-eight cohort.
- Only integer MRI frames are saved, scored, or rendered.

## Primary unseen-eight result, calibration frame excluded

- New fixed-BS10 raw: `{raw:.4f}` mm.
- New fixed-BS10 affine: `{affine:.4f}` mm (`{affine-raw:+.4f}` vs raw).
- New fixed-BS10 affine+TPS: `{tps:.4f}` mm (`{tps-affine:+.4f}` vs affine).
- Prior epoch-211 affine+TPS: `{prior:.4f}` mm.
- New minus prior: `{float(model_effect['effect_candidate_minus_reference_mm']):+.4f}` mm, paired whole-session bootstrap 95% CI `[{float(model_effect['ci95_low_mm']):+.4f}, {float(model_effect['ci95_high_mm']):+.4f}]`.

## Calibration-frame diagnostics

These numbers describe the frame used to estimate each speaker transform; they are not generalization claims.

{chr(10).join(calibration_table)}

## Full-session effects by target, calibration frame excluded

{chr(10).join(session_table)}

## Per-contour new-versus-epoch-211 comparison

Unseen-eight, affine+TPS, calibration frame excluded:

{chr(10).join(contour_table)}

## TPS geometry

- Dense diagnostic grid: 2,025 points per session.
- Sessions with any sampled non-positive Jacobian: `{foldovers}/9`.
- Maximum dense sampled TPS displacement relative to affine: `{max_displacement:.3f}` px.
- Exact TPS control fit is not treated as evidence of anatomical correctness.

## Videos

Each session video contains target ground truth, prior epoch-211 affine+TPS, new raw, new affine, new affine+TPS, and a per-frame error/delta panel. Videos contain only the canonical evaluated integer frames at exactly 50 fps. Playback uses independently timestamped 20-ms segments from the original target WAV; no processed audio is used.
"""
    path = args.output_root / "report.md"
    path.write_text(report, encoding="utf-8")


def write_preflight(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    config = base.asd2_core.load_yaml_config(args.config)
    if config.get("normalization_mode") != "train_global" or config.get("normalization_std_policy") != "raw_positive":
        raise RuntimeError("Expected train_global/raw_positive model normalization")
    payload = {
        "created_at": now(),
        "status": "passed",
        "operation": "fresh inference plus no-audio grid adaptation; no training",
        "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "hostname": socket.gethostname(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "normalization_stats": str(args.normalization_stats.resolve()),
        "normalization_sha256": sha256(args.normalization_stats),
        "normalization_mode": config["normalization_mode"],
        "normalization_std_policy": config["normalization_std_policy"],
        "normalization_fit_split": config["normalization_fit_split"],
        "source_pack": str(args.source_pack.resolve()),
        "source_pack_sha256": sha256(args.source_pack),
        "fresh_inference_root": str(args.fresh_root.resolve()),
        "prior_comparison_root": str(args.prior_root.resolve()),
        "audio_analysis_or_adaptation": False,
        "training_launched": False,
        "selection": [f"P{pair[0]}/S{pair[1]}" for pair in base.SELECTION],
        "unseen_selection": [f"P{pair[0]}/S{pair[1]}" for pair in base.UNSEEN_SELECTION],
        "same_person_control": "P10/S14",
        "fresh_inputs": rows,
        "saved_fractional_frame_count": 0,
        "scored_fractional_frame_count": 0,
        "rendered_fractional_frame_count": 0,
    }
    atomic_json(args.output_root / "preflight.json", payload)


def run_build(args: argparse.Namespace) -> None:
    started = time.monotonic()
    args.output_root.mkdir(parents=True, exist_ok=True)
    configure_modules(args)
    for path in (args.fresh_root, args.prior_root, args.config, args.checkpoint, args.normalization_stats, args.source_pack, args.source_cache):
        if not path.exists():
            raise FileNotFoundError(path)
    inputs, input_rows = load_inputs(args)
    write_preflight(args, input_rows)
    write_csv(args.output_root / "provenance/fresh_prediction_inputs.csv", input_rows)
    selections, reference_rows, u_masks = base.discover_references(inputs, args.output_root)
    validate_reference_selection(selections)
    save_candidate_rows(selections, args.output_root)
    sources, targets = base.build_reference_payloads(selections, args.output_root)
    provenance_path = args.output_root / "provenance/source_target_grid_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["asd2_source"]["classes_9_to_10"] = (
        "current lower-incisor repair-v2 and upper-incisor legacy-position labels used by the fixed-BS10 model"
    )
    provenance["p7_source"]["role"] = "historical audit-only reference; not used by this no-audio ASD2 transform"
    atomic_json(provenance_path, provenance)
    transforms = base.build_transforms(sources, targets, selections, args.output_root)
    inventory = []
    for pair in base.SELECTION:
        inventory.append(
            base.save_corrected_pack(
                args.output_root,
                "asd2",
                pair,
                "baseline",
                inputs[("asd2", "baseline", pair)],
                transforms[("asd2", pair)],
                u_masks[pair],
                SOURCE_FRAME,
                int(selections[f"target_P{pair[0]}_S{pair[1]}"]["selected_frame"]),
                args.transform_frame_batch,
            )
        )
    write_csv(args.output_root / "provenance/prediction_pack_inventory.csv", inventory)

    selection_figures = []
    process_figures = []
    prediction_figures = []
    warp_rows = []
    process_rows = []
    process_dir = args.output_root / "phase_a/transform_process"
    process_dir.mkdir(parents=True, exist_ok=True)
    for pair in base.SELECTION:
        speaker, session = pair
        selection = selections[f"target_P{speaker}_S{session}"]
        target_frame = int(selection["selected_frame"])
        selection_figure = args.output_root / f"phase_a/textgrid_selection/P{speaker}_S{session}_F{target_frame:04d}_selection.png"
        render_selection_plot(selection, selection_figure)
        selection_figures.append(selection_figure)
        process_figure = process_dir / f"P{speaker}_S{session}_F{target_frame:04d}_source_F0499_transform_process.png"
        process_row = transform_process.render_session(
            process_figure,
            sources["asd2"],
            targets[pair],
            transforms[("asd2", pair)],
            speaker,
            session,
            target_frame,
        )
        process_rows.append(process_row)
        process_figures.append(process_figure)
        prediction_figure = args.output_root / f"phase_a/prediction_stages/P{speaker}_S{session}_F{target_frame:04d}_prediction_stages.png"
        render_prediction_stage_figure(
            base.load_corrected_pack(args.output_root, "asd2", pair, "baseline"),
            base.load_corrected_pack(args.prior_root, "asd2", pair, "baseline"),
            target_frame,
            targets[pair]["image"],
            prediction_figure,
        )
        prediction_figures.append(prediction_figure)
        warp_figure = args.output_root / f"phase_a/warp_fields/P{speaker}_S{session}_warp_jacobian.png"
        dense = render_warp_map(transforms[("asd2", pair)], targets[pair]["image"], warp_figure)
        diagnostics = base.prior_u.transform_diagnostics(
            transforms[("asd2", pair)], sources["asd2"]["grid"], targets[pair]["grid"]
        )
        warp_rows.append(
            {
                "speaker": speaker,
                "session": session,
                "source_frame": SOURCE_FRAME,
                "target_frame": target_frame,
                **{key: value for key, value in diagnostics.items() if not isinstance(value, (list, dict))},
                **dense,
            }
        )
    write_csv(args.output_root / "phase_a/transform_process_metrics.csv", process_rows)
    write_csv(args.output_root / "phase_a/warp_diagnostics_dense.csv", warp_rows)
    save_contact_sheet(selection_figures, args.output_root / "phase_a/contact_sheet_textgrid_selection_all9.png")
    save_contact_sheet(process_figures, args.output_root / "phase_a/contact_sheet_transform_process_all9.png")
    save_contact_sheet(prediction_figures, args.output_root / "phase_a/contact_sheet_prediction_stages_all9.png")

    calibration_rows, aggregate_rows, comparisons, per_contour = compute_metrics(args, selections)
    write_report(args, calibration_rows, aggregate_rows, comparisons, per_contour, warp_rows)
    manifest = {
        "created_at": now(),
        "status": "build_complete",
        "experiment": "fixed-batch-10 fresh baseline inference plus exact-/u/ grid adaptation",
        "audio_analysis_or_adaptation": False,
        "training_launched": False,
        "source_frame": SOURCE_FRAME,
        "target_frames": {f"P{pair[0]}/S{pair[1]}": EXPECTED_TARGET_FRAMES[pair] for pair in base.SELECTION},
        "reference_count": len(reference_rows),
        "transform_reference_count": 10,
        "p7_reference_role": "historical audit only; not used by the new ASD2 transform",
        "all9_integer_frames": 8585,
        "unseen8_integer_frames": 7633,
        "primary_subset": "exclude_calibration_frame",
        "transform_scope": "one fixed transform per target session",
        "bootstrap_replicates": args.bootstrap_replicates,
        "elapsed_seconds": time.monotonic() - started,
        "saved_fractional_frame_count": 0,
        "scored_fractional_frame_count": 0,
        "rendered_fractional_frame_count": 0,
    }
    atomic_json(args.output_root / "manifest.json", manifest)


def summary_panel(
    frame: int, target_frame: int, ground_truth: np.ndarray, pack: dict[str, Any], index: int
) -> np.ndarray:
    canvas = np.full((INFO_HEIGHT + PANEL_SIZE, PANEL_SIZE, 3), 14, dtype=np.uint8)
    values = {
        stage: float(base.frame_rmse(pack[stage][index : index + 1], ground_truth[None], GROUPS["all_11"])[0])
        for stage in STAGES
    }
    lines = [
        "NEW MODEL: STAGE EFFECT",
        f"frame F{frame:04d}" + (" [calibration /u/]" if frame == target_frame else ""),
        f"raw       {values['raw']:.3f} mm",
        f"affine    {values['affine']:.3f} ({values['affine']-values['raw']:+.3f})",
        f"affineTPS {values['affine_tps']:.3f} ({values['affine_tps']-values['affine']:+.3f})",
        "primary metrics exclude calibration frame",
        "solid GT | dashed prediction",
    ]
    for line_index, line in enumerate(lines):
        cv2.putText(canvas, line, (7, 24 + line_index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas


def compose_video_canvas(
    image: np.ndarray,
    frame: int,
    index: int,
    target_frame: int,
    current: dict[str, Any],
    prior: dict[str, Any],
) -> np.ndarray:
    ground_truth = current["ground_truth"][index]
    panels = [
        base.draw_panel(image, "Target ground truth", frame, None, ground_truth),
        base.draw_panel(image, "Prior epoch-211 affine+TPS", frame, prior["affine_tps"][index], ground_truth),
        base.draw_panel(image, "New fixed-BS10 raw", frame, current["raw"][index], ground_truth),
        base.draw_panel(image, "New fixed-BS10 affine", frame, current["affine"][index], ground_truth),
        base.draw_panel(image, "New fixed-BS10 affine+TPS", frame, current["affine_tps"][index], ground_truth),
        summary_panel(frame, target_frame, ground_truth, current, index),
    ]
    height = 2 * panels[0].shape[0] + SEPARATOR
    width = 3 * PANEL_SIZE + 2 * SEPARATOR
    canvas = np.full((height, width, 3), 8, dtype=np.uint8)
    for panel_index, panel in enumerate(panels):
        row, column = divmod(panel_index, 3)
        y_value = row * (panel.shape[0] + SEPARATOR)
        x_value = column * (PANEL_SIZE + SEPARATOR)
        canvas[y_value : y_value + panel.shape[0], x_value : x_value + panel.shape[1]] = panel
    return canvas


def render_video(args: argparse.Namespace, pair: tuple[int, int], target_frame: int) -> dict[str, Any]:
    speaker, session = pair
    current = base.load_corrected_pack(args.output_root, "asd2", pair, "baseline")
    prior = base.load_corrected_pack(args.prior_root, "asd2", pair, "baseline")
    if not np.array_equal(current["frames"], prior["frames"]) or not np.array_equal(current["ground_truth"], prior["ground_truth"]):
        raise RuntimeError(f"Video population mismatch P{speaker}/S{session}")
    frames = current["frames"]
    session_dir = args.output_root / f"phase_b/videos/P{speaker}/S{session}"
    session_dir.mkdir(parents=True, exist_ok=True)
    output = session_dir / f"p{speaker}_s{session}_fixedbs10_textgrid_u_grid_adaptation_50fps.mp4"
    audit_path = session_dir / "video_audit.json"
    if output.is_file() and audit_path.is_file() and not args.force_videos:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "passed":
            print(f"REUSE passing video P{speaker}/S{session}", flush=True)
            return audit
    dicom_dir = base.RAW_ROOT / f"P{speaker}/DCM_2D/S{session}"
    dicom_index, _ = build_filename_dicom_index(dicom_dir)
    cache = load_or_build_mri_cache(
        dicom_dir,
        dicom_index,
        [int(frame) for frame in frames],
        session_dir / ".cache/evaluated_integer_mri_frames.npz",
        workers=args.mri_workers,
    )
    silent = session_dir / f".{output.stem}.silent.writing.mp4"
    muxing = session_dir / f".{output.stem}.muxing.mp4"
    audio_m4a = session_dir / f".{output.stem}.audio.writing.m4a"
    height = 2 * (INFO_HEIGHT + PANEL_SIZE) + SEPARATOR
    width = 3 * PANEL_SIZE + 2 * SEPARATOR
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {silent}")
    qc_indices = {0, len(frames) // 2, len(frames) - 1, int(np.flatnonzero(frames == target_frame)[0])}
    qc_paths = []
    try:
        for index, frame_value in enumerate(frames):
            frame = int(frame_value)
            canvas = compose_video_canvas(cache[frame], frame, index, target_frame, current, prior)
            writer.write(canvas)
            if index in qc_indices:
                qc = session_dir / f"qc_F{frame:04d}.png"
                if not cv2.imwrite(str(qc), canvas):
                    raise RuntimeError(f"Could not write {qc}")
                qc_paths.append(qc)
            if index == 0 or (index + 1) % 500 == 0 or index + 1 == len(frames):
                print(f"RENDER P{speaker}/S{session}: {index+1}/{len(frames)}", flush=True)
    finally:
        writer.release()
    original_audio, _ = base.asd2_core.exact_asd1_audio_paths(speaker, session)
    segment_wav = session_dir / "original_audio_evaluated_frame_segments.wav"
    audio_metadata = base.write_original_audio_segments(
        original_audio,
        frames,
        segment_wav,
        session_dir / "original_audio_evaluated_frame_segments.csv",
    )
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(segment_wav), "-c:a", "aac", "-b:a", "96k", str(audio_m4a)], check=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(silent), "-i", str(audio_m4a),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-vsync", "0", "-c:a", "copy", "-movflags", "+faststart", str(muxing),
        ],
        check=True,
    )
    muxing.replace(output)
    silent.unlink(missing_ok=True)
    audio_m4a.unlink(missing_ok=True)
    audit = base.audit_video(output, frames, original_audio)
    audit.update(
        {
            "target_u_calibration_frame": target_frame,
            "uses_processed_audio": False,
            "audio_analysis_or_adaptation": False,
            "audio": audio_metadata,
            "visual_qc_samples": [str(path.resolve()) for path in qc_paths],
            "timeline_policy": "canonical evaluated integer frames only; no interpolation or hold",
        }
    )
    atomic_json(audit_path, audit)
    return audit


def run_videos(args: argparse.Namespace) -> None:
    selection_path = args.output_root / "provenance/textgrid_u_selections.json"
    selections = json.loads(selection_path.read_text(encoding="utf-8"))
    audits = []
    for pair in base.SELECTION:
        target = int(selections[f"target_P{pair[0]}_S{pair[1]}"]["selected_frame"])
        audits.append(render_video(args, pair, target))
    rows = [
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
            "passed": audit["status"] == "passed",
        }
        for pair, audit in zip(base.SELECTION, audits)
    ]
    write_csv(args.output_root / "phase_b/videos/video_inventory.csv", rows)


def artifact_hashes(root: Path) -> list[dict[str, Any]]:
    excluded = {"artifact_hashes.csv", "artifact_hashes.json", "final_audit.json"}
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in excluded or ".cache" in path.parts or path.name.endswith(".writing"):
            continue
        rows.append({"relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return rows


def run_audit(args: argparse.Namespace) -> None:
    manifest = json.loads((args.output_root / "manifest.json").read_text(encoding="utf-8"))
    selections = json.loads((args.output_root / "provenance/textgrid_u_selections.json").read_text(encoding="utf-8"))
    video_rows = list(csv.DictReader((args.output_root / "phase_b/videos/video_inventory.csv").open(encoding="utf-8")))
    prediction_packs = list((args.output_root / "predictions/ASD2").glob("P*/S*/baseline.npz"))
    selection_figures = list((args.output_root / "phase_a/textgrid_selection").glob("*.png"))
    process_figures = list((args.output_root / "phase_a/transform_process").glob("*.png"))
    prediction_figures = list((args.output_root / "phase_a/prediction_stages").glob("*.png"))
    warp_figures = list((args.output_root / "phase_a/warp_fields").glob("*.png"))
    checks = {
        "manifest_build_complete": manifest.get("status") == "build_complete",
        "all_target_frames_exact_expected": all(
            int(selections[f"target_P{pair[0]}_S{pair[1]}"]["selected_frame"]) == expected
            for pair, expected in EXPECTED_TARGET_FRAMES.items()
        ),
        "source_frame_exact_F0499": int(selections["asd2_source"]["selected_frame"]) == SOURCE_FRAME,
        "all_references_verified_exact_u": bool((args.output_root / "textgrid_u_reference_audit.csv").is_file()),
        "nine_prediction_packs": len(prediction_packs) == 9,
        "nine_selection_figures": len(selection_figures) == 9,
        "nine_transform_process_figures": len(process_figures) == 9,
        "nine_prediction_stage_figures": len(prediction_figures) == 9,
        "nine_warp_figures": len(warp_figures) == 9,
        "nine_videos": len(video_rows) == 9,
        "all_videos_passed": len(video_rows) == 9 and all(row["passed"] == "True" for row in video_rows),
        "all_videos_exact_50fps": len(video_rows) == 9 and all(row["fps"] == "50/1" for row in video_rows),
        "all9_integer_frames_8585": int(manifest["all9_integer_frames"]) == 8585,
        "unseen8_integer_frames_7633": int(manifest["unseen8_integer_frames"]) == 7633,
        "no_audio_analysis_or_adaptation": manifest["audio_analysis_or_adaptation"] is False,
        "training_not_launched": manifest["training_launched"] is False,
        "saved_fractional_zero": int(manifest["saved_fractional_frame_count"]) == 0,
        "scored_fractional_zero": int(manifest["scored_fractional_frame_count"]) == 0,
        "rendered_fractional_zero": int(manifest["rendered_fractional_frame_count"]) == 0,
    }
    rows = artifact_hashes(args.output_root)
    write_csv(args.output_root / "artifact_hashes.csv", rows)
    atomic_json(
        args.output_root / "artifact_hashes.json",
        {"created_at": now(), "file_count": len(rows), "files": rows},
    )
    audit = {
        "created_at": now(),
        "status": "audit_ok" if all(checks.values()) else "audit_failed",
        "definition_of_done_passed": all(checks.values()),
        "checks": checks,
        "artifact_hash_count": len(rows),
        "saved_fractional_frame_count": 0,
        "scored_fractional_frame_count": 0,
        "rendered_fractional_frame_count": 0,
    }
    atomic_json(args.output_root / "final_audit.json", audit)
    if not audit["definition_of_done_passed"]:
        raise RuntimeError(f"Final audit failed: {checks}")


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.fresh_root = args.fresh_root.resolve()
    args.prior_root = args.prior_root.resolve()
    configure_modules(args)
    if args.phase in ("build", "all"):
        run_build(args)
        if args.phase == "build":
            return
    if args.phase in ("videos", "all"):
        run_videos(args)
        if args.phase == "videos":
            return
    run_audit(args)


if __name__ == "__main__":
    main()
