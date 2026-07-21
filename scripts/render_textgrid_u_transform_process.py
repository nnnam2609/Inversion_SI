#!/usr/bin/env python3
"""Render the fixed per-session TextGrid-/u/ transform construction stages.

This is a diagnostic renderer only.  It rebuilds the same source-reference to
target-reference transform used by the corrected selected-nine experiment; it
does not run inference and does not alter the canonical result bundle.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.distance import cdist

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_ROOT = REPO_ROOT / "external/grid-transform"
sys.path[:0] = [
    str(REPO_ROOT),
    str(REPO_ROOT / "src"),
    str(REPO_ROOT / "scripts"),
    str(GRID_ROOT),
]

import run_textgrid_u_corrected_selected9_experiment as experiment  # noqa: E402
from grid_transform.transfer import build_two_step_transform  # noqa: E402
from grid_transform.transform_helpers import (  # noqa: E402
    apply_transform,
    extract_true_landmarks,
)
from render_p7_grid_transform_selected_speakers import (  # noqa: E402
    CLASSES,
    FrameSpec,
    prepare_frame,
)
from src.utils.colors import COLORS  # noqa: E402
from src.utils.video_rendering import MM_PER_PIXEL  # noqa: E402


DEFAULT_INPUT_ROOT = (
    REPO_ROOT / "results/asd2_selected_9sessions_textgrid_u_corrected_20260720_194511"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/asd2_textgrid_u_transform_process_20260720"
LANDMARK_ORDER = tuple(f"I{index}" for index in range(1, 8)) + (
    "P1",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "M1",
    "L6",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render source native, target native, before affine, after affine, "
            "and affine+TPS for the corrected selected-nine TextGrid /u/ references."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def configure_axis(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlim(0, 136)
    ax.set_ylim(136, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_contours(
    ax: plt.Axes,
    contours: dict[str, np.ndarray],
    *,
    linestyle: str,
    linewidth: float,
    alpha: float,
) -> None:
    for name in CLASSES:
        points = np.asarray(contours[name], dtype=float)
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=COLORS.get(name, "white"),
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )


def mapped_contours(
    contours: dict[str, np.ndarray], mapping: Callable[[np.ndarray], np.ndarray]
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(mapping(np.asarray(points, dtype=float)), dtype=float)
        for name, points in contours.items()
    }


def named_landmarks(grid: Any) -> dict[str, np.ndarray]:
    landmarks = extract_true_landmarks(grid)
    return {
        name: np.asarray(landmarks[name], dtype=float)
        for name in LANDMARK_ORDER
        if landmarks.get(name) is not None
    }


def map_named_landmarks(
    landmarks: dict[str, np.ndarray], mapping: Callable[[np.ndarray], np.ndarray]
) -> dict[str, np.ndarray]:
    return {name: np.asarray(mapping(point), dtype=float) for name, point in landmarks.items()}


def landmark_color(name: str) -> str:
    if name in {"I1", "I2", "I3", "I4", "I5"}:
        return "#39ff14"
    if name in {"I6", "I7"}:
        return "#2fa4ff"
    if name.startswith("C"):
        return "#00ffff"
    return "#ff9f1c"


def draw_landmarks(
    ax: plt.Axes,
    landmarks: dict[str, np.ndarray],
    *,
    marker: str,
    annotate: bool,
    alpha: float = 0.95,
) -> None:
    for name, point in landmarks.items():
        color = landmark_color(name)
        scatter_options = {
            "s": 21,
            "marker": marker,
            "color": color,
            "linewidths": 0.35,
            "alpha": alpha,
            "zorder": 9,
        }
        if marker == "o":
            scatter_options["edgecolors"] = "black"
        ax.scatter(
            [point[0]],
            [point[1]],
            **scatter_options,
        )
        if annotate:
            ax.text(
                point[0] + 1.2,
                point[1] - 1.2,
                name,
                color=color,
                fontsize=5.5,
                fontweight="bold",
                clip_on=True,
                zorder=10,
            )


def draw_residual_arrows(
    ax: plt.Axes,
    mapped: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
) -> None:
    for name in LANDMARK_ORDER:
        if name not in mapped or name not in target:
            continue
        start = mapped[name]
        delta = target[name] - start
        ax.arrow(
            start[0],
            start[1],
            delta[0],
            delta[1],
            width=0.08,
            head_width=1.1,
            head_length=1.4,
            length_includes_head=True,
            color=landmark_color(name),
            alpha=0.75,
            zorder=8,
        )


def contour_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=float)
    target = np.asarray(target, dtype=float)
    delta = predicted - target
    distances = cdist(predicted, target)
    symmetric_nn_rms_px = np.sqrt(
        0.5
        * (
            np.mean(np.min(distances, axis=1) ** 2)
            + np.mean(np.min(distances, axis=0) ** 2)
        )
    )
    centroid_delta = np.mean(delta, axis=0)
    return {
        "coordinate_rmse_mm": float(np.sqrt(np.mean(delta * delta)) * MM_PER_PIXEL),
        "symmetric_nearest_curve_rms_mm": float(symmetric_nn_rms_px * MM_PER_PIXEL),
        "centroid_dx_mm": float(centroid_delta[0] * MM_PER_PIXEL),
        "centroid_dy_mm": float(centroid_delta[1] * MM_PER_PIXEL),
    }


def landmark_metrics(
    mapped: dict[str, np.ndarray], target: dict[str, np.ndarray], names: tuple[str, ...]
) -> tuple[float, float]:
    errors = [
        np.linalg.norm(mapped[name] - target[name]) * MM_PER_PIXEL
        for name in names
        if name in mapped and name in target
    ]
    return float(np.mean(errors)), float(np.max(errors))


def set_upper_incisor_zoom(ax: plt.Axes, *contours: np.ndarray) -> None:
    points = np.vstack([np.asarray(contour, dtype=float) for contour in contours])
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    x_pad = max(6.0, 0.18 * float(x_max - x_min))
    y_pad = max(6.0, 0.18 * float(y_max - y_min))
    ax.set_xlim(max(0.0, x_min - x_pad), min(136.0, x_max + x_pad))
    ax.set_ylim(min(136.0, y_max + y_pad), max(0.0, y_min - y_pad))


def render_step_image(
    output: Path,
    *,
    image: np.ndarray,
    title: str,
    candidate_contours: dict[str, np.ndarray],
    candidate_landmarks: dict[str, np.ndarray],
    target_contours: dict[str, np.ndarray] | None = None,
    target_landmarks: dict[str, np.ndarray] | None = None,
    metric_text: str = "",
) -> None:
    """Render one debug stage as a full overlay plus an upper-incisor zoom."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), dpi=170)
    configure_axis(axes[0], image, "Full frame: all contours and controls")
    configure_axis(axes[1], image, "Upper-incisor zoom: I1–I5 only")
    incisor_candidate_landmarks = {
        name: point
        for name, point in candidate_landmarks.items()
        if name in {"I1", "I2", "I3", "I4", "I5"}
    }

    if target_contours is None:
        draw_contours(axes[0], candidate_contours, linestyle="-", linewidth=1.25, alpha=0.96)
        draw_landmarks(axes[0], candidate_landmarks, marker="o", annotate=True)
        upper = np.asarray(candidate_contours["upper-incisor"], dtype=float)
        axes[1].plot(upper[:, 0], upper[:, 1], color="yellow", linewidth=2.5, label="native")
        draw_landmarks(axes[1], incisor_candidate_landmarks, marker="o", annotate=True)
        set_upper_incisor_zoom(axes[1], upper)
    else:
        draw_contours(axes[0], target_contours, linestyle="-", linewidth=1.35, alpha=0.94)
        draw_contours(axes[0], candidate_contours, linestyle="--", linewidth=1.0, alpha=0.90)
        target_upper = np.asarray(target_contours["upper-incisor"], dtype=float)
        candidate_upper = np.asarray(candidate_contours["upper-incisor"], dtype=float)
        axes[1].plot(
            target_upper[:, 0], target_upper[:, 1], color="yellow", linewidth=2.6, label="target"
        )
        axes[1].plot(
            candidate_upper[:, 0],
            candidate_upper[:, 1],
            color="#ff3b30",
            linewidth=2.0,
            linestyle="--",
            label="ASD2 source",
        )
        assert target_landmarks is not None
        draw_landmarks(axes[0], target_landmarks, marker="o", annotate=True)
        draw_landmarks(axes[0], candidate_landmarks, marker="x", annotate=False)
        draw_residual_arrows(axes[0], candidate_landmarks, target_landmarks)
        incisor_target_landmarks = {
            name: point
            for name, point in target_landmarks.items()
            if name in {"I1", "I2", "I3", "I4", "I5"}
        }
        draw_landmarks(axes[1], incisor_target_landmarks, marker="o", annotate=True)
        draw_landmarks(axes[1], incisor_candidate_landmarks, marker="x", annotate=False)
        draw_residual_arrows(axes[1], incisor_candidate_landmarks, incisor_target_landmarks)
        set_upper_incisor_zoom(axes[1], target_upper, candidate_upper)

    axes[1].legend(loc="best", fontsize=7, framealpha=0.82)
    if metric_text:
        fig.text(
            0.5,
            0.035,
            textwrap.fill(metric_text, width=145),
            ha="center",
            va="center",
            fontsize=8.5,
        )
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.13, top=0.86, wspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def render_session(
    output: Path,
    source: dict[str, Any],
    target: dict[str, Any],
    transform: dict[str, Any],
    speaker: int,
    session: int,
    target_frame: int,
) -> dict[str, Any]:
    source_contours = source["annotations"]
    target_contours = target["annotations"]
    affine_contours = mapped_contours(
        source_contours,
        lambda points: apply_transform(transform["step1_affine"], points),
    )
    final_contours = mapped_contours(source_contours, transform["apply_two_step"])

    source_lm = named_landmarks(source["grid"])
    target_lm = named_landmarks(target["grid"])
    affine_lm = map_named_landmarks(
        source_lm,
        lambda point: apply_transform(transform["step1_affine"], point),
    )
    final_lm = map_named_landmarks(source_lm, transform["apply_two_step"])

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=145)
    panels = (
        (source["image"], "1. Source ASD2 F0499 /u/\n(native)"),
        (target["image"], f"2. Target P{speaker}/S{session} F{target_frame:04d} /u/\n(native)"),
        (target["image"], "3. Before affine\n(source dashed, target solid)"),
        (target["image"], "4. After global affine\n(source dashed, target solid)"),
        (target["image"], "5. After affine + TPS\n(source dashed, target solid)"),
    )
    for column, (image, title) in enumerate(panels):
        configure_axis(axes[0, column], image, title)
        configure_axis(axes[1, column], image, title.replace("\n", " — "))

    draw_contours(axes[0, 0], source_contours, linestyle="-", linewidth=1.0, alpha=0.95)
    draw_contours(axes[0, 1], target_contours, linestyle="-", linewidth=1.0, alpha=0.95)
    for column, candidate in ((2, source_contours), (3, affine_contours), (4, final_contours)):
        draw_contours(axes[0, column], target_contours, linestyle="-", linewidth=1.15, alpha=0.92)
        draw_contours(axes[0, column], candidate, linestyle="--", linewidth=0.85, alpha=0.87)

    # The lower row isolates the upper incisor and the landmarks that construct
    # the transform. Target landmarks are circles; mapped source landmarks are x.
    source_upper = source_contours["upper-incisor"]
    target_upper = target_contours["upper-incisor"]
    axes[1, 0].plot(source_upper[:, 0], source_upper[:, 1], color="yellow", linewidth=2.0)
    draw_landmarks(axes[1, 0], source_lm, marker="o", annotate=True)
    axes[1, 1].plot(target_upper[:, 0], target_upper[:, 1], color="yellow", linewidth=2.0)
    draw_landmarks(axes[1, 1], target_lm, marker="o", annotate=True)

    for column, contour, landmarks in (
        (2, source_upper, source_lm),
        (3, affine_contours["upper-incisor"], affine_lm),
        (4, final_contours["upper-incisor"], final_lm),
    ):
        axes[1, column].plot(
            target_upper[:, 0], target_upper[:, 1], color="yellow", linewidth=2.2, label="target"
        )
        axes[1, column].plot(
            contour[:, 0], contour[:, 1], color="#ff3b30", linewidth=1.7, linestyle="--", label="ASD2"
        )
        draw_landmarks(axes[1, column], target_lm, marker="o", annotate=False)
        draw_landmarks(axes[1, column], landmarks, marker="x", annotate=False)
        draw_residual_arrows(axes[1, column], landmarks, target_lm)
        axes[1, column].legend(loc="lower right", fontsize=6, framealpha=0.75)

    affine_a = np.asarray(transform["step1_affine"]["A"], dtype=float)
    affine_t = np.asarray(transform["step1_affine"]["t"], dtype=float)
    fig.suptitle(
        (
            f"Fixed per-session transform: ASD2 F0499 /u/ → P{speaker}/S{session} "
            f"F{target_frame:04d} /u/\n"
            f"Affine A={np.array2string(affine_a, precision=3)}, "
            f"t={np.array2string(affine_t, precision=3)}; TPS smoothing=0"
        ),
        fontsize=11,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Landmarks: green I1–I5 = upper-incisor/hard-palate; blue I6–I7 = soft palate; "
            "cyan C1–C6 = cervical; orange = P1/M1/L6. Circles = target; x = source/mapped source; arrows = residual."
        ),
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    raw_upper = contour_metrics(source_upper, target_upper)
    affine_upper = contour_metrics(affine_contours["upper-incisor"], target_upper)
    final_upper = contour_metrics(final_contours["upper-incisor"], target_upper)
    affine_i_mean, affine_i_max = landmark_metrics(
        affine_lm, target_lm, ("I1", "I2", "I3", "I4", "I5")
    )
    final_i_mean, final_i_max = landmark_metrics(
        final_lm, target_lm, ("I1", "I2", "I3", "I4", "I5")
    )
    diagnostics = experiment.prior_u.transform_diagnostics(
        transform, source["grid"], target["grid"]
    )
    step_dir = output.with_name(f"{output.stem}_steps")
    stage_specs = (
        (
            "01_source_native.png",
            source["image"],
            "Step 1 — ASD2 source F0499 exact /u/ in native coordinates",
            source_contours,
            source_lm,
            None,
            None,
            "No transform has been applied.",
        ),
        (
            "02_target_native.png",
            target["image"],
            f"Step 2 — P{speaker}/S{session} target F{target_frame:04d} exact /u/ in native coordinates",
            target_contours,
            target_lm,
            None,
            None,
            "This frame supplies the target landmarks for the one fixed session transform.",
        ),
        (
            "03_before_affine.png",
            target["image"],
            "Step 3 — Source over target before affine",
            source_contours,
            source_lm,
            target_contours,
            target_lm,
            (
                f"Upper curve geometric RMS={raw_upper['symmetric_nearest_curve_rms_mm']:.3f} mm; "
                f"centroid dy={raw_upper['centroid_dy_mm']:+.3f} mm. Arrows point from source controls to target controls."
            ),
        ),
        (
            "04_after_affine.png",
            target["image"],
            "Step 4 — After global least-squares affine",
            affine_contours,
            affine_lm,
            target_contours,
            target_lm,
            (
                f"All-control RMS={diagnostics['affine_control_rmse_px']:.3f} px; "
                f"I1–I5 mean={affine_i_mean:.3f} mm; upper curve RMS="
                f"{affine_upper['symmetric_nearest_curve_rms_mm']:.3f} mm; "
                f"centroid dy={affine_upper['centroid_dy_mm']:+.3f} mm."
            ),
        ),
        (
            "05_after_affine_tps.png",
            target["image"],
            "Step 5 — After affine + zero-smoothing TPS",
            final_contours,
            final_lm,
            target_contours,
            target_lm,
            (
                f"TPS-control RMS={diagnostics['tps_control_rmse_px']:.3e} px; "
                f"I1–I5 mean={final_i_mean:.3e} mm; upper curve RMS="
                f"{final_upper['symmetric_nearest_curve_rms_mm']:.3f} mm. "
                "TPS interpolates controls, not every point of the upper-incisor contour."
            ),
        ),
    )
    step_paths: list[Path] = []
    for (
        filename,
        stage_image,
        stage_title,
        stage_contours,
        stage_landmarks,
        stage_target_contours,
        stage_target_landmarks,
        stage_metric_text,
    ) in stage_specs:
        stage_path = step_dir / filename
        render_step_image(
            stage_path,
            image=stage_image,
            title=stage_title,
            candidate_contours=stage_contours,
            candidate_landmarks=stage_landmarks,
            target_contours=stage_target_contours,
            target_landmarks=stage_target_landmarks,
            metric_text=stage_metric_text,
        )
        step_paths.append(stage_path)

    return {
        "speaker": speaker,
        "session": session,
        "source_frame": 499,
        "target_frame": target_frame,
        "transform_scope": "one fixed transform for the session",
        "affine_controls": ",".join(transform["step1_labels"]),
        "tps_controls": ",".join(transform["step2_labels"]),
        "affine_control_rmse_px": diagnostics["affine_control_rmse_px"],
        "affine_control_max_px": diagnostics["affine_control_max_px"],
        "tps_control_rmse_px": diagnostics["tps_control_rmse_px"],
        "tps_control_max_px": diagnostics["tps_control_max_px"],
        "affine_I1_I5_mean_mm": affine_i_mean,
        "affine_I1_I5_max_mm": affine_i_max,
        "tps_I1_I5_mean_mm": final_i_mean,
        "tps_I1_I5_max_mm": final_i_max,
        **{f"upper_raw_{key}": value for key, value in raw_upper.items()},
        **{f"upper_affine_{key}": value for key, value in affine_upper.items()},
        **{f"upper_tps_{key}": value for key, value in final_upper.items()},
        "affine_A_json": json.dumps(affine_a.tolist()),
        "affine_t_json": json.dumps(affine_t.tolist()),
        "figure": str(output.resolve()),
        "step_images_json": json.dumps([str(path.resolve()) for path in step_paths]),
    }


def save_contact_sheet(paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    thumb_width = 1200
    thumbs = []
    for image in images:
        height = int(round(image.height * thumb_width / image.width))
        thumbs.append(image.resize((thumb_width, height), Image.Resampling.LANCZOS))
    cell_height = max(image.height for image in thumbs) + 34
    canvas = Image.new("RGB", (thumb_width * 3, cell_height * 3), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (path, image) in enumerate(zip(paths, thumbs)):
        x_coord = (index % 3) * thumb_width
        y_coord = (index // 3) * cell_height
        canvas.paste(image, (x_coord, y_coord + 28))
        if path.parent.name.endswith("_steps"):
            label = path.parent.name.split("_source_", maxsplit=1)[0].replace("_", "/", 1)
        else:
            label = path.stem
        draw.text((x_coord + 8, y_coord + 7), label, fill="black")
    canvas.save(output)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    affine_curve = np.asarray(
        [row["upper_affine_symmetric_nearest_curve_rms_mm"] for row in rows], dtype=float
    )
    affine_point = np.asarray(
        [row["upper_affine_coordinate_rmse_mm"] for row in rows], dtype=float
    )
    affine_dy = np.asarray([row["upper_affine_centroid_dy_mm"] for row in rows], dtype=float)
    affine_i = np.asarray([row["affine_I1_I5_mean_mm"] for row in rows], dtype=float)
    tps_curve = np.asarray(
        [row["upper_tps_symmetric_nearest_curve_rms_mm"] for row in rows], dtype=float
    )
    table = [
        "| Session | Target /u/ | Affine controls RMS (px) | Upper curve affine (mm) | Upper centroid dy (mm) | Upper curve TPS (mm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            "| P{speaker}/S{session} | F{target_frame:04d} | {affine_control_rmse_px:.3f} | "
            "{upper_affine_symmetric_nearest_curve_rms_mm:.3f} | "
            "{upper_affine_centroid_dy_mm:+.3f} | "
            "{upper_tps_symmetric_nearest_curve_rms_mm:.3f} |".format(**row)
        )
    content = f"""# Corrected TextGrid `/u/` transform-process diagnostic

## What the current experiment actually builds

The corrected experiment builds **one fixed transform per target session**:

`ASD2 1791/S14/F0499 /u/ -> selected target-session /u/ reference frame`

It does not rebuild affine/TPS for every `/u/` frame. The fixed transform is then reused for every prediction frame in that session. The figures show the exact construction sequence: source native, target native, untransformed source in target coordinates, global affine, and affine+TPS.

Affine is a six-parameter least-squares fit over I1-I7, optional P1, and C1-C6. Only I1-I5 come from the upper-incisor/hard-palate contour; I6-I7 are soft-palate landmarks. TPS adds M1 and L6 and uses zero smoothing.

## Upper-incisor diagnosis

- Mean affine upper-incisor geometric curve residual: `{affine_curve.mean():.3f} mm` (symmetric nearest-curve RMS; range `{affine_curve.min():.3f}-{affine_curve.max():.3f}`).
- Mean TPS upper-incisor geometric curve residual: `{tps_curve.mean():.3f} mm`.
- Mean affine I1-I5 landmark error: `{affine_i.mean():.3f} mm`.
- The affine-mapped upper-incisor centroid is below the target in `{int(np.sum(affine_dy > 0))}/9` sessions; mean image-coordinate vertical offset is `{affine_dy.mean():+.3f} mm` (positive means lower on the image).
- The much larger point-index coordinate RMSE (`{affine_point.mean():.3f} mm`) should not be interpreted as the visible curve separation. Source and target upper-incisor contours use different 50-point starting phase/parameterization, so point `i` is often not the same anatomical position even when the two curves are spatially close.
- Source upper incisors come from the ASD2 VTLN-incisor training overlay. Target incisors come from each ASD1 session's `contours.before-roi-import` fallback. Their shapes and point parameterizations are not identical. A global affine cannot make the whole contour coincide while simultaneously fitting palate/soft-palate/spine landmarks.

## Per-session summary

{chr(10).join(table)}

## Reading the figures

- Top row: all 11 contours. Solid curves are target annotations; dashed curves are the ASD2 source at the indicated stage.
- Bottom row: upper incisor isolated. Yellow is target, dashed red is source/mapped source.
- Target landmarks are circles; source or mapped-source landmarks are crosses. Arrows show residual vectors to the target.
- Green I1-I5 are the only affine controls derived from upper-incisor/hard-palate. Blue I6-I7 belong to soft palate, cyan C1-C6 to cervical anatomy, and orange points are P1/M1/L6.
- Every session has a sibling `_steps/` directory containing five high-resolution debug images. Each image has a full-contour panel and a zoomed upper-incisor/landmark panel.
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    selection_path = args.input_root / "provenance/textgrid_u_selections.json"
    selections = json.loads(selection_path.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_root / "sessions"
    figure_dir.mkdir(parents=True, exist_ok=True)

    source_frame = int(selections["asd2_source"]["selected_frame"])
    if source_frame != 499:
        raise RuntimeError(f"Expected corrected ASD2 source F0499, found F{source_frame:04d}")
    source = experiment.prior_u.load_source_reference(source_frame)

    rows: list[dict[str, Any]] = []
    figures: list[Path] = []
    stage_figures: dict[str, list[Path]] = {
        "01_source_native": [],
        "02_target_native": [],
        "03_before_affine": [],
        "04_after_affine": [],
        "05_after_affine_tps": [],
    }
    for speaker, session in experiment.SELECTION:
        key = f"target_P{speaker}_S{session}"
        target_frame = int(selections[key]["selected_frame"])
        target = prepare_frame(
            FrameSpec(
                f"P{speaker}",
                f"S{session}",
                f"{target_frame:04d}",
                experiment.asd2_core.target_spec(speaker).vtln_anchor,
            ),
            experiment.asd2_core.DEFAULT_VTLN_DIR,
        )
        transform = build_two_step_transform(source["grid"], target["grid"])
        figure = figure_dir / (
            f"P{speaker}_S{session}_F{target_frame:04d}_source_F0499_transform_process.png"
        )
        row = render_session(
            figure, source, target, transform, speaker, session, target_frame
        )
        rows.append(row)
        for step_path_string in json.loads(row["step_images_json"]):
            step_path = Path(step_path_string)
            stage_figures[step_path.stem].append(step_path)
        figures.append(figure)

    write_csv(args.output_root / "transform_process_metrics.csv", rows)
    save_contact_sheet(figures, args.output_root / "contact_sheet_all_9_sessions.png")
    stage_contact_dir = args.output_root / "step_contact_sheets"
    stage_contact_dir.mkdir(parents=True, exist_ok=True)
    stage_contact_sheets = []
    for stage_name, paths in stage_figures.items():
        contact_sheet = stage_contact_dir / f"{stage_name}_all_9_sessions.png"
        save_contact_sheet(paths, contact_sheet)
        stage_contact_sheets.append(contact_sheet)
    write_report(args.output_root / "report.md", rows)
    manifest = {
        "input_corrected_root": str(args.input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "source": "ASD2 1791/S14/F0499 exact TextGrid /u/",
        "transform_scope": "one fixed affine+TPS transform per target session",
        "session_count": len(rows),
        "figures": [str(path.resolve()) for path in figures],
        "separate_step_figure_count": sum(len(paths) for paths in stage_figures.values()),
        "step_contact_sheets": [str(path.resolve()) for path in stage_contact_sheets],
        "metrics_csv": str((args.output_root / "transform_process_metrics.csv").resolve()),
        "report": str((args.output_root / "report.md").resolve()),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
