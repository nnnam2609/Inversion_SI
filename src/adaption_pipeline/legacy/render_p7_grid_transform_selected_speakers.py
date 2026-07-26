#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
GRID_ROOT = REPO_ROOT / "external" / "grid-transform"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(GRID_ROOT))

from .evaluate_grid_normalization import (  # noqa: E402
    BF_TO_GRID_LABEL,
    DEFAULT_VTLN_RELEASE_DIR,
    load_asd1_mri,
    load_vtln_c_contours,
)
from grid_transform.transfer import build_two_step_transform  # noqa: E402
from grid_transform.transform_helpers import (  # noqa: E402
    apply_transform,
    compute_grid_line_errors,
    compute_metrics,
    extract_true_landmarks,
    map_landmarks,
)
from grid_transform.vt import build_grid  # noqa: E402
from grid_transform.warp import warp_image_to_target_space  # noqa: E402
from src.utils.colors import COLORS  # noqa: E402


CLASSES = (
    "arytenoid-cartilage",
    "epiglottis",
    "lower-lip",
    "pharynx",
    "soft-palate-midline",
    "tongue",
    "upper-lip",
    "vocal-folds",
    "thyroid-cartilage",
    "lower-incisor",
    "upper-incisor",
)


@dataclass(frozen=True)
class FrameSpec:
    speaker: str
    session: str
    frame: str
    vtln_anchor: str

    @property
    def label(self) -> str:
        return f"{self.speaker}/{self.session}/F{self.frame}"

    @property
    def slug(self) -> str:
        return f"{self.speaker.lower()}_{self.session.lower()}_f{self.frame}"


SOURCE = FrameSpec("P7", "S2", "0828", "1640_P7_S2_F0829")
TARGETS = (
    FrameSpec("P1", "S16", "0954", "1612_P1_S16_F0952"),
    FrameSpec("P2", "S9", "1477", "1617_P2_S9_F1478"),
    FrameSpec("P3", "S14", "1553", "1618_P3_S14_F1556"),
    FrameSpec("P4", "S4", "0195", "1628_P4_S4_F0196"),
    FrameSpec("P5", "S6", "0322", "1635_P5_S6_F0324"),
    FrameSpec("P6", "S8", "0137", "1638_P6_S8_F0138"),
    FrameSpec("P8", "S2", "0153", "1653_P8_S2_F0159"),
    FrameSpec("P9", "S5", "0195", "1659_P9_S5_F0196"),
    FrameSpec("P10", "S14", "0107", "1662_P10_S14_F0110"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render P7 source-grid transfer stages for the selected /u/ frames."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/p7_grid_transform_selected_speakers_20260718",
    )
    parser.add_argument(
        "--vtln-dir",
        type=Path,
        default=(
            DEFAULT_VTLN_RELEASE_DIR
            if DEFAULT_VTLN_RELEASE_DIR.is_dir()
            else GRID_ROOT / "VTLN" / "data"
        ),
    )
    return parser.parse_args()


def contour_path(spec: FrameSpec, class_name: str) -> Path:
    session_root = WORKSPACE_ROOT / "bf" / "inference" / spec.speaker / spec.session
    primary = session_root / "contours" / f"{spec.frame}_{class_name}.npy"
    if primary.is_file():
        return primary
    original_incisor = (
        session_root / "contours.before-roi-import" / f"{spec.frame}_{class_name}.npy"
    )
    if original_incisor.is_file():
        return original_incisor
    raise FileNotFoundError(f"Missing annotation contour: {primary}")


def load_annotations(spec: FrameSpec) -> dict[str, np.ndarray]:
    contours: dict[str, np.ndarray] = {}
    for class_name in CLASSES:
        array = np.load(contour_path(spec, class_name)).astype(np.float64)
        if array.shape != (50, 2) or not np.isfinite(array).all():
            raise ValueError(
                f"Expected finite (50,2) contour for {spec.label}/{class_name}, got {array.shape}"
            )
        contours[class_name] = array
    return contours


def grid_contours(
    annotations: dict[str, np.ndarray],
    spec: FrameSpec,
    vtln_dir: Path,
    image_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    contours = {
        grid_name: annotations[bf_name].copy()
        for bf_name, grid_name in BF_TO_GRID_LABEL.items()
    }
    contours.update(load_vtln_c_contours(spec.vtln_anchor, vtln_dir, image_shape))
    return contours


def prepare_frame(spec: FrameSpec, vtln_dir: Path) -> dict[str, Any]:
    image = load_asd1_mri(spec.speaker, spec.session, spec.frame, (136, 136))
    annotations = load_annotations(spec)
    contours_for_grid = grid_contours(annotations, spec, vtln_dir, image.shape[:2])
    grid = build_grid(
        image,
        contours_for_grid,
        n_vert=9,
        n_points=250,
        frame_number=int(spec.frame),
    )
    return {
        "spec": spec,
        "image": image,
        "annotations": annotations,
        "grid_contours": contours_for_grid,
        "grid": grid,
    }


def inverse_affine(transform: dict[str, Any]) -> dict[str, Any]:
    inverse_a = np.linalg.inv(np.asarray(transform["A"], dtype=float))
    inverse_t = -inverse_a @ np.asarray(transform["t"], dtype=float)
    return {"A": inverse_a, "t": inverse_t, "type": "affine"}


def map_annotations(
    contours: dict[str, np.ndarray], mapping: Callable[[np.ndarray], np.ndarray]
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(mapping(points), dtype=np.float64)
        for name, points in contours.items()
    }


def map_grid(grid: Any, mapping: Callable[[np.ndarray], np.ndarray]) -> dict[str, Any]:
    return {
        "horiz_lines": [np.asarray(mapping(line), dtype=np.float64) for line in grid.horiz_lines],
        "vert_lines": [np.asarray(mapping(line), dtype=np.float64) for line in grid.vert_lines],
        "spine_curve": (
            None
            if grid.spine_curve is None
            else np.asarray(mapping(grid.spine_curve), dtype=np.float64)
        ),
        "vt_curve": (
            None
            if grid.vt_curve is None
            else np.asarray(mapping(grid.vt_curve), dtype=np.float64)
        ),
    }


def native_grid(grid: Any) -> dict[str, Any]:
    return map_grid(grid, lambda points: np.asarray(points, dtype=np.float64))


def draw_annotations(
    ax: plt.Axes,
    contours: dict[str, np.ndarray],
    *,
    linestyle: str = "-",
    linewidth: float = 1.25,
    alpha: float = 0.94,
) -> None:
    for name in CLASSES:
        points = contours[name]
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=COLORS.get(name, "white"),
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )


def draw_native_grid(ax: plt.Axes, payload: dict[str, Any], *, alpha: float = 0.85) -> None:
    for line in payload["horiz_lines"]:
        ax.plot(line[:, 0], line[:, 1], color="#ffe600", linewidth=0.75, alpha=alpha)
    for line in payload["vert_lines"]:
        ax.plot(line[:, 0], line[:, 1], color="#16d36b", linewidth=0.65, alpha=alpha)
    if payload["spine_curve"] is not None:
        line = payload["spine_curve"]
        ax.plot(line[:, 0], line[:, 1], color="#ff3344", linewidth=1.2, alpha=alpha)
    if payload["vt_curve"] is not None:
        line = payload["vt_curve"]
        ax.plot(line[:, 0], line[:, 1], color="#42a5ff", linewidth=1.0, alpha=alpha)


def draw_mapped_grid(
    ax: plt.Axes,
    payload: dict[str, Any],
    *,
    color: str = "#00e5ff",
    linestyle: str = "--",
    alpha: float = 0.9,
) -> None:
    for line in payload["horiz_lines"] + payload["vert_lines"]:
        ax.plot(
            line[:, 0],
            line[:, 1],
            color=color,
            linewidth=0.75,
            linestyle=linestyle,
            alpha=alpha,
        )
    for key in ("spine_curve", "vt_curve"):
        line = payload[key]
        if line is not None:
            ax.plot(
                line[:, 0],
                line[:, 1],
                color=color,
                linewidth=1.05,
                linestyle=linestyle,
                alpha=alpha,
            )


def format_axis(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_aspect("equal")
    ax.axis("off")


def render_native(
    ax: plt.Axes,
    image: np.ndarray,
    contours: dict[str, np.ndarray],
    grid_payload: dict[str, Any],
    title: str,
) -> None:
    format_axis(ax, image, title)
    draw_native_grid(ax, grid_payload)
    draw_annotations(ax, contours)


def render_warped(
    ax: plt.Axes,
    image: np.ndarray,
    contours: dict[str, np.ndarray],
    grid_payload: dict[str, Any],
    title: str,
) -> None:
    format_axis(ax, image, title)
    draw_mapped_grid(ax, grid_payload, color="#00e5ff", linestyle="-")
    draw_annotations(ax, contours, linestyle="--")


def render_overlay(
    ax: plt.Axes,
    target_image: np.ndarray,
    target_contours: dict[str, np.ndarray],
    target_grid: dict[str, Any],
    mapped_contours: dict[str, np.ndarray],
    mapped_grid: dict[str, Any],
    title: str,
) -> None:
    format_axis(ax, target_image, title)
    draw_native_grid(ax, target_grid, alpha=0.52)
    draw_mapped_grid(ax, mapped_grid, color="#00e5ff", linestyle="--", alpha=0.92)
    draw_annotations(ax, target_contours, linestyle="-", linewidth=1.25, alpha=0.86)
    draw_annotations(ax, mapped_contours, linestyle="--", linewidth=1.0, alpha=0.96)
    ax.text(
        0.015,
        0.985,
        "solid = target annotation/grid\ndashed/cyan = mapped P7",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        color="white",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.65},
    )


def save_panel(path: Path, renderer: Callable[[plt.Axes], None]) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    renderer(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_annotations(path: Path, contours: dict[str, np.ndarray]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name, points in contours.items():
        np.save(path / f"{name}.npy", np.asarray(points, dtype=np.float32))


def save_grid(path: Path, payload: dict[str, Any]) -> None:
    arrays: dict[str, np.ndarray] = {
        "horiz_lines": np.stack(payload["horiz_lines"]).astype(np.float32),
        "vert_lines": np.stack(payload["vert_lines"]).astype(np.float32),
    }
    for key in ("spine_curve", "vt_curve"):
        if payload[key] is not None:
            arrays[key] = np.asarray(payload[key], dtype=np.float32)
    np.savez_compressed(path, **arrays)


def jsonable_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    return {
        key: (None if value is None else float(value))
        for key, value in metrics.items()
    }


def annotation_rmse(
    mapped: dict[str, np.ndarray], target: dict[str, np.ndarray]
) -> tuple[float, dict[str, float]]:
    per_class = {
        name: float(np.sqrt(np.mean((mapped[name] - target[name]) ** 2)))
        for name in CLASSES
    }
    joined_mapped = np.stack([mapped[name] for name in CLASSES])
    joined_target = np.stack([target[name] for name in CLASSES])
    overall = float(np.sqrt(np.mean((joined_mapped - joined_target) ** 2)))
    return overall, per_class


def grid_error_summary(
    mapped_grid: dict[str, Any], target_grid: Any
) -> dict[str, Any]:
    horizontal, vertical = compute_grid_line_errors(
        mapped_grid["horiz_lines"], mapped_grid["vert_lines"], target_grid
    )
    return {
        "horizontal_rms_px": horizontal,
        "vertical_rms_px": vertical,
        "horizontal_mean_px": float(np.mean(list(horizontal.values()))),
        "vertical_mean_px": float(np.mean(list(vertical.values()))),
    }


def process_target(
    source: dict[str, Any], target: dict[str, Any], output_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_spec: FrameSpec = source["spec"]
    target_spec: FrameSpec = target["spec"]
    output_dir = output_root / f"{source_spec.speaker}_to_{target_spec.speaker}"
    output_dir.mkdir(parents=True, exist_ok=True)

    forward = build_two_step_transform(source["grid"], target["grid"])
    reverse = build_two_step_transform(target["grid"], source["grid"])
    affine_inverse = inverse_affine(forward["step1_affine"])

    affine_map = lambda points: apply_transform(forward["step1_affine"], points)
    affine_inverse_map = lambda points: apply_transform(affine_inverse, points)
    final_map = forward["apply_two_step"]

    source_grid = native_grid(source["grid"])
    target_grid = native_grid(target["grid"])
    affine_grid = map_grid(source["grid"], affine_map)
    final_grid = map_grid(source["grid"], final_map)
    affine_contours = map_annotations(source["annotations"], affine_map)
    final_contours = map_annotations(source["annotations"], final_map)

    affine_image, affine_mask = warp_image_to_target_space(
        source["image"], target["image"].shape, affine_inverse_map
    )
    final_image, final_mask = warp_image_to_target_space(
        source["image"], target["image"].shape, reverse["apply_two_step"]
    )

    Image.fromarray(source["image"]).save(output_dir / "00_source_mri.png")
    Image.fromarray(target["image"]).save(output_dir / "01_target_mri.png")
    Image.fromarray(affine_image).save(output_dir / "02_affine_warped_mri.png")
    Image.fromarray(affine_mask).save(output_dir / "02_affine_valid_mask.png")
    Image.fromarray(final_image).save(output_dir / "03_affine_tps_warped_mri.png")
    Image.fromarray(final_mask).save(output_dir / "03_affine_tps_valid_mask.png")

    save_annotations(output_dir / "contours/source_native", source["annotations"])
    save_annotations(output_dir / "contours/target_native", target["annotations"])
    save_annotations(output_dir / "contours/after_affine", affine_contours)
    save_annotations(output_dir / "contours/after_affine_tps", final_contours)
    save_grid(output_dir / "grid_source_native.npz", source_grid)
    save_grid(output_dir / "grid_target_native.npz", target_grid)
    save_grid(output_dir / "grid_after_affine.npz", affine_grid)
    save_grid(output_dir / "grid_after_affine_tps.npz", final_grid)

    panels = {
        "source": output_dir / "panel_00_source_native.png",
        "target": output_dir / "panel_01_target_native.png",
        "affine_warped": output_dir / "panel_02_affine_warped.png",
        "affine_overlay": output_dir / "panel_03_affine_on_target.png",
        "final_warped": output_dir / "panel_04_affine_tps_warped.png",
        "final_overlay": output_dir / "panel_05_affine_tps_on_target.png",
    }
    save_panel(
        panels["source"],
        lambda ax: render_native(
            ax,
            source["image"],
            source["annotations"],
            source_grid,
            f"Source native: {source_spec.label}\nannotation + source grid",
        ),
    )
    save_panel(
        panels["target"],
        lambda ax: render_native(
            ax,
            target["image"],
            target["annotations"],
            target_grid,
            f"Target native: {target_spec.label}\nannotation + target grid",
        ),
    )
    save_panel(
        panels["affine_warped"],
        lambda ax: render_warped(
            ax,
            affine_image,
            affine_contours,
            affine_grid,
            "Step 1: affine-warped P7 MRI\naffine contours + affine source grid",
        ),
    )
    save_panel(
        panels["affine_overlay"],
        lambda ax: render_overlay(
            ax,
            target["image"],
            target["annotations"],
            target_grid,
            affine_contours,
            affine_grid,
            "Step 1 comparison on target MRI",
        ),
    )
    save_panel(
        panels["final_warped"],
        lambda ax: render_warped(
            ax,
            final_image,
            final_contours,
            final_grid,
            "Step 2: affine + TPS-warped P7 MRI\nfinal contours + final source grid",
        ),
    )
    save_panel(
        panels["final_overlay"],
        lambda ax: render_overlay(
            ax,
            target["image"],
            target["annotations"],
            target_grid,
            final_contours,
            final_grid,
            "Step 2 comparison on target MRI",
        ),
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=170)
    render_native(
        axes[0, 0],
        source["image"],
        source["annotations"],
        source_grid,
        f"Source: {source_spec.label}\nannotation + source grid",
    )
    render_native(
        axes[0, 1],
        target["image"],
        target["annotations"],
        target_grid,
        f"Target: {target_spec.label}\nannotation + target grid",
    )
    render_warped(
        axes[0, 2],
        affine_image,
        affine_contours,
        affine_grid,
        "Step 1: affine-warped P7\nMRI + contours + grid",
    )
    render_overlay(
        axes[1, 0],
        target["image"],
        target["annotations"],
        target_grid,
        affine_contours,
        affine_grid,
        "Step 1: affine P7 vs target",
    )
    render_warped(
        axes[1, 1],
        final_image,
        final_contours,
        final_grid,
        "Step 2: affine + TPS-warped P7\nMRI + contours + grid",
    )
    render_overlay(
        axes[1, 2],
        target["image"],
        target["annotations"],
        target_grid,
        final_contours,
        final_grid,
        "Step 2: affine + TPS P7 vs target",
    )
    fig.suptitle(
        f"P7 anatomical-grid transfer: {source_spec.label} → {target_spec.label}",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    contact_sheet = output_dir / "all_steps_contact_sheet.png"
    fig.savefig(contact_sheet, bbox_inches="tight")
    plt.close(fig)

    source_landmarks = extract_true_landmarks(source["grid"])
    target_landmarks = extract_true_landmarks(target["grid"])
    affine_landmarks = map_landmarks(affine_map, source_landmarks)
    final_landmarks = map_landmarks(final_map, source_landmarks)
    affine_annotation_rmse, affine_per_class = annotation_rmse(
        affine_contours, target["annotations"]
    )
    final_annotation_rmse, final_per_class = annotation_rmse(
        final_contours, target["annotations"]
    )
    summary = {
        "source": {
            "speaker": source_spec.speaker,
            "session": source_spec.session,
            "dynamic_annotation_frame": source_spec.frame,
            "vtln_c_anchor": source_spec.vtln_anchor,
        },
        "target": {
            "speaker": target_spec.speaker,
            "session": target_spec.session,
            "dynamic_annotation_frame": target_spec.frame,
            "vtln_c_anchor": target_spec.vtln_anchor,
        },
        "coordinate_space": "136x136",
        "step1": "full 2D affine fitted from anatomical-axis landmarks",
        "step2": "thin-plate spline fitted after affine",
        "step1_labels": list(forward["step1_labels"]),
        "step2_labels": list(forward["step2_labels"]),
        "affine_A": np.asarray(forward["step1_affine"]["A"]).tolist(),
        "affine_t": np.asarray(forward["step1_affine"]["t"]).tolist(),
        "affine_landmark_metrics_px": jsonable_metrics(
            compute_metrics(affine_landmarks, target_landmarks)
        ),
        "affine_tps_landmark_metrics_px": jsonable_metrics(
            compute_metrics(final_landmarks, target_landmarks)
        ),
        "affine_grid_errors": grid_error_summary(affine_grid, target["grid"]),
        "affine_tps_grid_errors": grid_error_summary(final_grid, target["grid"]),
        "affine_annotation_rmse_px": affine_annotation_rmse,
        "affine_tps_annotation_rmse_px": final_annotation_rmse,
        "affine_per_class_annotation_rmse_px": affine_per_class,
        "affine_tps_per_class_annotation_rmse_px": final_per_class,
        "affine_valid_image_fraction": float(np.mean(affine_mask > 0)),
        "affine_tps_valid_image_fraction": float(np.mean(final_mask > 0)),
        "contact_sheet": str(contact_sheet),
        "panels": {name: str(path) for name, path in panels.items()},
        "saved_contour_stages": [
            "source_native",
            "target_native",
            "after_affine",
            "after_affine_tps",
        ],
        "saved_grid_stages": [
            "grid_source_native.npz",
            "grid_target_native.npz",
            "grid_after_affine.npz",
            "grid_after_affine_tps.npz",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    aggregate_item = {
        "target": target_spec.label,
        "target_speaker": target_spec.speaker,
        "target_session": target_spec.session,
        "target_frame": target_spec.frame,
        "contact_sheet": str(contact_sheet),
        "affine_annotation_rmse_px": affine_annotation_rmse,
        "affine_tps_annotation_rmse_px": final_annotation_rmse,
        "affine_horizontal_grid_mean_px": summary["affine_grid_errors"][
            "horizontal_mean_px"
        ],
        "affine_tps_horizontal_grid_mean_px": summary["affine_tps_grid_errors"][
            "horizontal_mean_px"
        ],
        "affine_vertical_grid_mean_px": summary["affine_grid_errors"][
            "vertical_mean_px"
        ],
        "affine_tps_vertical_grid_mean_px": summary["affine_tps_grid_errors"][
            "vertical_mean_px"
        ],
        "affine_valid_image_fraction": summary["affine_valid_image_fraction"],
        "affine_tps_valid_image_fraction": summary[
            "affine_tps_valid_image_fraction"
        ],
    }
    aggregate_visual = {
        "spec": target_spec,
        "image": target["image"],
        "target_contours": target["annotations"],
        "target_grid": target_grid,
        "mapped_contours": final_contours,
        "mapped_grid": final_grid,
    }
    return aggregate_item, aggregate_visual


def main() -> None:
    args = parse_args()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source = prepare_frame(SOURCE, args.vtln_dir)
    aggregate: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    for index, target_spec in enumerate(TARGETS, start=1):
        print(f"[{index}/{len(TARGETS)}] {SOURCE.label} -> {target_spec.label}", flush=True)
        target = prepare_frame(target_spec, args.vtln_dir)
        item, visual = process_target(source, target, output_root)
        aggregate.append(item)
        visuals.append(visual)

    fig, axes = plt.subplots(3, 3, figsize=(18, 18), dpi=170)
    for ax, visual in zip(axes.ravel(), visuals):
        spec: FrameSpec = visual["spec"]
        render_overlay(
            ax,
            visual["image"],
            visual["target_contours"],
            visual["target_grid"],
            visual["mapped_contours"],
            visual["mapped_grid"],
            f"{SOURCE.speaker} → {spec.speaker}: affine + TPS\n{spec.session} F{spec.frame}",
        )
    fig.suptitle(
        f"P7/S2 F{SOURCE.frame} affine + TPS transfer to selected /u/ speakers",
        fontsize=19,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    aggregate_image = output_root / "all_targets_affine_tps_overlay.png"
    fig.savefig(aggregate_image, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "speaker": SOURCE.speaker,
            "session": SOURCE.session,
            "dynamic_annotation_frame": SOURCE.frame,
            "vtln_c_anchor": SOURCE.vtln_anchor,
        },
        "target_count": len(aggregate),
        "targets": aggregate,
        "aggregate_affine_tps_overlay": str(aggregate_image),
        "contents_per_target": {
            "native_stages": ["source", "target"],
            "transform_stages": ["affine", "affine_tps"],
            "saved_mri": True,
            "saved_valid_masks": True,
            "saved_contours": True,
            "saved_grids": True,
            "individual_panels": 6,
            "contact_sheet": True,
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_fields = [
        "target",
        "target_speaker",
        "target_session",
        "target_frame",
        "affine_annotation_rmse_px",
        "affine_tps_annotation_rmse_px",
        "affine_tps_annotation_delta_px",
        "affine_horizontal_grid_mean_px",
        "affine_tps_horizontal_grid_mean_px",
        "affine_vertical_grid_mean_px",
        "affine_tps_vertical_grid_mean_px",
        "affine_valid_image_fraction",
        "affine_tps_valid_image_fraction",
        "contact_sheet",
    ]
    with (output_root / "transform_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for item in aggregate:
            writer.writerow(
                {
                    **item,
                    "affine_tps_annotation_delta_px": (
                        item["affine_tps_annotation_rmse_px"]
                        - item["affine_annotation_rmse_px"]
                    ),
                }
            )

    readme_lines = [
        "# P7 grid transfer to selected speakers",
        "",
        (
            f"Source: `{SOURCE.label}` dynamic BF annotations with fixed VTLN "
            f"C1-C6 from `{SOURCE.vtln_anchor}`."
        ),
        "",
        "Each target directory contains:",
        "",
        "- native source and target MRI, annotations, and anatomical grids;",
        "- affine-warped P7 MRI, valid mask, contours, and grid;",
        "- affine+TPS-warped P7 MRI, valid mask, contours, and grid;",
        "- six individual stage panels and one all-steps contact sheet;",
        "- contour arrays as finite float32 `(50, 2)` NPY files;",
        "- grid curves as compressed NPZ files;",
        "- transform parameters, landmark metrics, grid metrics, and per-class contour RMSE in `summary.json`.",
        "",
        "In comparison panels, solid colored contours and the yellow/green grid are the target. Dashed colored contours and the cyan dashed grid are mapped P7.",
        "",
        "TPS is fitted to anatomical-grid landmarks after affine; it is not fitted directly to all 11 contour RMSE values. A positive delta can therefore occur for non-grid annotations.",
        "",
        "| Target | Affine annotation RMSE px | Affine+TPS RMSE px | TPS delta px | Contact sheet |",
        "|---|---:|---:|---:|---|",
    ]
    for item in aggregate:
        delta = (
            item["affine_tps_annotation_rmse_px"]
            - item["affine_annotation_rmse_px"]
        )
        contact = Path(item["contact_sheet"])
        relative_contact = contact.relative_to(output_root)
        readme_lines.append(
            f"| {item['target']} | {item['affine_annotation_rmse_px']:.3f} | "
            f"{item['affine_tps_annotation_rmse_px']:.3f} | {delta:+.3f} | "
            f"[view]({relative_contact.as_posix()}) |"
        )
    readme_lines.extend(
        [
            "",
            "Negative TPS delta means the second step reduced whole-annotation RMSE.",
            "",
            f"[All-target affine+TPS overlay]({aggregate_image.name})",
            "",
        ]
    )
    (output_root / "README.md").write_text(
        "\n".join(readme_lines), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
