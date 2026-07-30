"""Render the required 2x4 anatomical-normalization diagnostic panels."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..contracts import ContractError, atomic_write_json, load_cohort
from ..io import load_mapping
from ..runtime import REPOSITORY_ROOT
from ..adapters.legacy_runtime import LegacyAnatomyAdapter


anatomy_core = LegacyAnatomyAdapter()


SOURCE_PACK = (
    REPOSITORY_ROOT
    / "cache_variants/asd2_11_bfincisor_sofiane153_s25_20260725/"
    "raw_contour_npz/asd2/1791/S14.npz"
)
P7_VTLN_ANCHOR = "1640_P7_S2_F0829"
MM_PER_PIXEL = 1.62


def target_anchor(speaker: int) -> str:
    if speaker == 7:
        return P7_VTLN_ANCHOR
    return anatomy_core.target_anchor(speaker, p7_anchor=P7_VTLN_ANCHOR)


def annotations(payload: Mapping[str, Any]) -> np.ndarray:
    return np.stack(
        [
            np.asarray(payload["annotations"][name], dtype=np.float64).reshape(50, 2)
            for name in anatomy_core.classes
        ]
    )


def rmse_mm(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ContractError(f"Annotation shape mismatch: {left.shape} != {right.shape}")
    return float(np.sqrt(np.mean(np.square(left - right))) * MM_PER_PIXEL)


def landmark_map(grid: Any) -> Dict[str, np.ndarray]:
    values = {name: np.asarray(point, dtype=float) for name, point in grid.I_points.items()}
    values["P1"] = np.asarray(grid.P1_point, dtype=float)
    values["M1"] = np.asarray(grid.M1_point, dtype=float)
    values["L6"] = np.asarray(grid.L6, dtype=float)
    values.update(
        {
            name.upper(): np.asarray(point, dtype=float)
            for name, point in grid.cervical_centers.items()
        }
    )
    return values


def draw_annotations(axis: Any, image: np.ndarray, values: np.ndarray, title: str) -> None:
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    colors = plt.cm.tab20(np.linspace(0, 1, len(values)))
    for index, contour in enumerate(values):
        axis.plot(contour[:, 0], contour[:, 1], color=colors[index], linewidth=0.75)
    axis.set_title(title, fontsize=9)
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.axis("off")


def draw_landmarks(
    axis: Any,
    image: np.ndarray,
    points: Mapping[str, np.ndarray],
    affine_labels: Iterable[str],
    tps_labels: Iterable[str],
    title: str,
) -> None:
    affine = set(affine_labels)
    tps = set(tps_labels)
    categories = {
        "both": (affine & tps, "#2b8cbe", "o"),
        "affine only": (affine - tps, "#fdae6b", "s"),
        "TPS only": (tps - affine, "#d7301f", "*"),
    }
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    for label, (names, color, marker) in categories.items():
        available = [name for name in sorted(names) if name in points]
        if not available:
            axis.scatter([], [], color=color, marker=marker, label=f"{label}: none")
            continue
        coordinates = np.stack([points[name] for name in available])
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            color=color,
            marker=marker,
            s=22 if marker != "*" else 45,
            label=label,
        )
        for name, point in zip(available, coordinates):
            axis.text(point[0] + 1, point[1] - 1, name, color=color, fontsize=5)
    axis.set_title(title, fontsize=9)
    axis.legend(loc="lower right", fontsize=5, framealpha=0.75)
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.axis("off")


def draw_grid(axis: Any, grid: Any, title: str, control_error: str) -> None:
    axis.imshow(grid.image, cmap="gray", vmin=0, vmax=255)
    for line in grid.horiz_lines:
        axis.plot(line[:, 0], line[:, 1], color="#00bcd4", linewidth=0.65)
    for line in grid.vert_lines:
        axis.plot(line[:, 0], line[:, 1], color="#ffca28", linewidth=0.65)
    axis.set_title(f"{title}\n{control_error}", fontsize=8)
    axis.set_xlim(0, grid.image.shape[1])
    axis.set_ylim(grid.image.shape[0], 0)
    axis.axis("off")


def draw_result(
    axis: Any,
    image: np.ndarray,
    predicted: np.ndarray,
    target: np.ndarray,
    title: str,
    error_mm: float,
) -> None:
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    colors = plt.cm.tab20(np.linspace(0, 1, len(predicted)))
    for index, (candidate, truth) in enumerate(zip(predicted, target)):
        axis.plot(
            truth[:, 0],
            truth[:, 1],
            color=colors[index],
            linewidth=0.6,
            linestyle="--",
            alpha=0.55,
        )
        axis.plot(
            candidate[:, 0],
            candidate[:, 1],
            color=colors[index],
            linewidth=0.9,
        )
    axis.set_title(f"{title}\nannotation RMSE={error_mm:.3f} mm", fontsize=8)
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.axis("off")


def render_pair(
    *,
    speaker: int,
    session: int,
    target_frame: int,
    output: Path,
) -> Dict[str, Any]:
    bundle = anatomy_core.build_transform_bundle(
        source_pack=SOURCE_PACK,
        speaker=speaker,
        session=session,
        target_frame=target_frame,
        target_anchor=target_anchor(speaker),
    )
    source = bundle.source
    target = bundle.target
    transform = bundle.transform
    diagnostics = bundle.diagnostics
    source_annotations = annotations(source)
    target_annotations = annotations(target)
    flat = source_annotations.reshape(-1, 2)
    affine_annotations = anatomy_core.apply_affine(transform, flat).reshape(
        source_annotations.shape
    )
    final_annotations = np.asarray(transform["apply_two_step"](flat)).reshape(
        source_annotations.shape
    )
    errors = {
        "annotation_raw_mm": rmse_mm(source_annotations, target_annotations),
        "annotation_affine_mm": rmse_mm(affine_annotations, target_annotations),
        "annotation_affine_tps_mm": rmse_mm(final_annotations, target_annotations),
        "affine_control_rmse_mm": float(diagnostics["affine_control_rmse_px"])
        * MM_PER_PIXEL,
        "tps_control_rmse_mm": float(diagnostics["tps_control_rmse_px"])
        * MM_PER_PIXEL,
    }
    affine_labels = list(diagnostics["step1_labels"])
    tps_labels = list(diagnostics["step2_labels"])
    source_points = landmark_map(source["grid"])
    target_points = landmark_map(target["grid"])

    figure, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=180)
    draw_annotations(
        axes[0, 0],
        source["image"],
        source_annotations,
        f"ASD2 1791/S14 F0499 /u/\nannotation; raw pair RMSE={errors['annotation_raw_mm']:.3f} mm",
    )
    draw_annotations(
        axes[1, 0],
        target["image"],
        target_annotations,
        f"P{speaker}/S{session} F{target_frame:04d} /u/\ntarget annotation",
    )
    draw_landmarks(
        axes[0, 1],
        source["image"],
        source_points,
        affine_labels,
        tps_labels,
        "Source selected landmarks",
    )
    draw_landmarks(
        axes[1, 1],
        target["image"],
        target_points,
        affine_labels,
        tps_labels,
        "Target selected landmarks",
    )
    draw_grid(
        axes[0, 2],
        source["grid"],
        "Source grid",
        f"affine control RMSE={errors['affine_control_rmse_mm']:.3f} mm",
    )
    draw_grid(
        axes[1, 2],
        target["grid"],
        "Target grid",
        f"TPS control RMSE={errors['tps_control_rmse_mm']:.3e} mm",
    )
    draw_result(
        axes[0, 3],
        target["image"],
        affine_annotations,
        target_annotations,
        "After affine (solid) vs target (dashed)",
        errors["annotation_affine_mm"],
    )
    draw_result(
        axes[1, 3],
        target["image"],
        final_annotations,
        target_annotations,
        "After affine+TPS (solid) vs target (dashed)",
        errors["annotation_affine_tps_mm"],
    )
    figure.suptitle(
        f"ASD2 reference → P{speaker}/S{session}: exact-/u/ anatomical normalization",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return {
        "speaker": f"P{speaker}",
        "session": f"S{session}",
        "source_frame": 499,
        "target_frame": target_frame,
        "vowel": "u",
        "panel_layout": "2x4",
        "affine_and_tps_landmarks": sorted(set(affine_labels) & set(tps_labels)),
        "affine_only_landmarks": sorted(set(affine_labels) - set(tps_labels)),
        "tps_only_landmarks": sorted(set(tps_labels) - set(affine_labels)),
        **errors,
        "figure": str(output.resolve()),
    }


def run(args: Any) -> int:
    pipeline = load_mapping(args.pipeline_config)
    cohort = load_cohort(pipeline["cohort"])
    cohort.validate()
    selected = set(args.speaker or [])
    rows = []
    for record in cohort.ordered_sessions():
        if selected and record.speaker not in selected:
            continue
        speaker = int(record.speaker.removeprefix("P"))
        session = int(record.session.removeprefix("S"))
        destination = (
            args.output_root.resolve()
            / record.speaker
            / record.session
            / "anatomical_normalization_8panel.png"
        )
        row = render_pair(
            speaker=speaker,
            session=session,
            target_frame=record.reference_frame,
            output=destination,
        )
        rows.append(row)
        atomic_write_json(destination.with_suffix(".json"), row)
        print(
            f"DONE {record.key}: affine={row['annotation_affine_mm']:.3f}, "
            f"affine+TPS={row['annotation_affine_tps_mm']:.3f} mm",
            flush=True,
        )
    if not rows:
        raise ContractError("Speaker filter selected no anatomical pairs")
    summary = args.output_root.resolve() / "anatomical_diagnostics.json"
    atomic_write_json(
        summary,
        {
            "status": "complete",
            "layout": "2x4",
            "pairs": rows,
            "training_launched": False,
        },
    )
    csv_path = args.output_root.resolve() / "anatomical_diagnostics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "speaker",
                "session",
                "source_frame",
                "target_frame",
                "vowel",
                "annotation_raw_mm",
                "annotation_affine_mm",
                "annotation_affine_tps_mm",
                "affine_control_rmse_mm",
                "tps_control_rmse_mm",
                "figure",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    return 0
