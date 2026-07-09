#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
GRID_ROOT = REPO_ROOT / "external" / "grid-transform"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(GRID_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_grid_normalization import (  # noqa: E402
    DEFAULT_VTLN_RELEASE_DIR,
    MM_PER_PIXEL,
    PRIMARY_CLASSES,
    aggregate_payload,
    load_asd1_mri,
    load_source_target_grids,
    load_yaml,
    rmse,
)
from grid_transform.transfer import build_two_step_transform  # noqa: E402
from grid_transform.transform_helpers import apply_transform  # noqa: E402
from src.utils.colors import COLORS  # noqa: E402


DEFAULT_PAYLOAD = (
    REPO_ROOT
    / "results/p7_stdfloor01_p7s15_p2s1_prediction_videos_20260705_225553/p2_s1/eval/cached_session_predictions.pt"
)
DEFAULT_CONFIG = REPO_ROOT / "config/train_config/asd1_p7_stdfloor01_trainstats_p2_s1_unseen_eval_st5_mfcc.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results/p7_to_p2_grid_transform_one_image_20260705"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one image visualizing P7->P2 grid transform on one P2 frame.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-anchor", default="1640_P7_S2_F0829")
    parser.add_argument("--target-anchor", default="1617_P2_S9_F1478")
    parser.add_argument("--source-speaker", default="P7")
    parser.add_argument("--source-session", default="S2")
    parser.add_argument("--source-frame", default="0829")
    parser.add_argument("--target-speaker-name", default="P2")
    parser.add_argument("--target-session-name", default="S9")
    parser.add_argument("--target-frame", default="1478")
    parser.add_argument("--vtln-dir", type=Path, default=DEFAULT_VTLN_RELEASE_DIR)
    parser.add_argument("--target-speaker", type=int, default=2)
    parser.add_argument("--target-session", type=int, default=1)
    parser.add_argument("--select", choices=("best_tps_primary", "median_tps_primary", "worst_tps_primary"), default="best_tps_primary")
    parser.add_argument("--frame", default=None, help="Optional frame token, e.g. 0219. Integer frames are preferred for MRI display.")
    parser.add_argument("--phoneme", default=None, help="Optional phoneme label filter, e.g. a, i, u, o, E/.")
    return parser.parse_args()


def frame_sort_key(token: str) -> float:
    if "p" in token:
        whole, frac = token.split("p", 1)
        return float(int(whole)) + float(int(frac)) / 10.0
    return float(int(token))


def transform_prediction(predicted: np.ndarray, transform: dict[str, Any], mode: str) -> np.ndarray:
    points = predicted.reshape(-1, 2)
    if mode == "raw":
        mapped = points
    elif mode == "affine":
        mapped = apply_transform(transform["step1_affine"], points)
    elif mode == "affine_tps":
        mapped = transform["apply_two_step"](points)
    else:
        raise ValueError(f"Unsupported transform mode: {mode}")
    return np.asarray(mapped, dtype=np.float32).reshape(predicted.shape)


def metric_summary(row: dict[str, Any], class_names: list[str], key: str) -> dict[str, float]:
    primary_indices = [idx for idx, name in enumerate(class_names) if name in PRIMARY_CLASSES]
    return {
        "overall_px": rmse(row[key], row["labels"]),
        "overall_mm": rmse(row[key], row["labels"]) * MM_PER_PIXEL,
        "primary_px": rmse(row[key][primary_indices], row["labels"][primary_indices]),
        "primary_mm": rmse(row[key][primary_indices], row["labels"][primary_indices]) * MM_PER_PIXEL,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "speaker",
        "session",
        "frame",
        "phoneme",
        "raw_overall_mm",
        "affine_overall_mm",
        "affine_tps_overall_mm",
        "affine_delta_mm",
        "affine_tps_delta_mm",
        "raw_primary_mm",
        "affine_primary_mm",
        "affine_tps_primary_mm",
        "affine_primary_delta_mm",
        "affine_tps_primary_delta_mm",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(np.mean([float(value) for value in values])) if values else float("nan")


def choose_row(rows: list[dict[str, Any]], requested_frame: str | None, select: str, phoneme: str | None) -> dict[str, Any]:
    candidates = rows
    if phoneme is not None:
        candidates = [row for row in candidates if row["phoneme"] == phoneme]
        if not candidates:
            available = sorted({row["phoneme"] for row in rows})
            raise ValueError(f"Phoneme {phoneme!r} was not found. Available phonemes: {available}")
    if requested_frame is not None:
        for row in candidates:
            if row["frame"] == requested_frame:
                return row
        raise ValueError(f"Requested frame {requested_frame} was not found in selected rows")
    integer_rows = [row for row in candidates if "p" not in row["frame"]]
    candidates = integer_rows or candidates
    ranked = sorted(candidates, key=lambda row: row["affine_tps_primary_delta_mm"])
    if select == "best_tps_primary":
        return ranked[0]
    if select == "worst_tps_primary":
        return ranked[-1]
    return ranked[len(ranked) // 2]


def plot_contours(ax, image: np.ndarray, row: dict[str, Any], class_names: list[str], pred_key: str, title: str, metrics: dict[str, float]) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    for idx, name in enumerate(class_names):
        color = COLORS.get(name, "white")
        label = row["labels"][idx].reshape(50, 2)
        pred = row[pred_key][idx].reshape(50, 2)
        ax.plot(label[:, 0], label[:, 1], color=color, linewidth=1.4, linestyle="-", alpha=0.9)
        ax.plot(pred[:, 0], pred[:, 1], color=color, linewidth=1.2, linestyle="--", alpha=0.95)
    ax.set_title(
        f"{title}\nall={metrics['overall_mm']:.3f}mm primary={metrics['primary_mm']:.3f}mm",
        fontsize=10,
    )
    ax.set_xlim(0, 136)
    ax.set_ylim(136, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def add_image_panel(ax, path: Path, title: str) -> None:
    image = Image.open(path)
    ax.imshow(image)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def write_contact_sheet(
    output_path: Path,
    source_grid_png: Path,
    target_grid_png: Path,
    image: np.ndarray,
    row: dict[str, Any],
    class_names: list[str],
    session_metrics: dict[str, float],
    phoneme_metrics: dict[str, float] | None,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=160)
    add_image_panel(axes[0, 0], source_grid_png, "P7 source grid\nBF contours + VTLN C1-C6")
    add_image_panel(axes[0, 1], target_grid_png, "P2 target grid\nBF contours + VTLN C1-C6")
    plot_contours(axes[0, 2], image, row, class_names, "predicted", "Raw P7-model pred vs P2 GT", row["raw_metrics"])
    plot_contours(axes[1, 0], image, row, class_names, "affine_predicted", "Affine only", row["affine_metrics"])
    plot_contours(axes[1, 1], image, row, class_names, "affine_tps_predicted", "Affine + TPS", row["affine_tps_metrics"])
    ax = axes[1, 2]
    ax.axis("off")
    lines = [
        f"Selected P{row['speaker']}/S{row['session']} frame {row['frame']}",
        f"phoneme: {row['phoneme']}",
        "",
        "Selected-frame deltas (negative = better):",
        f"affine overall: {row['affine_delta_mm']:+.3f} mm",
        f"affine+TPS overall: {row['affine_tps_delta_mm']:+.3f} mm",
        f"affine primary: {row['affine_primary_delta_mm']:+.3f} mm",
        f"affine+TPS primary: {row['affine_tps_primary_delta_mm']:+.3f} mm",
        "",
        "Whole P2/S1 session mean:",
        f"raw overall: {session_metrics['raw_overall_mm']:.3f} mm",
        f"affine overall: {session_metrics['affine_overall_mm']:.3f} mm ({session_metrics['affine_delta_mm']:+.3f})",
        f"affine+TPS overall: {session_metrics['affine_tps_overall_mm']:.3f} mm ({session_metrics['affine_tps_delta_mm']:+.3f})",
        f"raw primary: {session_metrics['raw_primary_mm']:.3f} mm",
        f"affine primary: {session_metrics['affine_primary_mm']:.3f} mm ({session_metrics['affine_primary_delta_mm']:+.3f})",
        f"affine+TPS primary: {session_metrics['affine_tps_primary_mm']:.3f} mm ({session_metrics['affine_tps_primary_delta_mm']:+.3f})",
    ]
    if phoneme_metrics is not None:
        lines.extend(
            [
                "",
                f"Selected phoneme mean ({row['phoneme']}, n={int(phoneme_metrics['count'])}):",
                f"raw overall: {phoneme_metrics['raw_overall_mm']:.3f} mm",
                f"affine overall: {phoneme_metrics['affine_overall_mm']:.3f} mm ({phoneme_metrics['affine_delta_mm']:+.3f})",
                f"affine+TPS overall: {phoneme_metrics['affine_tps_overall_mm']:.3f} mm ({phoneme_metrics['affine_tps_delta_mm']:+.3f})",
                f"raw primary: {phoneme_metrics['raw_primary_mm']:.3f} mm",
                f"affine primary: {phoneme_metrics['affine_primary_mm']:.3f} mm ({phoneme_metrics['affine_primary_delta_mm']:+.3f})",
                f"affine+TPS primary: {phoneme_metrics['affine_tps_primary_mm']:.3f} mm ({phoneme_metrics['affine_tps_primary_delta_mm']:+.3f})",
            ]
        )
    lines.extend(
        [
            "",
            "Solid = current P2 ground truth",
            "Dashed = prediction/transformed prediction",
        ]
    )
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=12, family="monospace")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml(args.config)
    class_names = list(config["classes"])

    class GridArgs:
        pass

    grid_args = GridArgs()
    grid_args.anchor_source = "bf_vtln_c"
    grid_args.source_anchor = args.source_anchor
    grid_args.target_anchor = args.target_anchor
    grid_args.source_speaker = args.source_speaker
    grid_args.source_session = args.source_session
    grid_args.source_frame = args.source_frame
    grid_args.target_speaker_name = args.target_speaker_name
    grid_args.target_session_name = args.target_session_name
    grid_args.target_frame = args.target_frame
    grid_args.vtln_dir = args.vtln_dir
    source_grid, target_grid, source_grid_png, target_grid_png, source_contours, target_contours = load_source_target_grids(grid_args, output_dir)
    transform = build_two_step_transform(source_grid, target_grid)

    rows = aggregate_payload(args.predictions, config, speaker=args.target_speaker, session=args.target_session)
    if not rows:
        raise RuntimeError("No matching P2 rows found in prediction payload")
    metric_rows = []
    for row in rows:
        row["affine_predicted"] = transform_prediction(row["predicted"], transform, "affine")
        row["affine_tps_predicted"] = transform_prediction(row["predicted"], transform, "affine_tps")
        row["raw_metrics"] = metric_summary(row, class_names, "predicted")
        row["affine_metrics"] = metric_summary(row, class_names, "affine_predicted")
        row["affine_tps_metrics"] = metric_summary(row, class_names, "affine_tps_predicted")
        row["affine_delta_mm"] = row["affine_metrics"]["overall_mm"] - row["raw_metrics"]["overall_mm"]
        row["affine_tps_delta_mm"] = row["affine_tps_metrics"]["overall_mm"] - row["raw_metrics"]["overall_mm"]
        row["affine_primary_delta_mm"] = row["affine_metrics"]["primary_mm"] - row["raw_metrics"]["primary_mm"]
        row["affine_tps_primary_delta_mm"] = row["affine_tps_metrics"]["primary_mm"] - row["raw_metrics"]["primary_mm"]
        metric_rows.append(
            {
                "speaker": row["speaker"],
                "session": row["session"],
                "frame": row["frame"],
                "phoneme": row["phoneme"],
                "raw_overall_mm": row["raw_metrics"]["overall_mm"],
                "affine_overall_mm": row["affine_metrics"]["overall_mm"],
                "affine_tps_overall_mm": row["affine_tps_metrics"]["overall_mm"],
                "affine_delta_mm": row["affine_delta_mm"],
                "affine_tps_delta_mm": row["affine_tps_delta_mm"],
                "raw_primary_mm": row["raw_metrics"]["primary_mm"],
                "affine_primary_mm": row["affine_metrics"]["primary_mm"],
                "affine_tps_primary_mm": row["affine_tps_metrics"]["primary_mm"],
                "affine_primary_delta_mm": row["affine_primary_delta_mm"],
                "affine_tps_primary_delta_mm": row["affine_tps_primary_delta_mm"],
            }
        )

    session_metrics = {
        "raw_overall_mm": mean([row["raw_overall_mm"] for row in metric_rows]),
        "affine_overall_mm": mean([row["affine_overall_mm"] for row in metric_rows]),
        "affine_tps_overall_mm": mean([row["affine_tps_overall_mm"] for row in metric_rows]),
        "raw_primary_mm": mean([row["raw_primary_mm"] for row in metric_rows]),
        "affine_primary_mm": mean([row["affine_primary_mm"] for row in metric_rows]),
        "affine_tps_primary_mm": mean([row["affine_tps_primary_mm"] for row in metric_rows]),
    }
    session_metrics["affine_delta_mm"] = session_metrics["affine_overall_mm"] - session_metrics["raw_overall_mm"]
    session_metrics["affine_tps_delta_mm"] = session_metrics["affine_tps_overall_mm"] - session_metrics["raw_overall_mm"]
    session_metrics["affine_primary_delta_mm"] = session_metrics["affine_primary_mm"] - session_metrics["raw_primary_mm"]
    session_metrics["affine_tps_primary_delta_mm"] = session_metrics["affine_tps_primary_mm"] - session_metrics["raw_primary_mm"]

    phoneme_metric_rows = [row for row in metric_rows if args.phoneme is not None and row["phoneme"] == args.phoneme]
    phoneme_metrics = None
    if phoneme_metric_rows:
        phoneme_metrics = {
            "count": len(phoneme_metric_rows),
            "raw_overall_mm": mean([row["raw_overall_mm"] for row in phoneme_metric_rows]),
            "affine_overall_mm": mean([row["affine_overall_mm"] for row in phoneme_metric_rows]),
            "affine_tps_overall_mm": mean([row["affine_tps_overall_mm"] for row in phoneme_metric_rows]),
            "raw_primary_mm": mean([row["raw_primary_mm"] for row in phoneme_metric_rows]),
            "affine_primary_mm": mean([row["affine_primary_mm"] for row in phoneme_metric_rows]),
            "affine_tps_primary_mm": mean([row["affine_tps_primary_mm"] for row in phoneme_metric_rows]),
        }
        phoneme_metrics["affine_delta_mm"] = phoneme_metrics["affine_overall_mm"] - phoneme_metrics["raw_overall_mm"]
        phoneme_metrics["affine_tps_delta_mm"] = phoneme_metrics["affine_tps_overall_mm"] - phoneme_metrics["raw_overall_mm"]
        phoneme_metrics["affine_primary_delta_mm"] = phoneme_metrics["affine_primary_mm"] - phoneme_metrics["raw_primary_mm"]
        phoneme_metrics["affine_tps_primary_delta_mm"] = phoneme_metrics["affine_tps_primary_mm"] - phoneme_metrics["raw_primary_mm"]

    selected = choose_row(rows, args.frame, args.select, args.phoneme)
    mri = load_asd1_mri(f"P{selected['speaker']}", f"S{selected['session']}", selected["frame"], (136, 136))
    safe_phoneme = "".join(char if char.isalnum() else "_" for char in str(selected["phoneme"]))
    contact_sheet = (
        output_dir
        / f"p7_to_p2_grid_transform_one_image_p{selected['speaker']}_s{selected['session']}_{selected['frame']}_{safe_phoneme}.png"
    )
    write_contact_sheet(
        contact_sheet,
        Path(source_grid_png),
        Path(target_grid_png),
        mri,
        selected,
        class_names,
        session_metrics,
        phoneme_metrics,
    )

    write_csv(output_dir / "frame_metrics.csv", metric_rows)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prediction_payload": str(args.predictions),
        "config": str(args.config),
        "output_image": str(contact_sheet),
        "source_grid_png": source_grid_png,
        "target_grid_png": target_grid_png,
        "source_reference": {
            "speaker": args.source_speaker,
            "session": args.source_session,
            "frame": args.source_frame,
            "anchor": args.source_anchor,
            "contour_point_counts": {name: int(len(points)) for name, points in source_contours.items()},
        },
        "target_reference": {
            "speaker": args.target_speaker_name,
            "session": args.target_session_name,
            "frame": args.target_frame,
            "anchor": args.target_anchor,
            "contour_point_counts": {name: int(len(points)) for name, points in target_contours.items()},
        },
        "vtln_dir": str(args.vtln_dir),
        "uses_bf_dynamic_contours": True,
        "uses_vtln_c1_to_c6": True,
        "coordinate_space": "136x136",
        "transform": {
            "step1": "affine",
            "step2": "thin_plate_spline",
            "step1_labels": transform["step1_labels"],
            "step2_labels": transform["step2_labels"],
        },
        "selection": args.select,
        "phoneme_filter": args.phoneme,
        "selected_frame": {
            "speaker": selected["speaker"],
            "session": selected["session"],
            "frame": selected["frame"],
            "phoneme": selected["phoneme"],
            "raw_overall_mm": selected["raw_metrics"]["overall_mm"],
            "affine_overall_mm": selected["affine_metrics"]["overall_mm"],
            "affine_tps_overall_mm": selected["affine_tps_metrics"]["overall_mm"],
            "affine_delta_mm": selected["affine_delta_mm"],
            "affine_tps_delta_mm": selected["affine_tps_delta_mm"],
            "raw_primary_mm": selected["raw_metrics"]["primary_mm"],
            "affine_primary_mm": selected["affine_metrics"]["primary_mm"],
            "affine_tps_primary_mm": selected["affine_tps_metrics"]["primary_mm"],
            "affine_primary_delta_mm": selected["affine_primary_delta_mm"],
            "affine_tps_primary_delta_mm": selected["affine_tps_primary_delta_mm"],
        },
        "session_metrics": session_metrics,
        "phoneme_metrics": phoneme_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# P7 -> P2 Grid Transform One-Image Check\n\n")
        handle.write(f"- output image: `{contact_sheet.name}`\n")
        handle.write(f"- selected frame: `P{selected['speaker']}/S{selected['session']}/{selected['frame']}`\n")
        handle.write(f"- phoneme: `{selected['phoneme']}`\n")
        handle.write("- negative delta means improvement.\n\n")
        if phoneme_metrics is not None:
            handle.write(f"- phoneme-filter count: `{int(phoneme_metrics['count'])}` frames\n\n")
        handle.write("## Selected Frame\n\n")
        handle.write("| Mode | Overall RMSE mm | Primary RMSE mm | Overall delta mm | Primary delta mm |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        handle.write(f"| raw | {selected['raw_metrics']['overall_mm']:.6f} | {selected['raw_metrics']['primary_mm']:.6f} | 0.000000 | 0.000000 |\n")
        handle.write(f"| affine | {selected['affine_metrics']['overall_mm']:.6f} | {selected['affine_metrics']['primary_mm']:.6f} | {selected['affine_delta_mm']:.6f} | {selected['affine_primary_delta_mm']:.6f} |\n")
        handle.write(f"| affine+TPS | {selected['affine_tps_metrics']['overall_mm']:.6f} | {selected['affine_tps_metrics']['primary_mm']:.6f} | {selected['affine_tps_delta_mm']:.6f} | {selected['affine_tps_primary_delta_mm']:.6f} |\n\n")
        handle.write("## Whole P2/S1 Session Mean\n\n")
        handle.write("| Mode | Overall RMSE mm | Primary RMSE mm | Overall delta mm | Primary delta mm |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        handle.write(f"| raw | {session_metrics['raw_overall_mm']:.6f} | {session_metrics['raw_primary_mm']:.6f} | 0.000000 | 0.000000 |\n")
        handle.write(f"| affine | {session_metrics['affine_overall_mm']:.6f} | {session_metrics['affine_primary_mm']:.6f} | {session_metrics['affine_delta_mm']:.6f} | {session_metrics['affine_primary_delta_mm']:.6f} |\n")
        handle.write(f"| affine+TPS | {session_metrics['affine_tps_overall_mm']:.6f} | {session_metrics['affine_tps_primary_mm']:.6f} | {session_metrics['affine_tps_delta_mm']:.6f} | {session_metrics['affine_tps_primary_delta_mm']:.6f} |\n")
        if phoneme_metrics is not None:
            handle.write(f"\n## Selected Phoneme Mean: `{selected['phoneme']}`\n\n")
            handle.write("| Mode | Overall RMSE mm | Primary RMSE mm | Overall delta mm | Primary delta mm |\n")
            handle.write("|---|---:|---:|---:|---:|\n")
            handle.write(f"| raw | {phoneme_metrics['raw_overall_mm']:.6f} | {phoneme_metrics['raw_primary_mm']:.6f} | 0.000000 | 0.000000 |\n")
            handle.write(f"| affine | {phoneme_metrics['affine_overall_mm']:.6f} | {phoneme_metrics['affine_primary_mm']:.6f} | {phoneme_metrics['affine_delta_mm']:.6f} | {phoneme_metrics['affine_primary_delta_mm']:.6f} |\n")
            handle.write(f"| affine+TPS | {phoneme_metrics['affine_tps_overall_mm']:.6f} | {phoneme_metrics['affine_tps_primary_mm']:.6f} | {phoneme_metrics['affine_tps_delta_mm']:.6f} | {phoneme_metrics['affine_tps_primary_delta_mm']:.6f} |\n")

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
