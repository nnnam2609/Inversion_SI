#!/usr/bin/env python3
"""Generate CSV, JSON, Markdown, and PNG tables from paired ASD2 RMSE rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel


DISPLAY_NAMES = {
    "arytenoid-cartilage": "Arytenoid cartilage",
    "epiglottis": "Epiglottis",
    "lower-lip": "Lower lip",
    "pharynx": "Pharynx",
    "soft-palate-midline": "Soft palate (midline)",
    "tongue": "Tongue",
    "upper-lip": "Upper lip",
    "vocal-folds": "Vocal folds",
    "thyroid-cartilage": "Thyroid cartilage",
    "lower-incisor": "Lower incisor",
    "upper-incisor": "Upper incisor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-npz", type=Path, required=True)
    parser.add_argument("--extraction-audit", type=Path, required=True)
    parser.add_argument("--global-summary", type=Path, required=True)
    parser.add_argument("--moving-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean_mm": float(flat.mean()),
        "std_mm": float(flat.std(ddof=0)),
        "median_mm": float(np.median(flat)),
    }


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values))
    adjusted = np.zeros(len(p_values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[int(index)]))
        adjusted[int(index)] = running
    return adjusted.tolist()


def metric_row(
    group_type: str,
    key: str,
    label: str,
    global_values: np.ndarray,
    moving_values: np.ndarray,
) -> dict[str, Any]:
    if global_values.shape != moving_values.shape:
        raise ValueError(f"Unpaired values for {group_type}/{key}")
    global_stats = describe(global_values)
    moving_stats = describe(moving_values)
    global_pairs = global_values if global_values.ndim == 1 else global_values.mean(axis=1)
    moving_pairs = moving_values if moving_values.ndim == 1 else moving_values.mean(axis=1)
    test = ttest_rel(moving_pairs, global_pairs, nan_policy="raise")
    improvement = global_stats["mean_mm"] - moving_stats["mean_mm"]
    return {
        "group_type": group_type,
        "key": key,
        "label": label,
        "n_aligned_integer_rows": int(global_pairs.size),
        "global": global_stats,
        "moving_average_oracle": moving_stats,
        "improvement_mm": improvement,
        "improvement_percent": 100.0 * improvement / global_stats["mean_mm"],
        "paired_t_statistic": float(test.statistic),
        "p_value_raw": float(test.pvalue),
        "p_value_holm": None,
        "significant_holm_0_05": None,
    }


def add_holm(rows: list[dict[str, Any]]) -> None:
    adjusted = holm([row["p_value_raw"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["p_value_holm"] = value
        row["significant_holm_0_05"] = bool(value < 0.05)


def fmt_p(value: float | None) -> str:
    if value is None:
        return ""
    if value < 1e-300:
        return "<1e-300"
    return f"{value:.3e}" if value < 1e-4 else f"{value:.6f}"


def csv_record(row: dict[str, Any]) -> dict[str, Any]:
    global_stats = row["global"]
    moving_stats = row["moving_average_oracle"]
    return {
        "group_type": row["group_type"],
        "key": row["key"],
        "label": row["label"],
        "n_aligned_integer_rows": row["n_aligned_integer_rows"],
        "global_mean_mm": f"{global_stats['mean_mm']:.9f}",
        "global_std_mm": f"{global_stats['std_mm']:.9f}",
        "global_median_mm": f"{global_stats['median_mm']:.9f}",
        "moving_average_oracle_mean_mm": f"{moving_stats['mean_mm']:.9f}",
        "moving_average_oracle_std_mm": f"{moving_stats['std_mm']:.9f}",
        "moving_average_oracle_median_mm": f"{moving_stats['median_mm']:.9f}",
        "improvement_mm": f"{row['improvement_mm']:.9f}",
        "improvement_percent": f"{row['improvement_percent']:.6f}",
        "paired_t_statistic": f"{row['paired_t_statistic']:.9f}",
        "p_value_raw": fmt_p(row["p_value_raw"]),
        "p_value_holm": fmt_p(row["p_value_holm"]),
        "significant_holm_0_05": (
            "" if row["significant_holm_0_05"] is None
            else str(row["significant_holm_0_05"]).lower()
        ),
    }


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# ASD2 Sofiane-153: global vs moving-average oracle",
        "",
        "> Moving-average prediction uses target-contour statistics and is an oracle/leaky baseline.",
        "",
        "| Group | Row | N | Global mean±SD (median), mm | Moving mean±SD (median), mm | Δ mm | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        g = row["global"]
        m = row["moving_average_oracle"]
        lines.append(
            f"| {row['group_type']} | {row['label']} | {row['n_aligned_integer_rows']} "
            f"| {g['mean_mm']:.3f}±{g['std_mm']:.3f} ({g['median_mm']:.3f}) "
            f"| {m['mean_mm']:.3f}±{m['std_mm']:.3f} ({m['median_mm']:.3f}) "
            f"| {row['improvement_mm']:.3f} | {fmt_p(row['p_value_holm'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_png(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = ["Group", "Row", "N", "Global mm", "Moving-oracle mm", "Δ mm", "Holm p"]
    cells = []
    for row in rows:
        g = row["global"]
        m = row["moving_average_oracle"]
        cells.append(
            [
                row["group_type"],
                row["label"],
                str(row["n_aligned_integer_rows"]),
                f"{g['mean_mm']:.2f}±{g['std_mm']:.2f}\nmed {g['median_mm']:.2f}",
                f"{m['mean_mm']:.2f}±{m['std_mm']:.2f}\nmed {m['median_mm']:.2f}",
                f"{row['improvement_mm']:.2f}",
                fmt_p(row["p_value_holm"]),
            ]
        )
    figure, axis = plt.subplots(figsize=(17, 0.48 * len(cells) + 2.2))
    axis.axis("off")
    axis.set_title(
        "ASD2 Sofiane-153 — Global vs Exact-Sofiane moving-average oracle\n"
        "Integer rows only; 1.62 mm/pixel; moving strategy uses target statistics",
        fontsize=13,
        pad=12,
    )
    table = axis.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.55)
    for (row_index, _), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#d9e8fb")
            cell.set_text_props(weight="bold")
        elif cells[row_index - 1][0] == "overall":
            cell.set_facecolor("#e8f5e9")
            cell.set_text_props(weight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    audit = json.loads(args.extraction_audit.read_text(encoding="utf-8"))
    if audit["output_sha256"] != sha256(args.paired_npz):
        raise ValueError("Paired NPZ checksum does not match extraction audit")
    with np.load(args.paired_npz, allow_pickle=False) as payload:
        frame_ids = payload["frame_ids"]
        global_values = payload["global_rmse_mm"].astype(np.float64)
        moving_values = payload["moving_average_oracle_rmse_mm"].astype(np.float64)
        classes = payload["classes"].tolist()
    if global_values.shape != moving_values.shape or frame_ids.shape[0] != global_values.shape[0]:
        raise ValueError("Paired artifact shapes do not align")

    overall = metric_row("overall", "overall", "Overall", global_values, moving_values)
    articulators = [
        metric_row(
            "articulator",
            class_name,
            DISPLAY_NAMES.get(class_name, class_name),
            global_values[:, index],
            moving_values[:, index],
        )
        for index, class_name in enumerate(classes)
    ]
    add_holm(articulators)
    sessions: list[dict[str, Any]] = []
    for speaker, session in np.unique(frame_ids[:, :2], axis=0):
        mask = (frame_ids[:, 0] == speaker) & (frame_ids[:, 1] == session)
        sessions.append(
            metric_row(
                "session",
                f"{speaker}/S{session}",
                f"{speaker}/S{session}",
                global_values[mask],
                moving_values[mask],
            )
        )
    if len(sessions) != 17:
        raise ValueError(f"Expected 17 test sessions, found {len(sessions)}")
    add_holm(sessions)
    rows = [overall, *articulators, *sessions]

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "global_vs_moving_oracle_metrics.csv"
    records = [csv_record(row) for row in rows]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    global_summary = json.loads(args.global_summary.read_text(encoding="utf-8"))
    moving_summary = json.loads(args.moving_summary.read_text(encoding="utf-8"))
    report = {
        "status": "complete",
        "dataset": "ASD2 Sofiane baseline 153 sessions with 1796/S25 replacement",
        "split_counts": {"train": 122, "validation": 14, "test": 17},
        "protocol": "paired aligned integer rows from all 17 test sessions",
        "unit": "mm",
        "mm_per_pixel": 1.62,
        "integer_rows": int(global_values.shape[0]),
        "fractional_rows_excluded": int(audit["pair_audit"]["num_valid_rows"] - global_values.shape[0]),
        "moving_average_warning": (
            "Exact-Sofiane oracle: prediction center uses per-chunk target-contour "
            "moving-average statistics; validation/test are target-statistics-leaky."
        ),
        "statistics": (
            "Population SD (ddof=0), median, paired two-sided t-test. Holm correction "
            "is applied separately across 11 articulators and across 17 sessions."
        ),
        "global_training_summary": global_summary,
        "moving_training_summary": moving_summary,
        "extraction_audit": audit,
        "rows": rows,
    }
    json_path = output / "global_vs_moving_oracle_metrics.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(output / "global_vs_moving_oracle_table.md", rows)
    write_png(output / "global_vs_moving_oracle_table.png", rows)
    manifest = {
        "status": "complete",
        "artifacts": {
            path.name: {"path": str(path), "sha256": sha256(path)}
            for path in (
                csv_path,
                json_path,
                output / "global_vs_moving_oracle_table.md",
                output / "global_vs_moving_oracle_table.png",
            )
        },
    }
    (output / "TABLE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
