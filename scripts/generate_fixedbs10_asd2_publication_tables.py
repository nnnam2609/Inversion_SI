#!/usr/bin/env python3
"""Create three publication-style ASD2-to-ASD1 adaptation tables.

The table layout follows the earlier P7 cross-speaker tables:

1. adaptation summary for all 11 and without three laryngeal contours;
2. per-articulator mean SD, median, and paired significance;
3. per-speaker mean SD and improvement.

P1--P9 form the unseen cohort. P10 is displayed as a same-person ASD2 control
but is excluded from the unseen aggregates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image
from scipy.stats import ttest_rel

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import generate_fixedbs10_adaptation_tables_10speakers as common  # noqa: E402


DEFAULT_SOURCE = common.DEFAULT_SOURCE
DEFAULT_RESULT = common.DEFAULT_OUTPUT
DEFAULT_OUTPUT = DEFAULT_RESULT / "publication_tables"
UNSEEN_LABELS = tuple(label for label in common.PAIRS if not label.startswith("P10/"))
CONTROL_LABEL = "P10/S14"
LARYNGEAL = {"epiglottis", "vocal-folds", "thyroid-cartilage"}
ALL_INDICES = tuple(range(len(common.EXPECTED_CLASSES)))
WITHOUT_LARYNGEAL_INDICES = tuple(
    index
    for index, name in enumerate(common.EXPECTED_CLASSES)
    if name not in LARYNGEAL
)
DISPLAY_NAMES = {
    "arytenoid-cartilage": "Arytenoid cartilage",
    "epiglottis": "Epiglottis",
    "lower-lip": "Lower lip",
    "pharynx": "Pharynx",
    "soft-palate-midline": "Soft palate",
    "tongue": "Tongue",
    "upper-lip": "Upper lip",
    "vocal-folds": "Vocal folds",
    "thyroid-cartilage": "Thyroid cartilage",
    "lower-incisor": "Lower incisor",
    "upper-incisor": "Upper incisor",
}
NAVY = "#192132"
BLUE = "#1f55d5"
GRID = "#aeb7c7"
LIGHT = "#f4f6f9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def describe(values: np.ndarray, ddof: int = 1) -> dict[str, float]:
    return {
        "mean_mm": float(np.mean(values)),
        "sd_mm": float(np.std(values, ddof=ddof)),
        "median_mm": float(np.median(values)),
    }


def group_frame_rmse(
    predicted: np.ndarray, ground_truth: np.ndarray, indices: tuple[int, ...]
) -> np.ndarray:
    delta = (
        predicted[:, list(indices)].astype(np.float64)
        - ground_truth[:, list(indices)].astype(np.float64)
    )
    return np.sqrt(np.mean(delta * delta, axis=(1, 2, 3))) * common.MM_PER_PIXEL


def per_contour_frame_rmse(
    predicted: np.ndarray, ground_truth: np.ndarray
) -> np.ndarray:
    delta = predicted.astype(np.float64) - ground_truth.astype(np.float64)
    return np.sqrt(np.mean(delta * delta, axis=(2, 3))) * common.MM_PER_PIXEL


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[int(original_index)])
        running_max = max(running_max, candidate)
        adjusted[int(original_index)] = running_max
    return adjusted


def safe_paired_ttest(candidate: np.ndarray, baseline: np.ndarray) -> tuple[float, float]:
    test = ttest_rel(candidate, baseline, nan_policy="raise")
    statistic = float(test.statistic)
    p_value = float(test.pvalue)
    if not math.isfinite(statistic) or not math.isfinite(p_value):
        raise RuntimeError("Paired t-test returned a non-finite result")
    return statistic, p_value


def load_store(
    source_root: Path, result_root: Path
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    store: dict[str, dict[str, np.ndarray]] = {}
    inventory: list[dict[str, Any]] = []
    for label, pair in common.PAIRS.items():
        baseline_path, audio_path = common.paths_for_pair(
            source_root, result_root, pair
        )
        baseline = common.load_npz(baseline_path)
        audio = common.load_npz(audio_path)
        common.validate_pair(label, baseline, audio)
        ground_truth = baseline["ground_truth"].astype(np.float32)
        store[label] = {
            "ground_truth": ground_truth,
            "baseline": baseline["predicted_raw"].astype(np.float32),
            "contour": baseline[
                "predicted_after_corrected_affine_tps"
            ].astype(np.float32),
            "contour_audio": audio[
                "predicted_after_corrected_affine_tps"
            ].astype(np.float32),
        }
        inventory.append(
            {
                "speaker_session": label,
                "frames": int(len(ground_truth)),
                "baseline_pack": str(baseline_path.resolve()),
                "baseline_pack_sha256": sha256(baseline_path),
                "contour_audio_pack": str(audio_path.resolve()),
                "contour_audio_pack_sha256": sha256(audio_path),
            }
        )
    return store, inventory


def concatenate(
    store: dict[str, dict[str, np.ndarray]], labels: tuple[str, ...], key: str
) -> np.ndarray:
    return np.concatenate([store[label][key] for label in labels], axis=0)


def change_fields(baseline: float, candidate: float) -> dict[str, float]:
    delta = candidate - baseline
    return {
        "change_mm_vs_baseline": delta,
        "change_pct_vs_baseline": 100.0 * delta / baseline,
        "improvement_pct_vs_baseline": -100.0 * delta / baseline,
    }


def build_summary(
    arrays: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    values: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for key, label, indices in (
        ("all_11", "All 11 contours RMSE (mm)", ALL_INDICES),
        (
            "without_laryngeal_3",
            "Without 3 laryngeal contours RMSE (mm)",
            WITHOUT_LARYNGEAL_INDICES,
        ),
    ):
        values[key] = {
            stage: group_frame_rmse(arrays[stage], arrays["ground_truth"], indices)
            for stage in ("baseline", "contour", "contour_audio")
        }
        baseline_mean = float(np.mean(values[key]["baseline"]))
        contour_mean = float(np.mean(values[key]["contour"]))
        audio_mean = float(np.mean(values[key]["contour_audio"]))
        rows.append(
            {
                "metric_key": key,
                "metric": label,
                "speakers": len(UNSEEN_LABELS),
                "frames": len(values[key]["baseline"]),
                "baseline_rmse_mm": baseline_mean,
                "contour_rmse_mm": contour_mean,
                **{
                    f"contour_{field}": value
                    for field, value in change_fields(
                        baseline_mean, contour_mean
                    ).items()
                },
                "contour_audio_rmse_mm": audio_mean,
                **{
                    f"contour_audio_{field}": value
                    for field, value in change_fields(
                        baseline_mean, audio_mean
                    ).items()
                },
            }
        )
    return rows, values


def build_articulator_rows(
    arrays: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    per_stage = {
        stage: per_contour_frame_rmse(arrays[stage], arrays["ground_truth"])
        for stage in ("baseline", "contour", "contour_audio")
    }
    raw_p_contour: list[float] = []
    raw_p_audio: list[float] = []
    rows: list[dict[str, Any]] = []
    for index, class_name in enumerate(common.EXPECTED_CLASSES):
        baseline_values = per_stage["baseline"][:, index]
        contour_values = per_stage["contour"][:, index]
        audio_values = per_stage["contour_audio"][:, index]
        contour_t, contour_p = safe_paired_ttest(contour_values, baseline_values)
        audio_t, audio_p = safe_paired_ttest(audio_values, baseline_values)
        raw_p_contour.append(contour_p)
        raw_p_audio.append(audio_p)
        rows.append(
            {
                "articulator_key": class_name,
                "articulator": DISPLAY_NAMES[class_name],
                "frames": len(baseline_values),
                "baseline_mean_rmse_mm": describe(baseline_values)["mean_mm"],
                "baseline_sd_rmse_mm": describe(baseline_values)["sd_mm"],
                "baseline_median_rmse_mm": describe(baseline_values)["median_mm"],
                "contour_mean_rmse_mm": describe(contour_values)["mean_mm"],
                "contour_sd_rmse_mm": describe(contour_values)["sd_mm"],
                "contour_median_rmse_mm": describe(contour_values)["median_mm"],
                "contour_paired_t_statistic": contour_t,
                "contour_p_value_raw": contour_p,
                "contour_audio_mean_rmse_mm": describe(audio_values)["mean_mm"],
                "contour_audio_sd_rmse_mm": describe(audio_values)["sd_mm"],
                "contour_audio_median_rmse_mm": describe(audio_values)["median_mm"],
                "contour_audio_paired_t_statistic": audio_t,
                "contour_audio_p_value_raw": audio_p,
            }
        )

    contour_adjusted = holm_adjust(raw_p_contour)
    audio_adjusted = holm_adjust(raw_p_audio)
    for row, contour_p, audio_p in zip(rows, contour_adjusted, audio_adjusted):
        row["contour_p_value_holm"] = contour_p
        row["contour_significant_holm_0_05"] = contour_p < 0.05
        row["contour_audio_p_value_holm"] = audio_p
        row["contour_audio_significant_holm_0_05"] = audio_p < 0.05

    baseline_flat = per_stage["baseline"].reshape(-1)
    contour_flat = per_stage["contour"].reshape(-1)
    audio_flat = per_stage["contour_audio"].reshape(-1)
    contour_t, contour_p = safe_paired_ttest(
        per_stage["contour"].mean(axis=1),
        per_stage["baseline"].mean(axis=1),
    )
    audio_t, audio_p = safe_paired_ttest(
        per_stage["contour_audio"].mean(axis=1),
        per_stage["baseline"].mean(axis=1),
    )
    rows.append(
        {
            "articulator_key": "mean",
            "articulator": "Mean",
            "frames": len(baseline_flat),
            "baseline_mean_rmse_mm": describe(baseline_flat)["mean_mm"],
            "baseline_sd_rmse_mm": describe(baseline_flat)["sd_mm"],
            "baseline_median_rmse_mm": describe(baseline_flat)["median_mm"],
            "contour_mean_rmse_mm": describe(contour_flat)["mean_mm"],
            "contour_sd_rmse_mm": describe(contour_flat)["sd_mm"],
            "contour_median_rmse_mm": describe(contour_flat)["median_mm"],
            "contour_paired_t_statistic": contour_t,
            "contour_p_value_raw": contour_p,
            "contour_p_value_holm": "",
            "contour_significant_holm_0_05": contour_p < 0.05,
            "contour_audio_mean_rmse_mm": describe(audio_flat)["mean_mm"],
            "contour_audio_sd_rmse_mm": describe(audio_flat)["sd_mm"],
            "contour_audio_median_rmse_mm": describe(audio_flat)["median_mm"],
            "contour_audio_paired_t_statistic": audio_t,
            "contour_audio_p_value_raw": audio_p,
            "contour_audio_p_value_holm": "",
            "contour_audio_significant_holm_0_05": audio_p < 0.05,
        }
    )
    return rows, per_stage


def build_speaker_rows(
    store: dict[str, dict[str, np.ndarray]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    per_speaker: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for label in common.PAIRS:
        payload = store[label]
        values = {
            stage: group_frame_rmse(
                payload[stage], payload["ground_truth"], ALL_INDICES
            )
            for stage in ("baseline", "contour", "contour_audio")
        }
        per_speaker[label] = values
        baseline = describe(values["baseline"])
        contour = describe(values["contour"])
        audio = describe(values["contour_audio"])
        rows.append(
            {
                "speaker_session": label,
                "cohort": (
                    "same_person_control"
                    if label == CONTROL_LABEL
                    else "unseen"
                ),
                "frames": len(values["baseline"]),
                "baseline_mean_rmse_mm": baseline["mean_mm"],
                "baseline_sd_rmse_mm": baseline["sd_mm"],
                "contour_mean_rmse_mm": contour["mean_mm"],
                "contour_sd_rmse_mm": contour["sd_mm"],
                **{
                    f"contour_{field}": value
                    for field, value in change_fields(
                        baseline["mean_mm"], contour["mean_mm"]
                    ).items()
                },
                "contour_audio_mean_rmse_mm": audio["mean_mm"],
                "contour_audio_sd_rmse_mm": audio["sd_mm"],
                **{
                    f"contour_audio_{field}": value
                    for field, value in change_fields(
                        baseline["mean_mm"], audio["mean_mm"]
                    ).items()
                },
            }
        )

    macro = {
        stage: np.asarray(
            [
                np.mean(per_speaker[label][stage])
                for label in UNSEEN_LABELS
            ],
            dtype=np.float64,
        )
        for stage in ("baseline", "contour", "contour_audio")
    }
    weighted = {
        stage: np.concatenate(
            [per_speaker[label][stage] for label in UNSEEN_LABELS]
        )
        for stage in ("baseline", "contour", "contour_audio")
    }
    for label, values, sd_unit in (
        ("Mean (unseen)", macro, "speaker means"),
        ("All frames", weighted, "frames"),
    ):
        baseline = describe(values["baseline"])
        contour = describe(values["contour"])
        audio = describe(values["contour_audio"])
        rows.append(
            {
                "speaker_session": label,
                "cohort": "unseen_aggregate",
                "frames": sum(
                    len(per_speaker[item]["baseline"]) for item in UNSEEN_LABELS
                ),
                "baseline_mean_rmse_mm": baseline["mean_mm"],
                "baseline_sd_rmse_mm": baseline["sd_mm"],
                "contour_mean_rmse_mm": contour["mean_mm"],
                "contour_sd_rmse_mm": contour["sd_mm"],
                **{
                    f"contour_{field}": value
                    for field, value in change_fields(
                        baseline["mean_mm"], contour["mean_mm"]
                    ).items()
                },
                "contour_audio_mean_rmse_mm": audio["mean_mm"],
                "contour_audio_sd_rmse_mm": audio["sd_mm"],
                **{
                    f"contour_audio_{field}": value
                    for field, value in change_fields(
                        baseline["mean_mm"], audio["mean_mm"]
                    ).items()
                },
                "sd_unit": sd_unit,
            }
        )
    return rows, per_speaker


def rmse_sd(mean: float, sd: float, star: bool = False) -> str:
    suffix = "*" if star else ""
    return f"{mean:.2f} ± {sd:.2f}{suffix}"


def change_text(delta: float, pct: float, decimals: int = 3) -> str:
    return f"{delta:+.{decimals}f} ({pct:+.2f}%)".replace("+", "+")


def improve_text(improvement: float) -> str:
    return f"↓{improvement:.2f}%" if improvement >= 0 else f"↑{-improvement:.2f}%"


def render_summary(rows: list[dict[str, Any]], output: Path) -> tuple[Path, Path]:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    figure, axis = plt.subplots(figsize=(16, 6.4))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(
        0.01,
        0.955,
        "ASD2 Speaker Adaptation Summary",
        fontsize=22,
        fontweight="bold",
        color=NAVY,
        va="top",
    )
    axis.text(
        0.01,
        0.895,
        "Unseen ASD1 speakers P1–P9; all evaluated integer frames.",
        fontsize=11.5,
        color="#4d596a",
        va="top",
    )

    x = [0.01, 0.43, 0.62, 0.81, 0.99]
    top, bottom = 0.80, 0.10
    table_rows = 5
    y = [top - index * (top - bottom) / table_rows for index in range(table_rows + 1)]
    for index, value in enumerate(y):
        axis.plot(
            [x[0], x[-1]],
            [value, value],
            color=GRID,
            lw=1.2 if index in (0, 1, len(y) - 1) else 0.65,
        )
    headers = ["Metric", "Baseline", "Contour-side", "Contour + audio-side"]
    for index, header in enumerate(headers):
        axis.text(
            x[index] + 0.002 if index == 0 else (x[index] + x[index + 1]) / 2,
            (y[0] + y[1]) / 2,
            header,
            ha="left" if index == 0 else "center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=NAVY,
        )

    body: list[tuple[str, str, str, str]] = []
    for row in rows:
        body.extend(
            [
                (
                    str(row["metric"]),
                    f"{float(row['baseline_rmse_mm']):.3f}",
                    f"{float(row['contour_rmse_mm']):.3f}",
                    f"{float(row['contour_audio_rmse_mm']):.3f}",
                ),
                (
                    "Change vs. baseline",
                    "—",
                    change_text(
                        float(row["contour_change_mm_vs_baseline"]),
                        float(row["contour_change_pct_vs_baseline"]),
                    ),
                    change_text(
                        float(row["contour_audio_change_mm_vs_baseline"]),
                        float(row["contour_audio_change_pct_vs_baseline"]),
                    ),
                ),
            ]
        )
    for row_index, row in enumerate(body, start=1):
        center_y = (y[row_index] + y[row_index + 1]) / 2
        for column, value in enumerate(row):
            axis.text(
                x[column] + 0.002 if column == 0 else (x[column] + x[column + 1]) / 2,
                center_y,
                value,
                ha="left" if column == 0 else "center",
                va="center",
                fontsize=12.5,
                fontweight="bold" if column >= 2 else "normal",
                color=BLUE if column >= 2 else NAVY,
            )
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    figure.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png, pdf


def render_articulators(
    rows: list[dict[str, Any]], output: Path
) -> tuple[Path, Path]:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    figure, axis = plt.subplots(figsize=(18, 10.5))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(
        0.01,
        0.975,
        "ASD2 Per-Articulator Comparison",
        fontsize=22,
        fontweight="bold",
        color=NAVY,
        va="top",
    )
    axis.text(
        0.01,
        0.925,
        "Baseline, contour-side adaptation, and contour + audio-side adaptation on unseen P1–P9.",
        fontsize=11.5,
        color="#4d596a",
        va="top",
    )
    x = [0.01, 0.23, 0.39, 0.48, 0.64, 0.73, 0.89, 0.99]
    top, bottom = 0.875, 0.13
    count = 1 + len(rows)
    y = [top - index * (top - bottom) / count for index in range(count + 1)]
    for index, value in enumerate(y):
        axis.plot(
            [x[0], x[-1]],
            [value, value],
            color=GRID,
            lw=1.2 if index in (0, 1, len(y) - 1) else 0.55,
        )
    headers = [
        "Articulator",
        "Baseline\nRMSE",
        "Median",
        "Contour-side\nRMSE",
        "Median",
        "Contour + audio\nRMSE",
        "Median",
    ]
    for column, header in enumerate(headers):
        axis.text(
            x[column] + 0.002 if column == 0 else (x[column] + x[column + 1]) / 2,
            (y[0] + y[1]) / 2,
            header,
            ha="left" if column == 0 else "center",
            va="center",
            fontsize=12.2,
            fontweight="bold",
            color=NAVY,
            linespacing=1.15,
        )
    for row_index, row in enumerate(rows, start=1):
        center_y = (y[row_index] + y[row_index + 1]) / 2
        mean_row = row["articulator_key"] == "mean"
        contour_star = bool(row["contour_significant_holm_0_05"])
        audio_star = bool(row["contour_audio_significant_holm_0_05"])
        cells = [
            str(row["articulator"]),
            rmse_sd(
                float(row["baseline_mean_rmse_mm"]),
                float(row["baseline_sd_rmse_mm"]),
            ),
            f"{float(row['baseline_median_rmse_mm']):.2f}",
            rmse_sd(
                float(row["contour_mean_rmse_mm"]),
                float(row["contour_sd_rmse_mm"]),
                contour_star,
            ),
            f"{float(row['contour_median_rmse_mm']):.2f}",
            rmse_sd(
                float(row["contour_audio_mean_rmse_mm"]),
                float(row["contour_audio_sd_rmse_mm"]),
                audio_star,
            ),
            f"{float(row['contour_audio_median_rmse_mm']):.2f}",
        ]
        for column, value in enumerate(cells):
            axis.text(
                x[column] + 0.002 if column == 0 else (x[column] + x[column + 1]) / 2,
                center_y,
                value,
                ha="left" if column == 0 else "center",
                va="center",
                fontsize=11.3,
                fontweight="bold" if mean_row or column >= 3 else "normal",
                color=BLUE if column >= 3 else NAVY,
            )
    footnote = (
        "* Significant difference vs. baseline. Articulator rows: paired "
        "two-sided frame-level t-test with Holm correction across 11 "
        "articulators within each branch (adjusted p < 0.05). Mean row: paired "
        "test on frame-level articulator means, outside the Holm family. "
        "Values are RMSE in mm, mean ± sample SD."
    )
    axis.text(0.01, 0.085, footnote, fontsize=10.3, color="#4d596a", va="top")
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    figure.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png, pdf


def render_speakers(rows: list[dict[str, Any]], output: Path) -> tuple[Path, Path]:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    figure, axis = plt.subplots(figsize=(19, 10.5))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(
        0.01,
        0.975,
        "ASD2 Per-Speaker Comparison",
        fontsize=22,
        fontweight="bold",
        color=NAVY,
        va="top",
    )
    axis.text(
        0.01,
        0.925,
        "ASD1 speakers under baseline, contour-side, and contour + audio-side adaptation.",
        fontsize=11.5,
        color="#4d596a",
        va="top",
    )
    x = [0.01, 0.16, 0.44, 0.72, 0.99]
    top, bottom = 0.875, 0.13
    count = 1 + len(rows)
    y = [top - index * (top - bottom) / count for index in range(count + 1)]
    for index, value in enumerate(y):
        axis.plot(
            [x[0], x[-1]],
            [value, value],
            color=GRID,
            lw=1.2 if index in (0, 1, len(y) - 1) else 0.55,
        )
    headers = [
        "Speaker",
        "Baseline Mean RMSE ± SD",
        "Contour Mean RMSE ± SD\n(Improve)",
        "Contour + Audio Mean RMSE ± SD\n(Improve)",
    ]
    for column, header in enumerate(headers):
        axis.text(
            x[column] + 0.002 if column == 0 else (x[column] + x[column + 1]) / 2,
            (y[0] + y[1]) / 2,
            header,
            ha="left" if column == 0 else "center",
            va="center",
            fontsize=12.2,
            fontweight="bold",
            color=NAVY,
            linespacing=1.15,
        )
    for row_index, row in enumerate(rows, start=1):
        center_y = (y[row_index] + y[row_index + 1]) / 2
        aggregate = row["cohort"] == "unseen_aggregate"
        label = str(row["speaker_session"])
        if label == CONTROL_LABEL:
            label += "†"
        cells = [
            label,
            rmse_sd(
                float(row["baseline_mean_rmse_mm"]),
                float(row["baseline_sd_rmse_mm"]),
            ),
            (
                rmse_sd(
                    float(row["contour_mean_rmse_mm"]),
                    float(row["contour_sd_rmse_mm"]),
                )
                + f" ({improve_text(float(row['contour_improvement_pct_vs_baseline']))})"
            ),
            (
                rmse_sd(
                    float(row["contour_audio_mean_rmse_mm"]),
                    float(row["contour_audio_sd_rmse_mm"]),
                )
                + f" ({improve_text(float(row['contour_audio_improvement_pct_vs_baseline']))})"
            ),
        ]
        for column, value in enumerate(cells):
            axis.text(
                x[column] + 0.002 if column == 0 else (x[column] + x[column + 1]) / 2,
                center_y,
                value,
                ha="left" if column == 0 else "center",
                va="center",
                fontsize=11.3,
                fontweight="bold" if aggregate or column >= 2 else "normal",
                color=BLUE if column >= 2 else NAVY,
            )
    footnote = (
        "† P10/S14 is the same-person ASD2 control and is shown but excluded "
        "from Mean (unseen) and All frames. All rows use evaluated integer frames."
    )
    axis.text(0.01, 0.085, footnote, fontsize=10.3, color="#4d596a", va="top")
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    figure.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png, pdf


def markdown_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Metric | Baseline | Contour-side | Contour + audio-side |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.extend(
            [
                (
                    f"| {row['metric']} | {float(row['baseline_rmse_mm']):.3f} | "
                    f"{float(row['contour_rmse_mm']):.3f} | "
                    f"{float(row['contour_audio_rmse_mm']):.3f} |"
                ),
                (
                    f"| Change vs. baseline | — | "
                    f"{change_text(float(row['contour_change_mm_vs_baseline']), float(row['contour_change_pct_vs_baseline']))} | "
                    f"{change_text(float(row['contour_audio_change_mm_vs_baseline']), float(row['contour_audio_change_pct_vs_baseline']))} |"
                ),
            ]
        )
    return "\n".join(lines)


def markdown_articulators(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Articulator | Baseline RMSE | Median | Contour-side RMSE | Median | Contour + audio RMSE | Median |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['articulator']} | "
            f"{rmse_sd(float(row['baseline_mean_rmse_mm']), float(row['baseline_sd_rmse_mm']))} | "
            f"{float(row['baseline_median_rmse_mm']):.2f} | "
            f"{rmse_sd(float(row['contour_mean_rmse_mm']), float(row['contour_sd_rmse_mm']), bool(row['contour_significant_holm_0_05']))} | "
            f"{float(row['contour_median_rmse_mm']):.2f} | "
            f"{rmse_sd(float(row['contour_audio_mean_rmse_mm']), float(row['contour_audio_sd_rmse_mm']), bool(row['contour_audio_significant_holm_0_05']))} | "
            f"{float(row['contour_audio_median_rmse_mm']):.2f} |"
        )
    return "\n".join(lines)


def markdown_speakers(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Speaker | Baseline Mean RMSE ± SD | Contour Mean RMSE ± SD (Improve) | Contour + Audio Mean RMSE ± SD (Improve) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        label = str(row["speaker_session"]) + (
            "†" if row["speaker_session"] == CONTROL_LABEL else ""
        )
        lines.append(
            f"| {label} | "
            f"{rmse_sd(float(row['baseline_mean_rmse_mm']), float(row['baseline_sd_rmse_mm']))} | "
            f"{rmse_sd(float(row['contour_mean_rmse_mm']), float(row['contour_sd_rmse_mm']))} "
            f"({improve_text(float(row['contour_improvement_pct_vs_baseline']))}) | "
            f"{rmse_sd(float(row['contour_audio_mean_rmse_mm']), float(row['contour_audio_sd_rmse_mm']))} "
            f"({improve_text(float(row['contour_audio_improvement_pct_vs_baseline']))}) |"
        )
    return "\n".join(lines)


def contact_sheet(images: list[Path], output: Path) -> Path:
    loaded = [Image.open(path).convert("RGB") for path in images]
    width = max(image.width for image in loaded)
    resized = []
    for image in loaded:
        if image.width == width:
            resized.append(image)
        else:
            height = int(round(image.height * width / image.width))
            resized.append(image.resize((width, height), Image.Resampling.LANCZOS))
    gap = 28
    canvas = Image.new(
        "RGB",
        (width, sum(image.height for image in resized) + gap * (len(resized) - 1)),
        "white",
    )
    y = 0
    for image in resized:
        canvas.paste(image, (0, y))
        y += image.height + gap
    canvas.save(output)
    return output


def main(args: argparse.Namespace) -> None:
    args.source_root = args.source_root.resolve()
    args.result_root = args.result_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store, inventory = load_store(args.source_root, args.result_root)
    unseen_arrays = {
        key: concatenate(store, UNSEEN_LABELS, key)
        for key in ("ground_truth", "baseline", "contour", "contour_audio")
    }
    summary_rows, summary_values = build_summary(unseen_arrays)
    articulator_rows, _per_stage = build_articulator_rows(unseen_arrays)
    speaker_rows, _per_speaker = build_speaker_rows(store)

    summary_csv = args.output_dir / "01_adaptation_summary.csv"
    articulator_csv = args.output_dir / "02_per_articulator_comparison.csv"
    speaker_csv = args.output_dir / "03_per_speaker_comparison.csv"
    write_csv(summary_csv, summary_rows)
    write_csv(articulator_csv, articulator_rows)
    write_csv(speaker_csv, speaker_rows)

    summary_md = markdown_summary(summary_rows)
    articulator_md = markdown_articulators(articulator_rows)
    speaker_md = markdown_speakers(speaker_rows)
    atomic_text(
        args.output_dir / "01_adaptation_summary.md",
        "# ASD2 Speaker Adaptation Summary\n\n" + summary_md + "\n",
    )
    atomic_text(
        args.output_dir / "02_per_articulator_comparison.md",
        "# ASD2 Per-Articulator Comparison\n\n" + articulator_md + "\n",
    )
    atomic_text(
        args.output_dir / "03_per_speaker_comparison.md",
        "# ASD2 Per-Speaker Comparison\n\n" + speaker_md + "\n",
    )

    summary_png, summary_pdf = render_summary(
        summary_rows, args.output_dir / "01_adaptation_summary"
    )
    articulator_png, articulator_pdf = render_articulators(
        articulator_rows, args.output_dir / "02_per_articulator_comparison"
    )
    speaker_png, speaker_pdf = render_speakers(
        speaker_rows, args.output_dir / "03_per_speaker_comparison"
    )
    sheet = contact_sheet(
        [summary_png, articulator_png, speaker_png],
        args.output_dir / "all_three_tables_contact_sheet.png",
    )

    statistical_tests = []
    for row in articulator_rows:
        statistical_tests.append(
            {
                "articulator": row["articulator"],
                "comparison": "contour_vs_baseline",
                "paired_t_statistic": row["contour_paired_t_statistic"],
                "p_value_raw": row["contour_p_value_raw"],
                "p_value_holm": row["contour_p_value_holm"],
                "significant_0_05": row["contour_significant_holm_0_05"],
            }
        )
        statistical_tests.append(
            {
                "articulator": row["articulator"],
                "comparison": "contour_audio_vs_baseline",
                "paired_t_statistic": row[
                    "contour_audio_paired_t_statistic"
                ],
                "p_value_raw": row["contour_audio_p_value_raw"],
                "p_value_holm": row["contour_audio_p_value_holm"],
                "significant_0_05": row[
                    "contour_audio_significant_holm_0_05"
                ],
            }
        )
    write_csv(args.output_dir / "statistical_tests.csv", statistical_tests)

    report = f"""# ASD2 adaptation publication tables

## Scope

- Model: ASD2 fixed-BS10 best human epoch 31.
- Unseen cohort: P1--P9, 8,547 integer frames.
- Same-person control: P10/S14, displayed only in the speaker table and excluded from unseen aggregates.
- Baseline: raw ASD2 prediction.
- Contour-side: affine+TPS exact-`/u/` adaptation.
- Contour + audio-side: RMS+VTLN input normalization followed by the same affine+TPS transform.
- Presentation uses all evaluated integer frames, including each fixed transform calibration frame, to match the earlier P7 table layout.
- No interpolation, held contour, fractional frame, fine-tuning, or retraining.

## Table 1 — Speaker Adaptation Summary

{summary_md}

## Table 2 — Per-Articulator Comparison

{articulator_md}

`*` denotes a significant paired difference versus baseline. Articulator rows use paired two-sided frame-level t-tests with Holm correction across 11 articulators separately for each adaptation branch. The Mean row uses a paired test on frame-level articulator means and is outside the Holm family.

## Table 3 — Per-Speaker Comparison

{speaker_md}

† P10/S14 is the same-person ASD2 control and is excluded from the two unseen aggregate rows.
"""
    atomic_text(args.output_dir / "REPORT.md", report)

    full_precision = {
        "summary": summary_rows,
        "per_articulator": articulator_rows,
        "per_speaker": speaker_rows,
        "summary_frame_values": {
            group: {
                stage: values.tolist()
                for stage, values in stages.items()
            }
            for group, stages in summary_values.items()
        },
    }
    atomic_json(args.output_dir / "metrics_full_precision.json", full_precision)

    outputs = [
        summary_csv,
        articulator_csv,
        speaker_csv,
        args.output_dir / "01_adaptation_summary.md",
        args.output_dir / "02_per_articulator_comparison.md",
        args.output_dir / "03_per_speaker_comparison.md",
        summary_png,
        summary_pdf,
        articulator_png,
        articulator_pdf,
        speaker_png,
        speaker_pdf,
        sheet,
        args.output_dir / "statistical_tests.csv",
        args.output_dir / "REPORT.md",
        args.output_dir / "metrics_full_precision.json",
    ]
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "passed",
        "generated_by": str(Path(__file__).resolve()),
        "model": "ASD2 fixed-BS10 best human epoch 31",
        "unseen_speakers": list(UNSEEN_LABELS),
        "same_person_control": CONTROL_LABEL,
        "unseen_frame_count": len(unseen_arrays["ground_truth"]),
        "control_frame_count": len(store[CONTROL_LABEL]["ground_truth"]),
        "all_frame_count": sum(
            len(store[label]["ground_truth"]) for label in common.PAIRS
        ),
        "training_launched": False,
        "inference_launched": False,
        "inputs": inventory,
        "outputs": [
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in outputs
        ],
    }
    atomic_json(args.output_dir / "MANIFEST.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_dir": str(args.output_dir),
                "unseen_frames": manifest["unseen_frame_count"],
                "images": [str(summary_png), str(articulator_png), str(speaker_png)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main(parse_args())
