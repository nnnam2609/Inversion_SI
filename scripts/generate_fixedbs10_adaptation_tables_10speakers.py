#!/usr/bin/env python3
"""Generate aggregate, per-contour, and per-speaker ASD2 adaptation tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "results/asd2_fixedbs10_selected_9sessions_textgrid_u_grid_adaptation_20260721_143350"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results/asd2_fixedbs10_selected_10speakers_adaptation_tables_20260724"
)
MM_PER_PIXEL = 1.62
PAIRS = OrderedDict(
    [
        ("P1/S16", (1, 16)),
        ("P2/S9", (2, 9)),
        ("P3/S14", (3, 14)),
        ("P4/S4", (4, 4)),
        ("P5/S6", (5, 6)),
        ("P6/S8", (6, 8)),
        ("P7/S2", (7, 2)),
        ("P8/S2", (8, 2)),
        ("P9/S5", (9, 5)),
        ("P10/S14", (10, 14)),
    ]
)
EXPECTED_CLASSES = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def paths_for_pair(
    source_root: Path, output_root: Path, pair: tuple[int, int]
) -> tuple[Path, Path]:
    speaker, session = pair
    if speaker == 7:
        base = output_root / "p7_extension/P7/S2"
        return base / "baseline.npz", base / "rms_vtln.npz"
    return (
        source_root / f"predictions/ASD2/P{speaker}/S{session}/baseline.npz",
        source_root
        / f"phase_b/demo_inputs/P{speaker}/S{session}/rms_vtln_exact_u_fixedbs10.npz",
    )


def validate_pair(
    label: str, baseline: dict[str, np.ndarray], audio: dict[str, np.ndarray]
) -> None:
    required = (
        "frame_numbers",
        "predicted_raw",
        "predicted_after_corrected_affine_tps",
        "ground_truth",
        "classes",
    )
    for branch_name, payload in (("baseline", baseline), ("audio", audio)):
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"{label} {branch_name} missing keys: {missing}")
        classes = tuple(str(value) for value in payload["classes"].tolist())
        if classes != EXPECTED_CLASSES:
            raise ValueError(f"{label} {branch_name} class order differs")
        shape = payload["ground_truth"].shape
        if shape[1:] != (11, 50, 2):
            raise ValueError(f"{label} {branch_name} unexpected shape: {shape}")
    baseline_frames = baseline["frame_numbers"].astype(np.int32)
    audio_frames = audio["frame_numbers"].astype(np.int32)
    if not np.array_equal(baseline_frames, audio_frames):
        raise RuntimeError(f"{label} branch timelines differ")
    if len(np.unique(baseline_frames)) != len(baseline_frames):
        raise RuntimeError(f"{label} contains duplicate frames")
    if not np.array_equal(
        baseline["ground_truth"].astype(np.float32),
        audio["ground_truth"].astype(np.float32),
    ):
        raise RuntimeError(f"{label} branch ground truth differs")
    for payload in (baseline, audio):
        for key in required[1:4]:
            if not np.isfinite(payload[key]).all():
                raise ValueError(f"{label} {key} contains non-finite values")


def frame_rmse(predicted: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    delta = predicted.astype(np.float64) - ground_truth.astype(np.float64)
    return np.sqrt(np.mean(delta * delta, axis=(1, 2, 3))) * MM_PER_PIXEL


def contour_frame_rmse(
    predicted: np.ndarray, ground_truth: np.ndarray
) -> np.ndarray:
    delta = predicted.astype(np.float64) - ground_truth.astype(np.float64)
    return np.sqrt(np.mean(delta * delta, axis=(2, 3))) * MM_PER_PIXEL


def mean_sd(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def improvement(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100.0


def metric_fields(
    baseline: np.ndarray, contour: np.ndarray, contour_audio: np.ndarray
) -> dict[str, float]:
    baseline_mean, baseline_sd = mean_sd(baseline)
    contour_mean, contour_sd = mean_sd(contour)
    audio_mean, audio_sd = mean_sd(contour_audio)
    return {
        "baseline_mean_rmse_mm": baseline_mean,
        "baseline_sd_rmse_mm": baseline_sd,
        "contour_mean_rmse_mm": contour_mean,
        "contour_sd_rmse_mm": contour_sd,
        "contour_improvement_pct_vs_baseline": improvement(
            baseline_mean, contour_mean
        ),
        "contour_audio_mean_rmse_mm": audio_mean,
        "contour_audio_sd_rmse_mm": audio_sd,
        "contour_audio_improvement_pct_vs_baseline": improvement(
            baseline_mean, audio_mean
        ),
        "audio_incremental_improvement_pct_vs_contour": improvement(
            contour_mean, audio_mean
        ),
    }


def formatted_metric(mean: float, sd: float) -> str:
    return f"{mean:.2f} ± {sd:.2f}"


def formatted_improvement(value: float) -> str:
    return f"↓{value:.2f}%" if value >= 0 else f"↑{-value:.2f}%"


def markdown_table(
    rows: list[dict[str, Any]], first_key: str, first_label: str
) -> str:
    lines = [
        f"| {first_label} | Frames | Baseline Mean RMSE ± SD | Contour Mean RMSE ± SD | Improve | Contour+Audio Mean RMSE ± SD | Improve | Audio vs Contour |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row[first_key]} | {int(row['frames'])} | "
            f"{formatted_metric(float(row['baseline_mean_rmse_mm']), float(row['baseline_sd_rmse_mm']))} | "
            f"{formatted_metric(float(row['contour_mean_rmse_mm']), float(row['contour_sd_rmse_mm']))} | "
            f"{formatted_improvement(float(row['contour_improvement_pct_vs_baseline']))} | "
            f"{formatted_metric(float(row['contour_audio_mean_rmse_mm']), float(row['contour_audio_sd_rmse_mm']))} | "
            f"{formatted_improvement(float(row['contour_audio_improvement_pct_vs_baseline']))} | "
            f"{formatted_improvement(float(row['audio_incremental_improvement_pct_vs_contour']))} |"
        )
    return "\n".join(lines)


def concatenate(
    store: dict[str, dict[str, np.ndarray]], labels: Iterable[str], stage: str
) -> np.ndarray:
    return np.concatenate([store[label][stage] for label in labels], axis=0)


def main(args: argparse.Namespace) -> None:
    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    store: dict[str, dict[str, np.ndarray]] = {}
    input_inventory = []
    for label, pair in PAIRS.items():
        baseline_path, audio_path = paths_for_pair(
            args.source_root, args.output_root, pair
        )
        baseline = load_npz(baseline_path)
        audio = load_npz(audio_path)
        validate_pair(label, baseline, audio)
        ground_truth = baseline["ground_truth"].astype(np.float32)
        store[label] = {
            "baseline": frame_rmse(
                baseline["predicted_raw"].astype(np.float32), ground_truth
            ),
            "contour": frame_rmse(
                baseline["predicted_after_corrected_affine_tps"].astype(np.float32),
                ground_truth,
            ),
            "contour_audio": frame_rmse(
                audio["predicted_after_corrected_affine_tps"].astype(np.float32),
                ground_truth,
            ),
            "baseline_per_contour": contour_frame_rmse(
                baseline["predicted_raw"].astype(np.float32), ground_truth
            ),
            "contour_per_contour": contour_frame_rmse(
                baseline["predicted_after_corrected_affine_tps"].astype(np.float32),
                ground_truth,
            ),
            "contour_audio_per_contour": contour_frame_rmse(
                audio["predicted_after_corrected_affine_tps"].astype(np.float32),
                ground_truth,
            ),
        }
        input_inventory.append(
            {
                "speaker_session": label,
                "frames": len(store[label]["baseline"]),
                "baseline_pack": str(baseline_path.resolve()),
                "baseline_pack_sha256": sha256(baseline_path),
                "contour_audio_pack": str(audio_path.resolve()),
                "contour_audio_pack_sha256": sha256(audio_path),
            }
        )

    speaker_rows = []
    for label in PAIRS:
        values = store[label]
        speaker_rows.append(
            {
                "speaker_session": label,
                "frames": len(values["baseline"]),
                **metric_fields(
                    values["baseline"],
                    values["contour"],
                    values["contour_audio"],
                ),
                "sd_unit": "frames",
                "notes": (
                    "same-person ASD2 control"
                    if label.startswith("P10/")
                    else "ASD1 target speaker"
                ),
            }
        )

    all_labels = list(PAIRS)
    unseen_labels = [label for label in all_labels if not label.startswith("P10/")]
    macro_values = {
        stage: np.asarray(
            [float(np.mean(store[label][stage])) for label in all_labels],
            dtype=np.float64,
        )
        for stage in ("baseline", "contour", "contour_audio")
    }
    aggregate_rows = [
        {
            "aggregate": "Macro mean (P1-P10)",
            "speakers": 10,
            "frames": sum(len(store[label]["baseline"]) for label in all_labels),
            **metric_fields(
                macro_values["baseline"],
                macro_values["contour"],
                macro_values["contour_audio"],
            ),
            "sd_unit": "speaker means",
            "notes": "equal weight per speaker/session",
        },
        {
            "aggregate": "All frames weighted (P1-P10)",
            "speakers": 10,
            "frames": sum(len(store[label]["baseline"]) for label in all_labels),
            **metric_fields(
                concatenate(store, all_labels, "baseline"),
                concatenate(store, all_labels, "contour"),
                concatenate(store, all_labels, "contour_audio"),
            ),
            "sd_unit": "frames",
            "notes": "frame weighted; includes P10 same-person control",
        },
        {
            "aggregate": "Unseen frames weighted (P1-P9)",
            "speakers": 9,
            "frames": sum(len(store[label]["baseline"]) for label in unseen_labels),
            **metric_fields(
                concatenate(store, unseen_labels, "baseline"),
                concatenate(store, unseen_labels, "contour"),
                concatenate(store, unseen_labels, "contour_audio"),
            ),
            "sd_unit": "frames",
            "notes": "frame weighted; P10 control excluded",
        },
    ]

    all_baseline_contours = concatenate(
        store, all_labels, "baseline_per_contour"
    )
    all_contour_contours = concatenate(
        store, all_labels, "contour_per_contour"
    )
    all_audio_contours = concatenate(
        store, all_labels, "contour_audio_per_contour"
    )
    contour_rows = []
    for class_index, class_name in enumerate(EXPECTED_CLASSES):
        contour_rows.append(
            {
                "contour": class_name,
                "frames": len(all_baseline_contours),
                **metric_fields(
                    all_baseline_contours[:, class_index],
                    all_contour_contours[:, class_index],
                    all_audio_contours[:, class_index],
                ),
                "sd_unit": "frames",
                "notes": "P1-P10 frame weighted",
            }
        )

    macro_speaker_row = {
        "speaker_session": "Macro mean (P1-P10)",
        "frames": aggregate_rows[0]["frames"],
        **{
            key: value
            for key, value in aggregate_rows[0].items()
            if key.endswith("_mm") or "_pct_" in key
        },
        "sd_unit": "speaker means",
        "notes": "equal weight per speaker/session",
    }
    weighted_speaker_row = {
        "speaker_session": "All frames weighted",
        "frames": aggregate_rows[1]["frames"],
        **{
            key: value
            for key, value in aggregate_rows[1].items()
            if key.endswith("_mm") or "_pct_" in key
        },
        "sd_unit": "frames",
        "notes": "P1-P10 frame weighted",
    }
    speaker_rows_with_summary = speaker_rows + [
        macro_speaker_row,
        weighted_speaker_row,
    ]

    write_csv(args.output_root / "aggregate_results.csv", aggregate_rows)
    write_csv(args.output_root / "per_contour_results.csv", contour_rows)
    write_csv(args.output_root / "per_speaker_results.csv", speaker_rows_with_summary)

    aggregate_md = markdown_table(aggregate_rows, "aggregate", "Aggregate")
    contour_md = markdown_table(contour_rows, "contour", "Contour")
    speaker_md = markdown_table(
        speaker_rows_with_summary, "speaker_session", "Speaker"
    )
    report = f"""# ASD2 fixed-BS10 adaptation on ASD1 P1-P10

Model: fixed-batch-10 ASD2 checkpoint, best human epoch 31.

- Baseline: raw ASD2 prediction.
- Contour: the same prediction after one fixed exact-`/u/` affine+TPS transform per session.
- Contour+Audio: RMS+VTLN model-input normalization followed by the same affine+TPS transform.
- Unit: millimetres. SD is the sample SD of per-frame RMSE, except the macro row where it is the sample SD across speaker means.
- Positive improvement is shown with `↓`; a regression is shown with `↑`.
- These tables use all evaluated integer frames to match the previous P7 table layout. No interpolation, held contour, fractional frame, fine-tuning, or retraining is used.

## Kết quả tổng hợp

{aggregate_md}

## Kết quả theo từng contour

{contour_md}

## Kết quả theo từng speaker

{speaker_md}
"""
    atomic_text(args.output_root / "report.md", report)
    atomic_text(
        args.output_root / "aggregate_results.md",
        "# Kết quả tổng hợp\n\n" + aggregate_md + "\n",
    )
    atomic_text(
        args.output_root / "per_contour_results.md",
        "# Kết quả theo từng contour\n\n" + contour_md + "\n",
    )
    atomic_text(
        args.output_root / "per_speaker_results.md",
        "# Kết quả theo từng speaker\n\n" + speaker_md + "\n",
    )
    atomic_json(
        args.output_root / "metrics_full_precision.json",
        {
            "aggregate": aggregate_rows,
            "per_contour": contour_rows,
            "per_speaker": speaker_rows_with_summary,
        },
    )
    outputs = [
        args.output_root / name
        for name in (
            "aggregate_results.csv",
            "aggregate_results.md",
            "per_contour_results.csv",
            "per_contour_results.md",
            "per_speaker_results.csv",
            "per_speaker_results.md",
            "report.md",
            "metrics_full_precision.json",
        )
    ]
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "passed",
        "model": "ASD2 fixed-BS10 best human epoch 31",
        "population": "ASD1 P1-P10 selected sessions",
        "speaker_count": 10,
        "frame_count": int(aggregate_rows[1]["frames"]),
        "metric": (
            "mean and sample SD of per-frame coordinate RMSE; "
            "50 points x 2 coordinates per contour; 1.62 mm/pixel"
        ),
        "table_scope": "all evaluated integer frames",
        "baseline": "raw model prediction",
        "contour": "baseline prediction after exact-/u/ affine+TPS",
        "contour_audio": "RMS+VTLN input prediction after exact-/u/ affine+TPS",
        "training_launched": False,
        "saved_fractional_frame_count": 0,
        "scored_fractional_frame_count": 0,
        "rendered_fractional_frame_count": 0,
        "inputs": input_inventory,
        "outputs": [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
            for path in outputs
        ],
    }
    atomic_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(parse_args())
