"""Strict paired evaluation for the four adaptation conditions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..contracts import ContractError, atomic_write_json, load_cohort
from ..domain import (
    CONDITIONS,
    GLOBAL,
    MOVING_AVERAGE,
    PREDICTION_ARRAY_KEYS,
    STRATEGIES,
    anatomical_result_is_primary,
    strategy_title,
)
from ..io import load_mapping, write_rows_csv
from ..metrics import coordinate_rmse_mm, p2cp_mm


def load_pack(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "frame_numbers",
            "classes",
            "ground_truth",
            *PREDICTION_ARRAY_KEYS.values(),
        }
        missing = required.difference(payload.files)
        if missing:
            raise ContractError(f"Prediction pack {path} is missing {sorted(missing)}")
        return {
            "path": path,
            "frames": np.asarray(payload["frame_numbers"], dtype=np.int32),
            "classes": [str(item) for item in payload["classes"].tolist()],
            "ground_truth": np.asarray(payload["ground_truth"], dtype=np.float32),
            "conditions": {
                condition: np.asarray(payload[key], dtype=np.float32)
                for condition, key in PREDICTION_ARRAY_KEYS.items()
            },
            "target_frame": int(payload["target_u_frame"]),
        }


def mean_and_sample_sd(values: np.ndarray) -> Tuple[float, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(flat.mean()), float(flat.std(ddof=1)) if len(flat) > 1 else 0.0


def metric_rows(
    *,
    strategy: str,
    speaker: str,
    session: str,
    session_order: int,
    pack: Mapping[str, Any],
    coordinate_scale_mm: float,
    metric_root: Path,
    valid_anatomical: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    frames = np.asarray(pack["frames"])
    primary_mask = frames != int(pack["target_frame"])
    if not primary_mask.any():
        raise ContractError(f"No primary frames after calibration exclusion: {speaker}/{session}")
    session_rows: List[Dict[str, Any]] = []
    articulator_rows: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {
        "frame_numbers": frames,
        "primary_mask": primary_mask,
        "classes": np.asarray(pack["classes"], dtype="U64"),
    }
    for condition, predicted in pack["conditions"].items():
        if predicted.shape != pack["ground_truth"].shape:
            raise ContractError(
                f"{strategy}/{speaker}/{session}/{condition} shape mismatch"
            )
        p2cp = p2cp_mm(
            predicted,
            pack["ground_truth"],
            symmetric=True,
            coordinate_scale_mm=coordinate_scale_mm,
        )
        rmse = coordinate_rmse_mm(
            predicted,
            pack["ground_truth"],
            coordinate_scale_mm=coordinate_scale_mm,
        )
        arrays[f"{condition}_symmetric_p2cp_mm"] = p2cp.astype(np.float32)
        arrays[f"{condition}_coordinate_rmse_mm"] = rmse.astype(np.float32)
        primary_valid = valid_anatomical or condition in {"original", "audio"}
        for subset, mask in (
            ("all_frames", np.ones(len(frames), dtype=bool)),
            ("exclude_calibration_frame", primary_mask),
        ):
            for metric_name, values in (
                ("symmetric_p2cp_mm", p2cp),
                ("coordinate_rmse_mm", rmse),
            ):
                per_frame = values[mask].mean(axis=1)
                mean, sample_sd = mean_and_sample_sd(per_frame)
                session_rows.append(
                    {
                        "strategy": strategy,
                        "speaker": speaker,
                        "session": session,
                        "session_order": session_order,
                        "subset": subset,
                        "condition": condition,
                        "metric": metric_name,
                        "frames": int(mask.sum()),
                        "mean_mm": mean,
                        "sample_sd_mm": sample_sd,
                        "primary_valid": primary_valid,
                    }
                )
                for class_index, class_name in enumerate(pack["classes"]):
                    class_mean, class_sd = mean_and_sample_sd(
                        values[mask, class_index]
                    )
                    articulator_rows.append(
                        {
                            "strategy": strategy,
                            "speaker": speaker,
                            "session": session,
                            "session_order": session_order,
                            "subset": subset,
                            "condition": condition,
                            "metric": metric_name,
                            "articulator": class_name,
                            "frames": int(mask.sum()),
                            "mean_mm": class_mean,
                            "sample_sd_mm": class_sd,
                            "primary_valid": primary_valid,
                        }
                    )
    destination = metric_root / strategy / speaker / session / "frame_metrics.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.writing.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(destination)
    return session_rows, articulator_rows


def aggregate_rows(
    session_packs: List[Tuple[Mapping[str, Any], Mapping[str, Any]]],
    metric_root: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for strategy in STRATEGIES:
        valid_anatomical = anatomical_result_is_primary(strategy)
        for condition in CONDITIONS:
            for metric_name in ("symmetric_p2cp_mm", "coordinate_rmse_mm"):
                values = []
                for session, pack in session_packs:
                    metric_path = (
                        metric_root
                        / strategy
                        / session.speaker
                        / session.session
                        / "frame_metrics.npz"
                    )
                    with np.load(metric_path, allow_pickle=False) as cached:
                        mask = np.asarray(cached["primary_mask"], dtype=bool)
                        metric = np.asarray(
                            cached[f"{condition}_{metric_name}"], dtype=np.float32
                        )
                    values.append(metric[mask].mean(axis=1))
                combined = np.concatenate(values)
                mean, sample_sd = mean_and_sample_sd(combined)
                rows.append(
                    {
                        "strategy": strategy,
                        "subset": "exclude_calibration_frame",
                        "condition": condition,
                        "metric": metric_name,
                        "sessions": len(session_packs),
                        "frames": len(combined),
                        "mean_mm": mean,
                        "sample_sd_mm": sample_sd,
                        "primary_valid": valid_anatomical
                        or condition in {"original", "audio"},
                    }
                )
    return rows


def strategy_table_rows(
    aggregate: List[Dict[str, Any]], strategy: str
) -> List[Dict[str, Any]]:
    rows = []
    for metric in ("symmetric_p2cp_mm", "coordinate_rmse_mm"):
        selected = {
            item["condition"]: item
            for item in aggregate
            if item["strategy"] == strategy and item["metric"] == metric
        }
        baseline = selected["original"]["mean_mm"]
        for condition in CONDITIONS:
            item = selected[condition]
            rows.append(
                {
                    "metric": metric,
                    "condition": condition,
                    "frames": item["frames"],
                    "mean_mm": item["mean_mm"],
                    "sample_sd_mm": item["sample_sd_mm"],
                    "change_vs_original_mm": item["mean_mm"] - baseline,
                    "improvement_vs_original_percent": (
                        100.0 * (baseline - item["mean_mm"]) / baseline
                    ),
                    "primary_valid": item["primary_valid"],
                }
            )
    return rows


def markdown_table(strategy: str, rows: List[Dict[str, Any]]) -> str:
    title = strategy_title(strategy)
    lines = [
        f"# {title} evaluation",
        "",
        "| Metric | Condition | Mean ± SD (mm) | Δ vs original (mm) | Improvement | Primary valid |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {row['condition']} | "
            f"{row['mean_mm']:.4f} ± {row['sample_sd_mm']:.4f} | "
            f"{row['change_vs_original_mm']:+.4f} | "
            f"{row['improvement_vs_original_percent']:+.2f}% | "
            f"{'yes' if row['primary_valid'] else 'no'} |"
        )
    if strategy == MOVING_AVERAGE:
        lines.extend(
            [
                "",
                "> Moving-average output is already target-native because it uses "
                "target contour statistics. Anatomical and anatomical+audio are "
                "double-transform diagnostics, not primary results.",
            ]
        )
    return "\n".join(lines) + "\n"


def render_table_png(strategy: str, rows: List[Dict[str, Any]], output: Path) -> None:
    data = [
        [
            row["metric"].replace("_mm", ""),
            row["condition"],
            f"{row['mean_mm']:.3f} ± {row['sample_sd_mm']:.3f}",
            f"{row['improvement_vs_original_percent']:+.2f}%",
            "yes" if row["primary_valid"] else "no",
        ]
        for row in rows
    ]
    fig, axis = plt.subplots(figsize=(11, 4.2), dpi=180)
    axis.axis("off")
    table = axis.table(
        cellText=data,
        colLabels=["Metric", "Condition", "Mean ± SD (mm)", "vs original", "Primary"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    title = strategy_title(strategy)
    axis.set_title(f"{title}: exact paired evaluation (calibration frame excluded)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def comparison_rows(strategy_tables: Mapping[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for metric in ("symmetric_p2cp_mm", "coordinate_rmse_mm"):
        for condition in CONDITIONS:
            global_row = next(
                item
                for item in strategy_tables[GLOBAL]
                if item["metric"] == metric and item["condition"] == condition
            )
            moving_row = next(
                item
                for item in strategy_tables[MOVING_AVERAGE]
                if item["metric"] == metric and item["condition"] == condition
            )
            rows.append(
                {
                    "metric": metric,
                    "condition": condition,
                    "global_mean_mm": global_row["mean_mm"],
                    "moving_average_mean_mm": moving_row["mean_mm"],
                    "moving_minus_global_mm": (
                        moving_row["mean_mm"] - global_row["mean_mm"]
                    ),
                    "paired_frames": global_row["frames"],
                    "comparison_primary_valid": condition in {"original", "audio"},
                    "warning": (
                        ""
                        if condition in {"original", "audio"}
                        else (
                            "moving-average anatomical output is "
                            "double-transformed"
                        )
                    ),
                }
            )
    return rows


def run(args: Any) -> int:
    pipeline = load_mapping(args.pipeline_config)
    cohort = load_cohort(pipeline["cohort"])
    cohort.validate()
    coordinate_scale_mm = float(pipeline["evaluation"]["coordinate_scale_mm"])
    prediction_root = args.prediction_root.resolve()
    output = args.output_root.resolve()
    session_rows: List[Dict[str, Any]] = []
    articulator_rows: List[Dict[str, Any]] = []
    session_packs = []
    for session in cohort.ordered_sessions():
        packs = {}
        for strategy in STRATEGIES:
            path = (
                prediction_root
                / strategy
                / session.speaker
                / session.session
                / "baseline_anatomical.npz"
            )
            pack = load_pack(path)
            if pack["target_frame"] != session.reference_frame:
                raise ContractError(f"Calibration frame mismatch in {path}")
            packs[strategy] = pack
            current_session_rows, current_articulator_rows = metric_rows(
                strategy=strategy,
                speaker=session.speaker,
                session=session.session,
                session_order=session.session_order,
                pack=pack,
                coordinate_scale_mm=coordinate_scale_mm,
                metric_root=output / "per_frame",
                valid_anatomical=anatomical_result_is_primary(strategy),
            )
            session_rows.extend(current_session_rows)
            articulator_rows.extend(current_articulator_rows)
        left = packs[GLOBAL]
        right = packs[MOVING_AVERAGE]
        if not np.array_equal(left["frames"], right["frames"]):
            raise ContractError(f"Strategy frame mismatch for {session.key}")
        if not np.array_equal(left["ground_truth"], right["ground_truth"]):
            raise ContractError(f"Strategy ground-truth mismatch for {session.key}")
        session_packs.append((session, packs))

    aggregate = aggregate_rows(session_packs, output / "per_frame")
    write_rows_csv(output / "per_session_metrics.csv", session_rows)
    write_rows_csv(output / "per_articulator_metrics.csv", articulator_rows)
    write_rows_csv(output / "aggregate_metrics.csv", aggregate)
    strategy_tables = {}
    for strategy in STRATEGIES:
        rows = strategy_table_rows(aggregate, strategy)
        strategy_tables[strategy] = rows
        strategy_dir = output / strategy
        write_rows_csv(strategy_dir / "evaluation_table.csv", rows)
        (strategy_dir / "evaluation_table.md").write_text(
            markdown_table(strategy, rows), encoding="utf-8"
        )
        render_table_png(strategy, rows, strategy_dir / "evaluation_table.png")
    comparison = comparison_rows(strategy_tables)
    write_rows_csv(output / "strategy_comparison.csv", comparison)
    atomic_write_json(
        output / "evaluation.json",
        {
            "status": "complete",
            "primary_subset": "exclude_calibration_frame",
            "primary_metric": "symmetric_p2cp_mm",
            "secondary_metric": "coordinate_rmse_mm",
            "p2cp_definition": (
                "symmetric mean point-to-curve distance between prediction and "
                "ground truth within each MRI frame, then aggregated across frames"
            ),
            "coordinate_scale_mm": coordinate_scale_mm,
            "strict_pairing": {
                "same_strategy_session_order": True,
                "same_frame_order": True,
                "same_ground_truth": True,
                "sessions": len(session_packs),
                "frames_per_strategy": int(
                    sum(
                        len(packs[GLOBAL]["frames"])
                        for _session, packs in session_packs
                    )
                ),
            },
            "strategy_tables": strategy_tables,
            "comparison": comparison,
            "training_launched": False,
        },
    )
    print(
        f"DONE evaluation: {len(session_packs)} sessions, "
        f"{sum(len(packs[GLOBAL]['frames']) for _session, packs in session_packs)} "
        "paired frames per strategy",
        flush=True,
    )
    return 0
