#!/usr/bin/env python3
"""RMS-only and VTLN-only ablation for the selected P7 cross-speaker run."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")]

from run_p7_all_nonp7_gridnorm import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_NORM_STATS,
    DEFAULT_RAW_CACHE,
    DEFAULT_VTLN_DIR,
    EXCLUDED_CLASSES,
    SOURCE,
    STAGES,
    build_two_step_transform,
    load_normalization,
    metric_payload,
    prepare_frame,
    target_spec_for_speaker,
    transform_contour_batch,
)
from run_p7_selected_audio_gridnorm import (  # noqa: E402
    DEFAULT_BASELINE_ROOT,
    DEFAULT_OUTPUT_ROOT as DEFAULT_COMBINED_ROOT,
    SELECTION,
    TARGET_RMS,
    build_audio_normalized_chunks,
    infer_with_features,
    load_audio_pack,
    load_baseline_pack,
    write_alignment,
)
from src.inference.session_inference import load_model  # noqa: E402
from src.utils.config_validation import load_yaml_config  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/p7_selected_nonp7_sessions_audio_ablation_20260718"
VARIANTS = ("baseline", "rms_only", "vtln_only", "rms_vtln")
VARIANT_LABELS = {
    "baseline": "Baseline",
    "rms_only": "RMS-only",
    "vtln_only": "VTLN-only",
    "rms_vtln": "RMS+VTLN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--normalization-stats", type=Path, default=DEFAULT_NORM_STATS)
    parser.add_argument("--raw-cache-root", type=Path, default=DEFAULT_RAW_CACHE)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--combined-root", type=Path, default=DEFAULT_COMBINED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vtln-dir", type=Path, default=DEFAULT_VTLN_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--transform-frame-batch", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    for path in (
        args.config,
        args.checkpoint,
        args.normalization_stats,
        args.raw_cache_root,
        args.baseline_root,
        args.combined_root,
        args.vtln_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    for speaker, session in SELECTION:
        paths = (
            args.raw_cache_root / f"P{speaker}/S{session}.pt",
            args.baseline_root / f"P{speaker}/S{session}/contours_and_ground_truth.npz",
            args.combined_root / f"P{speaker}/S{session}/audio_normalized_contours_and_ground_truth.npz",
        )
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)


def load_alphas(combined_root: Path) -> dict[str, float]:
    path = combined_root / "audio_normalization/alpha_to_p7.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    alphas = {str(key): float(value) for key, value in payload["alpha_to_p7"].items()}
    expected = {f"P{speaker}" for speaker, _ in SELECTION}
    if not expected <= set(alphas):
        raise KeyError(f"Missing alpha values in {path}: {sorted(expected - set(alphas))}")
    return alphas


def save_variant_pack(
    path: Path,
    variant: str,
    inferred: dict[str, np.ndarray],
    affine: np.ndarray,
    final: np.ndarray,
    classes: list[str],
    alpha: float,
    rms_target: float | None,
) -> None:
    np.savez_compressed(
        path,
        variant=np.asarray(variant),
        frame_numbers=inferred["frame_numbers"],
        phonemes=inferred["phonemes"],
        overlap_counts=inferred["overlap_counts"],
        predicted_raw=inferred["predicted_raw"],
        predicted_after_affine=affine,
        predicted_after_affine_tps=final,
        ground_truth=inferred["ground_truth"],
        classes=np.asarray(classes, dtype="U64"),
        excluded_classes=np.asarray(EXCLUDED_CLASSES, dtype="U64"),
        vtln_alpha_to_p7=np.asarray(alpha, dtype=np.float32),
        target_rms=np.asarray(np.nan if rms_target is None else rms_target, dtype=np.float32),
        num_input_rows=inferred["num_input_rows"],
        num_sequences=inferred["num_sequences"],
    )


def load_variant_pack(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "arrays": {
                "raw": np.asarray(payload["predicted_raw"], dtype=np.float32),
                "affine": np.asarray(payload["predicted_after_affine"], dtype=np.float32),
                "affine_tps": np.asarray(payload["predicted_after_affine_tps"], dtype=np.float32),
            },
            "ground_truth": np.asarray(payload["ground_truth"], dtype=np.float32),
            "frame_numbers": np.asarray(payload["frame_numbers"], dtype=np.float32),
            "phonemes": np.asarray(payload["phonemes"]),
            "num_input_rows": int(payload["num_input_rows"]),
            "num_sequences": int(payload["num_sequences"]),
        }


def write_frame_metrics(
    path: Path,
    frames: np.ndarray,
    phones: np.ndarray,
    metrics: dict[str, dict[str, dict[str, np.ndarray]]],
) -> None:
    fields = ["frame_number", "phoneme"]
    for mode in ("all_11", "without_laryngeal_3"):
        for variant in VARIANTS:
            fields.extend(f"{variant}_{stage}_{mode}_rmse_mm" for stage in STAGES)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, frame in enumerate(frames):
            row: dict[str, Any] = {"frame_number": float(frame), "phoneme": str(phones[index])}
            for mode in ("all_11", "without_laryngeal_3"):
                for variant in VARIANTS:
                    for stage in STAGES:
                        row[f"{variant}_{stage}_{mode}_rmse_mm"] = float(
                            metrics[variant][mode][stage][index]
                        )
            writer.writerow(row)


def build_variant(
    args: argparse.Namespace,
    variant: str,
    speaker: int,
    session: int,
    alpha_to_p7: float,
    config: dict[str, Any],
    classes: list[str],
    phonemes: list[str],
    normalization: dict[str, np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
    transform: dict[str, Any],
    raw: dict[str, Any],
    session_dir: Path,
) -> dict[str, Any]:
    if variant == "rms_only":
        alpha, rms_target = 1.0, TARGET_RMS
    elif variant == "vtln_only":
        alpha, rms_target = alpha_to_p7, None
    else:
        raise ValueError(variant)
    pack_path = session_dir / f"{variant}_contours_and_ground_truth.npz"
    metadata_path = session_dir / f"{variant}_extraction_metadata.json"
    alignment_path = session_dir / f"{variant}_chunk_alignment.csv"
    if pack_path.is_file() and metadata_path.is_file() and not args.force:
        print(f"REUSE P{speaker}/S{session} {variant}", flush=True)
        return load_variant_pack(pack_path)

    feature_chunks, alignment, metadata = build_audio_normalized_chunks(
        config,
        speaker,
        session,
        alpha,
        raw["features"],
        rms_target=rms_target,
    )
    metadata.update(
        {
            "variant": variant,
            "audio_operations": "RMS only" if variant == "rms_only" else "VTLN only",
            "model_frontend": "historical Inversion_SI 128-mel MFCC39",
        }
    )
    write_alignment(alignment_path, alignment)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    inferred = infer_with_features(
        model, device, raw, feature_chunks, normalization, phonemes, args.batch_size
    )
    affine, final = transform_contour_batch(
        inferred["predicted_raw"], transform, args.transform_frame_batch
    )
    save_variant_pack(
        pack_path, variant, inferred, affine, final, classes, alpha, rms_target
    )
    return {
        "arrays": {"raw": inferred["predicted_raw"], "affine": affine, "affine_tps": final},
        "ground_truth": inferred["ground_truth"],
        "frame_numbers": inferred["frame_numbers"],
        "phonemes": inferred["phonemes"],
        "num_input_rows": int(inferred["num_input_rows"]),
        "num_sequences": int(inferred["num_sequences"]),
    }


def process_session(
    args: argparse.Namespace,
    speaker: int,
    session: int,
    alpha: float,
    config: dict[str, Any],
    classes: list[str],
    phonemes: list[str],
    normalization: dict[str, np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
    transform: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    session_dir = args.output_root / f"P{speaker}/S{session}"
    session_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.baseline_root / f"P{speaker}/S{session}/contours_and_ground_truth.npz"
    combined_path = args.combined_root / f"P{speaker}/S{session}/audio_normalized_contours_and_ground_truth.npz"
    raw_path = args.raw_cache_root / f"P{speaker}/S{session}.pt"
    baseline = load_baseline_pack(baseline_path)
    combined = load_audio_pack(combined_path)
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)["raw"]
    payloads: dict[str, dict[str, Any]] = {
        "baseline": baseline,
        "rms_vtln": combined,
    }
    for variant in ("rms_only", "vtln_only"):
        payloads[variant] = build_variant(
            args,
            variant,
            speaker,
            session,
            alpha,
            config,
            classes,
            phonemes,
            normalization,
            model,
            device,
            transform,
            raw,
            session_dir,
        )

    reference_frames = baseline["frame_numbers"]
    reference_gt = baseline["ground_truth"]
    audit: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    frame_metrics: dict[str, Any] = {}
    for variant in VARIANTS:
        payload = payloads[variant]
        frames_equal = bool(np.array_equal(payload["frame_numbers"], reference_frames))
        gt_delta = float(np.max(np.abs(payload["ground_truth"] - reference_gt)))
        finite = bool(
            all(np.isfinite(payload["arrays"][stage]).all() for stage in STAGES)
        )
        if not frames_equal or gt_delta > 1e-5 or not finite:
            raise RuntimeError(
                f"Ablation audit failed P{speaker}/S{session} {variant}: "
                f"frames={frames_equal}, gt_delta={gt_delta}, finite={finite}"
            )
        audit[variant] = {
            "frame_timeline_equal": frames_equal,
            "ground_truth_max_abs_delta": gt_delta,
            "all_predictions_finite": finite,
        }
        summaries[variant], frame_metrics[variant] = metric_payload(
            payload["arrays"], reference_gt, classes
        )

    frame_csv = session_dir / "ablation_frame_metrics.csv"
    write_frame_metrics(
        frame_csv, reference_frames, baseline["phonemes"], frame_metrics
    )
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "speaker": speaker,
        "session": session,
        "num_unique_frames": int(len(reference_frames)),
        "vtln_alpha_to_p7": float(alpha),
        "target_rms": TARGET_RMS,
        "paths": {
            "baseline_pack": str(baseline_path.resolve()),
            "rms_only_pack": str((session_dir / "rms_only_contours_and_ground_truth.npz").resolve()),
            "vtln_only_pack": str((session_dir / "vtln_only_contours_and_ground_truth.npz").resolve()),
            "rms_vtln_pack": str(combined_path.resolve()),
            "frame_metrics": str(frame_csv.resolve()),
        },
        "audit": audit,
        "metrics": summaries,
        "elapsed_seconds": time.monotonic() - started,
    }
    (session_dir / "session_ablation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    final = {
        variant: summaries[variant]["modes"]["all_11"]["affine_tps"]["mean_frame_rmse_mm"]
        for variant in VARIANTS
    }
    print(
        f"DONE P{speaker}/S{session}: "
        + ", ".join(f"{variant}={final[variant]:.3f}" for variant in VARIANTS)
        + f" mm ({summary['elapsed_seconds']:.1f}s)",
        flush=True,
    )
    return summary


def stage_mean(summary: dict[str, Any], variant: str, mode: str, stage: str) -> float:
    return float(summary["metrics"][variant]["modes"][mode][stage]["mean_frame_rmse_mm"])


def weighted_metric(
    summaries: list[dict[str, Any]], variant: str, mode: str, stage: str
) -> float:
    total = sum(int(row["num_unique_frames"]) for row in summaries)
    return sum(
        int(row["num_unique_frames"]) * stage_mean(row, variant, mode, stage)
        for row in summaries
    ) / total


def weighted_class_metric(
    summaries: list[dict[str, Any]], variant: str, stage: str, class_name: str
) -> float:
    total = sum(int(row["num_unique_frames"]) for row in summaries)
    return sum(
        int(row["num_unique_frames"])
        * float(row["metrics"][variant]["per_class_mean_frame_rmse_mm"][stage][class_name])
        for row in summaries
    ) / total


def make_tables(
    summaries: list[dict[str, Any]], classes: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    groups = [(f"P{row['speaker']}", [row]) for row in summaries] + [("ALL", summaries)]
    for speaker_label, group in groups:
        session_label = f"S{group[0]['session']}" if len(group) == 1 else "selected_9"
        for mode in ("all_11", "without_laryngeal_3"):
            for variant in VARIANTS:
                values = {
                    stage: weighted_metric(group, variant, mode, stage) for stage in STAGES
                }
                long_rows.append(
                    {
                        "speaker": speaker_label,
                        "session": session_label,
                        "frames": sum(int(row["num_unique_frames"]) for row in group),
                        "metric_mode": mode,
                        "variant": variant,
                        "raw_rmse_mm": values["raw"],
                        "affine_rmse_mm": values["affine"],
                        "affine_tps_rmse_mm": values["affine_tps"],
                        "raw_to_affine_mm": values["affine"] - values["raw"],
                        "affine_to_tps_mm": values["affine_tps"] - values["affine"],
                        "raw_to_final_mm": values["affine_tps"] - values["raw"],
                    }
                )
            final = {
                variant: weighted_metric(group, variant, mode, "affine_tps")
                for variant in VARIANTS
            }
            comparison_rows.append(
                {
                    "speaker": speaker_label,
                    "session": session_label,
                    "frames": sum(int(row["num_unique_frames"]) for row in group),
                    "metric_mode": mode,
                    "baseline_final_mm": final["baseline"],
                    "rms_only_final_mm": final["rms_only"],
                    "rms_effect_mm": final["rms_only"] - final["baseline"],
                    "vtln_only_final_mm": final["vtln_only"],
                    "vtln_effect_mm": final["vtln_only"] - final["baseline"],
                    "rms_vtln_final_mm": final["rms_vtln"],
                    "combined_effect_mm": final["rms_vtln"] - final["baseline"],
                    "non_additive_interaction_mm": (
                        final["rms_vtln"] - final["rms_only"] - final["vtln_only"] + final["baseline"]
                    ),
                    "best_variant": min(final, key=final.get),
                }
            )

    class_rows: list[dict[str, Any]] = []
    for class_name in classes:
        for stage in STAGES:
            values = {
                variant: weighted_class_metric(summaries, variant, stage, class_name)
                for variant in VARIANTS
            }
            class_rows.append(
                {
                    "class": class_name,
                    "excluded_in_without_3": class_name in EXCLUDED_CLASSES,
                    "stage": stage,
                    "baseline_mm": values["baseline"],
                    "rms_only_mm": values["rms_only"],
                    "rms_effect_mm": values["rms_only"] - values["baseline"],
                    "vtln_only_mm": values["vtln_only"],
                    "vtln_effect_mm": values["vtln_only"] - values["baseline"],
                    "rms_vtln_mm": values["rms_vtln"],
                    "combined_effect_mm": values["rms_vtln"] - values["baseline"],
                }
            )
    return long_rows, comparison_rows, class_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def signed(value: float) -> str:
    return f"{value:+.3f}"


def generate_report(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    summaries = []
    for speaker, session in SELECTION:
        path = args.output_root / f"P{speaker}/S{session}/session_ablation_summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    classes = list(config["classes"])
    long_rows, comparisons, class_rows = make_tables(summaries, classes)
    long_path = args.output_root / "ablation_metrics_long.csv"
    comparison_path = args.output_root / "ablation_final_comparison.csv"
    class_path = args.output_root / "ablation_per_class_metrics.csv"
    write_csv(long_path, long_rows)
    write_csv(comparison_path, comparisons)
    write_csv(class_path, class_rows)

    report_path = args.output_root / "rms_vtln_ablation_detailed_report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# RMS-only and VTLN-only ablation — P7 model on selected unseen speakers\n\n")
        handle.write("## Experimental control\n\n")
        handle.write(
            "All four variants use the same P7 checkpoint, P7 train normalization statistics, cached "
            "annotated chunks, silence policy, ground truth, and P7→target affine/TPS transforms. Only "
            "the waveform operation before the inversion-compatible MFCC frontend changes:\n\n"
            "- **Baseline:** original waveform, alpha 1.0.\n"
            f"- **RMS-only:** waveform RMS set to {TARGET_RMS:.2f}, alpha 1.0.\n"
            "- **VTLN-only:** original waveform amplitude, speaker/session alpha estimated toward P7.\n"
            f"- **RMS+VTLN:** waveform RMS set to {TARGET_RMS:.2f} plus the same VTLN alpha.\n\n"
            "Negative effects below mean lower error than baseline. `Without 3` excludes vocal-folds, "
            "thyroid-cartilage, and epiglottis.\n\n"
        )

        handle.write("## Overall stage-by-stage result\n\n")
        for mode, title in (
            ("all_11", "All 11 contours"),
            ("without_laryngeal_3", "Without the three laryngeal contours"),
        ):
            handle.write(f"### {title}\n\n")
            handle.write("| Audio variant | Raw | Affine | Affine+TPS | Raw→final |\n|---|---:|---:|---:|---:|\n")
            rows = [
                row for row in long_rows
                if row["speaker"] == "ALL" and row["metric_mode"] == mode
            ]
            for row in rows:
                handle.write(
                    f"| {VARIANT_LABELS[row['variant']]} | {row['raw_rmse_mm']:.3f} | "
                    f"{row['affine_rmse_mm']:.3f} | {row['affine_tps_rmse_mm']:.3f} | "
                    f"{signed(row['raw_to_final_mm'])} |\n"
                )
            handle.write("\n")

        handle.write("## Final TPS result per speaker\n\n")
        for mode, title in (
            ("all_11", "All 11 contours"),
            ("without_laryngeal_3", "Without 3"),
        ):
            handle.write(f"### {title}\n\n")
            handle.write(
                "| Speaker/session | Baseline | RMS-only | Δ RMS | VTLN-only | Δ VTLN | "
                "RMS+VTLN | Δ combined | Best |\n"
                "|---|---:|---:|---:|---:|---:|---:|---:|---|\n"
            )
            rows = [row for row in comparisons if row["metric_mode"] == mode]
            for row in rows:
                label = "ALL" if row["speaker"] == "ALL" else f"{row['speaker']}/{row['session']}"
                handle.write(
                    f"| {label} | {row['baseline_final_mm']:.3f} | {row['rms_only_final_mm']:.3f} | "
                    f"{signed(row['rms_effect_mm'])} | {row['vtln_only_final_mm']:.3f} | "
                    f"{signed(row['vtln_effect_mm'])} | {row['rms_vtln_final_mm']:.3f} | "
                    f"{signed(row['combined_effect_mm'])} | {VARIANT_LABELS[row['best_variant']]} |\n"
                )
            handle.write("\n")

        handle.write("## Final TPS result by contour\n\n")
        handle.write(
            "| Contour | Baseline | RMS-only | Δ RMS | VTLN-only | Δ VTLN | RMS+VTLN | Δ combined |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for row in class_rows:
            if row["stage"] != "affine_tps":
                continue
            marker = " *" if row["excluded_in_without_3"] else ""
            handle.write(
                f"| {row['class']}{marker} | {row['baseline_mm']:.3f} | {row['rms_only_mm']:.3f} | "
                f"{signed(row['rms_effect_mm'])} | {row['vtln_only_mm']:.3f} | "
                f"{signed(row['vtln_effect_mm'])} | {row['rms_vtln_mm']:.3f} | "
                f"{signed(row['combined_effect_mm'])} |\n"
            )
        handle.write("\n`*` marks the three contours excluded from the without-3 metric.\n\n")

        handle.write("## Speaker-level diagnosis\n\n")
        all_comparison = {
            row["speaker"]: row
            for row in comparisons
            if row["metric_mode"] == "all_11" and row["speaker"] != "ALL"
        }
        without_comparison = {
            row["speaker"]: row
            for row in comparisons
            if row["metric_mode"] == "without_laryngeal_3" and row["speaker"] != "ALL"
        }
        for summary in summaries:
            speaker = f"P{summary['speaker']}"
            row = all_comparison[speaker]
            row_without = without_comparison[speaker]
            rms_word = "helps" if row["rms_effect_mm"] < 0 else "hurts"
            vtln_word = "helps" if row["vtln_effect_mm"] < 0 else "hurts"
            interaction = row["non_additive_interaction_mm"]
            interaction_word = "constructive" if interaction < 0 else "destructive"
            handle.write(
                f"- **{speaker}/S{summary['session']} (alpha {summary['vtln_alpha_to_p7']:.3f}):** "
                f"RMS-only {rms_word} by {signed(row['rms_effect_mm'])} mm; VTLN-only {vtln_word} by "
                f"{signed(row['vtln_effect_mm'])} mm; combined effect {signed(row['combined_effect_mm'])} mm. "
                f"The non-additive interaction is {interaction_word} ({signed(interaction)} mm). "
                f"Without three, RMS/VTLN/combined effects are {signed(row_without['rms_effect_mm'])}/"
                f"{signed(row_without['vtln_effect_mm'])}/{signed(row_without['combined_effect_mm'])} mm.\n"
            )

        overall_all = next(
            row for row in comparisons
            if row["speaker"] == "ALL" and row["metric_mode"] == "all_11"
        )
        overall_without = next(
            row for row in comparisons
            if row["speaker"] == "ALL" and row["metric_mode"] == "without_laryngeal_3"
        )
        rms_wins = sum(
            row["rms_effect_mm"] < 0 for row in comparisons
            if row["speaker"] != "ALL" and row["metric_mode"] == "all_11"
        )
        vtln_wins = sum(
            row["vtln_effect_mm"] < 0 for row in comparisons
            if row["speaker"] != "ALL" and row["metric_mode"] == "all_11"
        )
        combined_wins = sum(
            row["combined_effect_mm"] < 0 for row in comparisons
            if row["speaker"] != "ALL" and row["metric_mode"] == "all_11"
        )
        best_overall = overall_all["best_variant"]
        handle.write("\n## Final conclusion\n\n")
        handle.write(
            f"At final TPS with all 11 contours, RMS-only changes the frame-weighted error by "
            f"**{signed(overall_all['rms_effect_mm'])} mm**, VTLN-only by "
            f"**{signed(overall_all['vtln_effect_mm'])} mm**, and RMS+VTLN by "
            f"**{signed(overall_all['combined_effect_mm'])} mm**. The best aggregate variant is "
            f"**{VARIANT_LABELS[best_overall]}** at {overall_all[best_overall + '_final_mm'] if best_overall != 'baseline' else overall_all['baseline_final_mm']:.3f} mm. "
            f"RMS-only/VTLN-only/combined improve {rms_wins}/{vtln_wins}/{combined_wins} of 9 speakers.\n\n"
            f"Without the three laryngeal contours, their aggregate effects are "
            f"{signed(overall_without['rms_effect_mm'])}, {signed(overall_without['vtln_effect_mm'])}, "
            f"and {signed(overall_without['combined_effect_mm'])} mm respectively.\n\n"
            "The non-additive interaction is descriptive rather than causal: the recurrent model and grid "
            "transforms are nonlinear, so the combined result need not equal the sum of isolated effects.\n"
        )

        handle.write("\n## Deliverables\n\n")
        handle.write(
            "- `P*/S*/rms_only_contours_and_ground_truth.npz`: reusable RMS-only predictions.\n"
            "- `P*/S*/vtln_only_contours_and_ground_truth.npz`: reusable VTLN-only predictions.\n"
            "- `P*/S*/ablation_frame_metrics.csv`: all four variants, three grid stages, and both metric modes per frame.\n"
            "- `ablation_metrics_long.csv`: complete speaker/stage table.\n"
            "- `ablation_final_comparison.csv`: final TPS effects and best variant.\n"
            "- `ablation_per_class_metrics.csv`: stage-wise metrics for every contour.\n"
        )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report": str(report_path.resolve()),
        "selection": [f"P{speaker}/S{session}" for speaker, session in SELECTION],
        "variants": list(VARIANTS),
        "inputs": {
            "baseline_root": str(args.baseline_root.resolve()),
            "combined_root": str(args.combined_root.resolve()),
        },
        "outputs": {
            "long_metrics": str(long_path.resolve()),
            "final_comparison": str(comparison_path.resolve()),
            "per_class_metrics": str(class_path.resolve()),
        },
        "sessions": summaries,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"report": str(report_path.resolve()), "completed_sessions": len(summaries)}


def main() -> None:
    args = parse_args()
    validate(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = load_yaml_config(args.config)
    if args.report_only:
        print(json.dumps(generate_report(args, config), indent=2), flush=True)
        return
    classes = list(config["classes"])
    with open(config["phonemesdir"], "r", encoding="utf-8") as handle:
        phonemes = json.load(handle)
    alphas = load_alphas(args.combined_root)
    normalization = load_normalization(args.normalization_stats)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable for {device}")
    model = load_model(config, args.checkpoint, device)
    source = prepare_frame(SOURCE, args.vtln_dir)
    transforms = {}
    for speaker, session in SELECTION:
        if speaker not in transforms:
            target = prepare_frame(target_spec_for_speaker(speaker), args.vtln_dir)
            transforms[speaker] = build_two_step_transform(source["grid"], target["grid"])
        process_session(
            args,
            speaker,
            session,
            alphas[f"P{speaker}"],
            config,
            classes,
            phonemes,
            normalization,
            model,
            device,
            transforms[speaker],
        )
    print(json.dumps(generate_report(args, config), indent=2), flush=True)


if __name__ == "__main__":
    main()
