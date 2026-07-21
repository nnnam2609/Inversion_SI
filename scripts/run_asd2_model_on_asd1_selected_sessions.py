#!/usr/bin/env python3
"""Run the trained ASD2 model on selected ASD1 sessions using integer MRI frames only."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")]

from run_p7_all_nonp7_gridnorm import (  # noqa: E402
    EXCLUDED_CLASSES,
    RAW_ROOT,
    infer_session,
    load_normalization,
    mri_for_timestamp,
)
from src.inference.session_inference import load_model  # noqa: E402
from src.utils.colors import COLORS  # noqa: E402
from src.utils.config_validation import load_yaml_config  # noqa: E402
from src.utils.mri_rendering import build_filename_dicom_index, load_or_build_mri_cache  # noqa: E402
from src.utils.video_rendering import (  # noqa: E402
    MM_PER_PIXEL,
    draw_dashed_polyline,
    rgb_to_bgr255,
    scale_points,
)


SELECTION = ((1, 16), (2, 9), (3, 14), (4, 4), (5, 6), (6, 8), (7, 2), (8, 2), (9, 5), (10, 14))
DEFAULT_CONFIG = REPO_ROOT / "config/train_config/asd2_11contour_original_incisor_only_paper_st5_mfcc_500epoch.yaml"
DEFAULT_CHECKPOINT = REPO_ROOT / "mlruns/160619570814633961/ebce38c8f4134c57b537ff4558ea2930/artifacts/best_model.pth"
DEFAULT_NORMALIZATION = REPO_ROOT / "results/p7_s15_asd2_11contour_model_legacy_chunks_20260717/p7_mfcc_asd2_train_contour_normalization.npz"
DEFAULT_RAW_CACHE = REPO_ROOT / "cache/raw_sessions/asd1"
DEFAULT_OUTPUT = REPO_ROOT / "results/asd2_model_to_asd1_selected_sessions_integer_only_20260718"
INCISOR_CLASSES = ("lower-incisor", "upper-incisor")
INFO_HEIGHT = 105
INTEGER_FRAME_POLICY = "NEVER render fractional MRI frames (including .5); integer frames only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--normalization-stats", type=Path, default=DEFAULT_NORMALIZATION)
    parser.add_argument("--raw-cache-root", type=Path, default=DEFAULT_RAW_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection", nargs="+", default=None, metavar="P#:S#")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--mri-workers", type=int, default=4)
    parser.add_argument("--keep-mri-cache", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args()


def selected_pairs(args: argparse.Namespace) -> tuple[tuple[int, int], ...]:
    if not args.selection:
        return SELECTION
    allowed = set(SELECTION)
    parsed = []
    for token in args.selection:
        try:
            left, right = token.upper().split(":", maxsplit=1)
            pair = (int(left.removeprefix("P")), int(right.removeprefix("S")))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid selection {token!r}; expected P#:S#") from error
        if pair not in allowed:
            raise ValueError(f"Pair P{pair[0]}:S{pair[1]} is outside the fixed selection")
        parsed.append(pair)
    return tuple(parsed)


def validate(args: argparse.Namespace) -> None:
    for path in (args.config, args.checkpoint, args.normalization_stats, args.raw_cache_root):
        if not path.exists():
            raise FileNotFoundError(path)
    for speaker, session in selected_pairs(args):
        raw = args.raw_cache_root / f"P{speaker}/S{session}.pt"
        dicom = RAW_ROOT / f"P{speaker}/DCM_2D/S{session}"
        if not raw.is_file():
            raise FileNotFoundError(raw)
        if not dicom.is_dir():
            raise FileNotFoundError(dicom)


def frame_rmse_mm(predicted: np.ndarray, labels: np.ndarray, indices: list[int]) -> np.ndarray:
    diff = predicted[:, indices].astype(np.float64) - labels[:, indices].astype(np.float64)
    return np.sqrt(np.mean(diff * diff, axis=(1, 2, 3))) * MM_PER_PIXEL


def compute_metrics(
    predicted: np.ndarray, labels: np.ndarray, classes: list[str]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    class_sets = {
        "all_11": list(range(len(classes))),
        "without_laryngeal_3": [i for i, name in enumerate(classes) if name not in EXCLUDED_CLASSES],
        "without_incisors_2": [i for i, name in enumerate(classes) if name not in INCISOR_CLASSES],
        "without_laryngeal_3_and_incisors_2": [
            i for i, name in enumerate(classes)
            if name not in set(EXCLUDED_CLASSES) | set(INCISOR_CLASSES)
        ],
    }
    frame_metrics = {
        name: frame_rmse_mm(predicted, labels, indices) for name, indices in class_sets.items()
    }
    summary: dict[str, Any] = {"modes": {}, "per_class_mean_frame_rmse_mm": {}}
    for name, values in frame_metrics.items():
        summary["modes"][name] = {
            "mean_frame_rmse_mm": float(np.mean(values)),
            "median_frame_rmse_mm": float(np.median(values)),
            "std_frame_rmse_mm": float(np.std(values)),
        }
    per_class_diff = predicted.astype(np.float64) - labels.astype(np.float64)
    per_class = np.sqrt(np.mean(per_class_diff * per_class_diff, axis=(2, 3))) * MM_PER_PIXEL
    summary["per_class_mean_frame_rmse_mm"] = {
        name: float(np.mean(per_class[:, index])) for index, name in enumerate(classes)
    }
    return summary, frame_metrics


def retain_integer_frames(inferred: dict[str, Any]) -> dict[str, Any]:
    """Discard fractional timestamps before saving, scoring, or rendering."""
    frames = np.asarray(inferred["frame_numbers"], dtype=np.float32)
    integer_mask = np.isclose(frames, np.rint(frames), atol=1e-4)
    filtered = dict(inferred)
    for key in ("frame_numbers", "phonemes", "overlap_counts", "predicted_raw", "ground_truth"):
        filtered[key] = np.asarray(inferred[key])[integer_mask]
    filtered["frame_numbers"] = np.rint(filtered["frame_numbers"]).astype(np.int32)
    filtered["num_fractional_frames_discarded"] = int(np.count_nonzero(~integer_mask))
    if len(filtered["frame_numbers"]) == 0:
        raise RuntimeError("Integer-frame policy removed every prediction frame")
    if not np.allclose(filtered["frame_numbers"], np.rint(filtered["frame_numbers"]), atol=0.0):
        raise AssertionError(INTEGER_FRAME_POLICY)
    return filtered


def save_pack(path: Path, inferred: dict[str, np.ndarray], classes: list[str]) -> None:
    np.savez_compressed(
        path,
        frame_numbers=inferred["frame_numbers"],
        phonemes=inferred["phonemes"],
        overlap_counts=inferred["overlap_counts"],
        predicted=inferred["predicted_raw"],
        ground_truth=inferred["ground_truth"],
        classes=np.asarray(classes, dtype="U64"),
        num_input_rows=inferred["num_input_rows"],
        num_sequences=inferred["num_sequences"],
        num_fractional_frames_discarded=inferred["num_fractional_frames_discarded"],
        frame_policy=np.asarray(INTEGER_FRAME_POLICY),
    )


def load_pack(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "frame_numbers": np.asarray(payload["frame_numbers"], dtype=np.int32),
            "phonemes": np.asarray(payload["phonemes"]),
            "overlap_counts": np.asarray(payload["overlap_counts"]),
            "predicted_raw": np.asarray(payload["predicted"], dtype=np.float32),
            "ground_truth": np.asarray(payload["ground_truth"], dtype=np.float32),
            "classes": [str(value) for value in payload["classes"].tolist()],
            "num_input_rows": int(payload["num_input_rows"]),
            "num_sequences": int(payload["num_sequences"]),
            "num_fractional_frames_discarded": int(payload["num_fractional_frames_discarded"]),
            "frame_policy": str(payload["frame_policy"]),
        }


def draw_frame(
    image: np.ndarray,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    classes: list[str],
    speaker: int,
    session: int,
    frame_number: float,
    phoneme: str,
    all_rmse: float,
    no_incisor_rmse: float,
    scale: int,
) -> np.ndarray:
    size = 136 * scale
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_bgr = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((size + INFO_HEIGHT, size, 3), 15, dtype=np.uint8)
    canvas[INFO_HEIGHT:] = image_bgr
    for index, class_name in enumerate(classes):
        color = rgb_to_bgr255(COLORS.get(class_name, "white"))
        gt = scale_points(ground_truth[index], scale)
        pred = scale_points(predicted[index], scale)
        gt[:, 1] += INFO_HEIGHT
        pred[:, 1] += INFO_HEIGHT
        cv2.polylines(canvas, [gt], False, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.polylines(canvas, [gt], False, color, 1, cv2.LINE_AA)
        draw_dashed_polyline(canvas, pred, (0, 0, 0), 2, dash_length=7, gap_length=9)
        draw_dashed_polyline(canvas, pred, color, 1, dash_length=7, gap_length=9)
    if abs(frame_number - round(frame_number)) >= 1e-4:
        raise AssertionError(INTEGER_FRAME_POLICY)
    frame_text = f"{int(round(frame_number)):04d}"
    lines = [
        f"ASD2 model -> ASD1 P{speaker}/S{session}",
        f"frame {frame_text} | {phoneme}",
        f"RMSE all 11: {all_rmse:.3f} mm",
        f"RMSE no incisors: {no_incisor_rmse:.3f} mm",
        "solid GT | dashed prediction",
    ]
    for line_index, value in enumerate(lines):
        cv2.putText(
            canvas,
            value,
            (8, 17 + 19 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return canvas


def render_video(
    args: argparse.Namespace,
    speaker: int,
    session: int,
    session_dir: Path,
    inferred: dict[str, Any],
    frame_metrics: dict[str, np.ndarray],
    classes: list[str],
) -> Path:
    frames = inferred["frame_numbers"]
    if not np.allclose(frames, np.rint(frames), atol=0.0):
        raise AssertionError(INTEGER_FRAME_POLICY)
    dicom_dir = RAW_ROOT / f"P{speaker}/DCM_2D/S{session}"
    needed = sorted(int(frame) for frame in frames)
    dicom_index, _ = build_filename_dicom_index(dicom_dir)
    cache_path = session_dir / "mri_frames_cache.npz"
    mri_cache = load_or_build_mri_cache(
        dicom_dir, dicom_index, needed, cache_path, workers=args.mri_workers
    )
    size = 136 * args.scale
    video_path = session_dir / f"p{speaker}_s{session}_asd2_model_gt_vs_prediction_integer_only_50fps.mp4"
    temporary = session_dir / f".{video_path.stem}.writing.mp4"
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (size, size + INFO_HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open {temporary}")
    try:
        for index, frame in enumerate(frames):
            image = mri_for_timestamp(float(frame), mri_cache)
            writer.write(
                draw_frame(
                    image,
                    inferred["predicted_raw"][index],
                    inferred["ground_truth"][index],
                    classes,
                    speaker,
                    session,
                    float(frame),
                    str(inferred["phonemes"][index]),
                    float(frame_metrics["all_11"][index]),
                    float(frame_metrics["without_incisors_2"][index]),
                    args.scale,
                )
            )
    finally:
        writer.release()
    temporary.replace(video_path)
    if not args.keep_mri_cache and cache_path.exists():
        cache_path.unlink()
    return video_path


def write_frame_metrics(
    path: Path,
    frames: np.ndarray,
    phones: np.ndarray,
    frame_metrics: dict[str, np.ndarray],
) -> None:
    fields = ["frame_number", "phoneme"] + [f"{name}_rmse_mm" for name in frame_metrics]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, frame in enumerate(frames):
            row: dict[str, Any] = {"frame_number": float(frame), "phoneme": str(phones[index])}
            row.update({f"{name}_rmse_mm": float(values[index]) for name, values in frame_metrics.items()})
            writer.writerow(row)


def process_session(
    args: argparse.Namespace,
    speaker: int,
    session: int,
    classes: list[str],
    phonemes: list[str],
    normalization: dict[str, np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    started = time.monotonic()
    session_dir = args.output_root / f"P{speaker}/S{session}"
    session_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.raw_cache_root / f"P{speaker}/S{session}.pt"
    pack_path = session_dir / "asd2_model_predictions_and_ground_truth_integer_only.npz"
    video_path = session_dir / f"p{speaker}_s{session}_asd2_model_gt_vs_prediction_integer_only_50fps.mp4"
    if pack_path.is_file() and not args.force:
        inferred = load_pack(pack_path)
        if inferred["classes"] != classes:
            raise ValueError(f"Class mismatch in {pack_path}")
        print(f"REUSE P{speaker}/S{session} prediction pack", flush=True)
    else:
        inferred_full = infer_session(
            model, device, raw_path, normalization, phonemes, args.batch_size
        )
        inferred = retain_integer_frames(inferred_full)
        inferred["classes"] = classes
        save_pack(pack_path, inferred, classes)
    if not np.allclose(inferred["frame_numbers"], np.rint(inferred["frame_numbers"]), atol=0.0):
        raise AssertionError(INTEGER_FRAME_POLICY)
    metric_summary, frame_metrics = compute_metrics(
        inferred["predicted_raw"], inferred["ground_truth"], classes
    )
    frame_csv = session_dir / "frame_metrics_integer_only.csv"
    write_frame_metrics(frame_csv, inferred["frame_numbers"], inferred["phonemes"], frame_metrics)
    if not args.skip_video and (args.force or not video_path.is_file()):
        render_video(args, speaker, session, session_dir, inferred, frame_metrics, classes)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "speaker": speaker,
        "session": session,
        "num_unique_frames": int(len(inferred["frame_numbers"])),
        "num_fractional_frames_discarded": int(inferred["num_fractional_frames_discarded"]),
        "rendered_fractional_frame_count": 0,
        "frame_policy": INTEGER_FRAME_POLICY,
        "num_sequences": int(inferred["num_sequences"]),
        "num_input_rows": int(inferred["num_input_rows"]),
        "model_training_dataset": "ASD2",
        "target_dataset": "ASD1",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "normalization_stats": str(args.normalization_stats.resolve()),
        "normalization_policy": "P7 MFCC reference + ASD2 train contour denormalization; no target contours used for prediction normalization",
        "raw_asd1_session_cache": str(raw_path.resolve()),
        "contour_pack": str(pack_path.resolve()),
        "frame_metrics": str(frame_csv.resolve()),
        "video": None if args.skip_video else str(video_path.resolve()),
        "metrics": metric_summary,
        "elapsed_seconds": time.monotonic() - started,
    }
    (session_dir / "session_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"DONE P{speaker}/S{session}: {summary['num_unique_frames']} frames, "
        f"discarded_fractional={summary['num_fractional_frames_discarded']}, "
        f"all11={metric_summary['modes']['all_11']['mean_frame_rmse_mm']:.3f} mm, "
        f"no-incisors={metric_summary['modes']['without_incisors_2']['mean_frame_rmse_mm']:.3f} mm, "
        f"{summary['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return summary


def weighted(summaries: list[dict[str, Any]], mode: str) -> float:
    total = sum(int(row["num_unique_frames"]) for row in summaries)
    return sum(
        int(row["num_unique_frames"])
        * float(row["metrics"]["modes"][mode]["mean_frame_rmse_mm"])
        for row in summaries
    ) / total


def generate_report(args: argparse.Namespace) -> dict[str, Any]:
    summaries = []
    for speaker, session in SELECTION:
        path = args.output_root / f"P{speaker}/S{session}/session_summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    rows = []
    modes = (
        "all_11",
        "without_laryngeal_3",
        "without_incisors_2",
        "without_laryngeal_3_and_incisors_2",
    )
    for summary in summaries:
        row: dict[str, Any] = {
            "speaker": f"P{summary['speaker']}",
            "session": f"S{summary['session']}",
            "frames": summary["num_unique_frames"],
        }
        for mode in modes:
            row[f"{mode}_rmse_mm"] = summary["metrics"]["modes"][mode]["mean_frame_rmse_mm"]
        rows.append(row)
    rows.append(
        {
            "speaker": "ALL",
            "session": "selected_10",
            "frames": sum(int(row["num_unique_frames"]) for row in summaries),
            **{f"{mode}_rmse_mm": weighted(summaries, mode) for mode in modes},
        }
    )
    csv_path = args.output_root / "asd2_model_asd1_selected_session_metrics_integer_only.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path = args.output_root / "asd2_model_asd1_selected_sessions_integer_only_report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# ASD2-trained model on selected ASD1 sessions\n\n")
        handle.write(f"**Frame policy: {INTEGER_FRAME_POLICY}.** Fractional frames saved/rendered: **0**.\n\n")
        handle.write(
            "Checkpoint: `asd2_11contour_original_incisor_only`, trained only on ASD2. "
            "Prediction uses the established P7 MFCC reference normalization and ASD2-train contour "
            "denormalization; ASD1 target contours are used only as ground truth for the video/metrics.\n\n"
        )
        handle.write(
            "| Speaker/session | Frames | All 11 | Without laryngeal 3 | Without incisors 2 | Without both sets |\n"
            "|---|---:|---:|---:|---:|---:|\n"
        )
        for row in rows:
            label = "ALL" if row["speaker"] == "ALL" else f"{row['speaker']}/{row['session']}"
            handle.write(
                f"| {label} | {row['frames']} | {row['all_11_rmse_mm']:.3f} | "
                f"{row['without_laryngeal_3_rmse_mm']:.3f} | {row['without_incisors_2_rmse_mm']:.3f} | "
                f"{row['without_laryngeal_3_and_incisors_2_rmse_mm']:.3f} |\n"
            )
        handle.write("\n## Videos and contours\n\n")
        for summary in summaries:
            speaker, session = f"P{summary['speaker']}", f"S{summary['session']}"
            video = Path(summary["video"]).relative_to(args.output_root.resolve())
            pack = Path(summary["contour_pack"]).relative_to(args.output_root.resolve())
            handle.write(
                f"- [{speaker}/{session} video]({video.as_posix()}) — "
                f"[prediction contour pack]({pack.as_posix()})\n"
            )
        handle.write(
            "\nVideos use integer-numbered ASD1 MRI frames as background, solid ASD1 annotation, "
            "and dashed ASD2-model prediction.\n"
        )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection": [f"P{speaker}/S{session}" for speaker, session in SELECTION],
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "normalization_stats": str(args.normalization_stats.resolve()),
        "frame_policy": INTEGER_FRAME_POLICY,
        "rendered_fractional_frame_count": 0,
        "report": str(report_path.resolve()),
        "metrics_csv": str(csv_path.resolve()),
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
    if args.report_only:
        print(json.dumps(generate_report(args), indent=2), flush=True)
        return
    config = load_yaml_config(args.config)
    classes = list(config["classes"])
    if len(classes) != 11:
        raise ValueError(f"Expected 11 ASD2 classes, got {len(classes)}")
    with open(config["phonemesdir"], "r", encoding="utf-8") as handle:
        phonemes = json.load(handle)
    normalization = load_normalization(args.normalization_stats)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable for {device}")
    model = load_model(config, args.checkpoint, device)
    for speaker, session in selected_pairs(args):
        process_session(
            args, speaker, session, classes, phonemes, normalization, model, device
        )
    if not args.no_report:
        print(json.dumps(generate_report(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
