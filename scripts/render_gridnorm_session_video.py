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

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
GRID_ROOT = REPO_ROOT / "external" / "grid-transform"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(GRID_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from grid_transform.transfer import build_two_step_transform  # noqa: E402
from src.utils.colors import COLORS  # noqa: E402
from src.utils.gridnorm_rendering import (  # noqa: E402
    PRIMARY_GRIDNORM_CLASSES,
    build_dicom_mri_cache,
    mri_for_frame,
    transform_predictions,
    write_mode_contours,
)
from src.utils.prediction_motion_cli import (  # noqa: E402
    add_prediction_motion_arguments,
    assert_prediction_motion_from_args,
    prediction_motion_report_from_args,
)
from src.utils.session_rendering import (  # noqa: E402
    aggregate_state,
    build_timeline,
    load_config,
    prediction_denorm_summary,
)
from src.utils.video_rendering import (  # noqa: E402
    MM_PER_PIXEL,
    attach_audio,
    draw_dashed_polyline,
    mean_finite,
    rgb_to_bgr255,
    rmse_px,
    scale_points,
)

from evaluate_grid_normalization import load_source_target_grids  # noqa: E402


PRIMARY_CLASSES = PRIMARY_GRIDNORM_CLASSES
INFO_BAND_HEIGHT = 118
RAW_ROOT = Path(
    "/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/"
    "ArtSpeech_Database_1_raw"
)
DEFAULT_PREDICTIONS = (
    REPO_ROOT
    / "results/p7_ref_bf_vtlnc_to_p2_s1s3_gridnorm_20260705_170122/unseen_baseline/P2_S1/eval/cached_session_predictions.pt"
)
DEFAULT_CONFIG = REPO_ROOT / "config/train_config/asd1_p7_seen_trainstats_p2_s1s3_unseen_eval_st5_mfcc.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/p7_ref_bf_vtlnc_to_p2_s1_video_review_20260705"
DEFAULT_VTLN_DIR = WORKSPACE_ROOT / "_downloads/grid-transform-vtln/vtln-data-v0.1.14/extracted/VTLN/data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render P2 session videos for raw, affine, and affine+TPS gridnorm predictions.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--speaker", type=int, default=2)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--speaker-name", default="P2")
    parser.add_argument("--session-name", default="S1")
    parser.add_argument("--mri-dicom-dir", type=Path, default=RAW_ROOT / "P2/DCM_2D/S1")
    parser.add_argument("--audio", type=Path, default=RAW_ROOT / "P2/OTHER/S1/DENOISED_SOUND_P2_S1.wav")
    parser.add_argument("--source-anchor", default="1640_P7_S2_F0829")
    parser.add_argument("--target-anchor", default="1617_P2_S9_F1478")
    parser.add_argument("--source-speaker", default="P7")
    parser.add_argument("--source-session", default="S2")
    parser.add_argument("--source-frame", default="0829")
    parser.add_argument("--target-speaker-name", default="P2")
    parser.add_argument("--target-session-name", default="S9")
    parser.add_argument("--target-frame", default="1478")
    parser.add_argument("--vtln-dir", type=Path, default=DEFAULT_VTLN_DIR)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--ms-image", type=float, default=None)
    parser.add_argument("--timeline-step", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--modes", nargs="+", choices=("raw", "affine", "affine_tps"), default=["raw", "affine", "affine_tps"])
    add_prediction_motion_arguments(parser, include_diagnostic_flag=True, action_word="Fail render")
    return parser.parse_args()


def draw_frame(
    row: dict[str, Any],
    classes: list[str],
    primary_indices: list[int],
    image: np.ndarray,
    scale: int,
    mode_label: str,
    display_label: str,
) -> tuple[np.ndarray, dict[str, float]]:
    labels = row["labels"].reshape(len(classes), 50, 2) if row.get("labels") is not None else None
    pred = row["mode_prediction"].reshape(len(classes), 50, 2)
    all_rmse = rmse_px(pred, labels) if labels is not None else float("nan")
    primary_rmse = rmse_px(pred[primary_indices], labels[primary_indices]) if labels is not None else float("nan")

    image_canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_size = int(max(136, image.shape[0], image.shape[1]))
    image_canvas = cv2.resize(image_canvas, (image_size * scale, image_size * scale), interpolation=cv2.INTER_CUBIC)
    canvas = np.full(
        (image_canvas.shape[0] + INFO_BAND_HEIGHT, image_canvas.shape[1], 3),
        15,
        dtype=np.uint8,
    )
    canvas[INFO_BAND_HEIGHT : INFO_BAND_HEIGHT + image_canvas.shape[0], :] = image_canvas

    for idx, articulator in enumerate(classes):
        color = rgb_to_bgr255(COLORS.get(articulator, "white"))
        pred_points = scale_points(pred[idx], scale)
        pred_points[:, 1] += INFO_BAND_HEIGHT
        if labels is not None:
            gt_points = scale_points(labels[idx], scale)
            gt_points[:, 1] += INFO_BAND_HEIGHT
            cv2.polylines(canvas, [gt_points], isClosed=False, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [gt_points], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)
            draw_dashed_polyline(canvas, pred_points, color=(0, 0, 0), thickness=2)
            draw_dashed_polyline(canvas, pred_points, color=color, thickness=1)
        else:
            draw_dashed_polyline(canvas, pred_points, color=(0, 0, 0), thickness=2)
            draw_dashed_polyline(canvas, pred_points, color=color, thickness=1)

    lines = [f"{display_label} frame {row['frame']} | {mode_label}", f"phoneme: {row['phoneme']}"]
    if labels is not None:
        lines.extend(
            [
                f"RMSE all: {all_rmse * MM_PER_PIXEL:.3f} mm",
                f"RMSE primary: {primary_rmse * MM_PER_PIXEL:.3f} mm",
                "solid = ground truth | dashed = prediction",
            ]
        )
    else:
        lines.extend(["prediction only", "dashed = prediction"])
    if row["held"]:
        lines[-1] = f"{lines[-1]} | held frame"

    for line_index, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (16, 20 + line_index * 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return canvas, {"rmse_px": all_rmse, "primary_rmse_px": primary_rmse}


def render_mode(
    rows: list[dict[str, Any]],
    classes: list[str],
    mode: str,
    output_dir: Path,
    mri_cache: dict[int, np.ndarray],
    fps: float,
    scale: int,
    audio_path: Path,
    audio_start_seconds: float,
    speaker_name: str,
    session_name: str,
) -> dict[str, Any]:
    mode_labels = {
        "raw": "raw P7-model prediction",
        "affine": "affine only",
        "affine_tps": "affine + TPS",
    }
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    write_mode_contours(rows, classes, mode_dir / "predicted_contours")

    file_prefix = f"{speaker_name.lower()}_{session_name.lower()}_{mode}_prediction"
    silent_mp4 = mode_dir / f"{file_prefix}_silent.mp4"
    final_mp4 = mode_dir / f"{file_prefix}_audio.mp4"
    primary_indices = [idx for idx, name in enumerate(classes) if name in PRIMARY_CLASSES]
    first_image = mri_for_frame(rows[0]["frame_number"], mri_cache)
    base_size = int(max(136, first_image.shape[0], first_image.shape[1])) * scale
    writer = cv2.VideoWriter(
        str(silent_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (base_size, base_size + INFO_BAND_HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {silent_mp4}")

    metrics_rows = []
    try:
        for row in rows:
            image = mri_for_frame(row["frame_number"], mri_cache)
            canvas, metrics = draw_frame(
                row,
                classes,
                primary_indices,
                image,
                scale,
                mode_labels[mode],
                f"{speaker_name}/{session_name}",
            )
            writer.write(canvas)
            metrics_rows.append(
                {
                    "frame": row["frame"],
                    "frame_number": row["frame_number"],
                    "phoneme": row["phoneme"],
                    "held": row["held"],
                    "rmse_px": metrics["rmse_px"],
                    "rmse_mm": metrics["rmse_px"] * MM_PER_PIXEL if not math.isnan(metrics["rmse_px"]) else float("nan"),
                    "primary_rmse_px": metrics["primary_rmse_px"],
                    "primary_rmse_mm": metrics["primary_rmse_px"] * MM_PER_PIXEL if not math.isnan(metrics["primary_rmse_px"]) else float("nan"),
                }
            )
    finally:
        writer.release()

    with (mode_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["frame", "frame_number", "phoneme", "held", "rmse_px", "rmse_mm", "primary_rmse_px", "primary_rmse_mm"]
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(metrics_rows)

    duration_seconds = len(rows) / float(fps)
    audio_attached = attach_audio(silent_mp4, audio_path, final_mp4, audio_start_seconds, duration_seconds)
    output_mp4 = final_mp4 if audio_attached else silent_mp4
    return {
        "mode": mode,
        "video": str(output_mp4),
        "silent_video": str(silent_mp4),
        "audio_attached": audio_attached,
        "predicted_contours": str(mode_dir / "predicted_contours"),
        "frame_metrics": str(mode_dir / "frame_metrics.csv"),
        "mean_rmse_mm": mean_finite([row["rmse_mm"] for row in metrics_rows]),
        "mean_primary_rmse_mm": mean_finite([row["primary_rmse_mm"] for row in metrics_rows]),
        "num_frames": len(rows),
        "num_held_frames": int(sum(bool(row["held"]) for row in rows)),
        "fps": float(fps),
        "audio_start_seconds": float(audio_start_seconds),
        "duration_seconds": float(duration_seconds),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    classes = list(config["classes"])
    ms_image = float(args.ms_image if args.ms_image is not None else config.get("ms_image", 19.98))
    fps = 1000.0 / (ms_image * float(args.timeline_step))
    state = torch.load(args.predictions, map_location="cpu")
    motion_report = prediction_motion_report_from_args(
        state,
        classes,
        args,
        prediction_payload=str(args.predictions),
    )
    assert_prediction_motion_from_args(motion_report, args)
    rows = aggregate_state(state, config, args.speaker, args.session)
    denorm_summary = prediction_denorm_summary()
    timeline_rows = build_timeline(rows, float(args.timeline_step), args.max_frames)

    needs_grid_transform = any(mode in {"affine", "affine_tps"} for mode in args.modes)
    source_grid_png = None
    target_grid_png = None
    transform = None
    if needs_grid_transform:
        class ArgsForGrid:
            pass

        grid_args = ArgsForGrid()
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
        source_grid, target_grid, source_grid_png, target_grid_png, _, _ = load_source_target_grids(grid_args, output_dir)
        transform = build_two_step_transform(source_grid, target_grid)

    mri_cache = build_dicom_mri_cache(args.mri_dicom_dir, timeline_rows, output_dir)
    audio_start_seconds = (float(timeline_rows[0]["frame_number"]) - 1.0) * ms_image / 1000.0
    outputs = []
    for mode in args.modes:
        mode_rows = transform_predictions(timeline_rows, transform, mode, len(classes))
        outputs.append(
            render_mode(
                mode_rows,
                classes,
                mode,
                output_dir,
                mri_cache,
                fps,
                int(args.scale),
                args.audio,
                audio_start_seconds,
                args.speaker_name,
                args.session_name,
            )
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "predictions": str(args.predictions),
        "config": str(args.config),
        "speaker": args.speaker,
        "session": args.session,
        "note": "Raw predictions are cached model outputs for the requested session. Affine and affine_tps are optional morphology-normalized post-processing modes.",
        "prediction_denorm": denorm_summary,
        "source_grid_png": source_grid_png,
        "target_grid_png": target_grid_png,
        "audio": str(args.audio),
        "audio_start_seconds": audio_start_seconds,
        "ms_image": ms_image,
        "timeline_step": float(args.timeline_step),
        "fps": fps,
        "source_anchor": args.source_anchor,
        "target_anchor": args.target_anchor,
        "vtln_dir": str(args.vtln_dir),
        "has_labels": bool(timeline_rows and timeline_rows[0].get("labels") is not None),
        "num_original_prediction_frames": len(rows),
        "num_timeline_frames": len(timeline_rows),
        "frame_min": float(timeline_rows[0]["frame_number"]),
        "frame_max": float(timeline_rows[-1]["frame_number"]),
        "outputs": outputs,
        "prediction_motion": motion_report,
        "allow_static_prediction_diagnostic": bool(args.allow_static_prediction_diagnostic),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write(f"# Prediction Video For {args.speaker_name}/{args.session_name}\n\n")
        handle.write(
            "Raw predictions are produced by the cached model output already "
            "denormalized with the model training split stats. If labels are "
            "absent, the video is prediction-only and no RMSE is computed.\n\n"
        )
        handle.write("- prediction denorm mode: `payload_raw`\n")
        handle.write(f"- uses target session label stats: `{denorm_summary['uses_target_session_label_stats']}`\n")
        handle.write(f"- source prediction payload: `{args.predictions}`\n")
        if "motion_ratio_frame_diff" in motion_report:
            handle.write(f"- pred/label motion ratio: `{motion_report['motion_ratio_frame_diff']:.6f}`\n")
            handle.write(f"- static prediction flag: `{motion_report['is_static_prediction']}`\n")
        handle.write(f"- original prediction frames: `{len(rows)}`\n")
        handle.write(f"- rendered timeline frames: `{len(timeline_rows)}`\n")
        handle.write(f"- audio start: `{audio_start_seconds:.3f}s`\n")
        handle.write(f"- fps: `{fps:.3f}`\n\n")
        handle.write(f"- rendered session: `{args.speaker_name}/{args.session_name}`\n")
        handle.write(f"- has labels/RMSE: `{bool(timeline_rows and timeline_rows[0].get('labels') is not None)}`\n")
        handle.write("| Mode | Mean RMSE mm | Mean primary RMSE mm | Held frames | Video | Contours |\n")
        handle.write("|---|---:|---:|---:|---|---|\n")
        for item in outputs:
            handle.write(
                f"| `{item['mode']}` | {item['mean_rmse_mm']:.6f} | {item['mean_primary_rmse_mm']:.6f} | "
                f"{item['num_held_frames']} | `{Path(item['video']).relative_to(output_dir)}` | "
                f"`{Path(item['predicted_contours']).relative_to(output_dir)}` |\n"
            )
        handle.write("\n")
        if source_grid_png and target_grid_png:
            handle.write(f"- source grid: `{Path(source_grid_png).relative_to(output_dir)}`\n")
            handle.write(f"- target grid: `{Path(target_grid_png).relative_to(output_dir)}`\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
