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
    prediction_motion_report_from_args,
)
from src.utils.session_rendering import (  # noqa: E402
    aggregate_state,
    build_ground_truth_timeline,
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
    parser = argparse.ArgumentParser(description="Render one ASD1 session video from cached inversion predictions.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--speaker", type=int, default=2)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--speaker-name", default=None)
    parser.add_argument("--session-name", default=None)
    parser.add_argument(
        "--mri-dicom-dir",
        type=Path,
        default=None,
        help="Defaults to <ASD1 raw root>/<speaker-name>/DCM_2D/<session-name>.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Defaults to the denoised ASD1 WAV for <speaker-name>/<session-name>.",
    )
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
    contour_only_group = parser.add_mutually_exclusive_group()
    contour_only_group.add_argument(
        "--ground-truth-only",
        action="store_true",
        help="Render only the 11 ground-truth contours; requires --ground-truth-contour-dir.",
    )
    contour_only_group.add_argument(
        "--prediction-only",
        action="store_true",
        help="Render only cached model contour files; requires --prediction-contour-dir.",
    )
    parser.add_argument(
        "--ground-truth-contour-dir",
        type=Path,
        default=None,
        help=(
            "Per-integer-frame contour directory used by --ground-truth-only. "
            "Missing contours are listed on the video and are never held."
        ),
    )
    parser.add_argument(
        "--prediction-contour-dir",
        type=Path,
        default=None,
        help=(
            "Cached per-frame model contour directory used by --prediction-only. "
            "Missing contours are listed on the video and are never held."
        ),
    )
    add_prediction_motion_arguments(parser, include_diagnostic_flag=True, action_word="Report")
    return parser.parse_args()


def resolve_session_media(args: argparse.Namespace) -> argparse.Namespace:
    expected_speaker_name = f"P{int(args.speaker)}"
    expected_session_name = f"S{int(args.session)}"
    args.speaker_name = args.speaker_name or expected_speaker_name
    args.session_name = args.session_name or expected_session_name
    if args.speaker_name.upper() != expected_speaker_name.upper():
        raise ValueError(
            f"--speaker {args.speaker} requires --speaker-name {expected_speaker_name}, "
            f"got {args.speaker_name}"
        )
    if args.session_name.upper() != expected_session_name.upper():
        raise ValueError(
            f"--session {args.session} requires --session-name {expected_session_name}, "
            f"got {args.session_name}"
        )
    if args.mri_dicom_dir is None:
        args.mri_dicom_dir = RAW_ROOT / args.speaker_name / "DCM_2D" / args.session_name
    if args.audio is None:
        args.audio = (
            RAW_ROOT
            / args.speaker_name
            / "OTHER"
            / args.session_name
            / f"DENOISED_SOUND_{args.speaker_name}_{args.session_name}.wav"
        )
    return args


def draw_frame(
    row: dict[str, Any],
    classes: list[str],
    primary_indices: list[int],
    image: np.ndarray,
    scale: int,
    mode_label: str,
    display_label: str,
    ground_truth_only: bool = False,
    prediction_only: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    labels = row["labels"].reshape(len(classes), 50, 2) if row.get("labels") is not None else None
    if ground_truth_only and labels is None:
        raise RuntimeError("--ground-truth-only requires per-frame ground-truth contours")
    pred = None if ground_truth_only else row["mode_prediction"].reshape(len(classes), 50, 2)
    paired_indices = []
    if labels is not None and pred is not None:
        paired_indices = [
            index
            for index in range(len(classes))
            if np.isfinite(labels[index]).all() and np.isfinite(pred[index]).all()
        ]
    all_rmse = rmse_px(pred[paired_indices], labels[paired_indices]) if paired_indices else float("nan")
    paired_primary_indices = [index for index in paired_indices if index in primary_indices]
    primary_rmse = (
        rmse_px(pred[paired_primary_indices], labels[paired_primary_indices])
        if paired_primary_indices
        else float("nan")
    )

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
        if ground_truth_only:
            if not np.isfinite(labels[idx]).all():
                continue
            gt_points = scale_points(labels[idx], scale)
            gt_points[:, 1] += INFO_BAND_HEIGHT
            cv2.polylines(canvas, [gt_points], isClosed=False, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [gt_points], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)
            continue

        if prediction_only:
            if not np.isfinite(pred[idx]).all():
                continue
            pred_points = scale_points(pred[idx], scale)
            pred_points[:, 1] += INFO_BAND_HEIGHT
            cv2.polylines(canvas, [pred_points], isClosed=False, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [pred_points], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)
            continue

        if not np.isfinite(pred[idx]).all():
            continue
        pred_points = scale_points(pred[idx], scale)
        pred_points[:, 1] += INFO_BAND_HEIGHT
        if labels is not None and np.isfinite(labels[idx]).all():
            gt_points = scale_points(labels[idx], scale)
            gt_points[:, 1] += INFO_BAND_HEIGHT
            cv2.polylines(canvas, [gt_points], isClosed=False, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [gt_points], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)
            draw_dashed_polyline(canvas, pred_points, color=(0, 0, 0), thickness=2)
            draw_dashed_polyline(canvas, pred_points, color=color, thickness=1)
        else:
            draw_dashed_polyline(canvas, pred_points, color=(0, 0, 0), thickness=2)
            draw_dashed_polyline(canvas, pred_points, color=color, thickness=1)

    lines = [f"{display_label} frame {row['frame']} | {mode_label}"]
    if ground_truth_only:
        missing = list(row.get("missing_ground_truth_contours", []))
        lines.extend(
            [
                f"ground truth: {len(classes) - len(missing)}/{len(classes)} contours",
                f"source: {row.get('ground_truth_source', 'unknown')}",
            ]
        )
        if missing:
            missing_text = f"missing: {', '.join(missing)}" if len(missing) < len(classes) else f"missing: all {len(classes)} contours"
            lines.append(missing_text)
        lines.append("solid = ground truth")
    elif prediction_only:
        missing = list(row.get("missing_prediction_contours", []))
        lines.extend(
            [
                f"prediction: {len(classes) - len(missing)}/{len(classes)} contours",
                f"source: {row.get('prediction_source', 'unknown')}",
            ]
        )
        if missing:
            missing_text = f"missing: {', '.join(missing)}" if len(missing) < len(classes) else f"missing: all {len(classes)} contours"
            lines.append(missing_text)
        lines.append("solid = P7 model prediction")
    elif labels is not None:
        lines.append(f"phoneme: {row['phoneme']}")
        lines.extend(
            [
                f"RMSE all: {all_rmse * MM_PER_PIXEL:.3f} mm",
                f"RMSE primary: {primary_rmse * MM_PER_PIXEL:.3f} mm",
                "solid = ground truth | dashed = prediction",
            ]
        )
    else:
        lines.extend([f"phoneme: {row['phoneme']}", "prediction only", "dashed = prediction"])
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
    ground_truth_only: bool = False,
    prediction_only: bool = False,
) -> dict[str, Any]:
    mode_labels = {
        "raw": "raw P7-model prediction",
        "affine": "affine only",
        "affine_tps": "affine + TPS",
        "ground_truth": "ground truth only",
        "prediction": "P7 model prediction only",
    }
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    contour_dir = mode_dir / ("ground_truth_contours" if ground_truth_only else "predicted_contours")
    write_mode_contours(rows, classes, contour_dir)

    file_prefix = (
        f"{speaker_name.lower()}_{session_name.lower()}_ground_truth_11contour"
        if ground_truth_only
        else f"{speaker_name.lower()}_{session_name.lower()}_p7_model_prediction_11contour"
        if prediction_only
        else f"{speaker_name.lower()}_{session_name.lower()}_{mode}_prediction"
    )
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
                ground_truth_only,
                prediction_only,
            )
            writer.write(canvas)
            contour_only = ground_truth_only or prediction_only
            rmse_px_value = None if contour_only else metrics["rmse_px"]
            primary_rmse_px_value = None if contour_only else metrics["primary_rmse_px"]
            metrics_rows.append(
                {
                    "frame": row["frame"],
                    "frame_number": row["frame_number"],
                    "phoneme": row["phoneme"],
                    "held": row["held"],
                    "ground_truth_source": row.get("ground_truth_source"),
                    "prediction_source": row.get("prediction_source"),
                    "missing_ground_truth_contours": ";".join(row.get("missing_ground_truth_contours", [])),
                    "missing_prediction_contours": ";".join(row.get("missing_prediction_contours", [])),
                    "rmse_px": rmse_px_value,
                    "rmse_mm": (
                        None
                        if rmse_px_value is None
                        else rmse_px_value * MM_PER_PIXEL if not math.isnan(rmse_px_value) else float("nan")
                    ),
                    "primary_rmse_px": primary_rmse_px_value,
                    "primary_rmse_mm": (
                        None
                        if primary_rmse_px_value is None
                        else primary_rmse_px_value * MM_PER_PIXEL
                        if not math.isnan(primary_rmse_px_value)
                        else float("nan")
                    ),
                }
            )
    finally:
        writer.release()

    with (mode_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "frame",
            "frame_number",
            "phoneme",
            "held",
            "ground_truth_source",
            "prediction_source",
            "missing_ground_truth_contours",
            "missing_prediction_contours",
            "rmse_px",
            "rmse_mm",
            "primary_rmse_px",
            "primary_rmse_mm",
        ]
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
        "contours": str(contour_dir),
        "predicted_contours": None if ground_truth_only else str(contour_dir),
        "ground_truth_contours": str(contour_dir) if ground_truth_only else None,
        "ground_truth_only": bool(ground_truth_only),
        "prediction_only": bool(prediction_only),
        "frame_metrics": str(mode_dir / "frame_metrics.csv"),
        "mean_rmse_mm": None if ground_truth_only or prediction_only else mean_finite([row["rmse_mm"] for row in metrics_rows]),
        "mean_primary_rmse_mm": None if ground_truth_only or prediction_only else mean_finite([row["primary_rmse_mm"] for row in metrics_rows]),
        "num_frames": len(rows),
        "num_held_frames": int(sum(bool(row["held"]) for row in rows)),
        "num_missing_contour_frames": int(
            sum(bool(row.get("missing_ground_truth_contours") or row.get("missing_prediction_contours")) for row in rows)
        ),
        "num_missing_contour_instances": int(
            sum(len(row.get("missing_ground_truth_contours", [])) + len(row.get("missing_prediction_contours", [])) for row in rows)
        ),
        "fps": float(fps),
        "audio_start_seconds": float(audio_start_seconds),
        "duration_seconds": float(duration_seconds),
    }


def main() -> None:
    args = resolve_session_media(parse_args())
    if not args.mri_dicom_dir.is_dir():
        raise FileNotFoundError(f"Missing MRI DICOM directory: {args.mri_dicom_dir}")
    if not args.audio.is_file():
        raise FileNotFoundError(f"Missing session audio: {args.audio}")
    print(
        f"Rendering {args.speaker_name}/{args.session_name} "
        f"MRI={args.mri_dicom_dir.resolve()} audio={args.audio.resolve()}",
        flush=True,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    classes = list(config["classes"])
    ms_image = float(args.ms_image if args.ms_image is not None else config.get("ms_image", 19.98))
    fps = 1000.0 / (ms_image * float(args.timeline_step))
    state = torch.load(args.predictions, map_location="cpu")
    contour_only = args.ground_truth_only or args.prediction_only
    if contour_only and not math.isclose(float(args.timeline_step), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError(
            "--ground-truth-only and --prediction-only render integer frames only; use --timeline-step 1.0"
        )
    if contour_only and len(classes) != 11:
        raise RuntimeError(f"Contour-only rendering requires exactly 11 configured contours, got {len(classes)}")
    if args.ground_truth_only and args.ground_truth_contour_dir is None:
        raise RuntimeError("--ground-truth-only requires --ground-truth-contour-dir")
    if args.prediction_only and args.prediction_contour_dir is None:
        raise RuntimeError("--prediction-only requires --prediction-contour-dir")
    motion_report = None
    if not contour_only:
        motion_report = prediction_motion_report_from_args(
            state,
            classes,
            args,
            prediction_payload=str(args.predictions),
        )
    rows = aggregate_state(state, config, args.speaker, args.session)
    denorm_summary = prediction_denorm_summary()

    if args.ground_truth_only:
        timeline_rows = build_ground_truth_timeline(
            args.ground_truth_contour_dir,
            classes,
            float(rows[0]["frame_number"]),
            float(rows[-1]["frame_number"]),
            float(args.timeline_step),
            args.max_frames,
        )
        ground_truth_rows = timeline_rows
        prediction_only_rows = []
    elif args.prediction_only:
        timeline_rows = build_ground_truth_timeline(
            args.prediction_contour_dir,
            classes,
            float(rows[0]["frame_number"]),
            float(rows[-1]["frame_number"]),
            float(args.timeline_step),
            args.max_frames,
        )
        prediction_only_rows = []
        for row in timeline_rows:
            prediction_row = dict(row)
            prediction_row["labels"] = None
            prediction_row["predicted"] = row["mode_prediction"]
            prediction_row["prediction_source"] = row["ground_truth_source"]
            prediction_row["missing_prediction_contours"] = list(row["missing_ground_truth_contours"])
            prediction_row["missing_ground_truth_contours"] = []
            prediction_only_rows.append(prediction_row)
        timeline_rows = prediction_only_rows
        ground_truth_rows = []
    else:
        timeline_rows = build_timeline(rows, float(args.timeline_step), args.max_frames)
        ground_truth_rows = []
        prediction_only_rows = []

    needs_grid_transform = not contour_only and any(mode in {"affine", "affine_tps"} for mode in args.modes)
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
    modes = ["ground_truth"] if args.ground_truth_only else ["prediction"] if args.prediction_only else args.modes
    for mode in modes:
        if mode == "ground_truth":
            mode_rows = ground_truth_rows
        elif mode == "prediction":
            mode_rows = prediction_only_rows
        else:
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
                args.ground_truth_only,
                args.prediction_only,
            )
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "predictions": str(args.predictions),
        "config": str(args.config),
        "speaker": args.speaker,
        "session": args.session,
        "speaker_name": args.speaker_name,
        "session_name": args.session_name,
        "note": (
            "Ground-truth-only MRI overlay; predictions and RMSE are not rendered."
            if args.ground_truth_only
            else "P7-model prediction-only MRI overlay from existing cached contour files; ground truth and RMSE are not rendered."
            if args.prediction_only
            else "Raw predictions are cached model outputs for the requested session. Affine and affine_tps are optional morphology-normalized post-processing modes."
        ),
        "ground_truth_only": bool(args.ground_truth_only),
        "prediction_only": bool(args.prediction_only),
        "ground_truth_contour_dir": (
            str(args.ground_truth_contour_dir.resolve()) if args.ground_truth_contour_dir is not None else None
        ),
        "prediction_contour_dir": (
            str(args.prediction_contour_dir.resolve()) if args.prediction_contour_dir is not None else None
        ),
        "ground_truth_frame_policy": (
            "integer contours loaded directly; missing contours are shown as missing; no hold"
            if args.ground_truth_only
            else None
        ),
        "prediction_frame_policy": (
            "integer cached prediction contours loaded directly; missing contours are shown as missing; no hold"
            if args.prediction_only
            else None
        ),
        "prediction_denorm": None if args.ground_truth_only else denorm_summary,
        "source_grid_png": source_grid_png,
        "target_grid_png": target_grid_png,
        "audio": str(args.audio),
        "mri_dicom_dir": str(args.mri_dicom_dir.resolve()),
        "audio_start_seconds": audio_start_seconds,
        "ms_image": ms_image,
        "timeline_step": float(args.timeline_step),
        "fps": fps,
        "source_anchor": args.source_anchor,
        "target_anchor": args.target_anchor,
        "vtln_dir": str(args.vtln_dir),
        "has_labels": bool(timeline_rows and timeline_rows[0].get("labels") is not None),
        "num_original_prediction_frames": len(rows),
        "num_cached_ground_truth_frames": len(rows) if args.ground_truth_only else None,
        "num_direct_ground_truth_frames": (
            int(sum(row["ground_truth_lower_frame"] == row["ground_truth_upper_frame"] for row in timeline_rows))
            if args.ground_truth_only
            else None
        ),
        "num_interpolated_ground_truth_frames": (
            int(sum(row["ground_truth_lower_frame"] != row["ground_truth_upper_frame"] for row in timeline_rows))
            if args.ground_truth_only
            else None
        ),
        "num_timeline_frames": len(timeline_rows),
        "frame_min": float(timeline_rows[0]["frame_number"]),
        "frame_max": float(timeline_rows[-1]["frame_number"]),
        "outputs": outputs,
        "prediction_motion": motion_report,
        "motion_guard_enabled": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        title = "Ground Truth Video" if args.ground_truth_only else "P7 Model Prediction Video" if args.prediction_only else "Prediction Video"
        handle.write(f"# {title} For {args.speaker_name}/{args.session_name}\n\n")
        if args.ground_truth_only:
            handle.write(
                "Only integer-frame ground-truth contours are rendered. Missing contours are listed on the "
                "video and left empty. Contours are never interpolated or held from another frame.\n\n"
            )
            handle.write(f"- timeline range payload: `{args.predictions}`\n")
            handle.write(f"- ground-truth contour directory: `{args.ground_truth_contour_dir.resolve()}`\n")
        elif args.prediction_only:
            handle.write(
                "Only existing integer-frame P7-model prediction contours are rendered. Missing contours are "
                "listed on the video and left empty. Contours are never interpolated or held.\n\n"
            )
            handle.write(f"- source prediction payload: `{args.predictions}`\n")
            handle.write(f"- source prediction contour directory: `{args.prediction_contour_dir.resolve()}`\n")
        else:
            handle.write(
                "Raw predictions are produced by the cached model output already "
                "denormalized with the model training split stats. If labels are "
                "absent, the video is prediction-only and no RMSE is computed.\n\n"
            )
            handle.write("- prediction denorm mode: `payload_raw`\n")
            handle.write(f"- uses target session label stats: `{denorm_summary['uses_target_session_label_stats']}`\n")
            handle.write(f"- source prediction payload: `{args.predictions}`\n")
        if motion_report is not None and "motion_ratio_frame_diff" in motion_report:
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
            mean_rmse = "n/a" if item["mean_rmse_mm"] is None else f"{item['mean_rmse_mm']:.6f}"
            mean_primary_rmse = (
                "n/a" if item["mean_primary_rmse_mm"] is None else f"{item['mean_primary_rmse_mm']:.6f}"
            )
            handle.write(
                f"| `{item['mode']}` | {mean_rmse} | {mean_primary_rmse} | "
                f"{item['num_held_frames']} | `{Path(item['video']).relative_to(output_dir)}` | "
                f"`{Path(item['contours']).relative_to(output_dir)}` |\n"
            )
        handle.write("\n")
        if source_grid_png and target_grid_png:
            handle.write(f"- source grid: `{Path(source_grid_png).relative_to(output_dir)}`\n")
            handle.write(f"- target grid: `{Path(target_grid_png).relative_to(output_dir)}`\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
