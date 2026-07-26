#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
GRID_ROOT = REPO_ROOT / "external" / "grid-transform"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(GRID_ROOT))

from grid_transform.transfer import build_two_step_transform  # noqa: E402
from src.utils.colors import COLORS  # noqa: E402
from src.utils.gridnorm_rendering import (  # noqa: E402
    PRIMARY_GRIDNORM_CLASSES,
    build_dicom_mri_cache,
    mri_for_frame,
    needed_integer_frames,
    transform_predictions,
    write_mode_contours,
)
from src.utils.mri_rendering import load_or_build_npy_mri_cache  # noqa: E402
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

from src.adaption_pipeline.legacy.evaluate_grid_normalization import (  # noqa: E402
    load_source_target_grids,
)


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
DEFAULT_CONFIG = (
    REPO_ROOT
    / "config/inference_config/asd1_p7_seen_trainstats_p2_s1s3_unseen_eval_st5_mfcc.yaml"
)
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
        "--mri-npy-dir",
        type=Path,
        default=None,
        help="Directory of zero-padded integer MRI .npy frames; mutually exclusive with --mri-dicom-dir.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Defaults to the denoised ASD1 WAV for <speaker-name>/<session-name>.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Do not attach session audio. Required when skipped timeline frames would desynchronize audio.",
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
    parser.add_argument("--timeline-step", type=float, default=1.0)
    parser.add_argument(
        "--prediction-model-label",
        default="P7 model",
        help="Model name shown in prediction-only video text and output filenames.",
    )
    parser.add_argument(
        "--frame-min",
        type=int,
        default=None,
        help=(
            "Explicit first integer frame for contour-only rendering. Use together with "
            "--frame-max to avoid loading a prediction payload only to discover its range."
        ),
    )
    parser.add_argument(
        "--frame-max",
        type=int,
        default=None,
        help="Explicit last integer frame for contour-only rendering; requires --frame-min.",
    )
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
    contour_only_group.add_argument(
        "--ground-truth-prediction-compare",
        action="store_true",
        help=(
            "Compare direct integer-frame ground truth and predictions. Ground truth may come from "
            "--ground-truth-contour-dir or --ground-truth-contour-pack; predictions may come from "
            "--prediction-contour-dir or the cached --predictions payload."
        ),
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
        "--ground-truth-contour-pack",
        type=Path,
        default=None,
        help=(
            "Integer-frame contour NPZ with frame_numbers, contours, and articulators. "
            "In direct comparison mode this avoids scanning per-frame ground-truth files."
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
    parser.add_argument(
        "--skip-missing-prediction-frames",
        action="store_true",
        help=(
            "In prediction-only mode, omit every timeline frame missing one or more contours. "
            "This matches cached comparison rendering, which concatenates only predicted frames."
        ),
    )
    parser.add_argument(
        "--remove-silent-after-audio",
        action="store_true",
        help="Delete the intermediate silent MP4 only after the audio MP4 is attached successfully.",
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
    if args.mri_dicom_dir is None and args.mri_npy_dir is None:
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


def build_direct_comparison_timeline(
    ground_truth_dir: Path,
    prediction_dir: Path,
    classes: list[str],
    start_frame: float,
    end_frame: float,
    step: float,
    max_frames: int | None,
) -> list[dict[str, Any]]:
    ground_truth_rows = build_ground_truth_timeline(
        ground_truth_dir,
        classes,
        start_frame,
        end_frame,
        step,
        max_frames,
    )
    prediction_rows = build_ground_truth_timeline(
        prediction_dir,
        classes,
        start_frame,
        end_frame,
        step,
        max_frames,
    )
    if len(ground_truth_rows) != len(prediction_rows):
        raise AssertionError("Direct ground-truth/prediction timelines have different lengths")

    rows: list[dict[str, Any]] = []
    for ground_truth_row, prediction_row in zip(ground_truth_rows, prediction_rows):
        if ground_truth_row["frame_number"] != prediction_row["frame_number"]:
            raise AssertionError("Direct ground-truth/prediction frame mismatch")
        row = dict(ground_truth_row)
        row["labels"] = ground_truth_row["mode_prediction"]
        row["mode_prediction"] = prediction_row["mode_prediction"]
        row["predicted"] = prediction_row["mode_prediction"]
        row["prediction_source"] = prediction_row["ground_truth_source"]
        row["missing_prediction_contours"] = list(prediction_row["missing_ground_truth_contours"])
        row["missing_prediction_details"] = dict(prediction_row["missing_ground_truth_details"])
        row["direct_contour_comparison"] = True
        row["held"] = False
        rows.append(row)
    return rows


def build_packed_comparison_timeline(
    ground_truth_pack: Path,
    prediction_payload: Path | None,
    prediction_dir: Path | None,
    config: dict[str, Any],
    classes: list[str],
    speaker: int,
    session: int,
    start_frame: float,
    end_frame: float,
    step: float,
    max_frames: int | None,
) -> list[dict[str, Any]]:
    if not math.isclose(step, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("NEVER render fractional frames; packed comparison timeline step must be 1.0")
    start_integer = int(round(start_frame))
    end_integer = int(round(end_frame))
    if not math.isclose(start_frame, start_integer, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"NEVER render fractional start frame {start_frame}")
    if not math.isclose(end_frame, end_integer, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"NEVER render fractional end frame {end_frame}")
    if end_integer < start_integer:
        raise ValueError("Packed comparison end frame precedes start frame")

    ground_truth_pack = ground_truth_pack.resolve()
    if not ground_truth_pack.is_file():
        raise FileNotFoundError(f"Missing ground-truth contour pack: {ground_truth_pack}")
    if (prediction_payload is None) == (prediction_dir is None):
        raise ValueError("Packed comparison requires exactly one prediction payload or contour directory")

    with np.load(ground_truth_pack, allow_pickle=False) as pack:
        frame_numbers = np.asarray(pack["frame_numbers"])
        contours = np.asarray(pack["contours"], dtype=np.float32)
        articulators = tuple(str(value) for value in pack["articulators"].tolist())
    if articulators != tuple(classes):
        raise ValueError(f"Ground-truth pack contour order mismatch: {articulators}")
    if contours.shape != (len(frame_numbers), len(classes), 100):
        raise ValueError(
            f"Unexpected ground-truth contour pack shape: frames={frame_numbers.shape}, contours={contours.shape}"
        )
    if not np.isfinite(contours).all():
        raise ValueError("Ground-truth contour pack contains non-finite coordinates")
    if not np.allclose(frame_numbers, np.rint(frame_numbers), atol=1e-6, rtol=0.0):
        raise ValueError("Ground-truth contour pack contains fractional frames")
    ground_truth_by_frame = {
        int(round(float(frame_number))): contour
        for frame_number, contour in zip(frame_numbers, contours)
    }
    if len(ground_truth_by_frame) != len(frame_numbers):
        raise ValueError("Ground-truth contour pack contains duplicate integer frames")

    if prediction_dir is not None:
        prediction_dir = prediction_dir.resolve()
        if not prediction_dir.is_dir():
            raise FileNotFoundError(f"Missing prediction contour directory: {prediction_dir}")
        prediction_rows = build_ground_truth_timeline(
            prediction_dir,
            classes,
            start_frame,
            end_frame,
            step,
            max_frames,
        )
        for row in prediction_rows:
            row["predicted"] = row["mode_prediction"]
            row["prediction_source"] = row["ground_truth_source"]
            row["missing_prediction_contours"] = list(row["missing_ground_truth_contours"])
    else:
        prediction_payload = prediction_payload.resolve()
        if not prediction_payload.is_file():
            raise FileNotFoundError(f"Missing prediction payload: {prediction_payload}")
        prediction_state = torch.load(prediction_payload, map_location="cpu")
        prediction_rows = aggregate_state(prediction_state, config, speaker, session)
    prediction_by_frame = {
        int(round(float(row["frame_number"]))): row for row in prediction_rows
    }

    count = end_integer - start_integer + 1
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}")
        count = min(count, max_frames)
    rows: list[dict[str, Any]] = []
    for frame_number in range(start_integer, start_integer + count):
        ground_truth = ground_truth_by_frame.get(frame_number)
        prediction_row = prediction_by_frame.get(frame_number)
        prediction = None if prediction_row is None else np.asarray(prediction_row["predicted"], dtype=np.float32)
        missing_ground_truth = [] if ground_truth is not None else list(classes)
        missing_prediction = (
            list(classes)
            if prediction is None
            else list(prediction_row.get("missing_prediction_contours", []))
        )
        if prediction is None:
            prediction_source = "missing from prediction source"
        elif prediction_row.get("prediction_source"):
            prediction_source = prediction_row["prediction_source"]
        elif prediction_payload is not None:
            prediction_source = f"{prediction_payload.name} frame {frame_number:04d}"
        else:
            prediction_source = f"{prediction_dir.name} frame {frame_number:04d}"
        nan_contours = np.full((len(classes), 100), np.nan, dtype=np.float32)
        rows.append(
            {
                "frame_number": frame_number,
                "frame": f"{frame_number:04d}",
                "labels": nan_contours.copy() if ground_truth is None else ground_truth,
                "mode_prediction": nan_contours.copy() if prediction is None else prediction,
                "predicted": nan_contours.copy() if prediction is None else prediction,
                "phoneme": "n/a" if prediction_row is None else prediction_row["phoneme"],
                "held": False,
                "ground_truth_source": (
                    "missing from ground-truth pack"
                    if ground_truth is None
                    else f"{ground_truth_pack.name} frame {frame_number:04d}"
                ),
                "prediction_source": prediction_source,
                "ground_truth_lower_frame": frame_number,
                "ground_truth_upper_frame": frame_number,
                "ground_truth_interpolation_alpha": 0.0,
                "missing_ground_truth_contours": missing_ground_truth,
                "missing_prediction_contours": missing_prediction,
                "direct_contour_comparison": True,
            }
        )
    return rows


def draw_frame(
    row: dict[str, Any],
    classes: list[str],
    primary_indices: list[int],
    image: np.ndarray,
    scale: int,
    mode_label: str,
    display_label: str,
    prediction_model_label: str = "P7 model",
    ground_truth_only: bool = False,
    prediction_only: bool = False,
    direct_comparison: bool = False,
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

        ground_truth_valid = labels is not None and np.isfinite(labels[idx]).all()
        prediction_valid = np.isfinite(pred[idx]).all()
        if ground_truth_valid:
            gt_points = scale_points(labels[idx], scale)
            gt_points[:, 1] += INFO_BAND_HEIGHT
            cv2.polylines(canvas, [gt_points], isClosed=False, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [gt_points], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)
        if prediction_valid:
            pred_points = scale_points(pred[idx], scale)
            pred_points[:, 1] += INFO_BAND_HEIGHT
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
        lines.append(f"solid = {prediction_model_label} prediction")
    elif labels is not None:
        if direct_comparison:
            missing_ground_truth = list(row.get("missing_ground_truth_contours", []))
            missing_prediction = list(row.get("missing_prediction_contours", []))
            error_all = "N/A" if math.isnan(all_rmse) else f"{all_rmse * MM_PER_PIXEL:.3f} mm"
            error_primary = "N/A" if math.isnan(primary_rmse) else f"{primary_rmse * MM_PER_PIXEL:.3f} mm"
            lines.extend(
                [
                    f"RMSE: {error_all} | paired contours: {len(paired_indices)}/{len(classes)}",
                    f"RMSE primary: {error_primary}",
                    f"solid = ground truth | dashed = {prediction_model_label} prediction",
                ]
            )
            if missing_ground_truth or missing_prediction:
                lines.append(
                    f"missing GT/pred contours: {len(missing_ground_truth)}/{len(missing_prediction)}"
                )
        else:
            lines.append(f"phoneme: {row['phoneme']}")
            lines.extend(
                [
                    f"RMSE: {all_rmse * MM_PER_PIXEL:.3f} mm",
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
    fps_rational: str,
    scale: int,
    audio_path: Path | None,
    audio_start_seconds: float | None,
    speaker_name: str,
    session_name: str,
    prediction_model_label: str = "P7 model",
    ground_truth_only: bool = False,
    prediction_only: bool = False,
    direct_comparison: bool = False,
    remove_silent_after_audio: bool = False,
) -> dict[str, Any]:
    mode_labels = {
        "raw": "raw P7-model prediction",
        "affine": "affine only",
        "affine_tps": "affine + TPS",
        "ground_truth": "ground truth only",
        "prediction": f"{prediction_model_label} prediction only",
    }
    if direct_comparison:
        mode_labels["prediction"] = f"ground truth vs {prediction_model_label}"
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    contour_dir = mode_dir / ("ground_truth_contours" if ground_truth_only else "predicted_contours")
    write_mode_contours(rows, classes, contour_dir)

    prediction_model_tag = prediction_model_label.lower().replace(" ", "_").replace("/", "_")
    file_prefix = (
        f"{speaker_name.lower()}_{session_name.lower()}_ground_truth_11contour"
        if ground_truth_only
        else f"{speaker_name.lower()}_{session_name.lower()}_{prediction_model_tag}_prediction_11contour"
        if prediction_only or direct_comparison
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
                prediction_model_label,
                ground_truth_only,
                prediction_only,
                direct_comparison,
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
    audio_attached = bool(
        audio_path is not None
        and audio_start_seconds is not None
        and attach_audio(
            silent_mp4,
            audio_path,
            final_mp4,
            audio_start_seconds,
            duration_seconds,
            output_fps=fps_rational,
        )
    )
    output_mp4 = final_mp4 if audio_attached else silent_mp4
    silent_removed = False
    if remove_silent_after_audio and audio_attached:
        silent_mp4.unlink()
        silent_removed = True
    return {
        "mode": mode,
        "video": str(output_mp4),
        "silent_video": None if silent_removed else str(silent_mp4),
        "silent_video_removed": silent_removed,
        "audio_attached": audio_attached,
        "contours": str(contour_dir),
        "predicted_contours": None if ground_truth_only else str(contour_dir),
        "ground_truth_contours": str(contour_dir) if ground_truth_only else None,
        "ground_truth_only": bool(ground_truth_only),
        "prediction_only": bool(prediction_only),
        "ground_truth_prediction_compare": bool(direct_comparison),
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
        "fps_rational": fps_rational,
        "rendered_fractional_frame_count": 0,
        "audio_start_seconds": None if audio_start_seconds is None else float(audio_start_seconds),
        "duration_seconds": float(duration_seconds),
    }


def main() -> None:
    args = resolve_session_media(parse_args())
    if args.mri_dicom_dir is not None and args.mri_npy_dir is not None:
        raise ValueError("Use exactly one of --mri-dicom-dir or --mri-npy-dir")
    if args.mri_dicom_dir is None and args.mri_npy_dir is None:
        raise ValueError("One MRI source is required: --mri-dicom-dir or --mri-npy-dir")
    if args.mri_dicom_dir is not None and not args.mri_dicom_dir.is_dir():
        raise FileNotFoundError(f"Missing MRI DICOM directory: {args.mri_dicom_dir}")
    if args.mri_npy_dir is not None and not args.mri_npy_dir.is_dir():
        raise FileNotFoundError(f"Missing MRI NPY directory: {args.mri_npy_dir}")
    if not args.no_audio and not args.audio.is_file():
        raise FileNotFoundError(f"Missing session audio: {args.audio}")
    audio_description = "none (disabled)" if args.no_audio else str(args.audio.resolve())
    print(
        f"Rendering {args.speaker_name}/{args.session_name} "
        f"MRI={(args.mri_dicom_dir or args.mri_npy_dir).resolve()} audio={audio_description}",
        flush=True,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    classes = list(config["classes"])
    ms_image = float(args.ms_image if args.ms_image is not None else config.get("ms_image", 19.98))
    timeline_step = float(args.timeline_step)
    fps_fraction = Fraction(1000, 1) / Fraction(str(ms_image)) / Fraction(str(timeline_step))
    fps = float(fps_fraction)
    fps_rational = f"{fps_fraction.numerator}/{fps_fraction.denominator}"
    if not math.isclose(timeline_step, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError("NEVER render fractional frames; use --timeline-step 1.0")
    direct_comparison = bool(args.ground_truth_prediction_compare)
    contour_only = args.ground_truth_only or args.prediction_only or direct_comparison
    explicit_frame_range = args.frame_min is not None or args.frame_max is not None
    if explicit_frame_range and (args.frame_min is None or args.frame_max is None):
        raise RuntimeError("--frame-min and --frame-max must be provided together")
    if explicit_frame_range and not contour_only:
        raise RuntimeError("--frame-min/--frame-max are supported only for contour-only rendering")
    if explicit_frame_range and args.frame_max < args.frame_min:
        raise RuntimeError("--frame-max must be greater than or equal to --frame-min")
    if contour_only and len(classes) != 11:
        raise RuntimeError(f"Contour-only rendering requires exactly 11 configured contours, got {len(classes)}")
    if args.ground_truth_only and args.ground_truth_contour_dir is None:
        raise RuntimeError("--ground-truth-only requires --ground-truth-contour-dir")
    if args.prediction_only and args.prediction_contour_dir is None:
        raise RuntimeError("--prediction-only requires --prediction-contour-dir")
    if direct_comparison and args.ground_truth_contour_dir is None and args.ground_truth_contour_pack is None:
        raise RuntimeError(
            "--ground-truth-prediction-compare requires --ground-truth-contour-dir "
            "or --ground-truth-contour-pack"
        )
    if direct_comparison and args.ground_truth_contour_pack is None and args.prediction_contour_dir is None:
        raise RuntimeError(
            "Directory-based direct comparison requires --prediction-contour-dir; packed comparison "
            "uses the --predictions payload instead"
        )
    if (
        direct_comparison
        and args.ground_truth_contour_pack is not None
        and args.prediction_contour_dir is not None
        and not explicit_frame_range
    ):
        raise RuntimeError(
            "Packed ground truth plus prediction contour directory requires explicit --frame-min/--frame-max"
        )
    if args.skip_missing_prediction_frames and not args.prediction_only:
        raise RuntimeError("--skip-missing-prediction-frames requires --prediction-only")
    if args.skip_missing_prediction_frames and not args.no_audio:
        raise RuntimeError(
            "--skip-missing-prediction-frames requires --no-audio because timeline compression "
            "would desynchronize continuous session audio"
        )

    state = None
    rows = []
    packed_directory_comparison = bool(
        direct_comparison
        and args.ground_truth_contour_pack is not None
        and args.prediction_contour_dir is not None
    )
    if not explicit_frame_range:
        state = torch.load(args.predictions, map_location="cpu")
        rows = aggregate_state(state, config, args.speaker, args.session)
    range_min = float(args.frame_min) if explicit_frame_range else float(rows[0]["frame_number"])
    range_max = float(args.frame_max) if explicit_frame_range else float(rows[-1]["frame_number"])

    motion_report = None
    if not contour_only:
        assert state is not None
        motion_report = prediction_motion_report_from_args(
            state,
            classes,
            args,
            prediction_payload=str(args.predictions),
        )
    denorm_summary = prediction_denorm_summary()

    num_skipped_missing_prediction_frames = 0
    if args.ground_truth_only:
        timeline_rows = build_ground_truth_timeline(
            args.ground_truth_contour_dir,
            classes,
            range_min,
            range_max,
            float(args.timeline_step),
            args.max_frames,
        )
        ground_truth_rows = timeline_rows
        prediction_only_rows = []
    elif args.prediction_only:
        timeline_rows = build_ground_truth_timeline(
            args.prediction_contour_dir,
            classes,
            range_min,
            range_max,
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
        if args.skip_missing_prediction_frames:
            unfiltered_count = len(prediction_only_rows)
            prediction_only_rows = [
                row for row in prediction_only_rows if not row["missing_prediction_contours"]
            ]
            num_skipped_missing_prediction_frames = unfiltered_count - len(prediction_only_rows)
            if not prediction_only_rows:
                raise RuntimeError("No complete prediction frames remain after skipping missing contours")
        timeline_rows = prediction_only_rows
        ground_truth_rows = []
        direct_comparison_rows = []
    elif direct_comparison:
        if args.ground_truth_contour_pack is not None:
            timeline_rows = build_packed_comparison_timeline(
                args.ground_truth_contour_pack,
                None if packed_directory_comparison else args.predictions,
                args.prediction_contour_dir if packed_directory_comparison else None,
                config,
                classes,
                args.speaker,
                args.session,
                range_min,
                range_max,
                float(args.timeline_step),
                args.max_frames,
            )
        else:
            timeline_rows = build_direct_comparison_timeline(
                args.ground_truth_contour_dir,
                args.prediction_contour_dir,
                classes,
                range_min,
                range_max,
                float(args.timeline_step),
                args.max_frames,
            )
        direct_comparison_rows = timeline_rows
        ground_truth_rows = []
        prediction_only_rows = []
    else:
        timeline_rows = build_timeline(rows, float(args.timeline_step), args.max_frames)
        ground_truth_rows = []
        prediction_only_rows = []
        direct_comparison_rows = []

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

    if args.mri_dicom_dir is not None:
        mri_cache = build_dicom_mri_cache(args.mri_dicom_dir, timeline_rows, output_dir)
    else:
        mri_cache = load_or_build_npy_mri_cache(
            args.mri_npy_dir,
            needed_integer_frames(timeline_rows),
            output_dir / "mri_frames_cache.npz",
        )
    audio_start_seconds = (
        None
        if args.no_audio
        else (float(timeline_rows[0]["frame_number"]) - 1.0) * ms_image / 1000.0
    )
    outputs = []
    modes = (
        ["ground_truth"]
        if args.ground_truth_only
        else ["prediction"]
        if args.prediction_only or direct_comparison
        else args.modes
    )
    for mode in modes:
        if direct_comparison:
            mode_rows = direct_comparison_rows
        elif mode == "ground_truth":
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
                fps_rational,
                int(args.scale),
                None if args.no_audio else args.audio,
                audio_start_seconds,
                args.speaker_name,
                args.session_name,
                args.prediction_model_label,
                args.ground_truth_only,
                args.prediction_only,
                direct_comparison,
                bool(args.remove_silent_after_audio),
            )
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "predictions": (
            None
            if packed_directory_comparison
            else str(args.predictions.resolve())
            if direct_comparison and args.ground_truth_contour_pack is not None
            else None if explicit_frame_range else str(args.predictions)
        ),
        "prediction_payload_loaded": bool(
            not explicit_frame_range
            or (
                direct_comparison
                and args.ground_truth_contour_pack is not None
                and not packed_directory_comparison
            )
        ),
        "timeline_range_source": "explicit_frame_range" if explicit_frame_range else "prediction_payload",
        "uses_target_labels_for_rendering": bool(direct_comparison or not contour_only),
        "config": str(args.config),
        "speaker": args.speaker,
        "session": args.session,
        "speaker_name": args.speaker_name,
        "session_name": args.session_name,
        "note": (
            "Ground-truth-only MRI overlay; predictions and RMSE are not rendered."
            if args.ground_truth_only
            else f"Direct integer-frame ground truth versus {args.prediction_model_label} prediction; per-frame RMSE is rendered."
            if direct_comparison
            else f"{args.prediction_model_label} prediction-only MRI overlay from existing cached contour files; ground truth and RMSE are not rendered."
            if args.prediction_only
            else "Raw predictions are cached model outputs for the requested session. Affine and affine_tps are optional morphology-normalized post-processing modes."
        ),
        "ground_truth_only": bool(args.ground_truth_only),
        "prediction_only": bool(args.prediction_only),
        "ground_truth_prediction_compare": direct_comparison,
        "remove_silent_after_audio": bool(args.remove_silent_after_audio),
        "ground_truth_contour_dir": (
            str(args.ground_truth_contour_dir.resolve()) if args.ground_truth_contour_dir is not None else None
        ),
        "ground_truth_contour_pack": (
            str(args.ground_truth_contour_pack.resolve()) if args.ground_truth_contour_pack is not None else None
        ),
        "prediction_contour_dir": (
            str(args.prediction_contour_dir.resolve()) if args.prediction_contour_dir is not None else None
        ),
        "ground_truth_frame_policy": (
            "integer contours loaded from the versioned ground-truth NPZ; missing frames are explicit; no hold"
            if direct_comparison and args.ground_truth_contour_pack is not None
            else
            "integer contours loaded directly; missing contours are shown as missing; no hold"
            if args.ground_truth_only or direct_comparison
            else None
        ),
        "prediction_frame_policy": (
            "integer prediction contour files loaded directly; missing frames are explicit; no hold"
            if packed_directory_comparison
            else
            "integer predictions aggregated from the cached inference payload; missing frames are explicit; no hold"
            if direct_comparison and args.ground_truth_contour_pack is not None
            else
            "only complete integer prediction frames are concatenated; missing frames are skipped; no hold"
            if args.prediction_only and args.skip_missing_prediction_frames
            else "integer cached prediction contours loaded directly; missing contours are shown as missing; no hold"
            if args.prediction_only or direct_comparison
            else None
        ),
        "skip_missing_prediction_frames": bool(args.skip_missing_prediction_frames),
        "num_skipped_missing_prediction_frames": num_skipped_missing_prediction_frames,
        "prediction_denorm": (
            None
            if args.ground_truth_only
            else {
                "prediction_denorm_mode": "direct_contour_files_already_raw",
                "uses_target_session_label_stats": False,
                "description": (
                    "Prediction contour files were already de-normalized with the model training-split "
                    "statistics before rendering; no target-session statistics are used."
                ),
            }
            if packed_directory_comparison
            else denorm_summary
        ),
        "source_grid_png": source_grid_png,
        "target_grid_png": target_grid_png,
        "audio": None if args.no_audio else str(args.audio),
        "mri_dicom_dir": None if args.mri_dicom_dir is None else str(args.mri_dicom_dir.resolve()),
        "mri_npy_dir": None if args.mri_npy_dir is None else str(args.mri_npy_dir.resolve()),
        "audio_start_seconds": audio_start_seconds,
        "ms_image": ms_image,
        "timeline_step": float(args.timeline_step),
        "fps": fps,
        "fps_rational": fps_rational,
        "rendered_fractional_frame_count": 0,
        "fractional_frame_policy": "integer MRI frames only; no fractional frame is rendered, interpolated, or held",
        "source_anchor": args.source_anchor,
        "target_anchor": args.target_anchor,
        "vtln_dir": str(args.vtln_dir),
        "has_labels": bool(timeline_rows and timeline_rows[0].get("labels") is not None),
        "num_original_prediction_frames": (
            int(sum(not row.get("missing_prediction_contours") for row in timeline_rows))
            if direct_comparison
            else len(rows)
        ),
        "num_cached_ground_truth_frames": (
            int(sum(not row.get("missing_ground_truth_contours") for row in timeline_rows))
            if direct_comparison
            else len(rows) if args.ground_truth_only else None
        ),
        "num_direct_ground_truth_frames": (
            int(sum(not row.get("missing_ground_truth_contours") for row in timeline_rows))
            if direct_comparison
            else int(sum(row["ground_truth_lower_frame"] == row["ground_truth_upper_frame"] for row in timeline_rows))
            if args.ground_truth_only
            else None
        ),
        "num_interpolated_ground_truth_frames": (
            0
            if direct_comparison
            else int(sum(row["ground_truth_lower_frame"] != row["ground_truth_upper_frame"] for row in timeline_rows))
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
        title = (
            "Ground Truth Video"
            if args.ground_truth_only
            else f"Ground Truth Versus {args.prediction_model_label}"
            if direct_comparison
            else f"{args.prediction_model_label} Prediction Video"
            if args.prediction_only
            else "Prediction Video"
        )
        handle.write(f"# {title} For {args.speaker_name}/{args.session_name}\n\n")
        if args.ground_truth_only:
            handle.write(
                "Only integer-frame ground-truth contours are rendered. Missing contours are listed on the "
                "video and left empty. Contours are never interpolated or held from another frame.\n\n"
            )
            handle.write(f"- timeline range payload: `{args.predictions}`\n")
            handle.write(f"- ground-truth contour directory: `{args.ground_truth_contour_dir.resolve()}`\n")
        elif direct_comparison:
            handle.write(
                "Direct integer-frame ground-truth contours are drawn solid and prediction contours are drawn "
                "dashed. Per-frame error is computed only from contours available at that exact integer frame. "
                "Missing contours remain explicit; contours are never interpolated or held.\n\n"
            )
            if args.ground_truth_contour_pack is not None:
                handle.write(f"- ground-truth contour pack: `{args.ground_truth_contour_pack.resolve()}`\n")
                if packed_directory_comparison:
                    handle.write(f"- prediction contour directory: `{args.prediction_contour_dir.resolve()}`\n")
                else:
                    handle.write(f"- prediction payload: `{args.predictions.resolve()}`\n")
            else:
                handle.write(f"- ground-truth contour directory: `{args.ground_truth_contour_dir.resolve()}`\n")
                handle.write(f"- prediction contour directory: `{args.prediction_contour_dir.resolve()}`\n")
        elif args.prediction_only:
            if args.skip_missing_prediction_frames:
                handle.write(
                    f"Only complete integer-frame {args.prediction_model_label} prediction contours are rendered. "
                    "Missing timeline frames are omitted, matching cached comparison rendering. Contours are never "
                    "interpolated or held. Continuous audio is disabled because the timeline is compressed.\n\n"
                )
            else:
                handle.write(
                    f"Only existing integer-frame {args.prediction_model_label} prediction contours are rendered. Missing contours are "
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
        handle.write(
            "- audio start: `disabled`\n"
            if audio_start_seconds is None
            else f"- audio start: `{audio_start_seconds:.3f}s`\n"
        )
        handle.write(f"- fps: `{fps:.3f}`\n\n")
        handle.write(f"- exact fps rational: `{fps_rational}`\n\n")
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
