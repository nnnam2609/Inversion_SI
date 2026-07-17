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

from evaluate_grid_normalization import load_source_target_grids  # noqa: E402
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
from src.utils.session_rendering import aggregate_state, build_timeline, load_config  # noqa: E402
from src.utils.video_rendering import (  # noqa: E402
    MM_PER_PIXEL,
    attach_audio,
    draw_dashed_polyline,
    mean_finite,
    rgb_to_bgr255,
    rmse_px,
    scale_points,
)


RAW_ROOT = Path(
    "/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/"
    "ArtSpeech_Database_1_raw"
)
DEFAULT_PREDICTIONS = (
    REPO_ROOT
    / "results/p7_stdfloor01_p7s15_p2s1_prediction_videos_20260705_225553/p2_s1/eval/cached_session_predictions.pt"
)
DEFAULT_CONFIG = REPO_ROOT / "config/train_config/asd1_p7_stdfloor01_trainstats_p2_s1_unseen_eval_st5_mfcc.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/p7_to_p2_three_stage_mri_compare_p2_s1_20260706"
DEFAULT_DICOM = RAW_ROOT / "P2/DCM_2D/S1"
DEFAULT_AUDIO = RAW_ROOT / "P2/OTHER/S1/DENOISED_SOUND_P2_S1.wav"
DEFAULT_VTLN_DIR = WORKSPACE_ROOT / "_downloads/grid-transform-vtln/vtln-data-v0.1.14/extracted/VTLN/data"
INFO_BAND_HEIGHT = 118
STAGE_MODES = ("raw", "affine", "affine_tps")
STAGE_LABELS = {
    "raw": "Stage 1 Raw P7-model",
    "affine": "Stage 2 Affine P7->P2",
    "affine_tps": "Stage 3 Affine+TPS P7->P2",
}
PRIMARY_CLASSES = PRIMARY_GRIDNORM_CLASSES
ANTERIOR_VISIBLE_CLASSES = {
    "lower-lip",
    "upper-lip",
    "tongue",
    "lower-incisor",
    "upper-incisor",
    "soft-palate-midline",
}
POSTERIOR_CLASSES = {
    "pharynx",
    "arytenoid-cartilage",
    "epiglottis",
    "vocal-folds",
    "thyroid-cartilage",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render 3-stage raw/affine/affine+TPS MRI comparison video.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--speaker", type=int, default=2)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--speaker-name", default="P2")
    parser.add_argument("--session-name", default="S1")
    parser.add_argument("--mri-dicom-dir", type=Path, default=DEFAULT_DICOM)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
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
    parser.add_argument("--reference-vowel", default="i")
    parser.add_argument("--reference-frame", default="0589")
    parser.add_argument(
        "--overlay-rmse-mode",
        choices=("all_and_primary", "all"),
        default="all_and_primary",
        help="Choose whether panel overlays show both overall and primary RMSE or only the overall RMSE.",
    )
    parser.add_argument(
        "--overlay-overall-label",
        default="RMSE all",
        help="Text label for the overall RMSE line shown in video panels.",
    )
    parser.add_argument(
        "--final-video-name",
        default="p2_s1_mri_raw_affine_affine_tps_audio.mp4",
        help="Filename for the audio-attached comparison video.",
    )
    parser.add_argument(
        "--silent-video-name",
        default="p2_s1_mri_raw_affine_affine_tps_silent.mp4",
        help="Filename for the intermediate silent comparison video.",
    )
    parser.add_argument(
        "--allow-legacy-audio-vtln",
        action="store_true",
        help="Allow rendering legacy NPZ audio-VTLN payloads for diagnostics only.",
    )
    add_prediction_motion_arguments(parser, include_diagnostic_flag=True, action_word="Report")
    return parser.parse_args()


def draw_panel(
    row: dict[str, Any],
    classes: list[str],
    primary_indices: list[int],
    title: str,
    scale: int,
    mri_image: np.ndarray | None,
    overlay_rmse_mode: str = "all_and_primary",
    overlay_overall_label: str = "RMSE all",
) -> tuple[np.ndarray, dict[str, float]]:
    labels = row["labels"].reshape(len(classes), 50, 2)
    pred = row["mode_prediction"].reshape(len(classes), 50, 2)
    all_rmse = rmse_px(pred, labels)
    primary_rmse = rmse_px(pred[primary_indices], labels[primary_indices])
    base_size = 136 * scale
    if mri_image is None:
        image_canvas = np.full((base_size, base_size, 3), 44, dtype=np.uint8)
    else:
        image_canvas = cv2.cvtColor(mri_image, cv2.COLOR_GRAY2BGR)
        image_canvas = cv2.resize(image_canvas, (base_size, base_size), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((base_size + INFO_BAND_HEIGHT, base_size, 3), 15, dtype=np.uint8)
    canvas[:INFO_BAND_HEIGHT, :] = 15
    canvas[INFO_BAND_HEIGHT : INFO_BAND_HEIGHT + base_size, :] = image_canvas

    for idx, articulator in enumerate(classes):
        color = rgb_to_bgr255(COLORS.get(articulator, "white"))
        gt_points = scale_points(labels[idx], scale)
        pred_points = scale_points(pred[idx], scale)
        gt_points[:, 1] += INFO_BAND_HEIGHT
        pred_points[:, 1] += INFO_BAND_HEIGHT
        cv2.polylines(canvas, [gt_points], isClosed=False, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [gt_points], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)
        draw_dashed_polyline(canvas, pred_points, color=(0, 0, 0), thickness=2)
        draw_dashed_polyline(canvas, pred_points, color=color, thickness=1)

    lines = [
        title,
        f"frame {row['frame']} | phoneme {row['phoneme']}",
        f"{overlay_overall_label} {all_rmse * MM_PER_PIXEL:.3f} mm",
    ]
    if overlay_rmse_mode == "all_and_primary":
        lines.append(f"RMSE primary {primary_rmse * MM_PER_PIXEL:.3f} mm")
    lines.append("solid GT | dashed pred")
    if row["held"]:
        lines[-1] += " | held"
    for line_index, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (12, 21 + line_index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return canvas, {
        "rmse_mm": all_rmse * MM_PER_PIXEL,
        "primary_rmse_mm": primary_rmse * MM_PER_PIXEL,
    }


def selected_indices(classes: list[str], names: set[str]) -> list[int]:
    return [idx for idx, name in enumerate(classes) if name in names]


def mode_metric_rows(rows_by_mode: dict[str, list[dict[str, Any]]], classes: list[str]) -> list[dict[str, Any]]:
    metric_rows = []
    class_indices = list(range(len(classes)))
    groups: list[tuple[str, list[int], str]] = [
        *[(name, [idx], "articulator") for idx, name in enumerate(classes)],
        ("Overall (all 11)", class_indices, "overall"),
        ("Primary set", selected_indices(classes, PRIMARY_CLASSES), "group"),
        ("Anterior visible", selected_indices(classes, ANTERIOR_VISIBLE_CLASSES), "group"),
        ("Posterior/cartilage", selected_indices(classes, POSTERIOR_CLASSES), "group"),
    ]
    for label, indices, row_type in groups:
        if not indices:
            continue
        row: dict[str, Any] = {"row_type": row_type, "articulator": label}
        for mode, rows in rows_by_mode.items():
            values = []
            for item in rows:
                labels = item["labels"][indices]
                pred = item["mode_prediction"][indices]
                values.append(rmse_px(pred, labels) * MM_PER_PIXEL)
            row[f"{mode}_rmse_mm"] = mean_finite(values)
        row["affine_delta_mm"] = row["affine_rmse_mm"] - row["raw_rmse_mm"]
        row["affine_tps_delta_mm"] = row["affine_tps_rmse_mm"] - row["raw_rmse_mm"]
        metric_rows.append(row)
    return metric_rows


def write_per_articulator_outputs(output_dir: Path, metric_rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path = output_dir / "per_articulator_pc2p_rmse.csv"
    fieldnames = [
        "row_type",
        "articulator",
        "raw_rmse_mm",
        "affine_rmse_mm",
        "affine_tps_rmse_mm",
        "affine_delta_mm",
        "affine_tps_delta_mm",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    latex_path = output_dir / "per_articulator_pc2p_rmse_table.tex"
    with latex_path.open("w", encoding="utf-8") as handle:
        handle.write("% Requires \\usepackage{booktabs}\n")
        handle.write("\\begin{table}[ht]\n")
        handle.write("\\centering\n")
        handle.write("\\caption{P2/S1 point-to-point RMSE (mm) for raw, affine, and affine+TPS P7$\\rightarrow$P2 contour stages.}\n")
        handle.write("\\label{tab:p2s1_p7_to_p2_pc2p_rmse}\n")
        handle.write("\\begin{tabular}{lrrrrr}\n")
        handle.write("\\toprule\n")
        handle.write("Articulator & Raw & Affine & Affine+TPS & $\\Delta$ Affine & $\\Delta$ Affine+TPS \\\\\n")
        handle.write("\\midrule\n")
        previous_type = "articulator"
        for row in metric_rows:
            if row["row_type"] != previous_type:
                handle.write("\\midrule\n")
                previous_type = row["row_type"]
            label = str(row["articulator"]).replace("_", "\\_")
            if row["row_type"] != "articulator":
                label = f"\\textbf{{{label}}}"
            handle.write(
                f"{label} & {row['raw_rmse_mm']:.3f} & {row['affine_rmse_mm']:.3f} & "
                f"{row['affine_tps_rmse_mm']:.3f} & {row['affine_delta_mm']:+.3f} & "
                f"{row['affine_tps_delta_mm']:+.3f} \\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")
        handle.write("\\end{table}\n")
    return csv_path, latex_path


def resize_with_pad(image: np.ndarray, width: int, height: int, fill: int = 15) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    output = np.full((height, width, 3), fill, dtype=np.uint8)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    output[y : y + new_h, x : x + new_w] = resized
    return output


def add_story_title(panel: np.ndarray, title: str) -> np.ndarray:
    output = panel.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 38), (15, 15, 15), thickness=-1)
    cv2.putText(output, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
    return output


def load_grid_panel(path: str | Path, title: str, width: int, height: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read grid image: {path}")
    return add_story_title(resize_with_pad(image, width, height, fill=245), title)


def pick_reference_index(rows: list[dict[str, Any]], requested_frame: str) -> int:
    for index, row in enumerate(rows):
        if row["frame"] == requested_frame and not row["held"]:
            return index
    numeric = int(requested_frame.replace("p", ""))
    distances = []
    for index, row in enumerate(rows):
        if row["held"]:
            continue
        distances.append((abs(float(row["frame_number"]) - float(numeric)), index))
    return min(distances)[1]


def write_step_storyboard(
    output_dir: Path,
    rows_by_mode: dict[str, list[dict[str, Any]]],
    classes: list[str],
    primary_indices: list[int],
    mri_cache: dict[int, np.ndarray],
    source_grid_png: str,
    target_grid_png: str,
    reference_frame: str,
    scale: int,
    overlay_rmse_mode: str,
    overlay_overall_label: str,
) -> Path:
    ref_index = pick_reference_index(rows_by_mode["raw"], reference_frame)
    mri_image = mri_for_frame(rows_by_mode["raw"][ref_index]["frame_number"], mri_cache)
    panels = [
        load_grid_panel(source_grid_png, "Step 1: build P7 source grid", 136 * scale, 136 * scale + INFO_BAND_HEIGHT),
        load_grid_panel(target_grid_png, "Step 2: build P2 target grid", 136 * scale, 136 * scale + INFO_BAND_HEIGHT),
    ]
    for mode in STAGE_MODES:
        panel, _ = draw_panel(
            rows_by_mode[mode][ref_index],
            classes,
            primary_indices,
            f"Step {3 + STAGE_MODES.index(mode)}: {STAGE_LABELS[mode]}",
            scale,
            mri_image,
            overlay_rmse_mode,
            overlay_overall_label,
        )
        panels.append(panel)
    final_canvas = make_comparison_frame(
        {mode: rows_by_mode[mode][ref_index] for mode in STAGE_MODES},
        classes,
        primary_indices,
        mri_image,
        scale,
        overlay_rmse_mode,
        overlay_overall_label,
    )
    panels.append(add_story_title(resize_with_pad(final_canvas, 136 * scale, 136 * scale + INFO_BAND_HEIGHT), "Step 6: final 3-panel video frame"))

    width = panels[0].shape[1]
    height = panels[0].shape[0]
    separator = 18
    storyboard = np.full((height * 2 + separator, width * 3 + separator * 2, 3), 245, dtype=np.uint8)
    for index, panel in enumerate(panels):
        x = (index % 3) * (width + separator)
        y = (index // 3) * (height + separator)
        storyboard[y : y + height, x : x + width] = panel
    output_path = output_dir / "step_by_step_visualization.png"
    cv2.imwrite(str(output_path), storyboard)
    return output_path


def make_comparison_frame(
    rows_by_mode_for_frame: dict[str, dict[str, Any]],
    classes: list[str],
    primary_indices: list[int],
    mri_image: np.ndarray,
    scale: int,
    overlay_rmse_mode: str = "all_and_primary",
    overlay_overall_label: str = "RMSE all",
) -> np.ndarray:
    panels = []
    for mode in STAGE_MODES:
        panel, _ = draw_panel(
            rows_by_mode_for_frame[mode],
            classes,
            primary_indices,
            STAGE_LABELS[mode],
            scale,
            mri_image,
            overlay_rmse_mode,
            overlay_overall_label,
        )
        panels.append(panel)
    separator = 12
    panel_height, panel_width = panels[0].shape[:2]
    canvas = np.full((panel_height, panel_width * len(panels) + separator * (len(panels) - 1), 3), 15, dtype=np.uint8)
    for index, panel in enumerate(panels):
        x = index * (panel_width + separator)
        canvas[:, x : x + panel_width] = panel
    return canvas


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config, allow_legacy_audio_vtln=args.allow_legacy_audio_vtln)
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
    original_rows = aggregate_state(state, config, args.speaker, args.session)
    timeline_rows = build_timeline(original_rows, float(args.timeline_step), args.max_frames)

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

    rows_by_mode = {mode: transform_predictions(timeline_rows, transform, mode, len(classes)) for mode in STAGE_MODES}
    original_rows_by_mode = {mode: transform_predictions(original_rows, transform, mode, len(classes)) for mode in STAGE_MODES}
    contour_dirs = {}
    for mode, rows in rows_by_mode.items():
        contour_dir = output_dir / f"{mode}_predicted_contours"
        write_mode_contours(rows, classes, contour_dir)
        contour_dirs[mode] = contour_dir

    primary_indices = [idx for idx, name in enumerate(classes) if name in PRIMARY_CLASSES]
    panel_width = 136 * int(args.scale)
    panel_height = panel_width + INFO_BAND_HEIGHT
    separator = 12
    video_size = (panel_width * len(STAGE_MODES) + separator * (len(STAGE_MODES) - 1), panel_height)
    mri_cache = build_dicom_mri_cache(args.mri_dicom_dir, rows_by_mode["raw"], output_dir)

    silent_mp4 = output_dir / args.silent_video_name
    final_mp4 = output_dir / args.final_video_name
    writer = cv2.VideoWriter(str(silent_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), video_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {silent_mp4}")

    frame_metric_rows = []
    try:
        for frame_index in range(len(rows_by_mode["raw"])):
            row_by_mode = {mode: rows_by_mode[mode][frame_index] for mode in STAGE_MODES}
            mri_image = mri_for_frame(row_by_mode["raw"]["frame_number"], mri_cache)
            panels = []
            metrics = {}
            for mode in STAGE_MODES:
                panel, metric = draw_panel(
                    row_by_mode[mode],
                    classes,
                    primary_indices,
                    STAGE_LABELS[mode],
                    int(args.scale),
                    mri_image,
                    args.overlay_rmse_mode,
                    args.overlay_overall_label,
                )
                panels.append(panel)
                metrics[mode] = metric
            canvas = np.full((panel_height, video_size[0], 3), 15, dtype=np.uint8)
            for panel_index, panel in enumerate(panels):
                x = panel_index * (panel_width + separator)
                canvas[:, x : x + panel_width] = panel
            writer.write(canvas)
            raw = row_by_mode["raw"]
            frame_metric_rows.append(
                {
                    "frame": raw["frame"],
                    "frame_number": raw["frame_number"],
                    "phoneme": raw["phoneme"],
                    "held": raw["held"],
                    "raw_rmse_mm": metrics["raw"]["rmse_mm"],
                    "raw_primary_rmse_mm": metrics["raw"]["primary_rmse_mm"],
                    "affine_rmse_mm": metrics["affine"]["rmse_mm"],
                    "affine_primary_rmse_mm": metrics["affine"]["primary_rmse_mm"],
                    "affine_tps_rmse_mm": metrics["affine_tps"]["rmse_mm"],
                    "affine_tps_primary_rmse_mm": metrics["affine_tps"]["primary_rmse_mm"],
                    "affine_delta_rmse_mm": metrics["affine"]["rmse_mm"] - metrics["raw"]["rmse_mm"],
                    "affine_tps_delta_rmse_mm": metrics["affine_tps"]["rmse_mm"] - metrics["raw"]["rmse_mm"],
                    "affine_delta_primary_rmse_mm": metrics["affine"]["primary_rmse_mm"] - metrics["raw"]["primary_rmse_mm"],
                    "affine_tps_delta_primary_rmse_mm": metrics["affine_tps"]["primary_rmse_mm"] - metrics["raw"]["primary_rmse_mm"],
                }
            )
    finally:
        writer.release()

    duration_seconds = len(rows_by_mode["raw"]) / float(fps)
    audio_start_seconds = (float(rows_by_mode["raw"][0]["frame_number"]) - 1.0) * ms_image / 1000.0
    audio_attached = attach_audio(silent_mp4, args.audio, final_mp4, audio_start_seconds, duration_seconds)
    output_video = final_mp4 if audio_attached else silent_mp4

    frame_metrics_csv = output_dir / "frame_metrics.csv"
    with frame_metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "frame",
            "frame_number",
            "phoneme",
            "held",
            "raw_rmse_mm",
            "raw_primary_rmse_mm",
            "affine_rmse_mm",
            "affine_primary_rmse_mm",
            "affine_tps_rmse_mm",
            "affine_tps_primary_rmse_mm",
            "affine_delta_rmse_mm",
            "affine_tps_delta_rmse_mm",
            "affine_delta_primary_rmse_mm",
            "affine_tps_delta_primary_rmse_mm",
        ]
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(frame_metric_rows)

    per_artic_rows = mode_metric_rows(original_rows_by_mode, classes)
    per_artic_csv, per_artic_latex = write_per_articulator_outputs(output_dir, per_artic_rows)
    step_storyboard = write_step_storyboard(
        output_dir,
        rows_by_mode,
        classes,
        primary_indices,
        mri_cache,
        source_grid_png,
        target_grid_png,
        args.reference_frame,
        int(args.scale),
        args.overlay_rmse_mode,
        args.overlay_overall_label,
    )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "predictions": str(args.predictions),
        "config": str(args.config),
        "speaker": args.speaker,
        "session": args.session,
        "comparison_video": str(output_video),
        "silent_video": str(silent_mp4),
        "step_by_step_visualization": str(step_storyboard),
        "audio_attached": audio_attached,
        "audio": str(args.audio),
        "mri_dicom_dir": str(args.mri_dicom_dir),
        "audio_start_seconds": audio_start_seconds,
        "duration_seconds": duration_seconds,
        "fps": fps,
        "num_frames": len(rows_by_mode["raw"]),
        "num_original_prediction_frames": len(original_rows),
        "num_held_frames": int(sum(bool(row["held"]) for row in rows_by_mode["raw"])),
        "reference_vowel": args.reference_vowel,
        "reference_frame": args.reference_frame,
        "source_anchor": args.source_anchor,
        "target_anchor": args.target_anchor,
        "source_grid_png": source_grid_png,
        "target_grid_png": target_grid_png,
        "frame_metrics": str(frame_metrics_csv),
        "per_articulator_pc2p_rmse_csv": str(per_artic_csv),
        "per_articulator_pc2p_rmse_latex": str(per_artic_latex),
        "predicted_contours": {mode: str(path) for mode, path in contour_dirs.items()},
        "mean_raw_rmse_mm": mean_finite([row["raw_rmse_mm"] for row in frame_metric_rows]),
        "mean_affine_rmse_mm": mean_finite([row["affine_rmse_mm"] for row in frame_metric_rows]),
        "mean_affine_tps_rmse_mm": mean_finite([row["affine_tps_rmse_mm"] for row in frame_metric_rows]),
        "mean_affine_delta_rmse_mm": mean_finite([row["affine_delta_rmse_mm"] for row in frame_metric_rows]),
        "mean_affine_tps_delta_rmse_mm": mean_finite([row["affine_tps_delta_rmse_mm"] for row in frame_metric_rows]),
        "mean_raw_primary_rmse_mm": mean_finite([row["raw_primary_rmse_mm"] for row in frame_metric_rows]),
        "mean_affine_primary_rmse_mm": mean_finite([row["affine_primary_rmse_mm"] for row in frame_metric_rows]),
        "mean_affine_tps_primary_rmse_mm": mean_finite([row["affine_tps_primary_rmse_mm"] for row in frame_metric_rows]),
        "mean_affine_delta_primary_rmse_mm": mean_finite([row["affine_delta_primary_rmse_mm"] for row in frame_metric_rows]),
        "mean_affine_tps_delta_primary_rmse_mm": mean_finite([row["affine_tps_delta_primary_rmse_mm"] for row in frame_metric_rows]),
        "prediction_motion": motion_report,
        "motion_guard_enabled": False,
        "overlay_rmse_mode": args.overlay_rmse_mode,
        "overlay_overall_label": args.overlay_overall_label,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    overall_row = next(row for row in per_artic_rows if row["articulator"] == "Overall (all 11)")
    with (output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# P2/S1 MRI Overlay 3-Stage P7->P2 Grid Transform Video\n\n")
        handle.write(f"- comparison video: `{Path(output_video).name}`\n")
        handle.write(f"- step-by-step image: `{Path(step_storyboard).name}`\n")
        handle.write(f"- MRI DICOM dir: `{args.mri_dicom_dir}`\n")
        handle.write(f"- reference vowel/frame from previous check: `/{args.reference_vowel}/`, frame `{args.reference_frame}`\n")
        handle.write("- stages: raw, affine only, affine+TPS\n")
        handle.write("- solid contours: current P2 ground truth\n")
        handle.write("- dashed contours: prediction at each stage\n")
        handle.write("- negative delta means transformed is better.\n\n")
        if "motion_ratio_frame_diff" in motion_report:
            handle.write(f"- pred/label motion ratio: `{motion_report['motion_ratio_frame_diff']:.6f}`\n")
            handle.write(f"- static prediction flag: `{motion_report['is_static_prediction']}`\n\n")
        handle.write("## Video Timeline Metrics\n\n")
        handle.write("| Metric | Raw | Affine | Affine+TPS | Affine delta | Affine+TPS delta |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        handle.write(
            f"| overall RMSE mm | {summary['mean_raw_rmse_mm']:.6f} | {summary['mean_affine_rmse_mm']:.6f} | "
            f"{summary['mean_affine_tps_rmse_mm']:.6f} | {summary['mean_affine_delta_rmse_mm']:.6f} | "
            f"{summary['mean_affine_tps_delta_rmse_mm']:.6f} |\n"
        )
        handle.write(
            f"| primary RMSE mm | {summary['mean_raw_primary_rmse_mm']:.6f} | {summary['mean_affine_primary_rmse_mm']:.6f} | "
            f"{summary['mean_affine_tps_primary_rmse_mm']:.6f} | {summary['mean_affine_delta_primary_rmse_mm']:.6f} | "
            f"{summary['mean_affine_tps_delta_primary_rmse_mm']:.6f} |\n\n"
        )
        handle.write("## Original-Frame Overall PC2P RMSE\n\n")
        handle.write(
            f"- all 11 contours: raw `{overall_row['raw_rmse_mm']:.6f}`, affine `{overall_row['affine_rmse_mm']:.6f}`, "
            f"affine+TPS `{overall_row['affine_tps_rmse_mm']:.6f}`.\n"
        )
        handle.write(f"- frame metrics: `{frame_metrics_csv.name}`\n")
        handle.write(f"- per-articulator CSV: `{per_artic_csv.name}`\n")
        handle.write(f"- per-articulator LaTeX: `{per_artic_latex.name}`\n")
        for mode, path in contour_dirs.items():
            handle.write(f"- {mode} contours: `{path.name}/`\n")
        handle.write(f"- source grid: `{Path(source_grid_png).relative_to(output_dir)}`\n")
        handle.write(f"- target grid: `{Path(target_grid_png).relative_to(output_dir)}`\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
