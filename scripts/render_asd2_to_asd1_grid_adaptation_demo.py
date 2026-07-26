#!/usr/bin/env python3
"""Render a clean 2x3 ASD2-to-ASD1 grid-adaptation demo.

This is a rendering-only utility. It reuses an existing adapted prediction
pack, cached MRI frames, and the audited audio stream from the canonical
session video; it does not run inference or training.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")]

import run_asd2_fixedbs10_textgrid_u_grid_adaptation as source  # noqa: E402


DEFAULT_RESULT_ROOT = (
    REPO_ROOT
    / "results/asd2_fixedbs10_selected_9sessions_textgrid_u_grid_adaptation_20260721_143350"
)
PANEL_SIZE = source.PANEL_SIZE
INFO_HEIGHT = source.INFO_HEIGHT
SEPARATOR = source.SEPARATOR
FPS = source.FPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--speaker", type=int, default=1)
    parser.add_argument("--session", type=int, default=16)
    parser.add_argument("--target-frame", type=int)
    parser.add_argument("--rms-vtln-pack", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def draw_contour_panel(
    image: np.ndarray,
    title: str,
    frame: int,
    predicted: np.ndarray | None,
    ground_truth: np.ndarray,
) -> np.ndarray:
    """Draw a contour panel without model-name or integer-timeline wording."""
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_bgr = cv2.resize(
        image_bgr, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_CUBIC
    )
    canvas = np.full((INFO_HEIGHT + PANEL_SIZE, PANEL_SIZE, 3), 14, dtype=np.uint8)
    canvas[INFO_HEIGHT:] = image_bgr
    scale = PANEL_SIZE // image.shape[1]

    for index, class_name in enumerate(source.base.CLASSES):
        color = source.base.rgb_to_bgr255(source.base.COLORS.get(class_name, "white"))
        ground_truth_points = source.base.scale_points(ground_truth[index], scale)
        ground_truth_points[:, 1] += INFO_HEIGHT
        cv2.polylines(canvas, [ground_truth_points], False, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.polylines(canvas, [ground_truth_points], False, color, 1, cv2.LINE_AA)
        if predicted is not None:
            predicted_points = source.base.scale_points(predicted[index], scale)
            predicted_points[:, 1] += INFO_HEIGHT
            source.base.draw_dashed_polyline(
                canvas, predicted_points, (0, 0, 0), 2, dash_length=7, gap_length=9
            )
            source.base.draw_dashed_polyline(
                canvas, predicted_points, color, 1, dash_length=7, gap_length=9
            )

    if predicted is None:
        status = "Ground truth only"
        legend = "Solid ground truth"
    else:
        delta = predicted.astype(np.float64) - ground_truth.astype(np.float64)
        rmse = float(np.sqrt(np.mean(delta * delta)) * source.base.MM_PER_PIXEL)
        status = f"RMSE all 11: {rmse:.3f} mm"
        legend = "Solid GT | dashed prediction"

    lines = (title, f"MRI frame {frame:04d}", status, legend)
    for line_index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (7, 17 + line_index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return canvas


def draw_stage_effect_panel(
    frame: int,
    target_frame: int,
    ground_truth: np.ndarray,
    pack: dict[str, Any],
    rms_vtln_pack: dict[str, Any],
    index: int,
) -> np.ndarray:
    canvas = np.full((INFO_HEIGHT + PANEL_SIZE, PANEL_SIZE, 3), 14, dtype=np.uint8)
    values = {
        stage: float(
            source.base.frame_rmse(
                pack[stage][index : index + 1],
                ground_truth[None],
                source.GROUPS["all_11"],
            )[0]
        )
        for stage in source.STAGES
    }
    rms_vtln_value = float(
        source.base.frame_rmse(
            rms_vtln_pack["affine_tps"][index : index + 1],
            ground_truth[None],
            source.GROUPS["all_11"],
        )[0]
    )
    frame_line = f"MRI frame {frame:04d}"
    if frame == target_frame:
        frame_line += " [calibration /u/]"
    lines = (
        "Stage effect",
        frame_line,
        f"Raw              {values['raw']:.3f} mm",
        f"Affine           {values['affine']:.3f} mm",
        f"  effect vs raw  {values['affine'] - values['raw']:+.3f} mm",
        f"Affine + TPS     {values['affine_tps']:.3f} mm",
        f"  effect vs affine {values['affine_tps'] - values['affine']:+.3f} mm",
        "Contour slide + audio normalization",
        f"  RMSE           {rms_vtln_value:.3f} mm",
        f"  effect vs TPS  {rms_vtln_value - values['affine_tps']:+.3f} mm",
    )
    for line_index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (7, 22 + line_index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return canvas


def compose_canvas(
    image: np.ndarray,
    frame: int,
    index: int,
    target_frame: int,
    pack: dict[str, Any],
    rms_vtln_pack: dict[str, Any],
) -> np.ndarray:
    ground_truth = pack["ground_truth"][index]
    panels = (
        draw_contour_panel(image, "Raw", frame, pack["raw"][index], ground_truth),
        draw_contour_panel(image, "Affine", frame, pack["affine"][index], ground_truth),
        draw_contour_panel(
            image, "Affine + TPS", frame, pack["affine_tps"][index], ground_truth
        ),
        draw_contour_panel(image, "Ground truth only", frame, None, ground_truth),
        draw_contour_panel(
            image,
            "Contour slide + audio normalization",
            frame,
            rms_vtln_pack["affine_tps"][index],
            ground_truth,
        ),
        draw_stage_effect_panel(
            frame, target_frame, ground_truth, pack, rms_vtln_pack, index
        ),
    )
    height = 2 * (INFO_HEIGHT + PANEL_SIZE) + SEPARATOR
    width = 3 * PANEL_SIZE + 2 * SEPARATOR
    canvas = np.full((height, width, 3), 8, dtype=np.uint8)
    for panel_index, panel in enumerate(panels):
        row, column = divmod(panel_index, 3)
        y = row * (INFO_HEIGHT + PANEL_SIZE + SEPARATOR)
        x = column * (PANEL_SIZE + SEPARATOR)
        canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
    return canvas


def load_cached_mri(cache_path: Path, expected_frames: np.ndarray) -> dict[int, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as payload:
        cached_frames = np.asarray(payload["frame_numbers"], dtype=np.int32)
        images = np.asarray(payload["images"], dtype=np.uint8)
    if not np.array_equal(cached_frames, expected_frames):
        raise RuntimeError("Cached MRI timeline does not match the prediction pack")
    if len(images) != len(expected_frames):
        raise RuntimeError("Cached MRI image count does not match the prediction pack")
    return {int(frame): images[index] for index, frame in enumerate(cached_frames)}


def load_rms_vtln_pack(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "path": path,
            "frames": np.asarray(payload["frame_numbers"], dtype=np.int32),
            "affine_tps": np.asarray(
                payload["predicted_after_corrected_affine_tps"], dtype=np.float32
            ),
            "ground_truth": np.asarray(payload["ground_truth"], dtype=np.float32),
            "alpha": float(payload["vtln_alpha_to_asd2_train"].item()),
            "target_rms": float(payload["target_rms"].item()),
        }


def render(args: argparse.Namespace) -> tuple[Path, Path, list[Path]]:
    result_root = args.result_root.resolve()
    pair = (args.speaker, args.session)
    if pair not in source.EXPECTED_TARGET_FRAMES:
        choices = ", ".join(
            f"P{speaker}/S{session}"
            for speaker, session in source.EXPECTED_TARGET_FRAMES
        )
        raise ValueError(
            f"Unsupported selected pair P{args.speaker}/S{args.session}; "
            f"expected one of {choices}"
        )
    target_frame = (
        source.EXPECTED_TARGET_FRAMES[pair]
        if args.target_frame is None
        else args.target_frame
    )
    rms_vtln_path = (
        result_root
        / f"phase_b/demo_inputs/P{args.speaker}/S{args.session}"
        / "rms_vtln_exact_u_fixedbs10.npz"
        if args.rms_vtln_pack is None
        else args.rms_vtln_pack.resolve()
    )
    session_dir = result_root / f"phase_b/videos/P{args.speaker}/S{args.session}"
    output = session_dir / (
        f"p{args.speaker}_s{args.session}_asd2_to_asd1_grid_adaptation_2x3_50fps.mp4"
    )
    audit_path = output.with_name(f"{output.stem}_audit.json")
    source_video = session_dir / (
        f"p{args.speaker}_s{args.session}_fixedbs10_textgrid_u_grid_adaptation_50fps.mp4"
    )
    cache_path = session_dir / ".cache/evaluated_integer_mri_frames.npz"

    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing output without --force: {output}")
    if not source_video.is_file():
        raise FileNotFoundError(f"Missing canonical source video: {source_video}")

    pack = source.base.load_corrected_pack(result_root, "asd2", pair, "baseline")
    rms_vtln_pack = load_rms_vtln_pack(rms_vtln_path)
    frames = pack["frames"]
    if not np.array_equal(rms_vtln_pack["frames"], frames):
        raise RuntimeError("RMS+VTLN timeline does not match the baseline timeline")
    ground_truth_delta = float(
        np.max(np.abs(rms_vtln_pack["ground_truth"] - pack["ground_truth"]))
    )
    if ground_truth_delta > 1e-5:
        raise RuntimeError(
            f"RMS+VTLN ground truth differs from baseline: {ground_truth_delta}"
        )
    if target_frame not in frames:
        raise RuntimeError(f"Target frame {target_frame} is absent from the rendered timeline")
    mri = load_cached_mri(cache_path, frames)

    silent = session_dir / f".{output.stem}.silent.writing.mp4"
    muxing = session_dir / f".{output.stem}.muxing.mp4"
    silent.unlink(missing_ok=True)
    muxing.unlink(missing_ok=True)
    height = 2 * (INFO_HEIGHT + PANEL_SIZE) + SEPARATOR
    width = 3 * PANEL_SIZE + 2 * SEPARATOR
    writer = cv2.VideoWriter(
        str(silent), cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {silent}")

    qc_indices = {
        0,
        len(frames) // 2,
        len(frames) - 1,
        int(np.flatnonzero(frames == target_frame)[0]),
    }
    qc_paths: list[Path] = []
    try:
        for index, frame_value in enumerate(frames):
            frame = int(frame_value)
            canvas = compose_canvas(
                mri[frame], frame, index, target_frame, pack, rms_vtln_pack
            )
            writer.write(canvas)
            if index in qc_indices:
                qc_path = session_dir / f"{output.stem}_qc_F{frame:04d}.png"
                if not cv2.imwrite(str(qc_path), canvas):
                    raise RuntimeError(f"Could not write QC frame: {qc_path}")
                qc_paths.append(qc_path)
            if index == 0 or (index + 1) % 250 == 0 or index + 1 == len(frames):
                print(f"RENDER: {index + 1}/{len(frames)}", flush=True)
    finally:
        writer.release()

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(silent),
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-vsync",
            "0",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(muxing),
        ],
        check=True,
    )
    muxing.replace(output)
    silent.unlink(missing_ok=True)

    original_audio, _ = source.base.asd2_core.exact_asd1_audio_paths(
        args.speaker, args.session
    )
    audit = source.base.audit_video(output, frames, original_audio)
    audit.update(
        {
            "layout": [
                ["Raw", "Affine", "Affine + TPS"],
                [
                    "Ground truth only",
                    "Contour slide + audio normalization",
                    "Stage effect",
                ],
            ],
            "contour_convention": "solid ground truth; dashed prediction",
            "source_prediction_pack": str(pack["path"].resolve()),
            "rms_vtln_prediction_pack": str(rms_vtln_pack["path"].resolve()),
            "rms_vtln_alpha_to_asd2_train": rms_vtln_pack["alpha"],
            "rms_target": rms_vtln_pack["target_rms"],
            "audio_normalization_role": "model input only",
            "playback_audio_role": "original target audio",
            "audio_stream_copied_from": str(source_video.resolve()),
            "visual_qc_samples": [str(path.resolve()) for path in qc_paths],
            "visible_text_excludes": ["New fixed-BS10", "integer"],
        }
    )
    source.atomic_json(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return output, audit_path, qc_paths


def main() -> None:
    args = parse_args()
    render(args)


if __name__ == "__main__":
    main()
