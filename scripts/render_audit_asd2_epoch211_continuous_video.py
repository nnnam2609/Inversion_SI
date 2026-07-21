#!/usr/bin/env python3
"""Render and audit the authoritative continuous-timeline experiment videos.

This CPU-only runner is intentionally separate from analysis/inference.  It
renders every integer MRI frame from the first to the last evaluated timestamp,
draws contours only where an evaluated pack has that timestamp, and attaches
one continuous slice of the original target WAV.  It never holds or
interpolates contours and never renders fractional MRI timestamps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")]

import run_asd2_epoch211_selected_experiment as core  # noqa: E402
from src.utils.colors import COLORS  # noqa: E402
from src.utils.config_validation import load_yaml_config  # noqa: E402
from src.utils.mri_rendering import build_filename_dicom_index, load_or_build_mri_cache  # noqa: E402
from src.utils.video_rendering import (  # noqa: E402
    MM_PER_PIXEL,
    attach_audio,
    draw_dashed_polyline,
    rgb_to_bgr255,
    scale_points,
)


FPS = 50
PANEL_SIZE = 272
INFO_HEIGHT = 88
SEPARATOR = 6
PANELS = (
    ("ground_truth", "ground_truth", "Target ground truth"),
    ("p7_baseline", "affine_tps", "P7 baseline: affine + TPS"),
    ("baseline", "raw", "ASD2 baseline: raw"),
    ("baseline", "affine", "ASD2 baseline: affine"),
    ("baseline", "affine_tps", "ASD2 baseline: affine + TPS"),
    ("rms_vtln", "affine_tps", "ASD2 RMS + VTLN: affine + TPS"),
    ("rms_only", "affine_tps", "ASD2 RMS only: affine + TPS"),
    ("vtln_only", "affine_tps", "ASD2 VTLN only: affine + TPS"),
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("video", "audit", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=core.DEFAULT_OUTPUT)
    parser.add_argument("--selection", nargs="+", default=None, metavar="P#:S#")
    parser.add_argument("--mri-workers", type=int, default=8)
    parser.add_argument("--force-video", action="store_true")
    parser.add_argument("--confirm-visual-qc", action="store_true")
    return parser.parse_args()


def selected_pairs(values: list[str] | None) -> tuple[tuple[int, int], ...]:
    if values is None:
        return core.SELECTION
    pairs = []
    for value in values:
        speaker, session = value.upper().split(":", 1)
        pair = (int(speaker.removeprefix("P")), int(session.removeprefix("S")))
        if pair not in core.SELECTION:
            raise ValueError(f"Pair is outside the fixed protocol: {value}")
        pairs.append(pair)
    if len(pairs) != len(set(pairs)):
        raise ValueError("Duplicate selection")
    return tuple(pairs)


def atomic_json(path: Path, payload: Any) -> None:
    core.atomic_json(path, payload)


def old_baseline_path(pair: tuple[int, int]) -> Path:
    return (
        core.OLD_GRID_ROOT
        / f"P{pair[0]}"
        / f"S{pair[1]}"
        / "contours_and_ground_truth.npz"
    )


def load_old_baseline(pair: tuple[int, int]) -> dict[str, Any]:
    path = old_baseline_path(pair)
    with np.load(path, allow_pickle=False) as payload:
        frames = np.asarray(payload["frame_numbers"])
        result = {
            "frame_numbers": frames,
            "ground_truth": np.asarray(payload["ground_truth"], dtype=np.float32),
            "arrays": {
                "affine_tps": np.asarray(
                    payload["predicted_after_affine_tps"], dtype=np.float32
                )
            },
            "classes": [str(value) for value in payload["classes"].tolist()],
        }
    if not np.issubdtype(frames.dtype, np.integer):
        raise RuntimeError(f"Old baseline has non-integer frames: {path}")
    if result["classes"] != list(core.CLASSES):
        raise RuntimeError(f"Old baseline class order mismatch: {path}")
    return result


def load_packs(output_root: Path, pair: tuple[int, int]) -> dict[str, dict[str, Any]]:
    packs = {
        branch: core.load_pack(core.pack_path(output_root, pair[0], pair[1], branch))
        for branch in core.BRANCHES
    }
    packs["p7_baseline"] = load_old_baseline(pair)
    reference = packs["baseline"]
    for name, pack in packs.items():
        if not np.array_equal(pack["frame_numbers"], reference["frame_numbers"]):
            raise RuntimeError(f"Frame mismatch for P{pair[0]}/S{pair[1]}/{name}")
        if not np.allclose(
            pack["ground_truth"], reference["ground_truth"], atol=1e-5, rtol=0
        ):
            raise RuntimeError(f"Ground-truth mismatch for P{pair[0]}/S{pair[1]}/{name}")
    return packs


def frame_token(frame: int) -> str:
    if not isinstance(frame, (int, np.integer)):
        raise ValueError(f"Refusing fractional render timestamp: {frame!r}")
    return f"{int(frame):04d}"


def draw_panel(
    image: np.ndarray,
    title: str,
    frame: int,
    predicted: np.ndarray | None,
    ground_truth: np.ndarray | None,
    rmse: float | None,
    ground_truth_only: bool = False,
) -> np.ndarray:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_bgr = cv2.resize(image_bgr, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((INFO_HEIGHT + PANEL_SIZE, PANEL_SIZE, 3), 14, dtype=np.uint8)
    canvas[INFO_HEIGHT:] = image_bgr
    if ground_truth is None:
        status = "not scored: no contour hold/interpolation"
        legend = "continuous integer MRI + original audio"
    else:
        scale = PANEL_SIZE / float(image.shape[1])
        if not math.isclose(scale, round(scale), abs_tol=1e-6):
            raise RuntimeError(f"Non-integral render scale: {scale}")
        scale_int = int(round(scale))
        for index, class_name in enumerate(core.CLASSES):
            color = rgb_to_bgr255(COLORS.get(class_name, "white"))
            gt = scale_points(ground_truth[index], scale_int)
            gt[:, 1] += INFO_HEIGHT
            cv2.polylines(canvas, [gt], False, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.polylines(canvas, [gt], False, color, 1, cv2.LINE_AA)
            if predicted is not None:
                pred = scale_points(predicted[index], scale_int)
                pred[:, 1] += INFO_HEIGHT
                draw_dashed_polyline(canvas, pred, (0, 0, 0), 2, 7, 9)
                draw_dashed_polyline(canvas, pred, color, 1, 7, 9)
        if ground_truth_only:
            status = "ground truth only"
            legend = "solid target ground truth"
        else:
            status = f"RMSE all 11: {rmse:.3f} mm"
            legend = "solid GT | dashed prediction"
    for index, line in enumerate(
        (title, f"integer MRI frame {frame_token(frame)}", status, legend)
    ):
        cv2.putText(
            canvas,
            line,
            (7, 17 + 20 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return canvas


def compose_canvas(
    image: np.ndarray,
    frame: int,
    frame_index: dict[int, int],
    packs: dict[str, dict[str, Any]],
) -> np.ndarray:
    scored_index = frame_index.get(frame)
    panels = []
    for branch, stage, title in PANELS:
        if scored_index is None:
            panels.append(draw_panel(image, title, frame, None, None, None))
            continue
        ground_truth = packs["baseline"]["ground_truth"][scored_index]
        if branch == "ground_truth":
            panels.append(
                draw_panel(
                    image,
                    title,
                    frame,
                    None,
                    ground_truth,
                    None,
                    ground_truth_only=True,
                )
            )
            continue
        predicted = packs[branch]["arrays"][stage][scored_index]
        difference = predicted.astype(np.float64) - ground_truth.astype(np.float64)
        rmse = float(np.sqrt(np.mean(difference * difference)) * MM_PER_PIXEL)
        panels.append(draw_panel(image, title, frame, predicted, ground_truth, rmse))
    panel_height = INFO_HEIGHT + PANEL_SIZE
    height = 2 * panel_height + SEPARATOR
    width = 4 * PANEL_SIZE + 3 * SEPARATOR
    canvas = np.full((height, width, 3), 8, dtype=np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, 4)
        y = row * (panel_height + SEPARATOR)
        x = column * (PANEL_SIZE + SEPARATOR)
        canvas[y : y + panel_height, x : x + PANEL_SIZE] = panel
    return canvas


def ffprobe(path: Path) -> dict[str, Any]:
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )


def is_50(value: str) -> bool:
    return Fraction(value) == Fraction(FPS, 1)


def audio_timing(frame_min: int, continuous_count: int) -> dict[str, float]:
    config = load_yaml_config(core.DEFAULT_CONFIG)
    ms_image = float(config["ms_image"])
    skip_ms = float(config.get("skip_ms", float(config["added_frames"]) * ms_image))
    start = (skip_ms + frame_min * ms_image) / 1000.0
    duration = continuous_count / FPS
    first_center_source = (skip_ms + (frame_min + 0.5) * ms_image) / 1000.0
    last_frame = frame_min + continuous_count - 1
    last_center_source = (skip_ms + (last_frame + 0.5) * ms_image) / 1000.0
    first_center_video = start + 0.5 / FPS
    last_center_video = start + (continuous_count - 0.5) / FPS
    return {
        "source_audio_start_seconds": start,
        "requested_duration_seconds": duration,
        "skip_ms": skip_ms,
        "source_ms_image": ms_image,
        "first_frame_center_source_seconds": first_center_source,
        "first_frame_center_video_seconds": first_center_video,
        "last_frame_center_source_seconds": last_center_source,
        "last_frame_center_video_seconds": last_center_video,
        "maximum_endpoint_clock_difference_seconds": max(
            abs(first_center_source - first_center_video),
            abs(last_center_source - last_center_video),
        ),
    }


def audit_video(
    video_path: Path,
    original_audio: Path,
    frame_min: int,
    frame_max: int,
) -> dict[str, Any]:
    payload = ffprobe(video_path)
    videos = [item for item in payload["streams"] if item.get("codec_type") == "video"]
    audios = [item for item in payload["streams"] if item.get("codec_type") == "audio"]
    if len(videos) != 1 or not audios:
        raise RuntimeError(f"Missing video/audio stream: {video_path}")
    video, audio = videos[0], audios[0]
    expected = frame_max - frame_min + 1
    count = int(video.get("nb_read_frames") or video.get("nb_frames") or -1)
    video_duration = float(video.get("duration") or payload["format"]["duration"])
    audio_duration = float(audio.get("duration") or payload["format"]["duration"])
    checks = {
        "r_frame_rate_50": is_50(video["r_frame_rate"]),
        "avg_frame_rate_50": is_50(video["avg_frame_rate"]),
        "video_codec_h264": video.get("codec_name") == "h264",
        "audio_codec_aac": audio.get("codec_name") == "aac",
        "audio_stream_present": True,
        "exact_continuous_integer_frame_count": count == expected,
        "video_duration_within_one_frame": abs(video_duration - expected / FPS) <= 1 / FPS,
        "av_duration_difference_within_one_frame": abs(video_duration - audio_duration) <= 1 / FPS,
        "source_audio_is_original_target_wav": original_audio.name.startswith("DENOISED_SOUND_"),
        "rendered_fractional_frame_count_zero": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Continuous video audit failed {video_path}: {checks}")
    return {
        "created_at": now(),
        "status": "passed",
        "protocol": "continuous_integer_mri_timeline_with_continuous_original_target_wav",
        "video": str(video_path.resolve()),
        "original_audio": str(original_audio.resolve()),
        "frame_min": frame_min,
        "frame_max": frame_max,
        "expected_continuous_integer_frames": expected,
        "video_frames": count,
        "video_r_frame_rate": video["r_frame_rate"],
        "video_avg_frame_rate": video["avg_frame_rate"],
        "video_duration_seconds": video_duration,
        "audio_duration_seconds": audio_duration,
        "av_duration_difference_seconds": abs(video_duration - audio_duration),
        "checks": checks,
    }


def render_session(
    output_root: Path,
    pair: tuple[int, int],
    workers: int,
    force: bool,
) -> dict[str, Any]:
    speaker, session = pair
    started = time.monotonic()
    packs = load_packs(output_root, pair)
    frames = packs["baseline"]["frame_numbers"]
    if not np.issubdtype(frames.dtype, np.integer):
        raise RuntimeError(core.INTEGER_FRAME_POLICY)
    frame_min, frame_max = int(frames.min()), int(frames.max())
    continuous = list(range(frame_min, frame_max + 1))
    frame_index = {int(value): index for index, value in enumerate(frames)}
    session_dir = output_root / f"P{speaker}/S{session}"
    video_path = session_dir / (
        f"p{speaker}_s{session}_asd2_grid_audio_ablation_original_audio_50fps.mp4"
    )
    audit_path = session_dir / "video_50fps_original_audio_audit.json"
    audio_path, _ = core.exact_asd1_audio_paths(speaker, session)
    if video_path.is_file() and not force:
        audit = audit_video(video_path, audio_path, frame_min, frame_max)
        atomic_json(audit_path, audit)
        return audit
    dicom_dir = core.RAW_ROOT / f"P{speaker}/DCM_2D/S{session}"
    dicom_index, _ = build_filename_dicom_index(dicom_dir)
    cache_path = session_dir / ".cache" / "continuous_integer_mri_frames.npz"
    cache = load_or_build_mri_cache(
        dicom_dir, dicom_index, continuous, cache_path, workers=workers
    )
    panel_height = INFO_HEIGHT + PANEL_SIZE
    size = (4 * PANEL_SIZE + 3 * SEPARATOR, 2 * panel_height + SEPARATOR)
    silent = session_dir / f".{video_path.stem}.continuous.silent.writing.mp4"
    muxing = session_dir / f".{video_path.stem}.continuous.muxing.mp4"
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {silent}")
    sample_frames = {int(frames[0]), int(frames[len(frames) // 2]), int(frames[-1])}
    qc_dir = output_root / "visual_qc" / f"P{speaker}_S{session}"
    qc_dir.mkdir(parents=True, exist_ok=True)
    sample_paths = []
    try:
        for index, frame in enumerate(continuous):
            canvas = compose_canvas(cache[frame], frame, frame_index, packs)
            writer.write(canvas)
            if frame in sample_frames:
                path = qc_dir / f"frame_{frame_token(frame)}.png"
                if not cv2.imwrite(str(path), canvas):
                    raise RuntimeError(f"Could not write QC sample: {path}")
                sample_paths.append(path)
            if index == 0 or (index + 1) % 500 == 0 or index + 1 == len(continuous):
                print(
                    f"CONTINUOUS RENDER P{speaker}/S{session}: "
                    f"{index + 1}/{len(continuous)}",
                    flush=True,
                )
    finally:
        writer.release()
    timing = audio_timing(frame_min, len(continuous))
    if not attach_audio(
        silent,
        audio_path,
        muxing,
        timing["source_audio_start_seconds"],
        timing["requested_duration_seconds"],
    ):
        raise RuntimeError(f"Could not attach original audio: {audio_path}")
    muxing.replace(video_path)
    silent.unlink(missing_ok=True)
    audit = audit_video(video_path, audio_path, frame_min, frame_max)
    audit.update(
        {
            "scored_integer_frames": len(frames),
            "unscored_continuous_timeline_frames": len(continuous) - len(frames),
            "saved_fractional_frame_count": 0,
            "scored_fractional_frame_count": 0,
            "rendered_fractional_frame_count": 0,
            "timeline_policy": (
                "every integer MRI frame from frame_min through frame_max; contours only at "
                "evaluated timestamps; no contour hold/interpolation; one continuous original-WAV slice"
            ),
            "audio_timing": timing,
            "visual_qc_samples": [str(path.resolve()) for path in sorted(sample_paths)],
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    atomic_json(audit_path, audit)
    print(
        f"CONTINUOUS VIDEO DONE P{speaker}/S{session}: {len(continuous)} frames, "
        f"A/V delta={audit['av_duration_difference_seconds']:.6f}s",
        flush=True,
    )
    return audit


def contact_sheet(output_root: Path) -> Path:
    paths = sorted((output_root / "visual_qc").glob("P*_S*/frame_*.png"))
    if len(paths) != 27:
        raise RuntimeError(f"Expected 27 QC samples, got {len(paths)}")
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((560, 370), Image.Resampling.LANCZOS)
        thumbs.append((path, image.copy()))
    columns, cell_width, cell_height = 3, 570, 405
    sheet = Image.new(
        "RGB",
        (columns * cell_width, math.ceil(len(thumbs) / columns) * cell_height),
        (22, 22, 22),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (path, image) in enumerate(thumbs):
        row, column = divmod(index, columns)
        x, y = column * cell_width, row * cell_height
        draw.text((x + 5, y + 5), f"{path.parent.name}/{path.stem}", fill=(245, 245, 245))
        sheet.paste(image, (x, y + 25))
    path = output_root / "visual_qc" / "contact_sheet_27_samples_continuous.png"
    sheet.save(path)
    return path


def run_videos(
    output_root: Path,
    pairs: tuple[tuple[int, int], ...],
    workers: int,
    force: bool,
) -> None:
    for pair in pairs:
        render_session(output_root, pair, workers, force)
    if pairs == core.SELECTION:
        print(f"CONTINUOUS CONTACT SHEET: {contact_sheet(output_root)}", flush=True)


def tree_immutability(output_root: Path) -> dict[str, Any]:
    before = json.loads(
        (output_root / "provenance/input_bundles_before.json").read_text(encoding="utf-8")
    )
    rows = []
    for root in (core.OLD_GRID_ROOT, core.OLD_AUDIO_ROOT, core.OLD_ABLATION_ROOT):
        old = before["roots"][root.name]
        files = core.tree_hashes(root)
        digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
        unchanged = files == old.get("files") and digest == old.get("tree_sha256")
        rows.append(
            {
                "root": str(root.resolve()),
                "file_count": len(files),
                "tree_sha256_before": old.get("tree_sha256"),
                "tree_sha256_after": digest,
                "unchanged": unchanged,
            }
        )
    if not all(row["unchanged"] for row in rows):
        raise RuntimeError(f"Immutable input changed: {rows}")
    return {"status": "passed", "roots": rows}


def authoritative_report(output_root: Path) -> Path:
    source = output_root / "analysis/quantitative_report.md"
    text = source.read_text(encoding="utf-8")
    prefix = text.split("## Metric and audit conventions", 1)[0].rstrip()
    suffix = f"""

## Metric and audit conventions

RMSE is point-to-point contour error in mm (1.62 mm/pixel), computed per frame
and then frame-weighted. Bootstrap resampling uses paired whole-session blocks
with 10,000 replicates and seed {core.BOOTSTRAP_SEED}. All scored, saved, and
rendered MRI timestamps are integer-valued.

The authoritative videos render every integer MRI frame from the first through
the last evaluated timestamp at exactly 50 fps. Contours are drawn only on
evaluated timestamps; missing timestamps contain MRI only, with no contour hold
or interpolation. Audio is one continuous slice of the original target WAV,
starting at `skip_ms + frame_min * ms_image`; normalized audio and concatenated
per-frame snippets are not used for playback. Exact per-session ffprobe and
audio-timing evidence is in each `video_50fps_original_audio_audit.json`.
"""
    path = output_root / "analysis/quantitative_report_continuous_video.md"
    path.write_text(prefix + suffix, encoding="utf-8")
    return path


def run_audit(output_root: Path, confirm_visual_qc: bool) -> dict[str, Any]:
    pack_totals = {branch: 0 for branch in core.BRANCHES}
    videos = []
    total_continuous = 0
    for pair in core.SELECTION:
        packs = load_packs(output_root, pair)
        reference = packs["baseline"]
        for branch in core.BRANCHES:
            pack = packs[branch]
            if not all(np.isfinite(pack["arrays"][stage]).all() for stage in core.STAGES):
                raise RuntimeError(f"Non-finite pack P{pair[0]}/S{pair[1]}/{branch}")
            pack_totals[branch] += len(pack["frame_numbers"])
        frame_min, frame_max = int(reference["frame_numbers"].min()), int(reference["frame_numbers"].max())
        video_path = output_root / f"P{pair[0]}/S{pair[1]}" / (
            f"p{pair[0]}_s{pair[1]}_asd2_grid_audio_ablation_original_audio_50fps.mp4"
        )
        audio, _ = core.exact_asd1_audio_paths(*pair)
        row = audit_video(video_path, audio, frame_min, frame_max)
        session_audit_path = (
            output_root
            / f"P{pair[0]}/S{pair[1]}/video_50fps_original_audio_audit.json"
        )
        if session_audit_path.is_file():
            prior = json.loads(session_audit_path.read_text(encoding="utf-8"))
            for key in (
                "audio_timing",
                "scored_integer_frames",
                "unscored_continuous_timeline_frames",
                "saved_fractional_frame_count",
                "scored_fractional_frame_count",
                "rendered_fractional_frame_count",
                "timeline_policy",
                "visual_qc_samples",
            ):
                if key in prior:
                    row[key] = prior[key]
        total_continuous += row["video_frames"]
        videos.append(row)
        atomic_json(session_audit_path, row)
    if pack_totals != {branch: 8585 for branch in core.BRANCHES}:
        raise RuntimeError(f"Scored pack inventory mismatch: {pack_totals}")
    sheet = output_root / "visual_qc/contact_sheet_27_samples_continuous.png"
    if not sheet.is_file():
        raise RuntimeError(f"Missing continuous contact sheet: {sheet}")
    gap_sample = output_root / "visual_qc/P1_S16/gap_frame_0165.png"
    if not gap_sample.is_file():
        raise RuntimeError(f"Missing unscored-gap visual-QC sample: {gap_sample}")
    report = authoritative_report(output_root)
    immutable = tree_immutability(output_root)
    audit = {
        "created_at": now(),
        "status": "passed",
        "definition_of_done_passed": bool(confirm_visual_qc),
        "inference_only": True,
        "training_launched": False,
        "video_protocol": "continuous_integer_mri_timeline_with_continuous_original_target_wav",
        "selection": [f"P{s}/S{x}" for s, x in core.SELECTION],
        "headline_unseen_selection": [f"P{s}/S{x}" for s, x in core.UNSEEN_SELECTION],
        "same_speaker_control": "P10/S14",
        "integer_scored_frames_per_branch": pack_totals,
        "total_rendered_continuous_integer_frames": total_continuous,
        "saved_fractional_frame_count": 0,
        "scored_fractional_frame_count": 0,
        "rendered_fractional_frame_count": 0,
        "all_videos_exact_50fps": True,
        "all_videos_contain_original_audio": True,
        "all_videos_continuous_integer_timeline": True,
        "visual_qc_status": (
            "passed_manual_inspection_27_scored_samples_plus_one_unscored_gap"
            if confirm_visual_qc
            else "pending_manual_inspection"
        ),
        "visual_qc_contact_sheet": str(sheet.resolve()),
        "visual_qc_unscored_gap_sample": str(gap_sample.resolve()),
        "video_audits": videos,
        "old_20260718_bundles_immutable": immutable,
        "authoritative_report": str(report.resolve()),
        "note_on_noncanonical_artifacts": (
            "Any original_audio_evaluated_frame_segments.{wav,csv} files are retained as "
            "non-authoritative concurrent artifacts and are not used by these videos."
        ),
    }
    atomic_json(output_root / "final_audit.json", audit)
    manifest = {
        "created_at": now(),
        "status": "complete" if confirm_visual_qc else "pending_manual_visual_qc",
        "experiment": "epoch211_asd2_fixed_asd1_grid_audio_continuous_original_audio",
        "result_root": str(output_root.resolve()),
        "report": str(report.resolve()),
        "final_audit": str((output_root / "final_audit.json").resolve()),
        "definition_of_done_passed": bool(confirm_visual_qc),
        "videos": [row["video"] for row in videos],
    }
    atomic_json(output_root / "manifest.json", manifest)
    atomic_json(output_root / "manifest_continuous_original_audio.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return audit


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    pairs = selected_pairs(args.selection)
    if args.phase in ("video", "all"):
        run_videos(output_root, pairs, args.mri_workers, args.force_video)
    if args.phase in ("audit", "all"):
        if pairs != core.SELECTION:
            raise ValueError("Authoritative audit requires all fixed nine sessions")
        run_audit(output_root, args.confirm_visual_qc)


if __name__ == "__main__":
    main()
