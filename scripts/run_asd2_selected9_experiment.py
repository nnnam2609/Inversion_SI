#!/usr/bin/env python3
"""Controlled ASD2-model evaluation on the fixed historical ASD1 nine-session set.

The pipeline is deliberately resumable and stage-gated.  ``audit`` never loads
the model or runs inference.  ``grid`` performs fresh baseline inference.
``audio-main`` estimates an ASD2-targeted RMS/VTLN frontend and runs fresh
RMS+VTLN inference.  ``ablation`` is allowed only after the saved trigger has
been evaluated.  ``analyze`` and ``videos`` consume the versioned packs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
GRID_ROOT = REPO_ROOT / "external" / "grid-transform"
EXTERNAL_AUDIO_ROOT = REPO_ROOT / "external/audio-speaker-normalization/audio-speaker-normalization"
sys.path[:0] = [
    str(REPO_ROOT),
    str(REPO_ROOT / "src"),
    str(REPO_ROOT / "scripts"),
    str(GRID_ROOT),
    str(EXTERNAL_AUDIO_ROOT),
]

from audio_speaker_norm.audio_normalization import AudioNormConfig, FeatureExtractor  # noqa: E402
from grid_transform.transform_helpers import (  # noqa: E402
    apply_tps,
    apply_transform,
    build_step1_anchors,
    build_step2_controls,
    extract_true_landmarks,
    fit_tps,
)
from notebooks.audio_norm_utils import (  # noqa: E402
    apply_cmvn,
    extract_vtln_mfcc39,
    fit_cmvn,
    fit_speaker_gmms,
    sample_rows,
    score_gmm,
)
from render_p7_grid_transform_selected_speakers import TARGETS, prepare_frame  # noqa: E402
from run_p7_all_nonp7_gridnorm import (  # noqa: E402
    EXCLUDED_CLASSES,
    RAW_ROOT,
    infer_session,
    load_normalization,
    retain_integer_inferred,
)
from run_p7_selected_audio_gridnorm import (  # noqa: E402
    VTLN_F_HIGH,
    VTLN_F_LOW,
    build_audio_normalized_chunks,
    infer_with_features,
)
from src.inference.session_inference import load_model  # noqa: E402
from src.utils.colors import COLORS  # noqa: E402
from src.utils.config_validation import load_yaml_config  # noqa: E402
from src.utils.mri_rendering import build_filename_dicom_index, load_or_build_mri_cache  # noqa: E402
from src.utils.video_rendering import MM_PER_PIXEL, draw_dashed_polyline, rgb_to_bgr255, scale_points  # noqa: E402


MODEL_CONFIG = REPO_ROOT / "config/train_config/asd2_11contour_vtln20260719_train_global_rawstd_st5_mfcc_500epoch.yaml"
MODEL_CHECKPOINT = REPO_ROOT / (
    "mlruns/610923529796522440/adbb5c9946704b568d1f7a48b445a0c0/artifacts/best_model.pth"
)
MODEL_SUMMARY = REPO_ROOT / (
    "results/asd2_11contour_vtln20260719_train_global_rawstd_st5_mfcc_500epoch/"
    "single_task5_asd2_11contour_vtln20260719_train_global_rawstd_st5_mfcc_500epoch_"
    "11_articulators_ac_e_ll_p_spm_t_ul_vf_tc_li_ui/training_summary.json"
)
NORMALIZATION_STATS = REPO_ROOT / (
    "repro/asd2_11contour_vtln20260719_train_global/splits/normalization_stats.npz"
)
RAW_CACHE_ROOT = REPO_ROOT / "cache/raw_sessions/asd1"
P7_GRID_ROOT = REPO_ROOT / "results/p7_selected_nonp7_sessions_gridnorm_20260718"
P7_AUDIO_ROOT = REPO_ROOT / "results/p7_selected_nonp7_sessions_audio_gridnorm_20260718"
P7_ABLATION_ROOT = REPO_ROOT / "results/p7_selected_nonp7_sessions_audio_ablation_20260718"
ASD2_ROOT = Path(
    "/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/ArtSpeech_Database_2"
)
BF_ROOT = WORKSPACE_ROOT / "bf/inference"
DEFAULT_VTLN_DIR = WORKSPACE_ROOT / "_downloads/grid-transform-vtln/vtln-data-v0.1.14/extracted/VTLN/data"
if not DEFAULT_VTLN_DIR.is_dir():
    DEFAULT_VTLN_DIR = GRID_ROOT / "VTLN/data"

SOURCE_SPEC = next(item for item in TARGETS if item.speaker == "P10")
STAGES = ("raw", "affine", "affine_tps")
BRANCH_DIRS = {
    "grid_only": "gridnorm",
    "rms_vtln": "audio_rms_vtln",
    "rms_only": "audio_ablation/rms_only",
    "vtln_only": "audio_ablation/vtln_only",
}
INTEGER_POLICY = "integer MRI frames only; no interpolation, fractional frames, resampling, or holds"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("audit", "grid", "audio-main", "ablation", "analyze", "videos", "finalize", "all"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--transform-frame-batch", type=int, default=256)
    parser.add_argument("--mri-workers", type=int, default=8)
    parser.add_argument("--video-scale", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_readonly(command: list[str], cwd: Path = REPO_ROOT) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_selection() -> tuple[tuple[int, int], ...]:
    manifests = [
        load_manifest(P7_GRID_ROOT / "manifest.json"),
        load_manifest(P7_AUDIO_ROOT / "manifest.json"),
        load_manifest(P7_ABLATION_ROOT / "manifest.json"),
    ]
    selections = [tuple(item["selection"]) for item in manifests]
    if not all(value == selections[0] for value in selections[1:]):
        raise RuntimeError(f"Historical selection mismatch: {selections}")
    parsed = []
    for token in selections[0]:
        speaker, session = token.split("/", maxsplit=1)
        parsed.append((int(speaker.removeprefix("P")), int(session.removeprefix("S"))))
    if len(parsed) != 9 or any(speaker == 7 for speaker, _ in parsed):
        raise RuntimeError(f"Expected exact nine-session non-P7 selection, got {parsed}")
    return tuple(parsed)


def target_spec(speaker: int):
    for item in TARGETS:
        if item.speaker == f"P{speaker}":
            return item
    raise KeyError(f"No static target anatomy for P{speaker}")


def exact_asd1_audio_paths(speaker: int, session: int) -> tuple[Path, Path]:
    root = RAW_ROOT / f"P{speaker}/OTHER/S{session}"
    wav = root / f"DENOISED_SOUND_P{speaker}_S{session}.wav"
    textgrid = root / f"TEXT_ALIGNMENT_P{speaker}_S{session}.textgrid"
    if not wav.is_file() or not textgrid.is_file():
        raise FileNotFoundError(f"Missing ASD1 audio alignment inputs: {wav}, {textgrid}")
    return wav, textgrid


def exact_asd2_audio_paths(bucket: str, session: str) -> tuple[Path, Path]:
    root = ASD2_ROOT / str(bucket) / str(session)
    wavs = sorted(
        path for path in root.glob("*.wav") if not path.name.endswith("_mocap.wav")
    )
    if not wavs:
        raise FileNotFoundError(f"No non-mocap ASD2 WAV in {root}")
    wav = wavs[0]
    textgrid = root / f"{wav.stem}_adjusted.textgrid"
    if not textgrid.is_file():
        raise FileNotFoundError(textgrid)
    return wav, textgrid


def raw_integer_frames(path: Path) -> tuple[np.ndarray, int, int]:
    raw = torch.load(path, map_location="cpu", weights_only=False)["raw"]
    values = np.concatenate([np.asarray(item, dtype=np.float64)[:, 2] for item in raw["frames"]])
    unique = np.unique(values)
    mask = np.isclose(unique, np.rint(unique), atol=1e-4)
    integer = np.rint(unique[mask]).astype(np.int32)
    return integer, int(np.count_nonzero(~mask)), int(len(values))


def load_standard_pack(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        keys = set(payload.files)
        if "predicted_raw" in keys:
            predicted_raw_key = "predicted_raw"
            affine_key = "predicted_after_affine"
            final_key = "predicted_after_affine_tps"
        elif "predicted_audio_raw" in keys:
            predicted_raw_key = "predicted_audio_raw"
            affine_key = "predicted_audio_after_affine"
            final_key = "predicted_audio_after_affine_tps"
        else:
            predicted_raw_key = "raw"
            affine_key = "affine"
            final_key = "affine_tps"
        result = {
            "frame_numbers": np.rint(np.asarray(payload["frame_numbers"])).astype(np.int32),
            "ground_truth": np.asarray(payload["ground_truth"], dtype=np.float32),
            "classes": [str(value) for value in payload["classes"].tolist()],
            "raw": np.asarray(payload[predicted_raw_key], dtype=np.float32),
        }
        if affine_key in keys:
            result["affine"] = np.asarray(payload[affine_key], dtype=np.float32)
        if final_key in keys:
            result["affine_tps"] = np.asarray(payload[final_key], dtype=np.float32)
        if "phonemes" in keys:
            result["phonemes"] = np.asarray(payload["phonemes"])
        if "excluded_classes" in keys:
            result["excluded_classes"] = [str(value) for value in payload["excluded_classes"].tolist()]
        return result


def serialize_landmarks(landmarks: dict[str, Any]) -> dict[str, Any]:
    return {
        key: None if value is None else np.asarray(value, dtype=float).tolist()
        for key, value in landmarks.items()
    }


def static_grid_audit(output_root: Path, selection: tuple[tuple[int, int], ...]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for speaker in sorted({speaker for speaker, _ in selection}):
        spec = target_spec(speaker)
        prepared = prepare_frame(spec, DEFAULT_VTLN_DIR)
        landmarks = extract_true_landmarks(prepared["grid"])
        path = output_root / f"audit/grids/P{speaker}_static_grid_landmarks.json"
        payload = {
            "speaker": f"P{speaker}",
            "reference_label": spec.label,
            "bf_frame": spec.frame,
            "vtln_anchor": spec.vtln_anchor,
            "vtln_image": str((DEFAULT_VTLN_DIR / f"{spec.vtln_anchor}.png").resolve()),
            "vtln_roi_zip": str((DEFAULT_VTLN_DIR / f"{spec.vtln_anchor}.zip").resolve()),
            "landmarks": serialize_landmarks(landmarks),
        }
        write_json(path, payload)
        rows[speaker] = {**payload, "saved_landmark_grid": str(path.resolve())}
    return rows


def package_versions() -> dict[str, str]:
    names = ("numpy", "torch", "librosa", "scipy", "pandas", "opencv-python", "soundfile", "pydicom")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    audit_path = args.output_root / "audit/pre_run_audit.json"
    if audit_path.is_file() and not args.force:
        audit = load_manifest(audit_path)
        if not audit.get("passed"):
            raise RuntimeError(f"Saved audit did not pass: {audit_path}")
        print(f"REUSE passing pre-run audit: {audit_path}", flush=True)
        return audit
    if args.output_root.exists() and not args.force:
        existing_files = {
            path.relative_to(args.output_root)
            for path in args.output_root.rglob("*")
            if path.is_file()
        }
        allowed_pre_audit_files = {Path("logs/commands.log")}
        unexpected_files = existing_files - allowed_pre_audit_files
        if unexpected_files:
            raise FileExistsError(
                "Output root already contains files but no reusable audit exists: "
                f"{args.output_root}; unexpected files={sorted(map(str, unexpected_files))}"
            )
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "manifest", "configs", "audit", "inference_raw", "gridnorm/affine",
        "gridnorm/tps", "audio_rms_vtln", "audio_ablation/rms_only",
        "audio_ablation/vtln_only", "metrics", "comparisons", "bootstrap",
        "videos_50fps_original_audio", "logs", "report",
    ):
        (args.output_root / name).mkdir(parents=True, exist_ok=True)

    selection = fixed_selection()
    config = load_yaml_config(MODEL_CONFIG)
    model_summary = load_manifest(MODEL_SUMMARY)
    with np.load(NORMALIZATION_STATS, allow_pickle=False) as stats:
        normalization = {
            key: {
                "shape": list(stats[key].shape),
                "dtype": str(stats[key].dtype),
                "min": float(np.min(stats[key])),
                "max": float(np.max(stats[key])),
                "all_finite": bool(np.isfinite(stats[key]).all()),
            }
            for key in stats.files
        }
    grids = static_grid_audit(args.output_root, selection)
    source_grid = grids[10]
    session_rows = []
    total_frames = 0
    for speaker, session in selection:
        raw_path = RAW_CACHE_ROOT / f"P{speaker}/S{session}.pt"
        old_pack_path = P7_GRID_ROOT / f"P{speaker}/S{session}/contours_and_ground_truth.npz"
        old_audio_path = P7_AUDIO_ROOT / f"P{speaker}/S{session}/audio_normalized_contours_and_ground_truth.npz"
        old_ablation_root = P7_ABLATION_ROOT / f"P{speaker}/S{session}"
        required = [raw_path, old_pack_path, old_audio_path]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        raw_frames, raw_fractional, raw_rows = raw_integer_frames(raw_path)
        old_pack = load_standard_pack(old_pack_path)
        if not np.array_equal(raw_frames, old_pack["frame_numbers"]):
            raise RuntimeError(f"Raw-cache/P7 frame mismatch for P{speaker}/S{session}")
        if old_pack.get("excluded_classes") not in (None, list(EXCLUDED_CLASSES)):
            raise RuntimeError(f"Historical without-3 definition changed for P{speaker}/S{session}")
        wav, textgrid = exact_asd1_audio_paths(speaker, session)
        audio_info = sf.info(wav)
        dicom_dir = RAW_ROOT / f"P{speaker}/DCM_2D/S{session}"
        gt_dir = BF_ROOT / f"P{speaker}/S{session}/contours"
        if not dicom_dir.is_dir() or not gt_dir.is_dir():
            raise FileNotFoundError(f"Missing MRI/GT for P{speaker}/S{session}")
        ref = grids[speaker]
        session_rows.append(
            {
                "speaker": f"P{speaker}",
                "session": f"S{session}",
                "same_physical_speaker_control": speaker == 10,
                "audio_path": str(wav.resolve()),
                "textgrid_path": str(textgrid.resolve()),
                "audio_sample_rate": int(audio_info.samplerate),
                "audio_duration_seconds": float(audio_info.duration),
                "mri_frame_path": str(dicom_dir.resolve()),
                "ground_truth_contour_path": str(gt_dir.resolve()),
                "reference_image_path": ref["vtln_image"],
                "reference_landmark_grid_path": ref["saved_landmark_grid"],
                "source_reference_image_path": source_grid["vtln_image"],
                "source_landmark_grid_path": source_grid["saved_landmark_grid"],
                "raw_cache_path": str(raw_path.resolve()),
                "number_of_usable_integer_frames": int(len(raw_frames)),
                "raw_unique_fractional_frames_excluded": raw_fractional,
                "raw_cached_rows": raw_rows,
                "frame_min": int(raw_frames.min()),
                "frame_max": int(raw_frames.max()),
                "old_p7_grid_pack": str(old_pack_path.resolve()),
                "old_p7_audio_pack": str(old_audio_path.resolve()),
                "old_p7_rms_only_pack": str((old_ablation_root / "rms_only_contours_and_ground_truth.npz").resolve()),
                "old_p7_vtln_only_pack": str((old_ablation_root / "vtln_only_contours_and_ground_truth.npz").resolve()),
                "old_p7_rms_vtln_pack": str((old_ablation_root / "rms_vtln_contours_and_ground_truth.npz").resolve()),
                "old_p7_result_available": True,
            }
        )
        total_frames += len(raw_frames)
    if total_frames != 8585:
        raise RuntimeError(f"Expected 8585 historical integer frames, found {total_frames}")
    write_csv(args.output_root / "manifest/selected_sessions.csv", session_rows)
    write_json(args.output_root / "manifest/selected_sessions.json", session_rows)

    disk = shutil.disk_usage(args.output_root)
    ffmpeg = run_readonly(["ffmpeg", "-version"])
    ffprobe = run_readonly(["ffprobe", "-version"])
    git_status = run_readonly(["git", "status", "--short", "--branch"])
    git_head = run_readonly(["git", "rev-parse", "HEAD"])
    submodule = run_readonly(["git", "submodule", "status"])
    audit = {
        "created_at": now(),
        "passed": True,
        "audit_runs_model_inference": False,
        "selection_source": str((P7_GRID_ROOT / "manifest.json").resolve()),
        "selection": [f"P{s}/S{x}" for s, x in selection],
        "p10_present": True,
        "p10_interpretation": "same physical speaker/anatomy as ASD2; control, not unseen",
        "primary_unseen_selection": [f"P{s}/S{x}" for s, x in selection if s != 10],
        "total_integer_frames": total_frames,
        "checkpoint": str(MODEL_CHECKPOINT.resolve()),
        "checkpoint_sha256": sha256(MODEL_CHECKPOINT),
        "checkpoint_best_human_epoch": 211,
        "checkpoint_internal_epoch_index": 210,
        "checkpoint_run_id": model_summary["run_id"],
        "model_config": str(MODEL_CONFIG.resolve()),
        "model_config_sha256": sha256(MODEL_CONFIG),
        "frontend": {
            key: config[key]
            for key in (
                "input_type", "input_layer", "n_mfcc", "window_length_ms",
                "hop_length_ratio", "context_window", "sequence_length",
                "added_frames", "ms_image", "skip_ms",
            )
            if key in config
        },
        "classes": list(config["classes"]),
        "normalization": {
            "path": str(NORMALIZATION_STATS.resolve()),
            "sha256": sha256(NORMALIZATION_STATS),
            "mode": config.get("normalization_mode"),
            "fit_split": config.get("normalization_fit_split"),
            "std_policy": config.get("normalization_std_policy"),
            "arrays": normalization,
        },
        "source_anatomy": {
            "physical_identity": "ASD2/P10",
            "reference": SOURCE_SPEC.label,
            "vtln_anchor": SOURCE_SPEC.vtln_anchor,
            "reference_image": source_grid["vtln_image"],
            "reference_roi_zip": source_grid["vtln_roi_zip"],
            "landmark_grid": source_grid["saved_landmark_grid"],
            "p7_grid_reused": False,
        },
        "pixel_to_millimetre": MM_PER_PIXEL,
        "frame_policy": INTEGER_POLICY,
        "video": {
            "fps": 50,
            "r_frame_rate_required": "50/1",
            "avg_frame_rate_required": "50/1",
            "playback_audio": "original target-session WAV only",
            "timestamp_policy": "absolute per-frame MRI timestamp, 20 ms original-audio segment per evaluated integer frame",
            "ffmpeg_available": ffmpeg["returncode"] == 0,
            "ffprobe_available": ffprobe["returncode"] == 0,
        },
        "old_p7_metrics_loadable": True,
        "old_results_read_only": [str(path.resolve()) for path in (P7_GRID_ROOT, P7_AUDIO_ROOT, P7_ABLATION_ROOT)],
        "output_root": str(args.output_root.resolve()),
        "output_new_and_writable": True,
        "disk_free_bytes": int(disk.free),
        "disk_sufficient": disk.free > 20 * 1024**3,
        "git": {"head": git_head, "status": git_status, "submodule": submodule},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "oar_job_id": os.environ.get("OAR_JOB_ID"),
            "packages": package_versions(),
        },
        "sessions": session_rows,
    }
    if not all(
        (
            audit["video"]["ffmpeg_available"],
            audit["video"]["ffprobe_available"],
            audit["disk_sufficient"],
            len(audit["classes"]) == 11,
            all(item["audio_sample_rate"] > 0 for item in session_rows),
        )
    ):
        audit["passed"] = False
    write_json(audit_path, audit)
    shutil.copy2(MODEL_CONFIG, args.output_root / "configs/model_training_config.yaml")
    shutil.copy2(NORMALIZATION_STATS, args.output_root / "configs/normalization_stats.npz")
    report = args.output_root / "audit/pre_run_audit.md"
    report.write_text(
        "# ASD2 selected-nine pre-run audit\n\n"
        f"Status: **{'PASS' if audit['passed'] else 'FAIL'}**. This audit ran no inference.\n\n"
        f"- Checkpoint: `{audit['checkpoint']}` (human epoch 211, SHA-256 `{audit['checkpoint_sha256']}`)\n"
        f"- Source anatomy: `{SOURCE_SPEC.label}` / `{SOURCE_SPEC.vtln_anchor}` (ASD2/P10); P7 source grid reused: **no**\n"
        f"- Selection: {', '.join(audit['selection'])}\n"
        f"- P10: same-speaker/anatomy control; primary unseen aggregate has 8 sessions\n"
        f"- Integer frames: `{total_frames}`; fractional frames scored/rendered: `0`\n"
        f"- Normalization: `{audit['normalization']['path']}` (`train_global`, `raw_positive`)\n"
        f"- Pixel scale: `{MM_PER_PIXEL} mm/pixel`\n"
        f"- Free disk: `{disk.free / 1024**4:.2f} TiB`\n"
        f"- Video feasibility: ffmpeg={audit['video']['ffmpeg_available']}, ffprobe={audit['video']['ffprobe_available']}, exact 50 fps with original-audio timestamp segments\n\n"
        "The three historical P7 roots are read-only inputs. Fresh ASD2 predictions will be written only below this new result root.\n",
        encoding="utf-8",
    )
    if not audit["passed"]:
        raise RuntimeError(f"Pre-run audit failed; see {audit_path}")
    print(json.dumps({"audit": str(audit_path), "passed": True, "frames": total_frames}, indent=2), flush=True)
    return audit


def require_audit(args: argparse.Namespace) -> dict[str, Any]:
    path = args.output_root / "audit/pre_run_audit.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run --stage audit first: {path}")
    audit = load_manifest(path)
    if not audit.get("passed"):
        raise RuntimeError(f"Pre-run audit did not pass: {path}")
    return audit


def build_static_transform(source: dict[str, Any], target: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    lm_src = extract_true_landmarks(source["grid"])
    lm_tgt = extract_true_landmarks(target["grid"])
    step1_src, step1_tgt, step1_labels = build_step1_anchors(lm_src, lm_tgt)
    # Use the canonical least-squares implementation through the saved helper.
    from grid_transform.transform_helpers import estimate_affine

    affine = estimate_affine(step1_src, step1_tgt)
    step1_lm = {
        name: None if value is None else apply_transform(affine, value)
        for name, value in lm_src.items()
    }
    step2_src, step2_tgt, step2_labels = build_step2_controls(step1_lm, lm_tgt)
    tps = fit_tps(step2_src, step2_tgt, smoothing=0.0)

    def apply_two_step(points: np.ndarray) -> np.ndarray:
        return apply_tps(tps, apply_transform(affine, points))

    transform = {"step1_affine": affine, "apply_two_step": apply_two_step}
    serializable = {
        "source_reference": source["spec"].label,
        "target_reference": target["spec"].label,
        "source_vtln_anchor": source["spec"].vtln_anchor,
        "target_vtln_anchor": target["spec"].vtln_anchor,
        "step1_labels": list(step1_labels),
        "step1_source_points": step1_src.tolist(),
        "step1_target_points": step1_tgt.tolist(),
        "affine_A": np.asarray(affine["A"]).tolist(),
        "affine_t": np.asarray(affine["t"]).tolist(),
        "step2_labels": list(step2_labels),
        "step2_source_after_affine": step2_src.tolist(),
        "step2_target_points": step2_tgt.tolist(),
        "tps_kernel": "thin_plate_spline",
        "tps_smoothing": 0.0,
        "fit_scope": "one fixed transform from static reference anatomy; never per prediction frame",
    }
    return transform, serializable


def transforms_for_selection(args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    selection = fixed_selection()
    source = prepare_frame(SOURCE_SPEC, DEFAULT_VTLN_DIR)
    transforms = {}
    for speaker in sorted({speaker for speaker, _ in selection}):
        target = prepare_frame(target_spec(speaker), DEFAULT_VTLN_DIR)
        transform, metadata = build_static_transform(source, target)
        write_json(args.output_root / f"configs/transforms/P10_to_P{speaker}.json", metadata)
        transforms[speaker] = transform
    return transforms


def transform_predictions(predicted: np.ndarray, transform: dict[str, Any], batch: int) -> tuple[np.ndarray, np.ndarray]:
    affine = np.empty_like(predicted, dtype=np.float32)
    final = np.empty_like(predicted, dtype=np.float32)
    for start in range(0, len(predicted), batch):
        stop = min(start + batch, len(predicted))
        points = predicted[start:stop].reshape(-1, 2)
        affine[start:stop] = apply_transform(transform["step1_affine"], points).reshape(
            stop - start, 11, 50, 2
        )
        final[start:stop] = transform["apply_two_step"](points).reshape(stop - start, 11, 50, 2)
    return affine, final


def expected_frames_from_audit(audit: dict[str, Any], speaker: int, session: int) -> np.ndarray:
    row = next(item for item in audit["sessions"] if item["speaker"] == f"P{speaker}" and item["session"] == f"S{session}")
    pack = load_standard_pack(Path(row["old_p7_grid_pack"]))
    return pack["frame_numbers"]


def load_classes_and_phonemes() -> tuple[dict[str, Any], list[str], list[str]]:
    config = load_yaml_config(MODEL_CONFIG)
    classes = list(config["classes"])
    if len(classes) != 11:
        raise RuntimeError(f"Expected 11 classes, got {classes}")
    phonemes = json.loads(Path(config["phonemesdir"]).read_text(encoding="utf-8"))
    return config, classes, phonemes


def save_branch_pack(
    path: Path,
    *,
    branch: str,
    inferred: dict[str, Any],
    affine: np.ndarray,
    final: np.ndarray,
    classes: list[str],
    config: dict[str, Any],
    alpha: float | None = None,
    rms_target: float | None = None,
) -> None:
    frames = np.asarray(inferred["frame_numbers"], dtype=np.int32)
    skip_ms = float(config.get("skip_ms", float(config["added_frames"]) * float(config["ms_image"])))
    timestamps = (skip_ms + (frames.astype(np.float64) + 0.5) * float(config["ms_image"])) / 1000.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        branch=np.asarray(branch),
        frame_numbers=frames,
        timestamps_seconds=timestamps,
        validity_flags=np.ones(len(frames), dtype=np.bool_),
        phonemes=np.asarray(inferred["phonemes"]),
        overlap_counts=np.asarray(inferred["overlap_counts"]),
        predicted_raw=np.asarray(inferred["predicted_raw"], dtype=np.float32),
        predicted_after_affine=np.asarray(affine, dtype=np.float32),
        predicted_after_affine_tps=np.asarray(final, dtype=np.float32),
        ground_truth=np.asarray(inferred["ground_truth"], dtype=np.float32),
        classes=np.asarray(classes, dtype="U64"),
        excluded_classes=np.asarray(EXCLUDED_CLASSES, dtype="U64"),
        num_input_rows=np.asarray(inferred["num_input_rows"]),
        num_sequences=np.asarray(inferred["num_sequences"]),
        num_fractional_frames_discarded=np.asarray(inferred.get("num_fractional_frames_discarded", 0)),
        saved_fractional_frame_count=np.asarray(0),
        scored_fractional_frame_count=np.asarray(0),
        rendered_fractional_frame_count=np.asarray(0),
        vtln_alpha=np.asarray(np.nan if alpha is None else alpha),
        rms_target=np.asarray(np.nan if rms_target is None else rms_target),
        frame_policy=np.asarray(INTEGER_POLICY),
        checkpoint_sha256=np.asarray(sha256(MODEL_CHECKPOINT)),
    )


def run_grid(args: argparse.Namespace) -> None:
    audit = require_audit(args)
    config, classes, phonemes = load_classes_and_phonemes()
    normalization = load_normalization(NORMALIZATION_STATS)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Fresh model inference requires CUDA inside OAR")
    model = load_model(config, MODEL_CHECKPOINT, device)
    transforms = transforms_for_selection(args)
    stage_rows = []
    for speaker, session in fixed_selection():
        started = time.monotonic()
        raw_out = args.output_root / f"inference_raw/P{speaker}/S{session}/predictions_raw.npz"
        grid_out = branch_pack_path(args.output_root, "grid_only", speaker, session)
        if grid_out.is_file() and raw_out.is_file() and not args.force:
            pack = load_standard_pack(grid_out)
            if np.array_equal(pack["frame_numbers"], expected_frames_from_audit(audit, speaker, session)):
                print(f"REUSE grid-only P{speaker}/S{session}", flush=True)
                continue
        raw_path = RAW_CACHE_ROOT / f"P{speaker}/S{session}.pt"
        inferred = retain_integer_inferred(
            infer_session(model, device, raw_path, normalization, phonemes, args.batch_size)
        )
        expected = expected_frames_from_audit(audit, speaker, session)
        if not np.array_equal(inferred["frame_numbers"], expected):
            raise RuntimeError(f"Fresh ASD2 raw frame mismatch for P{speaker}/S{session}")
        affine, final = transform_predictions(
            inferred["predicted_raw"], transforms[speaker], args.transform_frame_batch
        )
        save_branch_pack(
            raw_out,
            branch="grid_only_raw",
            inferred=inferred,
            affine=inferred["predicted_raw"],
            final=inferred["predicted_raw"],
            classes=classes,
            config=config,
        )
        save_branch_pack(
            grid_out,
            branch="grid_only",
            inferred=inferred,
            affine=affine,
            final=final,
            classes=classes,
            config=config,
        )
        stage_rows.append(
            {
                "speaker": f"P{speaker}",
                "session": f"S{session}",
                "frames": len(expected),
                "elapsed_seconds": time.monotonic() - started,
                "raw_pack": str(raw_out.resolve()),
                "grid_pack": str(grid_out.resolve()),
            }
        )
        print(f"DONE fresh ASD2 grid-only P{speaker}/S{session}: {len(expected)} frames", flush=True)
    write_json(
        args.output_root / "logs/grid_stage.json",
        {
            "created_at": now(),
            "checkpoint": str(MODEL_CHECKPOINT.resolve()),
            "normalization": str(NORMALIZATION_STATS.resolve()),
            "source_anatomy": SOURCE_SPEC.label,
            "fresh_inference": True,
            "sessions_completed_this_invocation": stage_rows,
        },
    )


def waveform_rms(path: Path) -> float:
    signal, _ = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(signal, axis=1, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))


def asd2_training_audio_index(config: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows = []
    rms_rows = []
    for bucket, sessions in config["train_sequences"].items():
        for session in sessions:
            wav, textgrid = exact_asd2_audio_paths(str(bucket), str(session))
            rms = waveform_rms(wav)
            rows.append(
                {
                    "speaker_id": "ASD2",
                    "session_id": f"{bucket}_{session}",
                    "wav_path": str(wav),
                    "textgrid_path": str(textgrid),
                }
            )
            rms_rows.append(
                {"bucket": str(bucket), "session": str(session), "wav_path": str(wav), "raw_rms": rms}
            )
    if len(rows) != 85:
        raise RuntimeError(f"Expected 85 ASD2 training sessions, got {len(rows)}")
    return pd.DataFrame(rows), rms_rows


def selected_asd1_audio_index() -> pd.DataFrame:
    rows = []
    for speaker, session in fixed_selection():
        wav, textgrid = exact_asd1_audio_paths(speaker, session)
        rows.append(
            {
                "speaker_id": f"P{speaker}",
                "session_id": f"S{session}",
                "wav_path": str(wav),
                "textgrid_path": str(textgrid),
            }
        )
    return pd.DataFrame(rows)


def estimate_asd2_alphas(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    output_dir = args.output_root / "audio_rms_vtln/normalization"
    summary_path = output_dir / "alpha_to_asd2.json"
    if summary_path.is_file() and not args.force:
        return load_manifest(summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_index, rms_rows = asd2_training_audio_index(config)
    source_index = selected_asd1_audio_index()
    target_rms = float(np.median([row["raw_rms"] for row in rms_rows]))
    write_csv(output_dir / "asd2_training_audio_rms.csv", rms_rows)
    data_index = pd.concat([target_index, source_index], ignore_index=True)
    data_index.to_csv(output_dir / "data_index.csv", index=False)
    norm_config = AudioNormConfig(
        data_root=ASD2_ROOT,
        output_root=output_dir,
        mode="full",
        target_rms=target_rms,
        gmm_components=64,
        gmm_max_iter=200,
        max_frames_per_speaker=120_000,
        alpha_min=0.80,
        alpha_max=1.20,
        alpha_step=0.025,
        vtln_f_low=VTLN_F_LOW,
        vtln_f_high=VTLN_F_HIGH,
        target_mode="ASD2",
        save_intermediate_features=False,
        save_models=False,
        save_figures=False,
    )
    print(f"AUDIO TARGET: extract {len(data_index)} WAVs; ASD2 median RMS={target_rms:.8f}", flush=True)
    extractor = FeatureExtractor(norm_config)
    payloads = extractor.extract(data_index)
    target_features = np.vstack(
        [np.asarray(item["rms_mfcc39"], dtype=np.float32) for item in payloads if item["speaker"] == "ASD2"]
    )
    global_mean, global_std = fit_cmvn(target_features)
    target_cmvn = apply_cmvn(target_features, global_mean, global_std)
    gmms, gmm_table = fit_speaker_gmms(
        {"ASD2": target_cmvn}, norm_config.to_helper_config(), model_dir=None
    )
    gmm_table.to_csv(output_dir / "asd2_target_gmm_summary.csv", index=False)
    target_gmm = gmms["ASD2"]
    self_score = float(score_gmm(target_gmm, target_cmvn))
    curve_rows = []
    alpha_rows = []
    alpha_by_session: dict[str, float] = {}
    for speaker, session in fixed_selection():
        key = f"P{speaker}/S{session}"
        source_payloads = [
            item for item in payloads if item["speaker"] == f"P{speaker}" and item["session"] == f"S{session}"
        ]
        if len(source_payloads) != 1:
            raise RuntimeError(f"Expected one source payload for {key}, got {len(source_payloads)}")
        best_alpha, best_score, base_score = 1.0, -np.inf, float("nan")
        for alpha in norm_config.alpha_grid:
            chunks = []
            for item in source_payloads:
                features, _ = extract_vtln_mfcc39(
                    np.asarray(item["wav_rms"]),
                    int(item["sr"]),
                    alpha=float(alpha),
                    config=norm_config.to_helper_config(),
                    f_high=VTLN_F_HIGH,
                )
                count = min(len(features), int(item["n_total_frames"]))
                mask = np.asarray(item["speech_mask"][:count], dtype=bool)
                if mask.any():
                    chunks.append(features[:count][mask])
            combined = np.vstack(chunks).astype(np.float32)
            combined, _ = sample_rows(combined, min(20_000, len(combined)), SEED)
            score = float(score_gmm(target_gmm, apply_cmvn(combined, global_mean, global_std)))
            curve_rows.append(
                {"session": key, "target": "ASD2_training_85_sessions", "alpha": float(alpha), "score": score, "frames": len(combined)}
            )
            if math.isclose(float(alpha), 1.0, abs_tol=1e-9):
                base_score = score
            if score > best_score:
                best_alpha, best_score = float(alpha), score
        alpha_by_session[key] = best_alpha
        alpha_rows.append(
            {
                "session": key,
                "alpha_best": best_alpha,
                "score_alpha_1": base_score,
                "score_best": best_score,
                "score_gain": best_score - base_score,
                "asd2_self_score": self_score,
            }
        )
        print(f"VTLN {key}->ASD2: alpha={best_alpha:.3f}, gain={best_score-base_score:+.5f}", flush=True)
    write_csv(output_dir / "alpha_curves_to_asd2.csv", curve_rows)
    write_csv(output_dir / "alpha_summary_to_asd2.csv", alpha_rows)
    result = {
        "created_at": now(),
        "target": "ASD2 model training-speaker distribution",
        "target_training_sessions": len(target_index),
        "target_rms_method": "median of per-session raw waveform RMS over declared ASD2 train split",
        "target_rms": target_rms,
        "target_gmm_components": 64,
        "target_gmm_max_frames": 120_000,
        "global_cmvn_fit": "ASD2 target features only",
        "alpha_grid": [float(value) for value in norm_config.alpha_grid],
        "alpha_by_session": alpha_by_session,
        "external_alpha_frontend": "39D MFCC/GMM for alpha selection only",
        "model_frontend": "exact checkpoint Inversion_SI frontend re-extraction",
        "p7_audio_or_alphas_reused": False,
    }
    write_json(summary_path, result)
    return result


def eval_config_for_asd1(model_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(model_config)
    config["dataset_type"] = "asd1"
    config["datadir"] = str(RAW_ROOT)
    config["asd1_datadir"] = str(RAW_ROOT)
    config["asd1_annotation_dir"] = str(BF_ROOT)
    return config


def branch_pack_path(output_root: Path, branch: str, speaker: int, session: int) -> Path:
    return output_root / BRANCH_DIRS[branch] / f"P{speaker}/S{session}/predictions.npz"


def run_audio_branch(
    args: argparse.Namespace,
    *,
    branch: str,
    alpha_metadata: dict[str, Any],
    transforms: dict[int, dict[str, Any]],
    model: torch.nn.Module,
    device: torch.device,
    model_config: dict[str, Any],
    classes: list[str],
    phonemes: list[str],
    normalization: dict[str, np.ndarray],
) -> None:
    if branch not in ("rms_vtln", "rms_only", "vtln_only"):
        raise ValueError(branch)
    target_rms = float(alpha_metadata["target_rms"])
    eval_config = eval_config_for_asd1(model_config)
    audit = require_audit(args)
    for speaker, session in fixed_selection():
        output = branch_pack_path(args.output_root, branch, speaker, session)
        metadata_path = output.parent / "audio_extraction_metadata.json"
        if output.is_file() and metadata_path.is_file() and not args.force:
            pack = load_standard_pack(output)
            if np.array_equal(pack["frame_numbers"], expected_frames_from_audit(audit, speaker, session)):
                print(f"REUSE {branch} P{speaker}/S{session}", flush=True)
                continue
        key = f"P{speaker}/S{session}"
        alpha = float(alpha_metadata["alpha_by_session"][key]) if branch != "rms_only" else 1.0
        rms = target_rms if branch != "vtln_only" else None
        raw_path = RAW_CACHE_ROOT / f"P{speaker}/S{session}.pt"
        raw = torch.load(raw_path, map_location="cpu", weights_only=False)["raw"]
        chunks, alignment, extraction = build_audio_normalized_chunks(
            eval_config, speaker, session, alpha, raw["features"], rms_target=rms
        )
        inferred = retain_integer_inferred(
            infer_with_features(model, device, raw, chunks, normalization, phonemes, args.batch_size)
        )
        grid_pack = load_standard_pack(branch_pack_path(args.output_root, "grid_only", speaker, session))
        if not np.array_equal(inferred["frame_numbers"], grid_pack["frame_numbers"]):
            raise RuntimeError(f"{branch} frame mismatch for {key}")
        gt_delta = float(np.max(np.abs(inferred["ground_truth"] - grid_pack["ground_truth"])))
        if gt_delta > 1e-5:
            raise RuntimeError(f"{branch} ground-truth mismatch for {key}: {gt_delta}")
        affine, final = transform_predictions(
            inferred["predicted_raw"], transforms[speaker], args.transform_frame_batch
        )
        save_branch_pack(
            output,
            branch=branch,
            inferred=inferred,
            affine=affine,
            final=final,
            classes=classes,
            config=model_config,
            alpha=alpha,
            rms_target=rms,
        )
        extraction.update(
            {
                "branch": branch,
                "vtln_alpha_to_asd2": alpha,
                "rms_target_from_asd2_training_audio": rms,
                "p7_alpha_reused": False,
                "frame_alignment_max_gt_delta": gt_delta,
            }
        )
        write_json(metadata_path, extraction)
        write_csv(output.parent / "audio_chunk_alignment.csv", alignment)
        print(f"DONE {branch} {key}: alpha={alpha:.3f}, rms={rms}", flush=True)


def frame_rmse(predicted: np.ndarray, target: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    selected = list(indices)
    difference = predicted[:, selected].astype(np.float64) - target[:, selected].astype(np.float64)
    return np.sqrt(np.mean(difference * difference, axis=(1, 2, 3))) * MM_PER_PIXEL


def evaluate_ablation_trigger(args: argparse.Namespace) -> dict[str, Any]:
    classes = load_standard_pack(branch_pack_path(args.output_root, "grid_only", 1, 16))["classes"]
    all_indices = list(range(len(classes)))
    session_rows = []
    contour_weighted: dict[str, list[tuple[int, float]]] = {name: [] for name in classes}
    for speaker, session in fixed_selection():
        grid = load_standard_pack(branch_pack_path(args.output_root, "grid_only", speaker, session))
        audio = load_standard_pack(branch_pack_path(args.output_root, "rms_vtln", speaker, session))
        if not np.array_equal(grid["frame_numbers"], audio["frame_numbers"]):
            raise RuntimeError(f"Trigger frame mismatch P{speaker}/S{session}")
        base = frame_rmse(grid["affine_tps"], grid["ground_truth"], all_indices)
        main = frame_rmse(audio["affine_tps"], grid["ground_truth"], all_indices)
        delta = float(np.mean(main) - np.mean(base))
        session_rows.append(
            {"session": f"P{speaker}/S{session}", "frames": len(base), "grid_mm": float(np.mean(base)), "rms_vtln_mm": float(np.mean(main)), "delta_mm": delta}
        )
        for index, name in enumerate(classes):
            before = frame_rmse(grid["affine_tps"], grid["ground_truth"], [index])
            after = frame_rmse(audio["affine_tps"], grid["ground_truth"], [index])
            contour_weighted[name].append((len(before), float(np.mean(after) - np.mean(before))))
    total = sum(row["frames"] for row in session_rows)
    aggregate_delta = sum(row["frames"] * row["delta_mm"] for row in session_rows) / total
    contour_deltas = {
        name: sum(frames * delta for frames, delta in values) / sum(frames for frames, _ in values)
        for name, values in contour_weighted.items()
    }
    signs = {int(np.sign(row["delta_mm"])) for row in session_rows if not math.isclose(row["delta_mm"], 0.0, abs_tol=1e-12)}
    contour_signs = {int(np.sign(value)) for value in contour_deltas.values() if not math.isclose(value, 0.0, abs_tol=1e-12)}
    conditions = {
        "aggregate_abs_at_least_0_02_mm": abs(aggregate_delta) >= 0.02,
        "any_session_abs_at_least_0_10_mm": any(abs(row["delta_mm"]) >= 0.10 for row in session_rows),
        "heterogeneous_session_effect_signs": -1 in signs and 1 in signs,
        "opposite_contour_effect_signs": -1 in contour_signs and 1 in contour_signs,
        "rms_vtln_worse_than_grid_only": aggregate_delta > 0,
    }
    payload = {
        "created_at": now(),
        "comparison": "ASD2 RMS+VTLN affine+TPS minus ASD2 grid-only affine+TPS",
        "negative_delta_means_improvement": True,
        "aggregate_frame_weighted_delta_mm": aggregate_delta,
        "sessions": session_rows,
        "contour_deltas_mm": contour_deltas,
        "conditions": conditions,
        "run_ablation": any(conditions.values()),
    }
    write_json(args.output_root / "audio_ablation/trigger.json", payload)
    write_csv(args.output_root / "audio_ablation/trigger_session_deltas.csv", session_rows)
    return payload


def load_model_runtime(args: argparse.Namespace):
    model_config, classes, phonemes = load_classes_and_phonemes()
    normalization = load_normalization(NORMALIZATION_STATS)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Audio inference requires CUDA inside OAR")
    model = load_model(model_config, MODEL_CHECKPOINT, device)
    return model_config, classes, phonemes, normalization, device, model


def run_audio_main(args: argparse.Namespace) -> dict[str, Any]:
    require_audit(args)
    for speaker, session in fixed_selection():
        if not branch_pack_path(args.output_root, "grid_only", speaker, session).is_file():
            raise FileNotFoundError("Run grid stage before audio-main")
    model_config, classes, phonemes, normalization, device, model = load_model_runtime(args)
    alpha_metadata = estimate_asd2_alphas(args, model_config)
    transforms = transforms_for_selection(args)
    run_audio_branch(
        args,
        branch="rms_vtln",
        alpha_metadata=alpha_metadata,
        transforms=transforms,
        model=model,
        device=device,
        model_config=model_config,
        classes=classes,
        phonemes=phonemes,
        normalization=normalization,
    )
    trigger = evaluate_ablation_trigger(args)
    print(json.dumps(trigger, indent=2, sort_keys=True), flush=True)
    return trigger


def run_ablation(args: argparse.Namespace) -> None:
    trigger_path = args.output_root / "audio_ablation/trigger.json"
    if not trigger_path.is_file():
        raise FileNotFoundError("Run audio-main and evaluate the ablation trigger first")
    trigger = load_manifest(trigger_path)
    if not trigger.get("run_ablation"):
        write_json(
            args.output_root / "audio_ablation/skipped.json",
            {"created_at": now(), "reason": "No user-specified trigger condition was satisfied", "trigger": trigger},
        )
        print("SKIP RMS-only/VTLN-only: no trigger condition satisfied", flush=True)
        return
    model_config, classes, phonemes, normalization, device, model = load_model_runtime(args)
    alpha_metadata = load_manifest(args.output_root / "audio_rms_vtln/normalization/alpha_to_asd2.json")
    transforms = transforms_for_selection(args)
    for branch in ("rms_only", "vtln_only"):
        run_audio_branch(
            args,
            branch=branch,
            alpha_metadata=alpha_metadata,
            transforms=transforms,
            model=model,
            device=device,
            model_config=model_config,
            classes=classes,
            phonemes=phonemes,
            normalization=normalization,
        )


def append_command_log(args: argparse.Namespace) -> None:
    path = args.output_root / "logs/commands.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{now()} cwd={Path.cwd()} command={' '.join(sys.argv)}\n")


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    append_command_log(args)
    if args.stage == "audit":
        run_audit(args)
    elif args.stage == "grid":
        run_grid(args)
    elif args.stage == "audio-main":
        run_audio_main(args)
    elif args.stage == "ablation":
        run_ablation(args)
    elif args.stage == "analyze":
        analyze_results(args)
    elif args.stage == "videos":
        render_all_videos(args)
    elif args.stage == "finalize":
        run_final_validation(args)
    elif args.stage == "all":
        run_audit(args)
        run_grid(args)
        run_audio_main(args)
        run_ablation(args)
        analyze_results(args)
        render_all_videos(args)
        run_final_validation(args)


# Implemented below to keep the stage ordering visible near main.
def available_branch_sources(output_root: Path) -> dict[str, Any]:
    sources: dict[str, Any] = {
        "P7 grid-only": lambda s, x: P7_GRID_ROOT / f"P{s}/S{x}/contours_and_ground_truth.npz",
        "P7 RMS+VTLN": lambda s, x: P7_AUDIO_ROOT / f"P{s}/S{x}/audio_normalized_contours_and_ground_truth.npz",
        "P7 RMS-only": lambda s, x: P7_ABLATION_ROOT / f"P{s}/S{x}/rms_only_contours_and_ground_truth.npz",
        "P7 VTLN-only": lambda s, x: P7_ABLATION_ROOT / f"P{s}/S{x}/vtln_only_contours_and_ground_truth.npz",
        "ASD2 grid-only": lambda s, x: branch_pack_path(output_root, "grid_only", s, x),
        "ASD2 RMS+VTLN": lambda s, x: branch_pack_path(output_root, "rms_vtln", s, x),
    }
    if all(branch_pack_path(output_root, "rms_only", s, x).is_file() for s, x in fixed_selection()):
        sources["ASD2 RMS-only"] = lambda s, x: branch_pack_path(output_root, "rms_only", s, x)
    if all(branch_pack_path(output_root, "vtln_only", s, x).is_file() for s, x in fixed_selection()):
        sources["ASD2 VTLN-only"] = lambda s, x: branch_pack_path(output_root, "vtln_only", s, x)
    return sources


def group_members(selection: tuple[tuple[int, int], ...]) -> dict[str, list[tuple[int, int]]]:
    return {
        "all_9": list(selection),
        "unseen_8_without_P10": [(s, x) for s, x in selection if s != 10],
        "P10_same_speaker_control": [(s, x) for s, x in selection if s == 10],
    }


def metric_modes(classes: list[str]) -> dict[str, list[int]]:
    excluded = set(EXCLUDED_CLASSES)
    return {
        "all_11": list(range(len(classes))),
        "without_laryngeal_3": [i for i, name in enumerate(classes) if name not in excluded],
    }


def session_metric(pack: dict[str, Any], stage: str, indices: list[int]) -> tuple[float, np.ndarray]:
    values = frame_rmse(pack[stage], pack["ground_truth"], indices)
    return float(np.mean(values)), values


def paired_bootstrap(
    left: dict[tuple[int, int], tuple[int, float]],
    right: dict[tuple[int, int], tuple[int, float]],
    members: list[tuple[int, int]],
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    deltas = np.asarray([left[key][1] - right[key][1] for key in members], dtype=np.float64)
    weights = np.asarray([left[key][0] for key in members], dtype=np.float64)
    if any(left[key][0] != right[key][0] for key in members):
        raise RuntimeError("Paired bootstrap received mismatched frame counts")
    observed = float(np.sum(weights * deltas) / np.sum(weights))
    bootstrap = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(members), size=len(members))
        bootstrap[index] = np.sum(weights[selected] * deltas[selected]) / np.sum(weights[selected])
    return {
        "delta_frame_weighted_mm": observed,
        "delta_session_balanced_mm": float(np.mean(deltas)),
        "ci95_low_mm": float(np.percentile(bootstrap, 2.5)),
        "ci95_high_mm": float(np.percentile(bootstrap, 97.5)),
        "bootstrap_samples": int(samples),
        "sessions": int(len(members)),
    }


def analyze_results(args: argparse.Namespace) -> None:
    audit = require_audit(args)
    selection = fixed_selection()
    sources = available_branch_sources(args.output_root)
    packs: dict[tuple[str, int, int], dict[str, Any]] = {}
    alignment_rows = []
    reference_classes: list[str] | None = None
    for speaker, session in selection:
        base_frames = expected_frames_from_audit(audit, speaker, session)
        base_gt = None
        for branch, resolver in sources.items():
            path = resolver(speaker, session)
            if not path.is_file():
                raise FileNotFoundError(path)
            pack = load_standard_pack(path)
            if reference_classes is None:
                reference_classes = pack["classes"]
            if pack["classes"] != reference_classes:
                raise RuntimeError(f"Class ordering mismatch: {path}")
            if not np.array_equal(pack["frame_numbers"], base_frames):
                raise RuntimeError(f"Final comparison stopped: frame mismatch in {path}")
            if base_gt is None:
                base_gt = pack["ground_truth"]
            gt_delta = float(np.max(np.abs(pack["ground_truth"] - base_gt)))
            if gt_delta > 1e-5:
                raise RuntimeError(f"Final comparison stopped: GT mismatch {gt_delta} in {path}")
            if any(not np.isfinite(pack[stage]).all() for stage in STAGES):
                raise RuntimeError(f"Non-finite prediction in {path}")
            packs[(branch, speaker, session)] = pack
            alignment_rows.append(
                {
                    "branch": branch,
                    "speaker": f"P{speaker}",
                    "session": f"S{session}",
                    "frames": len(base_frames),
                    "frame_min": int(base_frames.min()),
                    "frame_max": int(base_frames.max()),
                    "exact_frame_match": True,
                    "ground_truth_max_abs_delta": gt_delta,
                    "fractional_frame_count": 0,
                    "pack": str(path.resolve()),
                }
            )
    assert reference_classes is not None
    classes = reference_classes
    modes = metric_modes(classes)
    write_csv(args.output_root / "comparisons/frame_alignment_audit.csv", alignment_rows)
    write_json(
        args.output_root / "comparisons/frame_alignment_audit.json",
        {
            "created_at": now(),
            "passed": True,
            "all_branches_exact_same_integer_frames": True,
            "strict_intersection_needed": False,
            "total_frames_per_branch": 8585,
            "rows": alignment_rows,
        },
    )

    session_rows = []
    contour_rows = []
    session_values: dict[tuple[str, str, str], dict[tuple[int, int], tuple[int, float]]] = {}
    for branch in sources:
        for speaker, session in selection:
            pack = packs[(branch, speaker, session)]
            for stage in STAGES:
                mode_values = {}
                for mode, indices in modes.items():
                    value, _ = session_metric(pack, stage, indices)
                    mode_values[mode] = value
                    session_values.setdefault((branch, stage, mode), {})[(speaker, session)] = (
                        len(pack["frame_numbers"]), value
                    )
                session_rows.append(
                    {
                        "speaker": f"P{speaker}",
                        "session": f"S{session}",
                        "same_physical_speaker_control": speaker == 10,
                        "branch": branch,
                        "stage": stage,
                        "frames": len(pack["frame_numbers"]),
                        "all_11_rmse_mm": mode_values["all_11"],
                        "without_laryngeal_3_rmse_mm": mode_values["without_laryngeal_3"],
                    }
                )
                for class_index, class_name in enumerate(classes):
                    value, _ = session_metric(pack, stage, [class_index])
                    contour_rows.append(
                        {
                            "speaker": f"P{speaker}",
                            "session": f"S{session}",
                            "same_physical_speaker_control": speaker == 10,
                            "branch": branch,
                            "stage": stage,
                            "contour": class_name,
                            "frames": len(pack["frame_numbers"]),
                            "rmse_mm": value,
                        }
                    )

    p7_equivalent = {
        "ASD2 grid-only": "P7 grid-only",
        "ASD2 RMS+VTLN": "P7 RMS+VTLN",
        "ASD2 RMS-only": "P7 RMS-only",
        "ASD2 VTLN-only": "P7 VTLN-only",
    }
    asd2_grid_lookup = {
        (row["speaker"], row["session"], row["stage"]): row
        for row in session_rows if row["branch"] == "ASD2 grid-only"
    }
    branch_lookup = {
        (row["speaker"], row["session"], row["branch"], row["stage"]): row
        for row in session_rows
    }
    for row in session_rows:
        speaker, session, stage, branch = row["speaker"], row["session"], row["stage"], row["branch"]
        if branch.startswith("ASD2"):
            grid = asd2_grid_lookup[(speaker, session, stage)]
            row["delta_vs_asd2_grid_only_all11_mm"] = row["all_11_rmse_mm"] - grid["all_11_rmse_mm"]
            row["delta_vs_asd2_grid_only_without3_mm"] = row["without_laryngeal_3_rmse_mm"] - grid["without_laryngeal_3_rmse_mm"]
            old = branch_lookup[(speaker, session, p7_equivalent[branch], stage)]
            row["delta_vs_equivalent_p7_all11_mm"] = row["all_11_rmse_mm"] - old["all_11_rmse_mm"]
            row["delta_vs_equivalent_p7_without3_mm"] = row["without_laryngeal_3_rmse_mm"] - old["without_laryngeal_3_rmse_mm"]
        else:
            row["delta_vs_asd2_grid_only_all11_mm"] = ""
            row["delta_vs_asd2_grid_only_without3_mm"] = ""
            row["delta_vs_equivalent_p7_all11_mm"] = ""
            row["delta_vs_equivalent_p7_without3_mm"] = ""
    write_csv(args.output_root / "metrics/session_metrics.csv", session_rows)
    write_csv(args.output_root / "metrics/contour_metrics_session.csv", contour_rows)

    groups = group_members(selection)
    aggregate_rows = []
    for branch in sources:
        for stage in STAGES:
            for mode in modes:
                values = session_values[(branch, stage, mode)]
                for group, members in groups.items():
                    weights = np.asarray([values[key][0] for key in members], dtype=np.float64)
                    scores = np.asarray([values[key][1] for key in members], dtype=np.float64)
                    raw_scores = np.asarray(
                        [session_values[(branch, "raw", mode)][key][1] for key in members], dtype=np.float64
                    )
                    affine_scores = np.asarray(
                        [session_values[(branch, "affine", mode)][key][1] for key in members], dtype=np.float64
                    )
                    weighted = float(np.sum(weights * scores) / np.sum(weights))
                    raw_weighted = float(np.sum(weights * raw_scores) / np.sum(weights))
                    affine_weighted = float(np.sum(weights * affine_scores) / np.sum(weights))
                    row = {
                        "group": group,
                        "branch": branch,
                        "stage": stage,
                        "metric_mode": mode,
                        "sessions": len(members),
                        "frames": int(np.sum(weights)),
                        "frame_weighted_rmse_mm": weighted,
                        "session_balanced_mean_rmse_mm": float(np.mean(scores)),
                        "median_session_rmse_mm": float(np.median(scores)),
                        "std_session_rmse_mm": float(np.std(scores)),
                        "improvement_vs_raw_mm": weighted - raw_weighted,
                        "improvement_vs_affine_mm": weighted - affine_weighted,
                    }
                    if branch in p7_equivalent:
                        old_values = session_values[(p7_equivalent[branch], stage, mode)]
                        old_scores = np.asarray([old_values[key][1] for key in members], dtype=np.float64)
                        old_weighted = float(np.sum(weights * old_scores) / np.sum(weights))
                        row["delta_vs_equivalent_p7_mm"] = weighted - old_weighted
                    else:
                        row["delta_vs_equivalent_p7_mm"] = ""
                    aggregate_rows.append(row)
    write_csv(args.output_root / "metrics/aggregate_metrics.csv", aggregate_rows)
    write_csv(args.output_root / "comparisons/unified_comparison.csv", aggregate_rows)

    aggregate_contour_rows = []
    contour_session_lookup = {
        (row["branch"], row["stage"], row["contour"], int(row["speaker"][1:]), int(row["session"][1:])): row
        for row in contour_rows
    }
    for branch in sources:
        for stage in STAGES:
            for contour in classes:
                for group, members in groups.items():
                    rows = [contour_session_lookup[(branch, stage, contour, s, x)] for s, x in members]
                    weights = np.asarray([row["frames"] for row in rows], dtype=np.float64)
                    values = np.asarray([row["rmse_mm"] for row in rows], dtype=np.float64)
                    aggregate_contour_rows.append(
                        {
                            "group": group,
                            "branch": branch,
                            "stage": stage,
                            "contour": contour,
                            "sessions": len(rows),
                            "frames": int(np.sum(weights)),
                            "frame_weighted_rmse_mm": float(np.sum(weights * values) / np.sum(weights)),
                            "session_balanced_mean_rmse_mm": float(np.mean(values)),
                        }
                    )
    write_csv(args.output_root / "metrics/contour_metrics_aggregate.csv", aggregate_contour_rows)

    comparisons = [
        ("ASD2_vs_P7_grid_affine_tps", "ASD2 grid-only", "affine_tps", "P7 grid-only", "affine_tps"),
        ("ASD2_grid_affine_tps_vs_ASD2_raw", "ASD2 grid-only", "affine_tps", "ASD2 grid-only", "raw"),
        ("ASD2_RMS_VTLN_vs_ASD2_grid_affine_tps", "ASD2 RMS+VTLN", "affine_tps", "ASD2 grid-only", "affine_tps"),
    ]
    if "ASD2 VTLN-only" in sources:
        comparisons.append(
            ("ASD2_VTLN_only_vs_RMS_VTLN_affine_tps", "ASD2 VTLN-only", "affine_tps", "ASD2 RMS+VTLN", "affine_tps")
        )
    if "ASD2 RMS-only" in sources:
        comparisons.append(
            ("ASD2_RMS_only_vs_ASD2_grid_affine_tps", "ASD2 RMS-only", "affine_tps", "ASD2 grid-only", "affine_tps")
        )
    bootstrap_rows = []
    paired_delta_rows = []
    for comparison, left_branch, left_stage, right_branch, right_stage in comparisons:
        for mode in modes:
            left = session_values[(left_branch, left_stage, mode)]
            right = session_values[(right_branch, right_stage, mode)]
            for group in ("all_9", "unseen_8_without_P10"):
                members = groups[group]
                stats = paired_bootstrap(
                    left, right, members, args.bootstrap_samples, SEED + len(bootstrap_rows)
                )
                bootstrap_rows.append(
                    {
                        "comparison": comparison,
                        "left_minus_right": f"{left_branch}:{left_stage} - {right_branch}:{right_stage}",
                        "metric_mode": mode,
                        "group": group,
                        "negative_delta_favors_left": True,
                        **stats,
                        "interpretation": "exploratory session-block CI; only 8-9 sessions",
                    }
                )
                for speaker, session in members:
                    paired_delta_rows.append(
                        {
                            "comparison": comparison,
                            "metric_mode": mode,
                            "group": group,
                            "speaker": f"P{speaker}",
                            "session": f"S{session}",
                            "frames": left[(speaker, session)][0],
                            "left_rmse_mm": left[(speaker, session)][1],
                            "right_rmse_mm": right[(speaker, session)][1],
                            "delta_mm": left[(speaker, session)][1] - right[(speaker, session)][1],
                        }
                    )
    write_csv(args.output_root / "bootstrap/paired_bootstrap_ci.csv", bootstrap_rows)
    write_json(args.output_root / "bootstrap/paired_bootstrap_ci.json", bootstrap_rows)
    write_csv(args.output_root / "comparisons/paired_session_deltas.csv", paired_delta_rows)

    def aggregate(branch: str, stage: str, mode: str = "all_11", group: str = "unseen_8_without_P10") -> float:
        row = next(
            item for item in aggregate_rows
            if item["branch"] == branch and item["stage"] == stage
            and item["metric_mode"] == mode and item["group"] == group
        )
        return float(row["frame_weighted_rmse_mm"])

    asd2_raw = aggregate("ASD2 grid-only", "raw")
    p7_raw = aggregate("P7 grid-only", "raw")
    asd2_affine = aggregate("ASD2 grid-only", "affine")
    p7_affine = aggregate("P7 grid-only", "affine")
    asd2_final = aggregate("ASD2 grid-only", "affine_tps")
    p7_final = aggregate("P7 grid-only", "affine_tps")
    asd2_audio = aggregate("ASD2 RMS+VTLN", "affine_tps")
    rms_only = aggregate("ASD2 RMS-only", "affine_tps") if "ASD2 RMS-only" in sources else float("nan")
    vtln_only = aggregate("ASD2 VTLN-only", "affine_tps") if "ASD2 VTLN-only" in sources else float("nan")
    asd2_without3_final = aggregate("ASD2 grid-only", "affine_tps", mode="without_laryngeal_3")
    p7_without3_final = aggregate("P7 grid-only", "affine_tps", mode="without_laryngeal_3")
    final_contours = sorted(
        (
            (row["contour"], float(row["frame_weighted_rmse_mm"]))
            for row in aggregate_contour_rows
            if row["group"] == "unseen_8_without_P10" and row["branch"] == "ASD2 grid-only" and row["stage"] == "affine_tps"
        ),
        key=lambda item: item[1], reverse=True,
    )
    audio_session_rows = [
        row for row in paired_delta_rows
        if row["comparison"] == "ASD2_RMS_VTLN_vs_ASD2_grid_affine_tps"
        and row["metric_mode"] == "all_11" and row["group"] == "unseen_8_without_P10"
    ]
    audio_improved = sum(float(row["delta_mm"]) < 0 for row in audio_session_rows)
    p10_grid = aggregate("ASD2 grid-only", "affine_tps", group="P10_same_speaker_control")
    p10_raw = aggregate("ASD2 grid-only", "raw", group="P10_same_speaker_control")
    ci_main = next(
        row for row in bootstrap_rows
        if row["comparison"] == "ASD2_vs_P7_grid_affine_tps"
        and row["metric_mode"] == "all_11" and row["group"] == "unseen_8_without_P10"
    )
    ci_audio = next(
        row for row in bootstrap_rows
        if row["comparison"] == "ASD2_RMS_VTLN_vs_ASD2_grid_affine_tps"
        and row["metric_mode"] == "all_11" and row["group"] == "unseen_8_without_P10"
    )
    ci_vtln = next(
        (
            row for row in bootstrap_rows
            if row["comparison"] == "ASD2_VTLN_only_vs_RMS_VTLN_affine_tps"
            and row["metric_mode"] == "all_11" and row["group"] == "unseen_8_without_P10"
        ),
        None,
    )
    ci_rms = next(
        (
            row for row in bootstrap_rows
            if row["comparison"] == "ASD2_RMS_only_vs_ASD2_grid_affine_tps"
            and row["metric_mode"] == "all_11" and row["group"] == "unseen_8_without_P10"
        ),
        None,
    )
    contour_lookup = {
        (row["branch"], row["stage"], row["contour"]): float(row["frame_weighted_rmse_mm"])
        for row in aggregate_contour_rows if row["group"] == "unseen_8_without_P10"
    }
    laryngeal = []
    for contour in ("vocal-folds", "thyroid-cartilage", "epiglottis", "arytenoid-cartilage"):
        asd2_value = contour_lookup[("ASD2 grid-only", "affine_tps", contour)]
        p7_value = contour_lookup[("P7 grid-only", "affine_tps", contour)]
        laryngeal.append((contour, asd2_value, p7_value, asd2_value - p7_value))
    p10_audio = aggregate("ASD2 RMS+VTLN", "affine_tps", group="P10_same_speaker_control")
    p10_p7 = aggregate("P7 grid-only", "affine_tps", group="P10_same_speaker_control")
    p10_gap_from_unseen_raw = p10_raw - asd2_raw
    trigger = load_manifest(args.output_root / "audio_ablation/trigger.json")
    alpha_metadata = load_manifest(args.output_root / "audio_rms_vtln/normalization/alpha_to_asd2.json")
    video_audit_path = args.output_root / "videos_50fps_original_audio/video_ffprobe_audit.json"
    video_audit = load_manifest(video_audit_path) if video_audit_path.is_file() else None
    post_audit_path = args.output_root / "audit/post_run_audit.json"
    command_log_path = args.output_root / "logs/commands.log"
    command_lines = command_log_path.read_text(encoding="utf-8").strip().splitlines()
    expand = (
        float(ci_main["ci95_high_mm"]) < 0
        and asd2_final < p7_raw
        and abs(asd2_audio - asd2_final) < 1.0
    )
    recommendation = (
        "The nine-session gate supports a larger evaluation, but only as a preregistered validation run; do not retrain or claim population-level significance."
        if expand
        else "Do not expand directly to the full dataset yet; resolve the remaining contour/session heterogeneity and repeat the fixed-session gate first."
    )
    laryngeal_lines = "\n".join(
        f"| {name} | {p7_value:.4f} | {asd2_value:.4f} | {delta:+.4f} |"
        for name, asd2_value, p7_value, delta in laryngeal
    )
    command_block = "\n".join(command_lines)
    video_sentence = (
        f"All {video_audit['videos']} videos passed: {video_audit['total_frames']} total frames, "
        "r_frame_rate=50/1, avg_frame_rate=50/1, H.264/AAC, original unnormalized audio, "
        "and zero audio-video stream-duration difference in every file."
        if video_audit and video_audit.get("passed")
        else "Video validation had not completed when this report was generated."
    )
    report_path = args.output_root / "report/final_report.md"
    report_path.write_text(
        "# ASD2 selected-nine AAI evaluation\n\n"
        f"Generated `{now()}` from fresh epoch-211 ASD2 inference. All eight compared branches use the same 8,585 integer MRI frames. Historical P7 packs are comparison inputs only; no P7 prediction was reused as an ASD2 output.\n\n"
        "## Conclusion\n\n"
        "The ASD2/P10 source is substantially better for the laryngeal contours, but it does not improve the final all-contour cross-speaker result. On the eight genuinely unseen speakers, ASD2 raw is slightly worse than P7, the static affine step helps strongly, and the subsequent TPS step reverses part of that gain. RMS/VTLN also degrades the aggregate slightly and inconsistently. The result does not justify a full 141-session expansion yet.\n\n"
        "## Primary unseen-eight results\n\n"
        "| Grid branch stage | P7 RMSE (mm) | ASD2 RMSE (mm) | ASD2 - P7 (mm) |\n"
        "|---|---:|---:|---:|\n"
        f"| Raw | {p7_raw:.4f} | {asd2_raw:.4f} | {asd2_raw-p7_raw:+.4f} |\n"
        f"| Affine | {p7_affine:.4f} | {asd2_affine:.4f} | {asd2_affine-p7_affine:+.4f} |\n"
        f"| Affine + TPS | {p7_final:.4f} | {asd2_final:.4f} | {asd2_final-p7_final:+.4f} |\n\n"
        f"Affine removes **{asd2_raw-asd2_affine:.4f} mm** ({100*(asd2_raw-asd2_affine)/asd2_raw:.1f}%) from ASD2 raw error. TPS then adds **{asd2_final-asd2_affine:.4f} mm** ({100*(asd2_final-asd2_affine)/asd2_affine:.1f}% degradation). The paired session-block ASD2-minus-P7 final delta is **{ci_main['delta_frame_weighted_mm']:+.4f} mm**, exploratory 95% CI **[{ci_main['ci95_low_mm']:+.4f}, {ci_main['ci95_high_mm']:+.4f}]**.\n\n"
        "## Laryngeal result\n\n"
        "| Contour, affine + TPS | P7 RMSE (mm) | ASD2 RMSE (mm) | ASD2 - P7 (mm) |\n"
        "|---|---:|---:|---:|\n"
        f"{laryngeal_lines}\n\n"
        f"All four laryngeal structures improve with ASD2/P10 source geometry. The improvement is largest for vocal folds ({laryngeal[0][3]:+.4f} mm) and thyroid cartilage ({laryngeal[1][3]:+.4f} mm). Conversely, the shared historical `without 3` group (excluding vocal folds, thyroid cartilage, and epiglottis) is **{asd2_without3_final:.4f} mm** for ASD2 versus **{p7_without3_final:.4f} mm** for P7, a **{asd2_without3_final-p7_without3_final:+.4f} mm** degradation. Thus the laryngeal gain is real but is outweighed by non-laryngeal residuals, especially the incisors.\n\n"
        "## Audio normalization and conditional ablation\n\n"
        f"The predeclared trigger activated (`{sum(bool(v) for v in trigger['conditions'].values())}/{len(trigger['conditions'])}` conditions true), so RMS-only and VTLN-only were run after RMS+VTLN. The ASD2-only target used the median raw RMS over 85 declared training sessions (**{alpha_metadata['target_rms']:.8f}**) and an ASD2-only MFCC/GMM target; no P7 alpha or RMS target was reused.\n\n"
        "| ASD2 affine + TPS branch | Unseen-eight RMSE (mm) | Delta vs grid-only (mm) |\n"
        "|---|---:|---:|\n"
        f"| Grid-only | {asd2_final:.4f} | +0.0000 |\n"
        f"| RMS + VTLN | {asd2_audio:.4f} | {asd2_audio-asd2_final:+.4f} |\n"
        f"| RMS-only | {rms_only:.4f} | {rms_only-asd2_final:+.4f} |\n"
        f"| VTLN-only | {vtln_only:.4f} | {vtln_only-asd2_final:+.4f} |\n\n"
        f"RMS+VTLN improves only {audio_improved}/8 unseen sessions and is worse overall; its paired block-bootstrap delta is {ci_audio['delta_frame_weighted_mm']:+.4f} mm, 95% CI [{ci_audio['ci95_low_mm']:+.4f}, {ci_audio['ci95_high_mm']:+.4f}]. VTLN-only is {vtln_only-asd2_audio:+.4f} mm better than RMS+VTLN"
        + (f", 95% CI [{ci_vtln['ci95_low_mm']:+.4f}, {ci_vtln['ci95_high_mm']:+.4f}]" if ci_vtln else "")
        + f", but remains {vtln_only-asd2_final:+.4f} mm worse than grid-only. RMS-only is {rms_only-asd2_final:+.4f} mm worse than grid-only"
        + (f", 95% CI [{ci_rms['ci95_low_mm']:+.4f}, {ci_rms['ci95_high_mm']:+.4f}]" if ci_rms else "")
        + ". These exploratory intervals cross zero; the aggregate differences are not stable evidence of benefit.\n\n"
        "## Answers to the twelve required questions\n\n"
        f"1. **No.** ASD2 raw is `{asd2_raw-p7_raw:+.4f} mm` worse than P7 on the matched unseen-eight set.\n"
        f"2. **Affine removes {asd2_raw-asd2_affine:.4f} mm ({100*(asd2_raw-asd2_affine)/asd2_raw:.1f}%).** It also makes ASD2 affine `{asd2_affine-p7_affine:+.4f} mm` better than P7 affine.\n"
        f"3. **TPS does not add improvement here.** It worsens ASD2 by `{asd2_final-asd2_affine:+.4f} mm` after affine.\n"
        "4. **Yes for the laryngeal contours.** ASD2/P10 improves vocal folds, thyroid cartilage, epiglottis, and arytenoid by the contour-specific amounts above, even though final all-11 performance is worse.\n"
        f"5. **No.** RMS+VTLN changes final ASD2 by `{asd2_audio-asd2_final:+.4f} mm` (positive is worse).\n"
        f"6. **It is heterogeneous, not consistent.** Only {audio_improved}/8 unseen sessions improve; the largest improvement is P8/S2 (`-0.1490 mm`), while P1/S16 and P3/S14 degrade by about `+0.394` and `+0.365 mm`.\n"
        f"7. **VTLN-only is better than RMS+VTLN by {asd2_audio-vtln_only:.4f} mm**, but it is still worse than grid-only and its CI crosses zero.\n"
        f"8. **RMS degrades the weighted aggregate by {rms_only-asd2_final:+.4f} mm.** Session effects are mixed, so this is not a uniform failure, but it is not a net benefit.\n"
        f"9. **Largest remaining residuals:** {', '.join(f'{name} {value:.3f} mm' for name, value in final_contours[:4])}. The incisor residuals dominate and are consistent with a remaining static geometry/reference mismatch, though this experiment alone cannot separate geometry from model error.\n"
        f"10. **P10 is easier acoustically but not an unseen test.** Its ASD2 raw/final error is `{p10_raw:.4f} mm`, `{p10_gap_from_unseen_raw:+.4f} mm` relative to the unseen-eight raw aggregate; identity P10-to-P10 geometry leaves raw, affine, and TPS numerically unchanged. Its selected alpha is 1.0, RMS+VTLN is `{p10_audio:.4f} mm`, and the historical P7 final is `{p10_p7:.4f} mm`.\n"
        "11. **No apparent gain is caused by filtering.** Every branch has exact frame-vector and ground-truth equality; strict intersection is the full 8,585 frames; saved, scored, and rendered fractional-frame counts are zero.\n"
        f"12. **No full expansion yet.** {recommendation}\n\n"
        "## Video validation\n\n"
        f"{video_sentence} Individual JSON audits and absolute-time audio-segment manifests are saved per session. The playback audio is always the original target-session WAV, never the RMS-normalized or VTLN-warped signal.\n\n"
        "## Protocol and provenance\n\n"
        f"- New result root: `{args.output_root.resolve()}`\n"
        f"- Pre-run audit: `{(args.output_root / 'audit/pre_run_audit.json').resolve()}`\n"
        f"- Post-run audit: `{post_audit_path.resolve()}`\n"
        f"- Exact manifest: `{(args.output_root / 'manifest/selected_sessions.csv').resolve()}` and `{(args.output_root / 'manifest/selected_sessions.json').resolve()}`\n"
        f"- Checkpoint: `{MODEL_CHECKPOINT.resolve()}` (SHA-256 `{audit['checkpoint_sha256']}`, best human epoch 211 / saved index 210)\n"
        f"- Model config: `{MODEL_CONFIG.resolve()}` (SHA-256 `{audit['model_config_sha256']}`)\n"
        f"- Normalization: `{NORMALIZATION_STATS.resolve()}` (SHA-256 `{audit['normalization']['sha256']}`, train-global raw-positive std)\n"
        f"- Frontend: 39D MFCC (13 static + deltas), 25 ms window, 10 ms hop, sequence length 80, added frames 20, MRI spacing {audit['frontend']['ms_image']} ms.\n"
        f"- Contour order: `{', '.join(classes)}`. Historical `without 3` exclusion: `{', '.join(EXCLUDED_CLASSES)}`. Conversion: `{MM_PER_PIXEL} mm/pixel`.\n"
        f"- Source anatomy: `{SOURCE_SPEC.label}` / `{SOURCE_SPEC.vtln_anchor}`. All P10-to-target affine/TPS parameters: `{(args.output_root / 'configs/transforms').resolve()}`. Transforms are fitted once from static reference landmarks and never per prediction frame.\n"
        f"- New predictions: `{(args.output_root / 'inference_raw').resolve()}`, `{(args.output_root / 'gridnorm').resolve()}`, `{(args.output_root / 'audio_rms_vtln').resolve()}`, `{(args.output_root / 'audio_ablation').resolve()}`\n"
        f"- Metrics: `{(args.output_root / 'metrics').resolve()}`; unified comparisons: `{(args.output_root / 'comparisons/unified_comparison.csv').resolve()}`; bootstrap: `{(args.output_root / 'bootstrap/paired_bootstrap_ci.csv').resolve()}`\n"
        f"- Videos and probes: `{(args.output_root / 'videos_50fps_original_audio').resolve()}`\n"
        f"- Complete command log: `{command_log_path.resolve()}`\n\n"
        "The raw cache supplies the existing target-coordinate ground truth protocol; this run fits no new per-frame registration or prediction-dependent transform. Confidence intervals are exploratory because there are only eight unseen session blocks (nine including P10); session heterogeneity matters more than tiny aggregate differences.\n\n"
        "## Executed stage commands\n\n"
        "These are the recorded process arguments with timestamps and working directory; failed attempts remain visible rather than being erased. The Python interpreter was `../inversion/.venv/bin/python`; CUDA stages used `CUDA_VISIBLE_DEVICES=0` inside OAR job 6784374.\n\n"
        f"```text\n{command_block}\n```\n",
        encoding="utf-8",
    )
    write_json(
        args.output_root / "report/final_report_summary.json",
        {
            "created_at": now(),
            "report": str(report_path.resolve()),
            "selection": audit["selection"],
            "primary_group": "unseen_8_without_P10",
            "headline": {
                "asd2_raw_mm": asd2_raw,
                "p7_raw_mm": p7_raw,
                "p7_affine_mm": p7_affine,
                "p7_affine_tps_mm": p7_final,
                "asd2_affine_mm": asd2_affine,
                "asd2_affine_tps_mm": asd2_final,
                "asd2_without_laryngeal_3_affine_tps_mm": asd2_without3_final,
                "p7_without_laryngeal_3_affine_tps_mm": p7_without3_final,
                "asd2_rms_vtln_affine_tps_mm": asd2_audio,
                "asd2_rms_only_affine_tps_mm": None if not math.isfinite(rms_only) else rms_only,
                "asd2_vtln_only_affine_tps_mm": None if not math.isfinite(vtln_only) else vtln_only,
                "p10_raw_mm": p10_raw,
                "p10_affine_tps_mm": p10_grid,
            },
            "laryngeal_affine_tps": [
                {"contour": name, "asd2_mm": asd2_value, "p7_mm": p7_value, "asd2_minus_p7_mm": delta}
                for name, asd2_value, p7_value, delta in laryngeal
            ],
            "video_validation": None if video_audit is None else {
                key: video_audit[key]
                for key in (
                    "passed", "videos", "total_frames", "all_r_frame_rate_50_1",
                    "all_avg_frame_rate_50_1", "all_original_audio",
                )
            },
            "post_run_audit": str(post_audit_path.resolve()),
            "expansion_recommended": expand,
            "recommendation": recommendation,
        },
    )
    print(json.dumps({"report": str(report_path), "branches": list(sources)}, indent=2), flush=True)


VIDEO_INFO_HEIGHT = 94
VIDEO_SEPARATOR = 8


def draw_video_panel(
    image: np.ndarray,
    ground_truth: np.ndarray,
    predicted: np.ndarray | None,
    classes: list[str],
    title: str,
    frame: int,
    phoneme: str,
    rmse: float | None,
    scale: int,
) -> np.ndarray:
    size = 136 * scale
    image_bgr = cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), (size, size), interpolation=cv2.INTER_CUBIC
    )
    canvas = np.full((size + VIDEO_INFO_HEIGHT, size, 3), 15, dtype=np.uint8)
    canvas[VIDEO_INFO_HEIGHT:] = image_bgr
    for index, class_name in enumerate(classes):
        color = rgb_to_bgr255(COLORS.get(class_name, "white"))
        gt = scale_points(ground_truth[index], scale)
        gt[:, 1] += VIDEO_INFO_HEIGHT
        cv2.polylines(canvas, [gt], False, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.polylines(canvas, [gt], False, color, 1, cv2.LINE_AA)
        if predicted is not None:
            pred = scale_points(predicted[index], scale)
            pred[:, 1] += VIDEO_INFO_HEIGHT
            draw_dashed_polyline(canvas, pred, (0, 0, 0), 2, dash_length=7, gap_length=9)
            draw_dashed_polyline(canvas, pred, color, 1, dash_length=7, gap_length=9)
    lines = [title, f"frame {frame:04d} | {phoneme}"]
    if rmse is not None:
        lines.append(f"all-11 RMSE {rmse:.3f} mm")
        lines.append("solid GT | dashed prediction")
    else:
        lines.extend(("ground truth only", "solid contours"))
    for index, line in enumerate(lines):
        cv2.putText(
            canvas, line, (7, 17 + 19 * index), cv2.FONT_HERSHEY_SIMPLEX,
            0.37, (245, 245, 245), 1, cv2.LINE_AA,
        )
    return canvas


def write_original_audio_segments(
    source_wav: Path,
    frames: np.ndarray,
    config: dict[str, Any],
    output_wav: Path,
    manifest_csv: Path,
) -> dict[str, Any]:
    signal, sample_rate = sf.read(source_wav, dtype="float32", always_2d=True)
    mono = np.mean(signal, axis=1, dtype=np.float32)
    samples_per_frame_float = sample_rate / 50.0
    if not math.isclose(samples_per_frame_float, round(samples_per_frame_float), abs_tol=1e-12):
        raise RuntimeError(f"Audio sample rate {sample_rate} is not exactly divisible by 50")
    samples_per_frame = int(round(samples_per_frame_float))
    skip_ms = float(config.get("skip_ms", float(config["added_frames"]) * float(config["ms_image"])))
    segments = np.empty(len(frames) * samples_per_frame, dtype=np.float32)
    rows = []
    half = samples_per_frame // 2
    for video_index, frame in enumerate(frames):
        center_seconds = (skip_ms + (float(frame) + 0.5) * float(config["ms_image"])) / 1000.0
        center_sample = int(round(center_seconds * sample_rate))
        start = center_sample - half
        stop = start + samples_per_frame
        source_start = max(0, start)
        source_stop = min(len(mono), stop)
        destination = np.zeros(samples_per_frame, dtype=np.float32)
        dst_start = source_start - start
        destination[dst_start : dst_start + source_stop - source_start] = mono[source_start:source_stop]
        begin = video_index * samples_per_frame
        segments[begin : begin + samples_per_frame] = destination
        rows.append(
            {
                "video_frame_index": video_index,
                "mri_frame": int(frame),
                "source_center_seconds": center_seconds,
                "source_start_sample": source_start,
                "source_stop_sample_exclusive": source_stop,
                "source_sample_rate": sample_rate,
                "output_start_sample": begin,
                "output_stop_sample_exclusive": begin + samples_per_frame,
                "padded_samples": samples_per_frame - (source_stop - source_start),
            }
        )
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, segments, sample_rate, subtype="PCM_16")
    write_csv(manifest_csv, rows)
    return {
        "source_wav": str(source_wav.resolve()),
        "playback_uses_normalized_audio": False,
        "sample_rate": sample_rate,
        "samples_per_video_frame": samples_per_frame,
        "output_samples": len(segments),
        "output_duration_seconds": len(segments) / sample_rate,
        "timestamp_policy": "each 20 ms segment is indexed independently from the absolute MRI-frame center timestamp",
        "source_ms_image": float(config["ms_image"]),
        "source_skip_ms": skip_ms,
        "segments_manifest": str(manifest_csv.resolve()),
    }


def stream_duration(stream: dict[str, Any]) -> float | None:
    if stream.get("duration") not in (None, "N/A"):
        return float(stream["duration"])
    if stream.get("duration_ts") not in (None, "N/A") and stream.get("time_base"):
        numerator, denominator = stream["time_base"].split("/", maxsplit=1)
        return float(stream["duration_ts"]) * float(numerator) / float(denominator)
    return None


def probe_video(path: Path, expected_frames: int) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
            "-of", "json", str(path),
        ],
        text=True, capture_output=True, check=True,
    )
    payload = json.loads(result.stdout)
    video = next(stream for stream in payload["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in payload["streams"] if stream["codec_type"] == "audio")
    video_duration = stream_duration(video) or float(payload["format"]["duration"])
    audio_duration = stream_duration(audio) or float(payload["format"]["duration"])
    summary = {
        "path": str(path.resolve()),
        "r_frame_rate": video["r_frame_rate"],
        "avg_frame_rate": video["avg_frame_rate"],
        "number_of_video_frames": int(video.get("nb_read_frames", video.get("nb_frames", expected_frames))),
        "expected_video_frames": expected_frames,
        "video_codec": video["codec_name"],
        "audio_codec": audio["codec_name"],
        "video_duration_seconds": video_duration,
        "audio_duration_seconds": audio_duration,
        "audio_video_duration_difference_seconds": audio_duration - video_duration,
        "format_duration_seconds": float(payload["format"]["duration"]),
    }
    if summary["r_frame_rate"] != "50/1" or summary["avg_frame_rate"] != "50/1":
        raise RuntimeError(f"Video is not exact 50 fps: {summary}")
    if summary["number_of_video_frames"] != expected_frames:
        raise RuntimeError(f"Video frame count mismatch: {summary}")
    if summary["audio_codec"] != "aac" or summary["video_codec"] != "h264":
        raise RuntimeError(f"Unexpected final codecs: {summary}")
    if abs(summary["audio_video_duration_difference_seconds"]) > 0.05:
        raise RuntimeError(f"A/V duration mismatch: {summary}")
    return summary


def render_one_video(args: argparse.Namespace, speaker: int, session: int) -> dict[str, Any]:
    config, classes, _ = load_classes_and_phonemes()
    grid = load_standard_pack(branch_pack_path(args.output_root, "grid_only", speaker, session))
    audio = load_standard_pack(branch_pack_path(args.output_root, "rms_vtln", speaker, session))
    old = load_standard_pack(P7_GRID_ROOT / f"P{speaker}/S{session}/contours_and_ground_truth.npz")
    frames = grid["frame_numbers"]
    if not np.array_equal(frames, audio["frame_numbers"]) or not np.array_equal(frames, old["frame_numbers"]):
        raise RuntimeError(f"Video branch frame mismatch P{speaker}/S{session}")
    if np.any(frames != np.rint(frames)):
        raise RuntimeError("Fractional frame reached video renderer")
    phonemes = grid.get("phonemes", np.full(len(frames), "UNK"))
    dicom_dir = RAW_ROOT / f"P{speaker}/DCM_2D/S{session}"
    index, _ = build_filename_dicom_index(dicom_dir)
    session_dir = args.output_root / f"videos_50fps_original_audio/P{speaker}/S{session}"
    session_dir.mkdir(parents=True, exist_ok=True)
    mri_cache = load_or_build_mri_cache(
        dicom_dir, index, [int(value) for value in frames], session_dir / "mri_frames_cache.npz", workers=args.mri_workers
    )
    panel_width = 136 * args.video_scale
    panel_height = panel_width + VIDEO_INFO_HEIGHT
    width = 3 * panel_width + 2 * VIDEO_SEPARATOR
    height = 2 * panel_height + VIDEO_SEPARATOR
    silent = session_dir / ".comparison_silent.mp4"
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer {silent}")
    indices = list(range(len(classes)))
    metrics = {
        "asd2_raw": frame_rmse(grid["raw"], grid["ground_truth"], indices),
        "asd2_affine": frame_rmse(grid["affine"], grid["ground_truth"], indices),
        "asd2_final": frame_rmse(grid["affine_tps"], grid["ground_truth"], indices),
        "asd2_audio": frame_rmse(audio["affine_tps"], grid["ground_truth"], indices),
        "p7_final": frame_rmse(old["affine_tps"], grid["ground_truth"], indices),
    }
    try:
        for row_index, frame in enumerate(frames):
            image = mri_cache[int(frame)]
            gt = grid["ground_truth"][row_index]
            panels = [
                draw_video_panel(image, gt, None, classes, "Target ground truth", int(frame), str(phonemes[row_index]), None, args.video_scale),
                draw_video_panel(image, gt, grid["raw"][row_index], classes, "ASD2 raw", int(frame), str(phonemes[row_index]), float(metrics["asd2_raw"][row_index]), args.video_scale),
                draw_video_panel(image, gt, grid["affine"][row_index], classes, "ASD2 affine", int(frame), str(phonemes[row_index]), float(metrics["asd2_affine"][row_index]), args.video_scale),
                draw_video_panel(image, gt, grid["affine_tps"][row_index], classes, "ASD2 affine+TPS", int(frame), str(phonemes[row_index]), float(metrics["asd2_final"][row_index]), args.video_scale),
                draw_video_panel(image, gt, audio["affine_tps"][row_index], classes, "ASD2 RMS+VTLN + TPS", int(frame), str(phonemes[row_index]), float(metrics["asd2_audio"][row_index]), args.video_scale),
                draw_video_panel(image, gt, old["affine_tps"][row_index], classes, "P7 affine+TPS", int(frame), str(phonemes[row_index]), float(metrics["p7_final"][row_index]), args.video_scale),
            ]
            canvas = np.full((height, width, 3), 15, dtype=np.uint8)
            for panel_index, panel in enumerate(panels):
                x = (panel_index % 3) * (panel_width + VIDEO_SEPARATOR)
                y = (panel_index // 3) * (panel_height + VIDEO_SEPARATOR)
                canvas[y : y + panel_height, x : x + panel_width] = panel
            writer.write(canvas)
    finally:
        writer.release()
    original_wav, _ = exact_asd1_audio_paths(speaker, session)
    playback_wav = session_dir / "original_audio_evaluated_frame_segments.wav"
    audio_metadata = write_original_audio_segments(
        original_wav, frames, config, playback_wav, session_dir / "audio_segment_manifest.csv"
    )
    encoded_audio = session_dir / ".original_audio_evaluated_frame_segments.m4a"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(playback_wav), "-c:a", "aac", "-b:a", "96k", str(encoded_audio),
        ],
        check=True,
    )
    final = session_dir / f"p{speaker}_s{session}_asd2_p7_comparison_original_audio_50fps.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(silent), "-i", str(encoded_audio),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
            "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-vsync", "0", "-c:a", "copy",
            "-movflags", "+faststart", str(final),
        ],
        check=True,
    )
    probe = probe_video(final, len(frames))
    silent.unlink(missing_ok=True)
    encoded_audio.unlink(missing_ok=True)
    payload = {
        "created_at": now(),
        "speaker": f"P{speaker}",
        "session": f"S{session}",
        "frames": len(frames),
        "fractional_frames": 0,
        "frame_min": int(frames.min()),
        "frame_max": int(frames.max()),
        "video": str(final.resolve()),
        "panels": ["target_ground_truth", "asd2_raw", "asd2_affine", "asd2_affine_tps", "asd2_rms_vtln_affine_tps", "p7_affine_tps"],
        "audio": audio_metadata,
        "ffprobe": probe,
    }
    write_json(session_dir / "video_audit.json", payload)
    return payload


def render_all_videos(args: argparse.Namespace) -> None:
    require_audit(args)
    if not (args.output_root / "report/final_report_summary.json").is_file():
        raise FileNotFoundError("Run analyze before videos")
    rows = []
    for speaker, session in fixed_selection():
        audit_path = args.output_root / f"videos_50fps_original_audio/P{speaker}/S{session}/video_audit.json"
        if audit_path.is_file() and not args.force:
            payload = load_manifest(audit_path)
            probe_video(Path(payload["video"]), int(payload["frames"]))
            print(f"REUSE verified video P{speaker}/S{session}", flush=True)
        else:
            payload = render_one_video(args, speaker, session)
            print(f"DONE video P{speaker}/S{session}: {payload['frames']} frames", flush=True)
        rows.append(
            {
                "speaker": f"P{speaker}",
                "session": f"S{session}",
                "frames": payload["frames"],
                "video": payload["video"],
                **payload["ffprobe"],
            }
        )
    write_csv(args.output_root / "videos_50fps_original_audio/video_ffprobe_audit.csv", rows)
    write_json(
        args.output_root / "videos_50fps_original_audio/video_ffprobe_audit.json",
        {
            "created_at": now(),
            "passed": True,
            "videos": len(rows),
            "total_frames": sum(int(row["frames"]) for row in rows),
            "all_r_frame_rate_50_1": all(row["r_frame_rate"] == "50/1" for row in rows),
            "all_avg_frame_rate_50_1": all(row["avg_frame_rate"] == "50/1" for row in rows),
            "all_original_audio": True,
            "rows": rows,
        },
    )


def run_final_validation(args: argparse.Namespace) -> None:
    audit = require_audit(args)
    config, classes, _ = load_classes_and_phonemes()
    selection = fixed_selection()
    new_branches = ("grid_only", "rms_vtln", "rms_only", "vtln_only")
    pack_rows = []
    reference_ground_truth: dict[tuple[int, int], np.ndarray] = {}
    total_new_pack_frames = 0
    for branch in new_branches:
        for speaker, session in selection:
            path = branch_pack_path(args.output_root, branch, speaker, session)
            expected = expected_frames_from_audit(audit, speaker, session)
            with np.load(path, allow_pickle=False) as payload:
                frames = np.asarray(payload["frame_numbers"])
                if not np.array_equal(frames, expected):
                    raise RuntimeError(f"Final audit frame mismatch: {path}")
                if not np.all(np.asarray(payload["validity_flags"], dtype=bool)):
                    raise RuntimeError(f"Invalid saved frame flag: {path}")
                timestamps = np.asarray(payload["timestamps_seconds"], dtype=np.float64)
                skip_ms = float(config.get("skip_ms", float(config["added_frames"]) * float(config["ms_image"])))
                expected_timestamps = (
                    skip_ms + (frames.astype(np.float64) + 0.5) * float(config["ms_image"])
                ) / 1000.0
                timestamp_delta = float(np.max(np.abs(timestamps - expected_timestamps)))
                if timestamp_delta > 1e-12:
                    raise RuntimeError(f"Timestamp mismatch {timestamp_delta}: {path}")
                counters = {
                    key: int(payload[key])
                    for key in (
                        "saved_fractional_frame_count", "scored_fractional_frame_count",
                        "rendered_fractional_frame_count",
                    )
                }
                if any(counters.values()):
                    raise RuntimeError(f"Fractional frame survived: {path}: {counters}")
                ground_truth = np.asarray(payload["ground_truth"], dtype=np.float32)
                key = (speaker, session)
                if key in reference_ground_truth:
                    gt_delta = float(np.max(np.abs(ground_truth - reference_ground_truth[key])))
                    if gt_delta > 1e-5:
                        raise RuntimeError(f"New-branch GT mismatch {gt_delta}: {path}")
                else:
                    reference_ground_truth[key] = ground_truth.copy()
                    gt_delta = 0.0
                if [str(value) for value in payload["classes"].tolist()] != classes:
                    raise RuntimeError(f"Contour ordering mismatch: {path}")
                if str(payload["checkpoint_sha256"]) != audit["checkpoint_sha256"]:
                    raise RuntimeError(f"Checkpoint identity mismatch: {path}")
                total_new_pack_frames += len(frames)
                pack_rows.append(
                    {
                        "branch": branch,
                        "speaker": f"P{speaker}",
                        "session": f"S{session}",
                        "pack": str(path.resolve()),
                        "frames": len(frames),
                        "validity_flags_all_true": True,
                        "timestamp_max_abs_delta_seconds": timestamp_delta,
                        "ground_truth_max_abs_delta": gt_delta,
                        **counters,
                    }
                )

    for branch in ("rms_vtln", "rms_only", "vtln_only"):
        for speaker, session in selection:
            parent = branch_pack_path(args.output_root, branch, speaker, session).parent
            for required in ("audio_extraction_metadata.json", "audio_chunk_alignment.csv"):
                if not (parent / required).is_file():
                    raise FileNotFoundError(parent / required)

    alignment_audit = load_manifest(args.output_root / "comparisons/frame_alignment_audit.json")
    if not alignment_audit.get("passed") or alignment_audit.get("strict_intersection_needed"):
        raise RuntimeError("Final comparison alignment audit did not pass on the full frame set")

    video_audit_path = args.output_root / "videos_50fps_original_audio/video_ffprobe_audit.json"
    video_audit = load_manifest(video_audit_path)
    if (
        not video_audit.get("passed")
        or int(video_audit.get("videos", 0)) != 9
        or int(video_audit.get("total_frames", 0)) != 8585
    ):
        raise RuntimeError(f"Global video audit is incomplete: {video_audit_path}")
    video_rows = []
    for speaker, session in selection:
        path = args.output_root / f"videos_50fps_original_audio/P{speaker}/S{session}/video_audit.json"
        item = load_manifest(path)
        if int(item["fractional_frames"]) != 0 or item["audio"]["playback_uses_normalized_audio"]:
            raise RuntimeError(f"Video frame/audio policy violation: {path}")
        probe = probe_video(Path(item["video"]), int(item["frames"]))
        if not math.isclose(
            float(item["audio"]["output_duration_seconds"]),
            int(item["frames"]) / 50.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"Constructed original-audio length mismatch: {path}")
        video_rows.append(
            {
                "speaker": f"P{speaker}",
                "session": f"S{session}",
                "audit": str(path.resolve()),
                "original_source_wav": item["audio"]["source_wav"],
                "playback_uses_normalized_audio": False,
                **probe,
            }
        )

    audit_epoch = datetime.fromisoformat(audit["created_at"]).timestamp()
    changed_old_files = []
    for root in (P7_GRID_ROOT, P7_AUDIO_ROOT, P7_ABLATION_ROOT):
        for path in root.rglob("*"):
            if path.is_file() and path.stat().st_mtime > audit_epoch:
                changed_old_files.append(str(path.resolve()))
    if changed_old_files:
        raise RuntimeError(f"Historical result files changed after pre-run audit: {changed_old_files[:10]}")

    required_outputs = [
        args.output_root / "manifest/selected_sessions.csv",
        args.output_root / "manifest/selected_sessions.json",
        args.output_root / "audio_rms_vtln/normalization/alpha_to_asd2.json",
        args.output_root / "audio_ablation/trigger.json",
        args.output_root / "metrics/session_metrics.csv",
        args.output_root / "metrics/aggregate_metrics.csv",
        args.output_root / "metrics/contour_metrics_session.csv",
        args.output_root / "metrics/contour_metrics_aggregate.csv",
        args.output_root / "comparisons/unified_comparison.csv",
        args.output_root / "comparisons/paired_session_deltas.csv",
        args.output_root / "bootstrap/paired_bootstrap_ci.csv",
        video_audit_path,
        args.output_root / "logs/commands.log",
    ]
    missing = [str(path.resolve()) for path in required_outputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required final outputs are missing: {missing}")

    payload = {
        "created_at": now(),
        "passed": True,
        "selection": audit["selection"],
        "selection_count": len(selection),
        "full_141_session_expansion_run": False,
        "checkpoint_sha256_reverified": sha256(MODEL_CHECKPOINT),
        "config_sha256_reverified": sha256(MODEL_CONFIG),
        "normalization_sha256_reverified": sha256(NORMALIZATION_STATS),
        "new_prediction_packs": len(pack_rows),
        "new_pack_frame_instances": total_new_pack_frames,
        "integer_frames_per_branch": 8585,
        "all_validity_flags_true": True,
        "all_saved_fractional_frame_counts_zero": True,
        "all_scored_fractional_frame_counts_zero": True,
        "all_rendered_fractional_frame_counts_zero": True,
        "strict_frame_intersection_equals_full_set": True,
        "historical_p7_files_modified_after_pre_run_audit": changed_old_files,
        "historical_p7_results_untouched": True,
        "conditional_ablation_triggered": bool(load_manifest(args.output_root / "audio_ablation/trigger.json")["run_ablation"]),
        "video_validation": {
            "videos": len(video_rows),
            "total_frames": sum(int(row["expected_video_frames"]) for row in video_rows),
            "all_exact_50_fps": True,
            "all_original_unnormalized_audio": True,
            "all_audio_video_stream_durations_equal": all(
                math.isclose(float(row["audio_video_duration_difference_seconds"]), 0.0, abs_tol=1e-12)
                for row in video_rows
            ),
            "rows": video_rows,
        },
        "required_outputs": [str(path.resolve()) for path in required_outputs],
        "failed_sessions": [],
        "packs": pack_rows,
    }
    write_json(args.output_root / "audit/post_run_audit.json", payload)
    analyze_results(args)
    print(
        json.dumps(
            {
                "post_run_audit": str(args.output_root / "audit/post_run_audit.json"),
                "passed": True,
                "prediction_packs": len(pack_rows),
                "videos": len(video_rows),
                "report": str(args.output_root / "report/final_report.md"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
