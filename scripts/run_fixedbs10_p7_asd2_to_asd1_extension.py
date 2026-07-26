#!/usr/bin/env python3
"""Add P7/S2 to the fixed-BS10 ASD2-to-ASD1 adaptation evaluation.

This is inference only. It evaluates the existing fixed-BS10 ASD2 checkpoint
on P7/S2 with (1) the unmodified model input and (2) RMS+VTLN-normalized
audio. Both prediction branches use the same exact-TextGrid-/u/ affine+TPS
transform from ASD2 1791/S14/F0499 to ASD1 P7/S2/F0500.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_ROOT = REPO_ROOT / "external/grid-transform"
sys.path[:0] = [
    str(REPO_ROOT),
    str(REPO_ROOT / "src"),
    str(REPO_ROOT / "scripts"),
    str(GRID_ROOT),
]

import run_asd2_epoch211_selected_experiment as audio_core  # noqa: E402
import run_asd2_fixedbs10_textgrid_u_grid_adaptation as fixed  # noqa: E402


SPEAKER = 7
SESSION = 2
SOURCE_FRAME = 499
TARGET_FRAME = 500
P7_VTLN_ANCHOR = "1640_P7_S2_F0829"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results/asd2_fixedbs10_selected_10speakers_adaptation_tables_20260724"
)
DEFAULT_OLD_P7_PACK = (
    REPO_ROOT
    / "results/asd2_model_to_asd1_selected_sessions_integer_only_20260718"
    / "P7/S2/asd2_model_predictions_and_ground_truth_integer_only.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=fixed.DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=fixed.DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--normalization-stats", type=Path, default=fixed.DEFAULT_NORMALIZATION
    )
    parser.add_argument("--source-pack", type=Path, default=fixed.DEFAULT_SOURCE_PACK)
    parser.add_argument(
        "--raw-cache-root", type=Path, default=audio_core.DEFAULT_RAW_CACHE
    )
    parser.add_argument("--old-p7-pack", type=Path, default=DEFAULT_OLD_P7_PACK)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--transform-frame-batch", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-alpha", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def serializable_selection(selection: dict[str, Any]) -> dict[str, Any]:
    """Remove the self-reference introduced by select_exact_u_reference."""
    result = {
        key: value for key, value in selection.items() if key != "candidate_runs"
    }
    result["candidate_runs"] = [
        {key: value for key, value in row.items() if key != "candidate_runs"}
        for row in selection["candidate_runs"]
    ]
    return result


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def validate_paths(args: argparse.Namespace) -> Path:
    raw_cache = args.raw_cache_root / f"P{SPEAKER}/S{SESSION}.pt"
    for path in (
        args.config,
        args.checkpoint,
        args.normalization_stats,
        args.source_pack,
        raw_cache,
        args.old_p7_pack,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = audio_core.load_yaml_config(args.config)
    if config.get("normalization_mode") != "train_global":
        raise ValueError("Expected train_global contour normalization")
    if config.get("normalization_std_policy") != "raw_positive":
        raise ValueError("Expected raw_positive contour normalization")
    if list(config["classes"]) != list(fixed.base.CLASSES):
        raise ValueError("Config class order differs from the canonical 11 classes")
    return raw_cache


def build_transform(args: argparse.Namespace) -> dict[str, Any]:
    fixed.base.ASD2_SOURCE_PACK = args.source_pack.resolve()
    fixed.base.prior_u.SOURCE_PACK = args.source_pack.resolve()
    source = fixed.base.prior_u.load_source_reference(SOURCE_FRAME)
    target = fixed.base.prepare_frame(
        fixed.base.FrameSpec(
            f"P{SPEAKER}",
            f"S{SESSION}",
            f"{TARGET_FRAME:04d}",
            P7_VTLN_ANCHOR,
        ),
        fixed.base.asd2_core.DEFAULT_VTLN_DIR,
    )
    return fixed.base.build_two_step_transform(source["grid"], target["grid"])


def direct_textgrid_u_selection(
    frames: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    _wav, textgrid_path = audio_core.exact_asd1_audio_paths(SPEAKER, SESSION)
    frame_set = {int(value) for value in frames}

    def valid_frame(frame: int) -> tuple[bool, dict[str, Any]]:
        if frame in frame_set:
            return True, {"frame_in_evaluation_timeline": True}
        return False, {"reason": "frame_not_in_evaluation_timeline"}

    selection, tier_inventory = fixed.base.select_exact_u_reference(
        dataset="asd1",
        speaker_session=f"P{SPEAKER}/S{SESSION}",
        textgrid_path=textgrid_path,
        valid_frame=valid_frame,
    )
    if int(selection["selected_frame"]) != TARGET_FRAME:
        raise RuntimeError(
            f"Direct TextGrid selection changed: expected F{TARGET_FRAME:04d}, "
            f"got F{int(selection['selected_frame']):04d}"
        )
    mask = fixed.base.textgrid_u_mask(
        frames.astype(np.int32), textgrid_path, int(selection["tier_index"])
    )
    matches = np.flatnonzero(frames.astype(np.int32) == TARGET_FRAME)
    if len(matches) != 1 or not bool(mask[int(matches[0])]):
        raise RuntimeError("P7/S2/F0500 is not a unique evaluated exact-/u/ frame")
    return mask, selection, tier_inventory


def save_prediction_pack(
    path: Path,
    *,
    branch: str,
    inferred: dict[str, np.ndarray],
    affine: np.ndarray,
    affine_tps: np.ndarray,
    u_mask: np.ndarray,
    alpha: float,
    target_rms: float | None,
) -> None:
    if path.exists() and not ARGS.force:
        raise FileExistsError(f"Refusing to overwrite without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    predicted_raw = np.asarray(inferred["predicted_raw"], dtype=np.float32)
    ground_truth = np.asarray(inferred["ground_truth"], dtype=np.float32)
    frames = np.asarray(inferred["frame_numbers"], dtype=np.int32)
    temporary = path.with_name(f".{path.stem}.writing.npz")
    np.savez_compressed(
        temporary,
        model=np.asarray("asd2_fixedbs10_best_human_epoch31"),
        branch=np.asarray(branch),
        frame_numbers=frames,
        textgrid_exact_u_mask=np.asarray(u_mask, dtype=bool),
        phonemes_cached_non_authoritative=np.asarray(inferred["phonemes"]),
        predicted_raw=predicted_raw,
        predicted_after_corrected_affine=np.asarray(affine, dtype=np.float32),
        predicted_after_corrected_affine_tps=np.asarray(
            affine_tps, dtype=np.float32
        ),
        ground_truth=ground_truth,
        classes=np.asarray(fixed.base.CLASSES, dtype="U64"),
        source_u_frame=np.asarray(SOURCE_FRAME, dtype=np.int32),
        target_u_frame=np.asarray(TARGET_FRAME, dtype=np.int32),
        source_verified_exact_u=np.asarray(True),
        target_verified_exact_u=np.asarray(True),
        vtln_alpha_to_asd2_train=np.asarray(alpha, dtype=np.float32),
        target_rms=np.asarray(
            np.nan if target_rms is None else target_rms, dtype=np.float32
        ),
        target_contours_or_labels_used_for_audio_normalization=np.asarray(False),
        raw_prediction_sha256=np.asarray(array_sha256(predicted_raw)),
        saved_fractional_frame_count=np.asarray(0, dtype=np.int64),
        scored_fractional_frame_count=np.asarray(0, dtype=np.int64),
        rendered_fractional_frame_count=np.asarray(0, dtype=np.int64),
        frame_policy=np.asarray(
            "integer MRI frames only; no interpolation, hold, or fractional rows"
        ),
    )
    temporary.replace(path)


def estimate_p7_alpha(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[float, Path]:
    original_selection = audio_core.SELECTION
    audio_core.SELECTION = ((SPEAKER, SESSION),)
    try:
        alphas = audio_core.estimate_alphas(
            SimpleNamespace(
                output_root=args.output_root / "p7_extension",
                force=args.force_alpha,
            ),
            config,
        )
    finally:
        audio_core.SELECTION = original_selection
    alpha_path = (
        args.output_root
        / "p7_extension/audio_normalization/alpha_to_asd2_train.json"
    )
    alpha = float(alphas[f"P{SPEAKER}"])
    if not 0.8 <= alpha <= 1.2:
        raise ValueError(f"Estimated P7 VTLN alpha is outside the search grid: {alpha}")
    return alpha, alpha_path


def all11_frame_rmse(predicted: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    return fixed.base.frame_rmse(
        predicted, ground_truth, fixed.GROUPS["all_11"]
    )


def summarize(
    raw: np.ndarray,
    contour: np.ndarray,
    contour_audio: np.ndarray,
    ground_truth: np.ndarray,
    frames: np.ndarray,
) -> dict[str, Any]:
    calibration = frames != TARGET_FRAME
    result: dict[str, Any] = {}
    for subset, mask in (
        ("all_frames", np.ones(len(frames), dtype=bool)),
        ("exclude_calibration_frame", calibration),
    ):
        stage_values = {
            "baseline": all11_frame_rmse(raw[mask], ground_truth[mask]),
            "contour": all11_frame_rmse(contour[mask], ground_truth[mask]),
            "contour_audio": all11_frame_rmse(
                contour_audio[mask], ground_truth[mask]
            ),
        }
        result[subset] = {
            "frames": int(mask.sum()),
            **{
                stage: {
                    "mean_rmse_mm": float(np.mean(values)),
                    "sample_sd_rmse_mm": float(np.std(values, ddof=1)),
                }
                for stage, values in stage_values.items()
            },
        }
    return result


def main(args: argparse.Namespace) -> None:
    if not os.environ.get("OAR_JOB_ID"):
        raise RuntimeError("GPU inference must run inside an active OAR allocation")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")

    started = time.monotonic()
    args.output_root.mkdir(parents=True, exist_ok=True)
    raw_cache = validate_paths(args)
    transform = build_transform(args)
    config = audio_core.load_yaml_config(args.config)
    normalization = audio_core.load_normalization(args.normalization_stats)
    with Path(config["phonemesdir"]).open("r", encoding="utf-8") as handle:
        phonemes = json.load(handle)
    model = audio_core.load_model(config, args.checkpoint, device)

    baseline_inferred = audio_core.retain_integer_inferred(
        audio_core.infer_session(
            model,
            device,
            raw_cache,
            normalization,
            phonemes,
            args.batch_size,
        )
    )
    baseline_frames = np.asarray(baseline_inferred["frame_numbers"], dtype=np.int32)
    u_mask, selection, tier_inventory = direct_textgrid_u_selection(baseline_frames)
    atomic_json(
        args.output_root / "p7_extension/reference_selection.json",
        {
            "selection": serializable_selection(selection),
            "tier_inventory": tier_inventory,
        },
    )
    baseline_affine, baseline_tps = fixed.base.transform_contour_batch(
        baseline_inferred["predicted_raw"],
        transform,
        args.transform_frame_batch,
    )
    baseline_path = args.output_root / "p7_extension/P7/S2/baseline.npz"
    save_prediction_pack(
        baseline_path,
        branch="baseline",
        inferred=baseline_inferred,
        affine=baseline_affine,
        affine_tps=baseline_tps,
        u_mask=u_mask,
        alpha=1.0,
        target_rms=None,
    )

    alpha, alpha_path = estimate_p7_alpha(args, config)
    raw_payload = torch.load(raw_cache, map_location="cpu", weights_only=False)["raw"]
    feature_chunks, alignment, extraction = audio_core.build_audio_normalized_chunks(
        audio_core.asd1_audio_config(config),
        SPEAKER,
        SESSION,
        alpha,
        raw_payload["features"],
        rms_target=audio_core.TARGET_RMS,
    )
    extraction.pop("vtln_alpha_to_p7", None)
    extraction.update(
        {
            "branch": "rms_vtln",
            "vtln_alpha_to_asd2_train": alpha,
            "audio_reference": "ASD2 train audio only (85 sessions)",
            "target_contours_or_labels_used_for_audio_normalization": False,
            "normalized_waveform_role": "model input only",
            "playback_audio": "not rendered in this table-only extension",
        }
    )
    audio_dir = args.output_root / "p7_extension/P7/S2"
    audio_core.write_csv(audio_dir / "rms_vtln_chunk_alignment.csv", alignment)
    atomic_json(audio_dir / "rms_vtln_audio_extraction.json", extraction)
    audio_inferred = audio_core.retain_integer_inferred(
        audio_core.infer_with_features(
            model,
            device,
            raw_payload,
            feature_chunks,
            normalization,
            phonemes,
            args.batch_size,
        )
    )
    audio_frames = np.asarray(audio_inferred["frame_numbers"], dtype=np.int32)
    if not np.array_equal(audio_frames, baseline_frames):
        raise RuntimeError("P7 RMS+VTLN timeline differs from baseline")
    baseline_gt = np.asarray(baseline_inferred["ground_truth"], dtype=np.float32)
    audio_gt = np.asarray(audio_inferred["ground_truth"], dtype=np.float32)
    ground_truth_delta = float(np.max(np.abs(audio_gt - baseline_gt)))
    if ground_truth_delta > 1e-5:
        raise RuntimeError(
            f"P7 RMS+VTLN ground truth differs from baseline: {ground_truth_delta}"
        )
    audio_affine, audio_tps = fixed.base.transform_contour_batch(
        audio_inferred["predicted_raw"],
        transform,
        args.transform_frame_batch,
    )
    audio_path = args.output_root / "p7_extension/P7/S2/rms_vtln.npz"
    save_prediction_pack(
        audio_path,
        branch="rms_vtln",
        inferred=audio_inferred,
        affine=audio_affine,
        affine_tps=audio_tps,
        u_mask=u_mask,
        alpha=alpha,
        target_rms=audio_core.TARGET_RMS,
    )

    old = load_npz(args.old_p7_pack)
    old_frames = np.asarray(old["frame_numbers"], dtype=np.int32)
    if not np.array_equal(old_frames, baseline_frames):
        raise RuntimeError("Current P7 timeline differs from the July-18 integer timeline")
    old_gt = np.asarray(old["ground_truth"], dtype=np.float32)
    old_gt_delta = float(np.max(np.abs(old_gt - baseline_gt)))
    if old_gt_delta > 1e-5:
        raise RuntimeError(
            f"Current P7 ground truth differs from the July-18 pack: {old_gt_delta}"
        )

    metrics = summarize(
        np.asarray(baseline_inferred["predicted_raw"], dtype=np.float32),
        np.asarray(baseline_tps, dtype=np.float32),
        np.asarray(audio_tps, dtype=np.float32),
        baseline_gt,
        baseline_frames,
    )
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "passed",
        "operation": "inference-only P7/S2 extension; no training",
        "training_launched": False,
        "speaker_session": "P7/S2",
        "frames": int(len(baseline_frames)),
        "frame_min": int(baseline_frames.min()),
        "frame_max": int(baseline_frames.max()),
        "source_reference": "ASD2 1791/S14/F0499 exact TextGrid /u/",
        "target_reference": "ASD1 P7/S2/F0500 exact TextGrid /u/",
        "transform_scope": "one fixed affine+TPS transform reused over P7/S2",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "normalization_stats": str(args.normalization_stats.resolve()),
        "normalization_stats_sha256": sha256(args.normalization_stats),
        "source_pack": str(args.source_pack.resolve()),
        "source_pack_sha256": sha256(args.source_pack),
        "raw_cache": str(raw_cache.resolve()),
        "raw_cache_sha256": sha256(raw_cache),
        "baseline_pack": str(baseline_path.resolve()),
        "baseline_pack_sha256": sha256(baseline_path),
        "rms_vtln_pack": str(audio_path.resolve()),
        "rms_vtln_pack_sha256": sha256(audio_path),
        "p7_vtln_alpha_to_asd2_train": alpha,
        "alpha_metadata": str(alpha_path.resolve()),
        "alpha_metadata_sha256": sha256(alpha_path),
        "target_rms": float(audio_core.TARGET_RMS),
        "target_contours_or_labels_used_for_audio_normalization": False,
        "timeline_equal_between_branches": True,
        "ground_truth_max_abs_delta_between_branches": ground_truth_delta,
        "timeline_equal_to_july18_pack": True,
        "ground_truth_max_abs_delta_vs_july18_pack": old_gt_delta,
        "metrics": metrics,
        "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "hostname": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(device),
        "elapsed_seconds": time.monotonic() - started,
        "saved_fractional_frame_count": 0,
        "scored_fractional_frame_count": 0,
        "rendered_fractional_frame_count": 0,
    }
    atomic_json(args.output_root / "p7_extension/manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    ARGS = parse_args()
    ARGS.output_root = ARGS.output_root.resolve()
    ARGS.config = ARGS.config.resolve()
    ARGS.checkpoint = ARGS.checkpoint.resolve()
    ARGS.normalization_stats = ARGS.normalization_stats.resolve()
    ARGS.source_pack = ARGS.source_pack.resolve()
    ARGS.raw_cache_root = ARGS.raw_cache_root.resolve()
    ARGS.old_p7_pack = ARGS.old_p7_pack.resolve()
    main(ARGS)
