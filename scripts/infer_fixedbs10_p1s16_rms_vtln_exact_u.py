#!/usr/bin/env python3
"""Infer the fixed-BS10 model on RMS+VTLN-normalized selected ASD1 audio.

The output uses the same integer timeline and ground truth as the canonical
baseline pack, then applies the exact TextGrid-/u/ source-to-target affine+TPS
transform. This is inference only; it never trains or modifies source data.
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


RESULT_ROOT = fixed.DEFAULT_OUTPUT
DEFAULT_ALPHA_METADATA = (
    REPO_ROOT
    / "results/asd2_epoch211_asd1_selected_grid_audio_20260719"
    / "audio_normalization/alpha_to_asd2_train.json"
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--config", type=Path, default=fixed.DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=fixed.DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--normalization-stats", type=Path, default=fixed.DEFAULT_NORMALIZATION
    )
    parser.add_argument("--source-pack", type=Path, default=fixed.DEFAULT_SOURCE_PACK)
    parser.add_argument("--raw-cache-root", type=Path, default=audio_core.DEFAULT_RAW_CACHE)
    parser.add_argument("--alpha-metadata", type=Path, default=DEFAULT_ALPHA_METADATA)
    parser.add_argument("--speaker", type=int, default=1)
    parser.add_argument("--session", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--transform-frame-batch", type=int, default=256)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
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


def build_exact_u_transform(args: argparse.Namespace) -> dict[str, Any]:
    fixed.base.ASD2_SOURCE_PACK = args.source_pack.resolve()
    fixed.base.prior_u.SOURCE_PACK = args.source_pack.resolve()
    source = fixed.base.prior_u.load_source_reference(fixed.SOURCE_FRAME)
    target = fixed.base.prepare_frame(
        fixed.base.FrameSpec(
            f"P{args.speaker}",
            f"S{args.session}",
            f"{args.target_frame:04d}",
            fixed.base.asd2_core.target_spec(args.speaker).vtln_anchor,
        ),
        fixed.base.asd2_core.DEFAULT_VTLN_DIR,
    )
    return fixed.base.build_two_step_transform(source["grid"], target["grid"])


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def validate_exact_transform(
    args: argparse.Namespace, transform: dict[str, Any], frame_batch: int
) -> dict[str, float]:
    pair_path = f"P{args.speaker}/S{args.session}"
    fresh_path = args.result_root / f"fresh_inference/{pair_path}/baseline.npz"
    canonical_path = args.result_root / f"predictions/ASD2/{pair_path}/baseline.npz"
    fresh = load_npz(fresh_path)
    canonical = load_npz(canonical_path)
    raw = np.asarray(fresh["predicted_raw"], dtype=np.float32)
    affine, affine_tps = fixed.base.transform_contour_batch(raw, transform, frame_batch)
    checks = {
        "raw_max_abs_delta": float(
            np.max(np.abs(raw - canonical["predicted_raw"].astype(np.float32)))
        ),
        "affine_max_abs_delta": float(
            np.max(
                np.abs(
                    affine.astype(np.float32)
                    - canonical["predicted_after_corrected_affine"].astype(np.float32)
                )
            )
        ),
        "affine_tps_max_abs_delta": float(
            np.max(
                np.abs(
                    affine_tps.astype(np.float32)
                    - canonical["predicted_after_corrected_affine_tps"].astype(np.float32)
                )
            )
        ),
    }
    if max(checks.values()) > 1e-5:
        raise RuntimeError(f"Exact-/u/ transform reproduction failed: {checks}")
    return checks


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], float, dict[str, float]]:
    pair_path = f"P{args.speaker}/S{args.session}"
    raw_cache_path = args.raw_cache_root / f"{pair_path}.pt"
    required = (
        args.result_root,
        args.config,
        args.checkpoint,
        args.normalization_stats,
        args.source_pack,
        raw_cache_path,
        args.alpha_metadata,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    config = audio_core.load_yaml_config(args.config)
    if config.get("normalization_mode") != "train_global":
        raise ValueError("Expected train_global contour normalization")
    if config.get("normalization_std_policy") != "raw_positive":
        raise ValueError("Expected raw_positive contour std policy")
    if list(config["classes"]) != list(fixed.base.CLASSES):
        raise ValueError("Config class order differs from the canonical 11 classes")

    alpha_payload = json.loads(args.alpha_metadata.read_text(encoding="utf-8"))
    if alpha_payload.get("target_reference") != "ASD2 training audio only":
        raise ValueError("VTLN alpha target is not the ASD2 training-audio reference")
    if int(alpha_payload.get("target_reference_session_count", -1)) != 85:
        raise ValueError("VTLN alpha reference does not contain the expected 85 sessions")
    if bool(alpha_payload.get("target_contours_or_labels_used", True)):
        raise ValueError("Audio normalization unexpectedly used target contours or labels")
    alpha_key = f"P{args.speaker}"
    alpha = float(alpha_payload["alpha_to_asd2_train"][alpha_key])
    if not 0.8 <= alpha <= 1.2:
        raise ValueError(
            f"{alpha_key} VTLN alpha is outside the declared grid: {alpha}"
        )

    transform = build_exact_u_transform(args)
    transform_checks = validate_exact_transform(args, transform, args.transform_frame_batch)
    payload = {
        "status": "passed",
        "inference_only": True,
        "training_launched": False,
        "speaker_session": pair_path,
        "source_u_frame": fixed.SOURCE_FRAME,
        "target_u_frame": args.target_frame,
        "audio_normalization": "RMS + VTLN",
        "target_rms": float(audio_core.TARGET_RMS),
        "vtln_alpha_to_asd2_train": alpha,
        "alpha_metadata": str(args.alpha_metadata.resolve()),
        "alpha_metadata_sha256": sha256(args.alpha_metadata),
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "normalization_stats": str(args.normalization_stats.resolve()),
        "normalization_stats_sha256": sha256(args.normalization_stats),
        "raw_cache": str(raw_cache_path.resolve()),
        "exact_u_transform_reproduction": transform_checks,
        "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "hostname": socket.gethostname(),
    }
    return payload, alpha, transform


def run_inference(
    args: argparse.Namespace,
    preflight_payload: dict[str, Any],
    alpha: float,
    transform: dict[str, Any],
) -> dict[str, Any]:
    output = args.output.resolve()
    manifest_path = output.with_name(f"{output.stem}_manifest.json")
    alignment_path = output.with_name(f"{output.stem}_chunk_alignment.csv")
    extraction_path = output.with_name(f"{output.stem}_audio_extraction.json")
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite without --force: {output}")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("GPU inference must run inside OAR; CPU fallback is forbidden")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the OAR allocation")

    started = time.monotonic()
    config = audio_core.load_yaml_config(args.config)
    normalization = audio_core.load_normalization(args.normalization_stats)
    with Path(config["phonemesdir"]).open("r", encoding="utf-8") as handle:
        phonemes = json.load(handle)
    model = audio_core.load_model(config, args.checkpoint, device)

    raw_cache_path = args.raw_cache_root / f"P{args.speaker}/S{args.session}.pt"
    raw = torch.load(raw_cache_path, map_location="cpu", weights_only=False)["raw"]
    feature_chunks, alignment, extraction = audio_core.build_audio_normalized_chunks(
        audio_core.asd1_audio_config(config),
        args.speaker,
        args.session,
        alpha,
        raw["features"],
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
            "playback_audio": "original target WAV",
        }
    )
    audio_core.write_csv(alignment_path, alignment)
    atomic_json(extraction_path, extraction)

    inferred = audio_core.retain_integer_inferred(
        audio_core.infer_with_features(
            model,
            device,
            raw,
            feature_chunks,
            normalization,
            phonemes,
            args.batch_size,
        )
    )
    affine, affine_tps = fixed.base.transform_contour_batch(
        inferred["predicted_raw"], transform, args.transform_frame_batch
    )

    baseline = load_npz(
        args.result_root
        / f"predictions/ASD2/P{args.speaker}/S{args.session}/baseline.npz"
    )
    frames = np.asarray(inferred["frame_numbers"], dtype=np.int32)
    ground_truth = np.asarray(inferred["ground_truth"], dtype=np.float32)
    if not np.array_equal(frames, baseline["frame_numbers"].astype(np.int32)):
        raise RuntimeError("RMS+VTLN timeline differs from the canonical baseline")
    ground_truth_delta = float(
        np.max(np.abs(ground_truth - baseline["ground_truth"].astype(np.float32)))
    )
    if ground_truth_delta > 1e-5:
        raise RuntimeError(f"RMS+VTLN ground truth differs from baseline: {ground_truth_delta}")
    arrays = {
        "raw": np.asarray(inferred["predicted_raw"], dtype=np.float32),
        "affine": np.asarray(affine, dtype=np.float32),
        "affine_tps": np.asarray(affine_tps, dtype=np.float32),
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("RMS+VTLN prediction contains non-finite coordinates")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.writing.npz")
    np.savez_compressed(
        temporary,
        model=np.asarray("asd2_fixedbs10_best_human_epoch31"),
        branch=np.asarray("rms_vtln"),
        frame_numbers=frames,
        textgrid_exact_u_mask=baseline["textgrid_exact_u_mask"].astype(bool),
        phonemes_cached_non_authoritative=np.asarray(inferred["phonemes"]),
        predicted_raw=arrays["raw"],
        predicted_after_corrected_affine=arrays["affine"],
        predicted_after_corrected_affine_tps=arrays["affine_tps"],
        ground_truth=ground_truth,
        classes=np.asarray(fixed.base.CLASSES, dtype="U64"),
        source_u_frame=np.asarray(fixed.SOURCE_FRAME, dtype=np.int32),
        target_u_frame=np.asarray(args.target_frame, dtype=np.int32),
        source_verified_exact_u=np.asarray(True),
        target_verified_exact_u=np.asarray(True),
        vtln_alpha_to_asd2_train=np.asarray(alpha, dtype=np.float32),
        target_rms=np.asarray(audio_core.TARGET_RMS, dtype=np.float32),
        target_contours_or_labels_used_for_audio_normalization=np.asarray(False),
        raw_prediction_sha256=np.asarray(array_sha256(arrays["raw"])),
        saved_fractional_frame_count=np.asarray(0, dtype=np.int64),
        scored_fractional_frame_count=np.asarray(0, dtype=np.int64),
        rendered_fractional_frame_count=np.asarray(0, dtype=np.int64),
    )
    temporary.replace(output)

    baseline_tps = baseline["predicted_after_corrected_affine_tps"].astype(np.float32)
    baseline_rmse = fixed.base.frame_rmse(
        baseline_tps, ground_truth, fixed.GROUPS["all_11"]
    )
    audio_rmse = fixed.base.frame_rmse(
        arrays["affine_tps"], ground_truth, fixed.GROUPS["all_11"]
    )
    calibration_index = int(
        np.flatnonzero(frames == args.target_frame)[0]
    )
    manifest = {
        **preflight_payload,
        "status": "passed",
        "output": str(output),
        "output_sha256": sha256(output),
        "frames": int(len(frames)),
        "frame_min": int(frames.min()),
        "frame_max": int(frames.max()),
        "timeline_equal_to_baseline": True,
        "ground_truth_max_abs_delta_vs_baseline": ground_truth_delta,
        "saved_fractional_frame_count": 0,
        "scored_fractional_frame_count": 0,
        "rendered_fractional_frame_count": 0,
        "mean_affine_tps_rmse_mm": float(np.mean(audio_rmse)),
        "mean_effect_vs_baseline_affine_tps_mm": float(
            np.mean(audio_rmse - baseline_rmse)
        ),
        "calibration_frame_affine_tps_rmse_mm": float(audio_rmse[calibration_index]),
        "calibration_frame_effect_vs_baseline_affine_tps_mm": float(
            audio_rmse[calibration_index] - baseline_rmse[calibration_index]
        ),
        "gpu": torch.cuda.get_device_name(device),
        "elapsed_seconds": time.monotonic() - started,
        "audio_extraction": str(extraction_path.resolve()),
        "chunk_alignment": str(alignment_path.resolve()),
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def main() -> None:
    args = parse_args()
    pair = (args.speaker, args.session)
    if pair not in fixed.EXPECTED_TARGET_FRAMES:
        choices = ", ".join(
            f"P{speaker}/S{session}"
            for speaker, session in fixed.EXPECTED_TARGET_FRAMES
        )
        raise ValueError(f"Unsupported selected pair P{args.speaker}/S{args.session}; expected one of {choices}")
    args.target_frame = fixed.EXPECTED_TARGET_FRAMES[pair]
    args.result_root = args.result_root.resolve()
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.normalization_stats = args.normalization_stats.resolve()
    args.source_pack = args.source_pack.resolve()
    args.raw_cache_root = args.raw_cache_root.resolve()
    args.alpha_metadata = args.alpha_metadata.resolve()
    if args.output is None:
        args.output = (
            args.result_root
            / f"phase_b/demo_inputs/P{args.speaker}/S{args.session}"
            / "rms_vtln_exact_u_fixedbs10.npz"
        )
    args.output = args.output.resolve()
    preflight_payload, alpha, transform = preflight(args)
    print(json.dumps(preflight_payload, indent=2, sort_keys=True), flush=True)
    if args.preflight_only:
        return
    run_inference(args, preflight_payload, alpha, transform)


if __name__ == "__main__":
    main()
