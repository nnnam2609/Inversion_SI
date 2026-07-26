"""Inference and exact-/u/ anatomical adaptation for selected ASD1 sessions."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from ..contracts import ContractError, atomic_write_json, load_cohort, load_model_bundle
from ..domain import GLOBAL, MOVING_AVERAGE
from ..io import load_mapping, write_rows_csv
from ..metrics import contour_metric_summary
from ..runtime import REPOSITORY_ROOT
from ..adapters.legacy_runtime import (
    LegacyAnatomyAdapter,
    LegacyInversionAdapter,
)
from ..strategies import (
    InferredSession,
    from_legacy_inferred,
    infer_moving_average,
)


from src.inference.session_inference import load_model  # noqa: E402

audio_core = LegacyInversionAdapter()
anatomy_core = LegacyAnatomyAdapter()


SOURCE_PACK = (
    REPOSITORY_ROOT
    / "cache_variants/asd2_11_bfincisor_sofiane153_s25_20260725/"
    "raw_contour_npz/asd2/1791/S14.npz"
)
RAW_CACHE_ROOT = REPOSITORY_ROOT / "cache/raw_sessions/asd1"
P7_VTLN_ANCHOR = "1640_P7_S2_F0829"


def sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("utf-8"))
    digest.update(json.dumps(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _target_anchor(speaker_number: int) -> str:
    if speaker_number == 7:
        return P7_VTLN_ANCHOR
    return anatomy_core.target_anchor(
        speaker_number, p7_anchor=P7_VTLN_ANCHOR
    )


def build_transform(
    speaker_number: int, session_number: int, target_frame: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    bundle = anatomy_core.build_transform_bundle(
        source_pack=SOURCE_PACK,
        speaker=speaker_number,
        session=session_number,
        target_frame=target_frame,
        target_anchor=_target_anchor(speaker_number),
    )
    transform = bundle.transform
    diagnostics = bundle.diagnostics
    metadata = {
        "source_frame": 499,
        "target_frame": target_frame,
        "source_vowel": "u",
        "target_vowel": "u",
        "affine_controls": diagnostics["step1_labels"],
        "tps_controls": diagnostics["step2_labels"],
        "diagnostics": {
            key: value
            for key, value in diagnostics.items()
            if not isinstance(value, (list, dict, np.ndarray))
        },
    }
    return transform, metadata


def validate_target_u(
    speaker_number: int,
    session_number: int,
    expected_frame: int,
    frames: np.ndarray,
) -> np.ndarray:
    _wav, textgrid_path = audio_core.exact_asd1_audio_paths(
        speaker_number, session_number
    )
    allowed = {int(value) for value in frames}

    def valid_frame(frame: int):
        if frame in allowed:
            return True, {"frame_in_evaluation_timeline": True}
        return False, {"reason": "frame_not_in_evaluation_timeline"}

    selection, _inventory = anatomy_core.select_exact_u_reference(
        dataset="ASD1",
        speaker_session=f"P{speaker_number}/S{session_number}",
        textgrid_path=textgrid_path,
        valid_frame=valid_frame,
    )
    observed = int(selection["selected_frame"])
    if observed != expected_frame:
        raise ContractError(
            f"Exact-/u/ reference changed for P{speaker_number}/S{session_number}: "
            f"expected F{expected_frame:04d}, got F{observed:04d}"
        )
    mask = anatomy_core.textgrid_u_mask(
        frames.astype(np.int32), textgrid_path, int(selection["tier_index"])
    )
    if not mask.any():
        raise ContractError("No exact-/u/ frames in the evaluation timeline")
    return mask


def infer_baseline(
    *,
    strategy: str,
    model: torch.nn.Module,
    device: torch.device,
    raw_cache: Path,
    config: Dict[str, Any],
    split_root: Path,
    phonemes: List[str],
    batch_size: int,
) -> InferredSession:
    if strategy == MOVING_AVERAGE:
        return infer_moving_average(
            model=model,
            device=device,
            raw_cache=raw_cache,
            phonemes=phonemes,
            batch_size=batch_size,
        )
    normalization = audio_core.load_normalization(
        split_root / "normalization_stats.npz"
    )
    inferred = audio_core.retain_integer_inferred(
        audio_core.infer_session(
            model, device, raw_cache, normalization, phonemes, batch_size
        )
    )
    return from_legacy_inferred(
        inferred,
        strategy=GLOBAL,
        metadata={
            "uses_target_statistics": False,
            "uses_target_labels": False,
            "causal": True,
            "blind_inference_compatible": True,
        },
    )


def infer_audio_normalized(
    *,
    strategy: str,
    model: torch.nn.Module,
    device: torch.device,
    raw_cache: Path,
    config: Dict[str, Any],
    split_root: Path,
    phonemes: List[str],
    batch_size: int,
    speaker_number: int,
    session_number: int,
    alpha: float,
    rms_target: float,
) -> Tuple[InferredSession, List[Dict[str, Any]], Dict[str, Any]]:
    raw = torch.load(raw_cache, map_location="cpu", weights_only=False)["raw"]
    feature_chunks, alignment, extraction = audio_core.build_audio_normalized_chunks(
        audio_core.asd1_audio_config(config),
        speaker_number,
        session_number,
        alpha,
        raw["features"],
        rms_target=rms_target,
    )
    extraction.pop("vtln_alpha_to_p7", None)
    extraction.update(
        {
            "vtln_alpha_to_asd2_train": alpha,
            "target_rms": rms_target,
            "audio_reference": "ASD2 training sessions from selected model config",
            "target_contours_or_labels_used_for_audio_normalization": False,
            "normalized_waveform_role": "model input only",
        }
    )
    if strategy == MOVING_AVERAGE:
        inferred = infer_moving_average(
            model=model,
            device=device,
            raw_cache=raw_cache,
            phonemes=phonemes,
            batch_size=batch_size,
            feature_override=feature_chunks,
        )
        inferred.metadata.update(
            {
                "audio_normalization": "RMS+VTLN",
                "vtln_alpha": alpha,
                "target_rms": rms_target,
            }
        )
        return inferred, alignment, extraction
    normalization = audio_core.load_normalization(
        split_root / "normalization_stats.npz"
    )
    legacy = audio_core.retain_integer_inferred(
        audio_core.infer_with_features(
            model,
            device,
            raw,
            feature_chunks,
            normalization,
            phonemes,
            batch_size,
        )
    )
    return (
        from_legacy_inferred(
            legacy,
            strategy=GLOBAL,
            metadata={
                "uses_target_statistics": False,
                "uses_target_labels": False,
                "causal": True,
                "blind_inference_compatible": True,
                "audio_normalization": "RMS+VTLN",
                "vtln_alpha": alpha,
                "target_rms": rms_target,
            },
        ),
        alignment,
        extraction,
    )


def save_pack(
    path: Path,
    *,
    inferred: InferredSession,
    affine: np.ndarray,
    affine_tps: np.ndarray,
    u_mask: np.ndarray,
    strategy: str,
    target_frame: int,
    coordinate_scale_mm: float,
    transform_metadata: Dict[str, Any],
    output_coordinate_space: str,
    audio_inferred: InferredSession = None,
    audio_affine: np.ndarray = None,
    audio_affine_tps: np.ndarray = None,
    audio_metadata: Dict[str, Any] = None,
) -> Dict[str, Any]:
    if not np.array_equal(inferred.frame_numbers, np.sort(inferred.frame_numbers)):
        raise ContractError("Inference frame timeline is not ordered")
    conditions = {
        "original": inferred.predicted_raw,
        "anatomical": affine_tps,
    }
    if audio_inferred is not None:
        if not np.array_equal(inferred.frame_numbers, audio_inferred.frame_numbers):
            raise ContractError("Raw/audio frame timeline mismatch")
        ground_truth_delta = float(
            np.max(np.abs(inferred.ground_truth - audio_inferred.ground_truth))
        )
        if ground_truth_delta > 1e-5:
            raise ContractError(
                f"Raw/audio ground truth mismatch: max delta={ground_truth_delta}"
            )
        if audio_affine is None or audio_affine_tps is None:
            raise ContractError("Audio anatomical arrays are incomplete")
        conditions.update(
            {
                "audio": audio_inferred.predicted_raw,
                "anatomical_audio": audio_affine_tps,
            }
        )
    metrics = {
        condition: contour_metric_summary(
            predicted,
            inferred.ground_truth,
            coordinate_scale_mm=coordinate_scale_mm,
        )
        for condition, predicted in conditions.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.writing.npz")
    arrays = {
        "schema_version": np.asarray("1.0"),
        "strategy": np.asarray(strategy),
        "frame_numbers": inferred.frame_numbers.astype(np.int32),
        "classes": np.asarray(anatomy_core.classes, dtype="U64"),
        "predicted_original": inferred.predicted_raw.astype(np.float32),
        "predicted_after_affine": affine.astype(np.float32),
        "predicted_anatomical": affine_tps.astype(np.float32),
        "ground_truth": inferred.ground_truth.astype(np.float32),
        "phonemes": inferred.phonemes,
        "overlap_counts": inferred.overlap_counts,
        "textgrid_exact_u_mask": u_mask.astype(bool),
        "source_u_frame": np.asarray(499, dtype=np.int32),
        "target_u_frame": np.asarray(target_frame, dtype=np.int32),
        "coordinate_scale_mm": np.asarray(
            coordinate_scale_mm, dtype=np.float32
        ),
        "saved_fractional_frame_count": np.asarray(0, dtype=np.int64),
    }
    if audio_inferred is not None:
        arrays.update(
            {
                "predicted_audio": audio_inferred.predicted_raw.astype(np.float32),
                "predicted_audio_after_affine": audio_affine.astype(np.float32),
                "predicted_anatomical_audio": audio_affine_tps.astype(np.float32),
            }
        )
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    manifest = {
        "status": "complete",
        "schema_version": "1.0",
        "strategy": strategy,
        "pack": str(path.resolve()),
        "frame_count": int(len(inferred.frame_numbers)),
        "frame_min": int(inferred.frame_numbers.min()),
        "frame_max": int(inferred.frame_numbers.max()),
        "frame_sha256": sha256_array(inferred.frame_numbers),
        "ground_truth_sha256": sha256_array(inferred.ground_truth),
        "conditions": list(conditions),
        "output_coordinate_space": output_coordinate_space,
        "anatomical_condition_valid_for_primary_comparison": (
            output_coordinate_space == "reference_native"
        ),
        "anatomical_condition_warning": (
            None
            if output_coordinate_space == "reference_native"
            else (
                "The moving-average output was restored with target-contour "
                "statistics into target-native coordinates. Applying the "
                "reference-to-target anatomical transform is a diagnostic "
                "double adaptation and is not a valid primary result."
            )
        ),
        "metrics": metrics,
        "inference": inferred.metadata,
        "audio_inference": (
            None if audio_inferred is None else audio_inferred.metadata
        ),
        "audio_normalization": audio_metadata,
        "transform": transform_metadata,
        "source_u_frame": 499,
        "target_u_frame": target_frame,
        "training_launched": False,
        "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "hostname": socket.gethostname(),
    }
    atomic_write_json(path.with_suffix(".manifest.json"), manifest)
    path.with_suffix(".SUCCESS").write_text("complete\n", encoding="utf-8")
    return manifest


def run(args: Any) -> int:
    if not os.environ.get("OAR_JOB_ID"):
        raise RuntimeError("GPU inference requires an active OAR allocation")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    payload = load_mapping(args.pipeline_config)
    if payload.get("training_enabled") is not False:
        raise ContractError("This pipeline must explicitly disable training")
    cohort = load_cohort(payload["cohort"])
    cohort.validate()
    model_bundle = next(
        load_model_bundle(item)
        for item in payload["models"]
        if item["strategy"] == args.strategy
    )
    model_bundle.validate()
    config = audio_core.load_yaml_config(Path(model_bundle.config))
    if list(config["classes"]) != list(anatomy_core.classes):
        raise ContractError("Model class order differs from anatomical adapter")
    with Path(config["phonemesdir"]).open("r", encoding="utf-8") as handle:
        phonemes = json.load(handle)
    model = load_model(config, Path(model_bundle.checkpoint), device)
    audio_payload = None
    if args.audio_normalization is not None:
        audio_payload = load_mapping(args.audio_normalization)
        if audio_payload.get("status") != "complete":
            raise ContractError("Audio normalization artifact is not complete")
        pairing = audio_payload.get("strict_session_pairing", {})
        if not pairing.get("same_number_and_order"):
            raise ContractError(
                "Audio normalization did not pass exact session-order pairing"
            )
    selected = set(args.speaker or [])
    sessions = [
        item
        for item in cohort.ordered_sessions()
        if not selected or item.speaker in selected
    ]
    if not sessions:
        raise ContractError("Speaker filter selected no cohort sessions")

    started = time.monotonic()
    manifests = []
    for session in sessions:
        speaker_number = int(session.speaker.removeprefix("P"))
        session_number = int(session.session.removeprefix("S"))
        raw_cache = RAW_CACHE_ROOT / session.speaker / f"{session.session}.pt"
        if not raw_cache.is_file():
            raise FileNotFoundError(raw_cache)
        inferred = infer_baseline(
            strategy=args.strategy,
            model=model,
            device=device,
            raw_cache=raw_cache,
            config=config,
            split_root=Path(model_bundle.split_root),
            phonemes=phonemes,
            batch_size=args.batch_size,
        )
        u_mask = validate_target_u(
            speaker_number,
            session_number,
            session.reference_frame,
            inferred.frame_numbers,
        )
        transform, transform_metadata = build_transform(
            speaker_number, session_number, session.reference_frame
        )
        affine, affine_tps = anatomy_core.transform_contour_batch(
            inferred.predicted_raw, transform, args.transform_frame_batch
        )
        output = (
            args.output_root.resolve()
            / args.strategy
            / session.speaker
            / session.session
            / "baseline_anatomical.npz"
        )
        audio_inferred = None
        audio_affine = None
        audio_affine_tps = None
        session_audio_metadata = None
        if audio_payload is not None:
            alpha = float(audio_payload["alphas"][session.speaker])
            rms_target = float(audio_payload["target_rms"])
            audio_inferred, alignment, extraction = infer_audio_normalized(
                strategy=args.strategy,
                model=model,
                device=device,
                raw_cache=raw_cache,
                config=config,
                split_root=Path(model_bundle.split_root),
                phonemes=phonemes,
                batch_size=args.batch_size,
                speaker_number=speaker_number,
                session_number=session_number,
                alpha=alpha,
                rms_target=rms_target,
            )
            audio_affine, audio_affine_tps = (
                anatomy_core.transform_contour_batch(
                    audio_inferred.predicted_raw,
                    transform,
                    args.transform_frame_batch,
                )
            )
            write_rows_csv(
                output.parent / "rms_vtln_chunk_alignment.csv", alignment
            )
            atomic_write_json(
                output.parent / "rms_vtln_audio_extraction.json", extraction
            )
            session_audio_metadata = {
                "alpha": alpha,
                "target_rms": rms_target,
                "fit_artifact": str(args.audio_normalization.resolve()),
                "target_contours_or_labels_used_for_audio_normalization": False,
            }
        manifest = save_pack(
            output,
            inferred=inferred,
            affine=affine,
            affine_tps=affine_tps,
            u_mask=u_mask,
            strategy=args.strategy,
            target_frame=session.reference_frame,
            coordinate_scale_mm=float(payload["evaluation"]["coordinate_scale_mm"]),
            transform_metadata=transform_metadata,
            output_coordinate_space=model_bundle.output_coordinate_space,
            audio_inferred=audio_inferred,
            audio_affine=audio_affine,
            audio_affine_tps=audio_affine_tps,
            audio_metadata=session_audio_metadata,
        )
        manifests.append(manifest)
        print(
            f"DONE {args.strategy} {session.key}: "
            f"{manifest['frame_count']} frames; "
            f"P2CP original={manifest['metrics']['original']['symmetric_p2cp_mean']:.3f}, "
            f"anatomical={manifest['metrics']['anatomical']['symmetric_p2cp_mean']:.3f} mm",
            flush=True,
        )
    summary = {
        "status": "complete",
        "strategy": args.strategy,
        "sessions": manifests,
        "training_launched": False,
        "elapsed_seconds": time.monotonic() - started,
        "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "gpu": torch.cuda.get_device_name(device),
        "model_bundle": dataclasses.asdict(model_bundle),
    }
    atomic_write_json(
        args.output_root.resolve() / args.strategy / "baseline_anatomical_summary.json",
        summary,
    )
    return 0
