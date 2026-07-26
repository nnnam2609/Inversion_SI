"""Fit RMS+VTLN and measure signed audio correlation before versus after VTLN."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from ..contracts import ContractError, atomic_write_json, load_cohort
from ..io import load_mapping, write_rows_csv
from ..metrics import (
    correlation_change,
    pearson_correlation,
    signed_correlation_change,
)
from ..pairing import validate_speaker_session_matrix
from ..runtime import (
    activate_audio_normalization_project,
)
from ..adapters.legacy_runtime import LegacyInversionAdapter


activate_audio_normalization_project()
inversion_audio = LegacyInversionAdapter()
from audio_speaker_norm.audio_normalization import (  # noqa: E402
    AudioNormConfig,
    FeatureExtractor,
)
from notebooks.audio_norm_utils import (  # noqa: E402
    apply_cmvn,
    extract_vtln_mfcc39,
    fit_cmvn,
    fit_speaker_gmms,
    sample_rows,
    score_gmm,
)


def build_index(
    model_config: Mapping[str, Any], sessions: Sequence[Any]
) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for bucket, bucket_sessions in model_config["train_sequences"].items():
        for session in bucket_sessions:
            wav, textgrid = inversion_audio.exact_asd2_audio_paths(
                str(bucket), str(session)
            )
            rows.append(
                {
                    "speaker_id": "ASD2_REFERENCE",
                    "session_id": f"{bucket}_{session}",
                    "wav_path": str(wav),
                    "textgrid_path": str(textgrid),
                }
            )
    for record in sessions:
        speaker = int(record.speaker.removeprefix("P"))
        session = int(record.session.removeprefix("S"))
        wav, textgrid = inversion_audio.exact_asd1_audio_paths(speaker, session)
        rows.append(
            {
                "speaker_id": record.speaker,
                "session_id": record.session,
                "wav_path": str(wav),
                "textgrid_path": str(textgrid),
            }
        )
    return pd.DataFrame(rows)


def vtln_features(
    payload: Mapping[str, Any],
    alpha: float,
    audio_config: AudioNormConfig,
) -> np.ndarray:
    features, _times = extract_vtln_mfcc39(
        np.asarray(payload["wav_rms"]),
        int(payload["sr"]),
        alpha=float(alpha),
        config=audio_config.to_helper_config(),
        f_high=audio_config.vtln_f_high,
    )
    count = min(len(features), int(payload["n_total_frames"]))
    mask = np.asarray(payload["speech_mask"][:count], dtype=bool)
    if not mask.any():
        mask = np.ones(count, dtype=bool)
    return np.asarray(features[:count][mask], dtype=np.float32)


def mean_absolute_speaker_indicator_correlation(
    features_by_speaker: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    """Mean |point-biserial r| using equal deterministic frame counts."""

    speakers = sorted(features_by_speaker)
    if len(speakers) < 2:
        raise ContractError("Speaker-correlation report requires at least two speakers")
    counts = {speaker: len(features_by_speaker[speaker]) for speaker in speakers}
    balanced_count = min(counts.values())
    if balanced_count < 2:
        raise ContractError("Not enough frames to balance speaker correlation")
    balanced = []
    labels = []
    for index, speaker in enumerate(speakers):
        values = np.asarray(features_by_speaker[speaker], dtype=np.float64)
        selected = np.linspace(0, len(values) - 1, balanced_count, dtype=np.int64)
        balanced.append(values[selected])
        labels.append(np.full(balanced_count, index, dtype=np.int32))
    matrix = np.vstack(balanced)
    identity = np.concatenate(labels)
    correlations = []
    per_speaker = {}
    for index, speaker in enumerate(speakers):
        indicator = (identity == index).astype(np.float64)
        values = []
        for dimension in range(matrix.shape[1]):
            feature = matrix[:, dimension]
            if np.std(feature) == 0:
                continue
            values.append(float(np.corrcoef(feature, indicator)[0, 1]))
        per_speaker[speaker] = float(np.mean(np.abs(values)))
        correlations.extend(values)
    return {
        "statistic": "mean_absolute_point_biserial_speaker_indicator_correlation",
        "value": float(np.mean(np.abs(correlations))),
        "balanced_frames_per_speaker": balanced_count,
        "total_balanced_frames": int(len(matrix)),
        "feature_dimensions": int(matrix.shape[1]),
        "speakers": speakers,
        "source_frame_counts": counts,
        "per_speaker_mean_absolute_correlation": per_speaker,
        "sampling": "deterministic evenly spaced frames; identical count per speaker",
    }


def target_reference_correlations(
    *,
    raw_by_speaker: Mapping[str, np.ndarray],
    rms_by_speaker: Mapping[str, np.ndarray],
    normalized_by_speaker: Mapping[str, np.ndarray],
    reference_raw: np.ndarray,
    reference_rms: np.ndarray,
) -> Dict[str, Any]:
    """Correlation of target feature centroid with the ASD2 reference centroid."""

    raw_reference_centroid = np.asarray(reference_raw).mean(axis=0)
    rms_reference_centroid = np.asarray(reference_rms).mean(axis=0)
    rows = []
    for speaker in sorted(raw_by_speaker):
        raw_value = pearson_correlation(
            np.asarray(raw_by_speaker[speaker]).mean(axis=0),
            raw_reference_centroid,
        )
        rms_value = pearson_correlation(
            np.asarray(rms_by_speaker[speaker]).mean(axis=0),
            rms_reference_centroid,
        )
        normalized_value = pearson_correlation(
            np.asarray(normalized_by_speaker[speaker]).mean(axis=0),
            rms_reference_centroid,
        )
        rows.append(
            {
                "speaker": speaker,
                "raw_to_raw_asd2_reference": raw_value,
                "rms_to_rms_asd2_reference": rms_value,
                "rms_vtln_to_rms_asd2_reference": normalized_value,
                "vtln_signed_change": normalized_value - rms_value,
                "vtln_signed_change_percent": (
                    None if rms_value == 0 else
                    100.0 * (normalized_value - rms_value) / abs(rms_value)
                ),
            }
        )
    macro_raw = float(
        np.mean([row["raw_to_raw_asd2_reference"] for row in rows])
    )
    macro_rms = float(
        np.mean([row["rms_to_rms_asd2_reference"] for row in rows])
    )
    macro_normalized = float(
        np.mean([row["rms_vtln_to_rms_asd2_reference"] for row in rows])
    )
    return {
        "primary_measurement": "target_to_asd2_reference_correlation_change",
        "feature_summary": "Pearson r between 39-D session and reference centroids",
        "reference_population": "ASD2 model training sessions only",
        "per_speaker": rows,
        "macro": {
            "raw_to_raw_asd2_reference": macro_raw,
            "rms_to_rms_asd2_reference": macro_rms,
            "rms_vtln_to_rms_asd2_reference": macro_normalized,
            "vtln_signed_change_after_rms": signed_correlation_change(
                macro_rms, macro_normalized
            ),
        },
        "primary_objective": (
            "measure_signed_target_to_asd2_reference_correlation_change"
        ),
    }


def run(args: Any) -> int:
    started = time.monotonic()
    pipeline = load_mapping(args.pipeline_config)
    if pipeline.get("training_enabled") is not False:
        raise ContractError("Audio normalization pipeline must not enable training")
    cohort = load_cohort(pipeline["cohort"])
    cohort.validate()
    sessions = cohort.ordered_sessions()
    by_speaker = {
        speaker: [item for item in sessions if item.speaker == speaker]
        for speaker in sorted({item.speaker for item in sessions})
    }
    pairing_slots = validate_speaker_session_matrix(by_speaker)

    global_model = next(
        item for item in pipeline["models"] if item["strategy"] == "global"
    )
    model_config = inversion_audio.load_yaml_config(Path(global_model["config"]))
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audio_config = AudioNormConfig(
        data_root=Path(model_config["datadir"]),
        output_root=output,
        mode="full",
        target_rms=float(pipeline["audio_normalization"]["rms_target"]),
        gmm_components=args.gmm_components,
        gmm_max_iter=args.gmm_max_iter,
        max_frames_per_speaker=args.max_reference_frames,
        alpha_min=0.80,
        alpha_max=1.20,
        alpha_step=0.025,
        vtln_f_low=60.0,
        vtln_f_high=3200.0,
        target_mode="ASD2_REFERENCE",
        save_intermediate_features=False,
        save_models=False,
        save_figures=False,
    )
    index = build_index(model_config, sessions)
    train_count = int((index["speaker_id"] == "ASD2_REFERENCE").sum())
    expected_train_count = sum(
        len(items) for items in model_config["train_sequences"].values()
    )
    if train_count != expected_train_count:
        raise ContractError(
            f"ASD2 audio reference count mismatch: {train_count} != {expected_train_count}"
        )
    index.to_csv(output / "data_index.csv", index=False)
    extractor = FeatureExtractor(audio_config)
    payloads = extractor.extract(index)
    reference_payloads = [
        item for item in payloads if item["speaker"] == "ASD2_REFERENCE"
    ]
    target_payloads = [
        item for item in payloads if item["speaker"] != "ASD2_REFERENCE"
    ]
    if len(target_payloads) != len(sessions):
        raise ContractError(
            f"Expected {len(sessions)} target audio payloads, got {len(target_payloads)}"
        )

    reference_raw = np.vstack(
        [np.asarray(item["raw_mfcc39"], dtype=np.float32) for item in reference_payloads]
    )
    reference = np.vstack(
        [np.asarray(item["rms_mfcc39"], dtype=np.float32) for item in reference_payloads]
    )
    global_mean, global_std = fit_cmvn(reference)
    reference_normalized = apply_cmvn(reference, global_mean, global_std)
    reference_sample, _indices = sample_rows(
        reference_normalized,
        min(args.max_reference_frames, len(reference_normalized)),
        audio_config.random_state,
    )
    gmms, _table = fit_speaker_gmms(
        {"ASD2_REFERENCE": reference_sample},
        audio_config.to_helper_config(),
        model_dir=None,
    )
    target_gmm = gmms["ASD2_REFERENCE"]

    curve_rows: List[Dict[str, Any]] = []
    alpha_rows: List[Dict[str, Any]] = []
    alphas: Dict[str, float] = {}
    raw_by_speaker: Dict[str, np.ndarray] = {}
    rms_by_speaker: Dict[str, np.ndarray] = {}
    normalized_by_speaker: Dict[str, np.ndarray] = {}
    for payload in target_payloads:
        speaker = str(payload["speaker"])
        session = str(payload["session"])
        raw_by_speaker[speaker] = np.asarray(payload["raw_mfcc39"], dtype=np.float32)
        rms_by_speaker[speaker] = np.asarray(payload["rms_mfcc39"], dtype=np.float32)
        best_alpha = 1.0
        best_score = -np.inf
        base_score = float("nan")
        for alpha in audio_config.alpha_grid:
            features = vtln_features(payload, float(alpha), audio_config)
            sampled, _indices = sample_rows(
                features,
                min(20_000, len(features)),
                audio_config.random_state,
            )
            normalized = apply_cmvn(sampled, global_mean, global_std)
            score = float(score_gmm(target_gmm, normalized))
            curve_rows.append(
                {
                    "speaker": speaker,
                    "session": session,
                    "session_order": 0,
                    "alpha": float(alpha),
                    "score": score,
                    "frames": int(len(sampled)),
                }
            )
            if np.isclose(alpha, 1.0):
                base_score = score
            if score > best_score:
                best_score = score
                best_alpha = float(alpha)
        alphas[speaker] = best_alpha
        normalized_by_speaker[speaker] = vtln_features(
            payload, best_alpha, audio_config
        )
        alpha_rows.append(
            {
                "speaker": speaker,
                "session": session,
                "session_order": 0,
                "alpha_best": best_alpha,
                "score_alpha_1": base_score,
                "score_best": best_score,
                "score_gain": best_score - base_score,
            }
        )
        print(
            f"VTLN {speaker}/{session}: alpha={best_alpha:.3f}, "
            f"score_gain={best_score - base_score:+.5f}",
            flush=True,
        )

    raw_correlation = mean_absolute_speaker_indicator_correlation(raw_by_speaker)
    rms_correlation = mean_absolute_speaker_indicator_correlation(rms_by_speaker)
    normalized_correlation = mean_absolute_speaker_indicator_correlation(
        normalized_by_speaker
    )
    target_reference = target_reference_correlations(
        raw_by_speaker=raw_by_speaker,
        rms_by_speaker=rms_by_speaker,
        normalized_by_speaker=normalized_by_speaker,
        reference_raw=reference_raw,
        reference_rms=reference,
    )
    report = {
        "status": "complete",
        "method": "RMS plus VTLN alpha search against ASD2 training audio GMM",
        "target_reference": "ASD2 training sessions from the selected model config",
        "target_reference_session_count": train_count,
        "target_contours_or_labels_used": False,
        "target_rms": audio_config.target_rms,
        "alphas": alphas,
        "strict_session_pairing": {
            "passed": True,
            "same_number_and_order": True,
            "comparison_slots": pairing_slots,
            "sessions_per_speaker": len(pairing_slots),
        },
        "target_reference_correlation": target_reference,
        "secondary_speaker_identity_leakage": {
            "raw": raw_correlation,
            "rms_only": rms_correlation,
            "rms_vtln": normalized_correlation,
            "raw_to_rms_vtln": correlation_change(
                raw_correlation["value"], normalized_correlation["value"]
            ),
            "rms_to_rms_vtln": correlation_change(
                rms_correlation["value"], normalized_correlation["value"]
            ),
            "interpretation": (
                "Positive reduction means less linear speaker-identity leakage. "
                "The statistic uses one-vs-rest speaker indicators and balanced "
                "frame counts; it does not ordinally encode speaker IDs."
            ),
        },
        "elapsed_seconds": time.monotonic() - started,
        "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "training_launched": False,
    }
    write_rows_csv(output / "alpha_curves.csv", curve_rows)
    write_rows_csv(output / "alpha_summary.csv", alpha_rows)
    atomic_write_json(output / "audio_normalization.json", report)
    print(json.dumps(report["target_reference_correlation"], indent=2), flush=True)
    return 0
