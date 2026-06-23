#!/usr/bin/env python3
"""Generate ASD1+ASD2 mixed ST-5 preprocessing configs and session manifests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import textgrid
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent

ASD1_RAW = Path(
    "/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/ArtSpeech_Database_1_raw"
)
ASD2_ROOT = Path(
    "/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/ArtSpeech_Database_2"
)
BF_INFERENCE = WORKSPACE_ROOT / "bf" / "inference"

CLASSES = [
    "arytenoid-cartilage",
    "epiglottis",
    "lower-lip",
    "pharynx",
    "soft-palate-midline",
    "tongue",
    "upper-lip",
    "vocal-folds",
]


def contour_frame_count(contour_dir: Path, classes: List[str]) -> Tuple[int, str | None]:
    if not contour_dir.is_dir():
        return 0, f"missing_contour_dir:{contour_dir}"
    counts_by_articulator = {articulator: 0 for articulator in classes}
    with os.scandir(contour_dir) as entries:
        for entry in entries:
            if not entry.name.endswith(".npy") or "_" not in entry.name:
                continue
            articulator = entry.name[:-4].split("_", 1)[1]
            if articulator in counts_by_articulator:
                counts_by_articulator[articulator] += 1
    counts = list(counts_by_articulator.values())
    min_count = min(counts) if counts else 0
    if min_count <= 1:
        return min_count, f"too_few_complete_contour_frames:{min_count}"
    return min_count, None


def textgrid_readable(path: Path) -> str | None:
    if not path.exists():
        return f"missing_textgrid:{path}"
    try:
        textgrid.TextGrid.fromFile(str(path))
    except Exception as exc:  # noqa: BLE001 - manifest should preserve the real parser error.
        return f"textgrid_parse_error:{type(exc).__name__}:{exc}"
    return None


def discover_asd1() -> Tuple[Dict[str, List[str]], List[dict]]:
    valid: Dict[str, List[str]] = {}
    skipped = []
    for speaker_dir in sorted(BF_INFERENCE.glob("P*")):
        if not speaker_dir.is_dir():
            continue
        speaker = speaker_dir.name
        sessions = []
        for session_dir in sorted(speaker_dir.glob("S*"), key=lambda p: (len(p.name), p.name)):
            session = session_dir.name
            if ".partial-" in session:
                skipped.append({"dataset_type": "asd1", "bucket": speaker, "session": session, "reason": "partial_output"})
                continue
            other_dir = ASD1_RAW / speaker / "OTHER" / session
            dcm_dir = ASD1_RAW / speaker / "DCM_2D" / session
            audio = other_dir / f"DENOISED_SOUND_{speaker}_{session}.wav"
            tg = other_dir / f"TEXT_ALIGNMENT_{speaker}_{session}.textgrid"
            contour_dir = session_dir / "contours"
            reason = None
            if not dcm_dir.is_dir():
                reason = f"missing_dcm_dir:{dcm_dir}"
            elif not audio.exists():
                reason = f"missing_audio:{audio}"
            else:
                reason = textgrid_readable(tg)
            if reason is None:
                _, reason = contour_frame_count(contour_dir, CLASSES)
            if reason:
                skipped.append({"dataset_type": "asd1", "bucket": speaker, "session": session, "reason": reason})
                continue
            sessions.append(session)
        if sessions:
            valid[speaker] = sessions
    return valid, skipped


def discover_asd2() -> Tuple[Dict[str, List[str]], List[dict]]:
    valid: Dict[str, List[str]] = {}
    skipped = []
    for bucket_dir in sorted(ASD2_ROOT.iterdir()):
        if not bucket_dir.is_dir() or not bucket_dir.name.isdigit():
            continue
        bucket = bucket_dir.name
        sessions = []
        for session_dir in sorted(bucket_dir.glob("S*"), key=lambda p: (len(p.name), p.name)):
            session = session_dir.name
            if not session_dir.is_dir():
                continue
            wavs = [
                p
                for p in sorted(session_dir.glob("*.wav"))
                if not p.name.endswith("_mocap.wav")
            ]
            contour_dir = (
                session_dir / "inference_contours_registered"
                if (session_dir / "inference_contours_registered").is_dir()
                else session_dir / "inference_contours"
            )
            reason = None
            if not wavs:
                reason = f"missing_audio:{session_dir}"
            else:
                tg = session_dir / f"{wavs[0].stem}_adjusted.textgrid"
                reason = textgrid_readable(tg)
            if reason is None:
                _, reason = contour_frame_count(contour_dir, CLASSES)
            if reason:
                skipped.append({"dataset_type": "asd2", "bucket": bucket, "session": session, "reason": reason})
                continue
            sessions.append(session)
        if sessions:
            valid[bucket] = sessions
    return valid, skipped


def base_config(name: str, cache_name: str) -> dict:
    return {
        "experiment_name": f"experiment_{name}",
        "dataset_type": "mixed",
        "datadir": str(ASD2_ROOT),
        "asd1_datadir": str(ASD1_RAW),
        "asd2_datadir": str(ASD2_ROOT),
        "asd1_annotation_dir": str(BF_INFERENCE),
        "phonemesdir": str(REPO_ROOT / "config" / "list_phonemes.json"),
        "data_save": str(REPO_ROOT),
        "folder_save": name,
        "n_epochs": 1,
        "batch_size": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "optimizer": "adam",
        "patience": 5,
        "save_every": 1,
        "num_layers": 1,
        "input_layer": 39,
        "hidden_layer": 300,
        "output_layer": 100,
        "context_window": 0,
        "sequence_length": 80,
        "added_frames": 20,
        "ms_image": 19.98,
        "n_mfcc": 13,
        "hop_length_ratio": 10,
        "window_length_ms": 25,
        "skip_ms": 400,
        "nbr_phonemes": 1,
        "phonemes": False,
        "alpha": 1,
        "loss_function": "mse_all_articulators_full_asd2_single_task5",
        "mode": "train",
        "model": name,
        "tag": name,
        "classes": CLASSES,
        "input_type": "mfcc",
        "phoneme_tier_index": 1,
        "skip_outlier_detection": True,
        "skip_test_plots": True,
        "skip_tract_variables": True,
        "use_mri": False,
        "cache_dataset": True,
        "rebuild_dataset_cache": False,
        "dataset_cache_dir": str(REPO_ROOT / "repro" / cache_name / "cache"),
        "load_labels_progress_every": 25,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }


def first_session(mapping: Dict[str, List[str]], bucket: str, index: int) -> str:
    sessions = mapping[bucket]
    if index >= len(sessions):
        raise ValueError(f"Need at least {index + 1} sessions for {bucket}, got {len(sessions)}")
    return sessions[index]


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    os.replace(tmp, path)


def main() -> None:
    asd1, skipped_asd1 = discover_asd1()
    asd2, skipped_asd2 = discover_asd2()
    if "P1" not in asd1:
        raise RuntimeError("ASD1 discovery did not find valid P1 sessions")
    if "1775" not in asd2:
        raise RuntimeError("ASD2 discovery did not find valid 1775 sessions")

    validate = base_config(
        "asd1_asd2_mixed_first3_st5_mfcc",
        "asd1_asd2_mixed_first3_st5_mfcc",
    )
    validate["max_chunks_per_session"] = 5
    validate["train_sequences"] = {
        "P1": [first_session(asd1, "P1", 0)],
        "1775": [first_session(asd2, "1775", 0)],
    }
    validate["valid_sequences"] = {
        "P1": [first_session(asd1, "P1", 1)],
        "1775": [first_session(asd2, "1775", 1)],
    }
    validate["test_sequences"] = {
        "P1": [first_session(asd1, "P1", 2)],
        "1775": [first_session(asd2, "1775", 2)],
    }
    validate["inference_sequences"] = validate["test_sequences"]

    full = base_config(
        "asd1_asd2_full_raw_st5_mfcc",
        "asd1_asd2_full_raw_st5_mfcc",
    )
    full["train_sequences"] = {**asd1, **asd2}
    full["valid_sequences"] = {}
    full["test_sequences"] = {}
    full["inference_sequences"] = {}

    config_dir = REPO_ROOT / "config" / "train_config"
    write_yaml(config_dir / "asd1_asd2_mixed_first3_st5_mfcc.yaml", validate)
    write_yaml(config_dir / "asd1_asd2_full_raw_st5_mfcc.yaml", full)

    manifest = {
        "asd1_valid_session_count": sum(len(v) for v in asd1.values()),
        "asd2_valid_session_count": sum(len(v) for v in asd2.values()),
        "asd1_valid_sessions": asd1,
        "asd2_valid_sessions": asd2,
        "skipped_sessions": skipped_asd1 + skipped_asd2,
        "validate_config": str(config_dir / "asd1_asd2_mixed_first3_st5_mfcc.yaml"),
        "full_raw_config": str(config_dir / "asd1_asd2_full_raw_st5_mfcc.yaml"),
    }
    manifest_path = REPO_ROOT / "repro" / "asd1_asd2_full_raw_st5_mfcc" / "session_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    with tmp_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp_manifest, manifest_path)
    print(json.dumps({
        "asd1_valid_session_count": manifest["asd1_valid_session_count"],
        "asd2_valid_session_count": manifest["asd2_valid_session_count"],
        "skipped_session_count": len(manifest["skipped_sessions"]),
        "validate_config": manifest["validate_config"],
        "full_raw_config": manifest["full_raw_config"],
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
