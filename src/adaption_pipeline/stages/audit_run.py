"""Audit a completed non-training ASD2-to-ASD1 adaptation run."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from ..contracts import (
    ContractError,
    atomic_write_json,
    load_cohort,
    load_model_bundle,
    sha256_file,
    utc_now,
)
from ..domain import CONDITIONS, GLOBAL, MOVING_AVERAGE, STRATEGIES
from ..io import load_mapping
from ..provenance import capture_external_repo


VIDEO_LAYOUTS = {
    "single",
    "pair",
    "global_four",
    "moving_four",
    "strategy_compare",
}


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise ContractError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def audit_predictions(
    root: Path, cohort: Any
) -> Dict[str, Any]:
    total_frames: Dict[str, int] = {strategy: 0 for strategy in STRATEGIES}
    reference_frames: Dict[str, List[int]] = {}
    reference_gt_hashes: Dict[str, str] = {}
    packs = 0
    for strategy in STRATEGIES:
        for record in cohort.ordered_sessions():
            base = root / strategy / record.speaker / record.session
            manifest = load_mapping(base / "baseline_anatomical.manifest.json")
            pack = base / "baseline_anatomical.npz"
            if manifest.get("status") != "complete":
                raise ContractError(f"Incomplete prediction: {strategy}/{record.key}")
            if manifest.get("strategy") != strategy:
                raise ContractError(
                    f"Strategy manifest mismatch: {strategy}/{record.key}"
                )
            if manifest.get("training_launched") is not False:
                raise ContractError("Prediction manifest indicates training")
            if tuple(manifest.get("conditions", ())) != CONDITIONS:
                raise ContractError(
                    f"Condition mismatch for {strategy}/{record.key}"
                )
            if not pack.is_file():
                raise ContractError(f"Missing prediction pack: {pack}")
            with np.load(pack, allow_pickle=False) as payload:
                required = {
                    "frame_numbers",
                    "strategy",
                    "ground_truth",
                    "predicted_original",
                    "predicted_anatomical",
                    "predicted_audio",
                    "predicted_anatomical_audio",
                }
                missing = required.difference(payload.files)
                if missing:
                    raise ContractError(f"{pack} missing {sorted(missing)}")
                frames = np.asarray(payload["frame_numbers"])
                if str(np.asarray(payload["strategy"]).item()) != strategy:
                    raise ContractError(f"Strategy pack mismatch: {pack}")
                if frames.ndim != 1 or not np.issubdtype(frames.dtype, np.integer):
                    raise ContractError(f"Non-integer frame timeline: {pack}")
                if len(frames) != int(manifest["frame_count"]):
                    raise ContractError(f"Frame count mismatch: {pack}")
                if len(frames) and (
                    not np.all(np.diff(frames) > 0)
                    or int(frames[0]) != int(manifest["frame_min"])
                    or int(frames[-1]) != int(manifest["frame_max"])
                ):
                    raise ContractError(f"Invalid ordered frame timeline: {pack}")
            key = record.key
            frame_list = frames.astype(int).tolist()
            gt_hash = str(manifest["ground_truth_sha256"])
            if strategy == STRATEGIES[0]:
                reference_frames[key] = frame_list
                reference_gt_hashes[key] = gt_hash
            elif (
                frame_list != reference_frames[key]
                or gt_hash != reference_gt_hashes[key]
            ):
                raise ContractError(
                    f"Strategy pairing mismatch for {record.key}"
                )
            total_frames[strategy] += len(frames)
            packs += 1
    if len(set(total_frames.values())) != 1:
        raise ContractError(f"Strategy frame totals differ: {total_frames}")
    return {
        "status": "passed",
        "prediction_packs": packs,
        "sessions_per_strategy": len(cohort.sessions),
        "frames_per_strategy": total_frames,
        "exact_frame_and_ground_truth_pairing": True,
        "conditions": list(CONDITIONS),
    }


def audit_evaluation(root: Path, expected_frames: int, session_count: int) -> Dict[str, Any]:
    report = load_mapping(root / "evaluation/evaluation.json")
    expected_primary = expected_frames - session_count
    if report.get("status") != "complete":
        raise ContractError("Evaluation is incomplete")
    if report.get("primary_metric") != "symmetric_p2cp_mm":
        raise ContractError("Primary contour metric is not symmetric P2CP")
    comparison = report.get("comparison", [])
    if len(comparison) != len(STRATEGIES) * len(CONDITIONS):
        raise ContractError("Evaluation comparison row count mismatch")
    if any(int(row["paired_frames"]) != expected_primary for row in comparison):
        raise ContractError("Evaluation paired-frame total mismatch")
    for relative in (
        "evaluation/global/evaluation_table.csv",
        "evaluation/global/evaluation_table.png",
        f"evaluation/{MOVING_AVERAGE}/evaluation_table.csv",
        f"evaluation/{MOVING_AVERAGE}/evaluation_table.png",
        "evaluation/strategy_comparison.csv",
    ):
        if not (root / relative).is_file():
            raise ContractError(f"Missing evaluation artifact: {relative}")
    return {
        "status": "passed",
        "primary_metric": report["primary_metric"],
        "secondary_metric": report["secondary_metric"],
        "metric_pairing": "within the same MRI frame",
        "paired_primary_frames_per_strategy": expected_primary,
        "calibration_frames_excluded": session_count,
    }


def audit_audio(root: Path, cohort: Any) -> Dict[str, Any]:
    report = load_mapping(
        root / "audio_normalization/audio_correlation_before_after.json"
    )
    if report.get("status") != "passed" or report.get("scope") != "audio_only":
        raise ContractError("Audio-only correlation report did not pass")
    if report.get("target_contours_or_labels_used") is not False:
        raise ContractError("Audio correlation report used contour information")
    pairing = report["strict_same_number_and_session_order"]
    expected_order = [item.speaker for item in cohort.ordered_sessions()]
    if (
        not pairing.get("passed")
        or pairing.get("speaker_order") != expected_order
        or len(report.get("per_session", [])) != len(cohort.sessions)
    ):
        raise ContractError("Audio correlation session/speaker pairing mismatch")
    return {
        "status": "passed",
        "scope": "audio_only",
        "target_contours_or_labels_used": False,
        "macro": report["macro"],
        "direction_counts": report["direction_counts"],
        "same_number_and_session_order": True,
    }


def audit_anatomy(root: Path, cohort: Any) -> Dict[str, Any]:
    report = load_mapping(
        root / "anatomical_diagnostics/anatomical_diagnostics.json"
    )
    pairs = report.get("pairs", [])
    if (
        report.get("status") != "complete"
        or report.get("layout") != "2x4"
        or len(pairs) != len(cohort.sessions)
    ):
        raise ContractError("Anatomical diagnostic summary mismatch")
    for pair in pairs:
        if pair.get("vowel") != "u" or pair.get("panel_layout") != "2x4":
            raise ContractError("Anatomical diagnostic contract mismatch")
        if not Path(pair["figure"]).is_file():
            raise ContractError(f"Missing anatomical figure: {pair['figure']}")
        for field in (
            "annotation_raw_mm",
            "annotation_affine_mm",
            "annotation_affine_tps_mm",
            "affine_control_rmse_mm",
            "tps_control_rmse_mm",
        ):
            if not np.isfinite(float(pair[field])):
                raise ContractError(f"Invalid anatomical error {field}")
    return {
        "status": "passed",
        "pairs": len(pairs),
        "figures": len(pairs),
        "layout": "2x4",
        "reference_vowel": "u",
        "errors_present": True,
    }


def audit_videos(root: Path) -> Dict[str, Any]:
    manifest = load_mapping(root / "videos/P1/S16/video_manifest.json")
    videos = manifest.get("videos", [])
    if manifest.get("status") != "complete" or len(videos) != len(VIDEO_LAYOUTS):
        raise ContractError("Video manifest is incomplete")
    layouts = {item["layout"] for item in videos}
    if layouts != VIDEO_LAYOUTS:
        raise ContractError(f"Video layout mismatch: {layouts}")
    for item in videos:
        if item.get("status") != "passed" or not all(item["checks"].values()):
            raise ContractError(f"Video audit failed: {item['layout']}")
        path = Path(item["video"])
        if not path.is_file() or sha256_file(path) != item["video_sha256"]:
            raise ContractError(f"Video hash mismatch: {path}")
    return {
        "status": "passed",
        "speaker": manifest["speaker"],
        "session": manifest["session"],
        "videos": len(videos),
        "layouts": sorted(layouts),
        "frames": manifest["frame_count"],
        "fps": manifest["fps"],
        "audio": "original target audio once",
    }


def audit_external_repositories(config: Dict[str, Any], root: Path) -> Dict[str, Any]:
    provenance = load_mapping(root / "provenance/external_repositories.json")
    expected = provenance["external_repositories"]
    current = {
        "grid_transform": capture_external_repo(
            "grid-transform", Path(config["paths"]["grid_transform_repo"])
        ),
        "audio_normalization": capture_external_repo(
            "audio-speaker-normalization",
            Path(config["paths"]["audio_normalization_repo"]),
        ),
    }
    payload: Dict[str, Any] = {}
    for key, state in current.items():
        recorded = expected[key]
        if state.head != recorded["head"] or state.dirty != recorded["dirty"]:
            raise ContractError(
                f"External repository state changed since inference: {key}"
            )
        payload[key] = {
            "path": state.path,
            "head": state.head,
            "dirty": state.dirty,
            "unchanged_since_capture": True,
        }
    return payload


def report_markdown(manifest: Dict[str, Any]) -> str:
    audio = manifest["audits"]["audio"]["macro"]
    return f"""# ASD2-to-ASD1 non-training adaptation run

Status: **{manifest['status']}**

- Branch: `{manifest['repository']['branch']}`
- OAR job: `{manifest['runtime']['oar_job_id']}`
- Training launched: **no**
- Strategies: global and moving average
- Prediction frames per strategy: {manifest['audits']['predictions']['frames_per_strategy']['global']:,}
- Primary paired frames per strategy: {manifest['audits']['evaluation']['paired_primary_frames_per_strategy']:,}

## Audio-only VTLN correlation

This measurement uses audio MFCC only; it does not use contours, P2CP, or
contour RMSE. Sessions have the same count and exact order across speakers.

- Before VTLN: `{audio['correlation_before']:.12f}`
- After VTLN: `{audio['correlation_after']:.12f}`
- Signed change: `{audio['signed_change']:+.12f}`
- Signed change percent: `{audio['signed_change_percent']:+.6f}%`
- Direction: **{audio['direction']}**

## Audited outputs

- Two strategy-specific evaluation tables plus one comparison table.
- Ten `2x4` anatomical diagnostic figures using exact `/u/` references.
- Five synchronized P1/S16 videos: `1x1`, `1x2`, global `2x2`, moving-average
  `2x2`, and strategy comparison `2x4`.
- Moving-average anatomical outputs are retained only as invalid
  double-transform diagnostics because those predictions are already in
  target-native coordinates after using target contour statistics.
"""


def run(args: Any) -> int:
    config = load_mapping(args.pipeline_config)
    if config.get("training_enabled") is not False:
        raise ContractError("Final audit refuses a training-enabled config")
    cohort = load_cohort(config["cohort"])
    cohort.validate()
    models = [load_model_bundle(item) for item in config["models"]]
    for model in models:
        model.validate(verify_files=True)
    root = args.artifact_root.resolve()
    predictions = audit_predictions(root, cohort)
    frame_total = int(predictions["frames_per_strategy"][GLOBAL])
    audits = {
        "external_repositories": audit_external_repositories(config, root),
        "predictions": predictions,
        "audio": audit_audio(root, cohort),
        "evaluation": audit_evaluation(root, frame_total, len(cohort.sessions)),
        "anatomical_diagnostics": audit_anatomy(root, cohort),
        "videos": audit_videos(root),
    }
    repository = Path(__file__).resolve().parents[3]
    manifest = {
        "schema_version": "1.0",
        "status": "passed",
        "created_at": utc_now(),
        "experiment_id": config["experiment_id"],
        "training_enabled": False,
        "training_launched": False,
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "oar_job_id": os.environ.get("OAR_JOB_ID"),
        },
        "repository": {
            "path": str(repository),
            "branch": git(repository, "branch", "--show-current"),
            "head": git(repository, "rev-parse", "HEAD"),
            "dirty": bool(git(repository, "status", "--porcelain")),
        },
        "models": [
            {
                "strategy": model.strategy,
                "checkpoint": model.checkpoint,
                "checkpoint_sha256": model.checkpoint_sha256,
                "capabilities": {
                    "uses_target_labels": model.capabilities.uses_target_labels,
                    "uses_target_statistics": (
                        model.capabilities.uses_target_statistics
                    ),
                    "causal": model.capabilities.causal,
                    "blind_inference_compatible": (
                        model.capabilities.blind_inference_compatible
                    ),
                },
                "output_coordinate_space": model.output_coordinate_space,
            }
            for model in models
        ],
        "audits": audits,
    }
    atomic_write_json(root / "RUN_MANIFEST.json", manifest)
    (root / "REPORT.md").write_text(report_markdown(manifest), encoding="utf-8")
    (root / "_SUCCESS").write_text("passed\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0
