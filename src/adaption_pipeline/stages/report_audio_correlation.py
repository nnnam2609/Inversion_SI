"""Create the contour-independent before/after VTLN audio correlation report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..contracts import ContractError, atomic_write_json, load_cohort
from ..io import load_mapping, write_rows_csv
from ..metrics import signed_correlation_change
from ..pairing import validate_speaker_session_matrix


def run(args: Any) -> int:
    pipeline = load_mapping(args.pipeline_config)
    if pipeline.get("training_enabled") is not False:
        raise ContractError("Audio reporting must not enable training")
    cohort = load_cohort(pipeline["cohort"])
    cohort.validate()
    sessions = cohort.ordered_sessions()
    by_speaker = {
        speaker: [item for item in sessions if item.speaker == speaker]
        for speaker in dict.fromkeys(item.speaker for item in sessions)
    }
    comparison_slots = validate_speaker_session_matrix(by_speaker)
    if any(len(items) != 1 for items in by_speaker.values()):
        raise ContractError(
            "This report requires one selected session per speaker; "
            "multi-session correlation must be emitted per session upstream"
        )

    source = load_mapping(args.audio_normalization)
    if source.get("target_contours_or_labels_used") is not False:
        raise ContractError("Audio correlation must not consume target contours")
    pairing = source.get("strict_session_pairing", {})
    if not (
        pairing.get("passed")
        and pairing.get("same_number_and_order")
        and pairing.get("sessions_per_speaker") == len(comparison_slots)
    ):
        raise ContractError("Upstream audio session-pairing gate did not pass")

    target_reference = source["target_reference_correlation"]
    source_rows = {
        str(item["speaker"]): item for item in target_reference["per_speaker"]
    }
    expected_speakers = list(by_speaker)
    if set(source_rows) != set(expected_speakers):
        raise ContractError(
            "Audio correlation speaker mismatch: "
            f"expected={expected_speakers}, actual={sorted(source_rows)}"
        )

    rows: List[Dict[str, Any]] = []
    for record in sessions:
        source_row = source_rows[record.speaker]
        before = float(source_row["rms_to_rms_asd2_reference"])
        after = float(source_row["rms_vtln_to_rms_asd2_reference"])
        change = signed_correlation_change(before, after)
        rows.append(
            {
                "speaker_order": record.speaker_order,
                "session_order": record.session_order,
                "speaker": record.speaker,
                "session": record.session,
                "correlation_before_vtln": before,
                "correlation_after_vtln": after,
                "signed_change": change["signed_change"],
                "signed_change_percent": change["signed_change_percent"],
                "direction": change["direction"],
            }
        )

    macro_source = target_reference["macro"]
    macro_before = float(macro_source["rms_to_rms_asd2_reference"])
    macro_after = float(macro_source["rms_vtln_to_rms_asd2_reference"])
    macro_change = signed_correlation_change(macro_before, macro_after)
    directions = {
        direction: sum(row["direction"] == direction for row in rows)
        for direction in ("increased", "decreased", "unchanged")
    }
    report = {
        "status": "passed",
        "scope": "audio_only",
        "target_contours_or_labels_used": False,
        "comparison": "RMS audio before VTLN versus RMS+VTLN audio after",
        "feature": target_reference["feature_summary"],
        "reference_population": target_reference["reference_population"],
        "reference_session_count": source["target_reference_session_count"],
        "strict_same_number_and_session_order": {
            "passed": True,
            "sessions_per_speaker": len(comparison_slots),
            "comparison_slots": comparison_slots,
            "speaker_order": expected_speakers,
        },
        "macro": macro_change,
        "direction_counts": directions,
        "per_session": rows,
        "interpretation": (
            "The signed change is after minus before. Positive means the "
            "audio correlation with ASD2 increased after VTLN; negative means "
            "it decreased. No contour, contour prediction, P2CP, or contour "
            "RMSE is used in this report."
        ),
        "training_launched": False,
    }
    write_rows_csv(args.output_csv, rows)
    atomic_write_json(args.output_json, report)
    print(json.dumps(report, indent=2), flush=True)
    return 0
