#!/usr/bin/env python3
"""Validate the nine 2x3 demo videos, preserve one inventory, and clean QC files."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = (
    REPO_ROOT
    / "results/asd2_fixedbs10_selected_9sessions_textgrid_u_grid_adaptation_20260721_143350"
)
PAIRS = ((1, 16), (2, 9), (3, 14), (4, 4), (5, 6), (6, 8), (8, 2), (9, 5), (10, 14))
EXPECTED_LAYOUT = [
    ["Raw", "Affine", "Affine + TPS"],
    ["Ground truth only", "Contour slide + audio normalization", "Stage effect"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--delete-reviewed-qc", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def validate_session(result_root: Path, speaker: int, session: int) -> dict[str, Any]:
    session_dir = (
        result_root / f"phase_b/videos/P{speaker}/S{session}"
    ).resolve()
    stem = f"p{speaker}_s{session}_asd2_to_asd1_grid_adaptation_2x3_50fps"
    video = session_dir / f"{stem}.mp4"
    audit_path = session_dir / f"{stem}_audit.json"
    if not video.is_file() or not audit_path.is_file():
        raise FileNotFoundError(f"Missing video or audit for P{speaker}/S{session}")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise RuntimeError(f"Audit failed for P{speaker}/S{session}")
    if not all(audit.get("checks", {}).values()):
        raise RuntimeError(f"One or more video checks failed for P{speaker}/S{session}")
    if audit.get("layout") != EXPECTED_LAYOUT:
        raise RuntimeError(f"Unexpected panel layout for P{speaker}/S{session}")
    if audit.get("contour_convention") != "solid ground truth; dashed prediction":
        raise RuntimeError(f"Unexpected contour convention for P{speaker}/S{session}")
    if audit.get("visible_text_excludes") != ["New fixed-BS10", "integer"]:
        raise RuntimeError(f"Unexpected visible-text exclusions for P{speaker}/S{session}")
    if Path(audit["video"]).resolve() != video:
        raise RuntimeError(f"Audit points to the wrong video for P{speaker}/S{session}")
    current_hash = sha256(video)
    if current_hash != audit.get("video_sha256"):
        raise RuntimeError(f"Video hash changed after audit for P{speaker}/S{session}")

    qc_paths = [Path(value).resolve() for value in audit["visual_qc_samples"]]
    if not 3 <= len(qc_paths) <= 4:
        raise RuntimeError(f"Unexpected QC sample count for P{speaker}/S{session}")
    for qc_path in qc_paths:
        if qc_path.parent != session_dir:
            raise RuntimeError(f"QC path escapes its session directory: {qc_path}")
        if not qc_path.name.startswith(f"{stem}_qc_F") or qc_path.suffix != ".png":
            raise RuntimeError(f"Unexpected QC filename: {qc_path}")
        if not qc_path.is_file():
            raise FileNotFoundError(qc_path)

    return {
        "speaker_session": f"P{speaker}/S{session}",
        "video": str(video),
        "video_sha256": current_hash,
        "frames": int(audit["video_frames"]),
        "fps": audit["avg_frame_rate"],
        "video_codec": audit["video_codec"],
        "audio_codec": audit["audio_codec"],
        "duration_seconds": float(audit["video_duration_seconds"]),
        "av_duration_difference_seconds": float(audit["av_duration_difference_seconds"]),
        "rms_vtln_alpha_to_asd2_train": float(audit["rms_vtln_alpha_to_asd2_train"]),
        "status": "passed",
        "audit_path": str(audit_path),
        "reviewed_qc_paths": [str(path) for path in qc_paths],
    }


def main() -> None:
    args = parse_args()
    result_root = args.result_root.resolve()
    sessions = [
        validate_session(result_root, speaker, session)
        for speaker, session in PAIRS
    ]
    inventory_path = result_root / "phase_b/demo_inputs/2x3_video_inventory.json"
    inventory = {
        "status": "passed",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "visual_qc": "calibration frame reviewed for all nine sessions",
        "reviewed_qc_and_session_audits_deleted": bool(args.delete_reviewed_qc),
        "layout": EXPECTED_LAYOUT,
        "contour_convention": "solid ground truth; dashed prediction",
        "session_count": len(sessions),
        "total_frames": sum(row["frames"] for row in sessions),
        "sessions": sessions,
    }
    atomic_json(inventory_path, inventory)

    deleted: list[str] = []
    if args.delete_reviewed_qc:
        for row in sessions:
            for value in row["reviewed_qc_paths"]:
                path = Path(value)
                path.unlink()
                deleted.append(str(path))
            audit_path = Path(row["audit_path"])
            audit_path.unlink()
            deleted.append(str(audit_path))

    print(
        json.dumps(
            {
                "status": "passed",
                "inventory": str(inventory_path.resolve()),
                "session_count": len(sessions),
                "total_frames": inventory["total_frames"],
                "deleted_file_count": len(deleted),
                "deleted_files": deleted,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
