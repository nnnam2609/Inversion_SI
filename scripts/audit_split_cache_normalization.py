#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.utils.config_validation import load_yaml_config  # noqa: E402
from src.utils.normalization import load_validated_split_cache_state  # noqa: E402
from src.utils.split_cache_overrides import SPLIT_FILES, split_cache_dir, split_filename  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit split-cache contour normalization floors before inference/rendering. "
            "This catches stale caches that can make de-normalized prediction contours look under-moving."
        )
    )
    parser.add_argument("configs", nargs="+", type=Path, help="YAML config files to audit.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLIT_FILES),
        choices=sorted(SPLIT_FILES),
        help="Split cache names to audit. Defaults to train_sequences valid_sequences test_sequences.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Report missing split cache files without failing the audit.",
    )
    parser.add_argument(
        "--allow-legacy-audio-vtln",
        action="store_true",
        help="Allow legacy audio_vtln_feature_npz configs while auditing split caches.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def resolve_workspace_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def audit_split(
    config_path: Path,
    config: dict[str, Any],
    cache_dir: Path,
    split: str,
    *,
    allow_missing: bool,
) -> dict[str, Any]:
    cache_path = cache_dir / split_filename(split)
    row: dict[str, Any] = {
        "config_path": str(config_path),
        "split": split,
        "split_cache_dir": str(cache_dir),
        "cache_path": str(cache_path),
    }
    if not cache_path.exists():
        row.update(
            {
                "status": "missing_allowed" if allow_missing else "error",
                "error": None if allow_missing else f"Missing split cache: {cache_path}",
            }
        )
        return row

    try:
        _state, summary = load_validated_split_cache_state(cache_path, config)
        row.update({"status": "ok", "error": None, **summary})
        if summary.get("cache_metadata"):
            row["cache_metadata"] = summary["cache_metadata"]
    except Exception as exc:  # noqa: BLE001 - audit should report every split failure.
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return row


def audit_config(
    config_path: Path,
    splits: list[str],
    *,
    allow_missing: bool,
    allow_legacy_audio_vtln: bool,
) -> list[dict[str, Any]]:
    try:
        config = load_yaml_config(
            config_path,
            allow_legacy_audio_vtln=allow_legacy_audio_vtln,
        )
        cache_dir = resolve_workspace_path(split_cache_dir(config))
    except Exception as exc:  # noqa: BLE001 - audit should report config failures.
        return [
            {
                "config_path": str(config_path),
                "split": None,
                "split_cache_dir": None,
                "cache_path": None,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        ]

    return [
        audit_split(
            config_path,
            config,
            cache_dir,
            split,
            allow_missing=allow_missing,
        )
        for split in splits
    ]


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for config_path in args.configs:
        rows.extend(
            audit_config(
                config_path,
                list(args.splits),
                allow_missing=bool(args.allow_missing),
                allow_legacy_audio_vtln=bool(args.allow_legacy_audio_vtln),
            )
        )

    summary = {
        "num_rows": len(rows),
        "num_ok": sum(1 for row in rows if row["status"] == "ok"),
        "num_errors": sum(1 for row in rows if row["status"] == "error"),
        "num_missing_allowed": sum(1 for row in rows if row["status"] == "missing_allowed"),
        "allow_missing": bool(args.allow_missing),
        "allow_legacy_audio_vtln": bool(args.allow_legacy_audio_vtln),
        "rows": rows,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["num_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
