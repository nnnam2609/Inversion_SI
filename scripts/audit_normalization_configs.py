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

from src.utils.audio_vtln import LEGACY_AUDIO_VTLN_CONFIG_KEY  # noqa: E402
from src.utils.config_validation import load_yaml_mapping, validate_runtime_config  # noqa: E402


DEFAULT_PATHS = (
    REPO_ROOT / "config/train_config",
    REPO_ROOT / "config/inference_config",
    REPO_ROOT / "config/preprocess_config",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit YAML configs for contour std-floor and legacy audio-VTLN guard violations."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=list(DEFAULT_PATHS),
        help="YAML files or directories to audit. Defaults to config/*_config directories.",
    )
    parser.add_argument(
        "--allow-legacy-audio-vtln",
        action="store_true",
        help="Inventory legacy exported-NPZ audio-VTLN configs instead of treating them as errors.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def iter_yaml_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.yaml")))
            files.extend(sorted(path.rglob("*.yml")))
        elif path.suffix.lower() in {".yaml", ".yml"}:
            files.append(path)
    return sorted(dict.fromkeys(files))


def audit_file(path: Path, allow_legacy_audio_vtln: bool) -> dict[str, Any]:
    precheck: dict[str, Any] = {}
    try:
        config = load_yaml_mapping(path)
        legacy_value = config.get(LEGACY_AUDIO_VTLN_CONFIG_KEY)
        precheck = {
            "legacy_audio_vtln": legacy_value is not None,
            "legacy_audio_vtln_feature_npz": legacy_value,
        }
        validation = validate_runtime_config(
            config,
            path,
            allow_legacy_audio_vtln=allow_legacy_audio_vtln,
        )
        status = "ok"
        error = None
    except Exception as exc:  # noqa: BLE001 - audit reports all config failures.
        validation = {}
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    return {
        "path": str(path),
        "status": status,
        "error": error,
        **precheck,
        **validation,
    }


def main() -> None:
    args = parse_args()
    files = iter_yaml_files(list(args.paths))
    rows = [audit_file(path, bool(args.allow_legacy_audio_vtln)) for path in files]
    summary = {
        "num_files": len(rows),
        "num_ok": sum(1 for row in rows if row["status"] == "ok"),
        "num_errors": sum(1 for row in rows if row["status"] == "error"),
        "num_legacy_audio_vtln": sum(1 for row in rows if bool(row.get("legacy_audio_vtln", False))),
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
