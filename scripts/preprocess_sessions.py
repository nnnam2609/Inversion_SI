#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from preprocessing.session_cache import (  # noqa: E402
    build_raw_session,
    contour_pack_path,
    iter_raw_jobs,
    load_config,
    parse_session_filter,
    raw_session_part_path,
    write_metadata,
)
from src.utils.config_validation import load_yaml_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-session .pt/.npz preprocessing cache from YAML.")
    parser.add_argument("--config", type=Path, required=True, help="Preprocess YAML.")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild raw .pt session caches.")
    parser.add_argument("--rebuild-contour-packs", action="store_true", help="Rebuild per-session .npz packs.")
    parser.add_argument("--only-sessions", nargs="*", default=None, metavar="BUCKET/SESSION")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first failed session.")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    return load_yaml_config(path)


def merge_base_config(config: Dict[str, Any]) -> Dict[str, Any]:
    base_path = config.get("base_config")
    if not base_path:
        return config
    path = Path(base_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    merged = load_config(path)
    merged.update({key: value for key, value in config.items() if key != "base_config"})
    return merged


def normalize_preprocess_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(config)
    sessions = normalized.get("sessions")
    if sessions is None and normalized.get("sessions_from_base_splits", False):
        sessions = {}
        for split_key in ("train_sequences", "valid_sequences", "test_sequences"):
            for bucket, split_sessions in normalized.get(split_key, {}).items():
                bucket_key = str(bucket)
                sessions.setdefault(bucket_key, [])
                for session in split_sessions:
                    session_name = str(session)
                    if session_name not in sessions[bucket_key]:
                        sessions[bucket_key].append(session_name)
    if not isinstance(sessions, dict) or not sessions:
        raise ValueError("Preprocess config must define non-empty sessions: {bucket: [S..]}")
    normalized["train_sequences"] = sessions
    normalized["contour_pack_format"] = "npz"
    normalized["rebuild_contour_packs"] = bool(normalized.get("rebuild_contour_packs", False))
    normalized["cache_dataset"] = False
    return normalized


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = normalize_preprocess_config(merge_base_config(load_yaml(config_path)))
    if args.rebuild_contour_packs:
        config["rebuild_contour_packs"] = True
    cache_dir_value = config.get("session_cache_dir")
    if not cache_dir_value:
        raise KeyError("Preprocess config must define session_cache_dir")
    cache_dir = Path(cache_dir_value).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress_every = int(config.get("load_labels_progress_every", 25))
    only_sessions = parse_session_filter(args.only_sessions)

    started_at = time.time()
    raw_jobs = list(iter_raw_jobs(config, cache_dir, ("train_sequences",), progress_every))
    if only_sessions is not None:
        raw_jobs = [
            job for job in raw_jobs
            if (job["bucket"], job["session"]) in only_sessions
        ]
    to_build = [
        job for job in raw_jobs
        if args.rebuild or not Path(job["part_path"]).exists()
    ]
    workers = int(args.max_workers or config.get("max_workers", 1))
    workers = max(1, min(workers, len(to_build) or 1))

    print(f"config={config_path}", flush=True)
    print(f"session_cache_dir={cache_dir}", flush=True)
    print(f"declared_sessions={len(raw_jobs)} to_build={len(to_build)} workers={workers}", flush=True)

    successes = []
    failures = []
    if to_build:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_job = {
                executor.submit(build_raw_session, **job): job
                for job in to_build
            }
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    successes.append(future.result())
                except Exception as exc:
                    failure = {
                        "bucket": job["bucket"],
                        "session": job["session"],
                        "part_path": job["part_path"],
                        "contour_npz_path": job["contour_pack_path"],
                        "error": repr(exc),
                    }
                    failures.append(failure)
                    print(f"[failed] {job['bucket']}/{job['session']}: {exc!r}", file=sys.stderr, flush=True)
                    if args.fail_fast:
                        raise

    missing_after = []
    for job in raw_jobs:
        raw_path = raw_session_part_path(cache_dir, config, job["bucket"], job["session"])
        npz_path = contour_pack_path(cache_dir, config, job["bucket"], job["session"])
        if not raw_path.exists() or not npz_path.exists():
            missing_after.append(
                {
                    "bucket": job["bucket"],
                    "session": job["session"],
                    "raw_session_path": str(raw_path),
                    "contour_npz_path": str(npz_path),
                    "raw_exists": raw_path.exists(),
                    "npz_exists": npz_path.exists(),
                }
            )

    report = {
        "config": str(config_path),
        "session_cache_dir": str(cache_dir),
        "declared_sessions": len(raw_jobs),
        "built_sessions": len(successes),
        "failed_sessions": len(failures),
        "missing_after_preprocess": len(missing_after),
        "successes": successes,
        "failures": failures,
        "missing": missing_after,
        "elapsed_seconds": time.time() - started_at,
    }
    report_path = cache_dir / f"preprocess_report_{config_path.stem}.json"
    write_metadata(report_path, report)
    print(
        "preprocess_summary "
        f"declared={report['declared_sessions']} built={report['built_sessions']} "
        f"failed={report['failed_sessions']} missing={report['missing_after_preprocess']} "
        f"report={report_path}",
        flush=True,
    )
    if failures or missing_after:
        sys.exit(2)


if __name__ == "__main__":
    main()
