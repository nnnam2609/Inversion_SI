#!/usr/bin/env python3
"""Refresh only ASD2 incisor labels in reusable 11-contour session caches.

The source feature cache is immutable. Classes 0..8 and every feature/timeline
payload are copied exactly; classes 9 and 10 are replaced from the generated
integer-frame VTLN incisor contours. Fractional N.5 rows are reconstructed as
the float32 arithmetic mean of integer frames N and N+1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_CONFIG = REPO_ROOT / "config/train_config/asd2_11contour_full_preprocessed_paper_st5_mfcc_500epoch.yaml"
DEFAULT_SOURCE_CACHE = REPO_ROOT / "cache"
DEFAULT_TARGET_CACHE = REPO_ROOT / "cache_variants/asd2_11_vtln_20260719"
DEFAULT_INCISOR_ROOT = WORKSPACE_ROOT / "bf/inference"
SPLITS = ("train_sequences", "valid_sequences", "test_sequences")
EXPECTED_CLASSES = (
    "arytenoid-cartilage",
    "epiglottis",
    "lower-lip",
    "pharynx",
    "soft-palate-midline",
    "tongue",
    "upper-lip",
    "vocal-folds",
    "thyroid-cartilage",
    "lower-incisor",
    "upper-incisor",
)
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--target-cache", type=Path, default=DEFAULT_TARGET_CACHE)
    parser.add_argument("--incisor-root", type=Path, default=DEFAULT_INCISOR_ROOT)
    parser.add_argument(
        "--variant-name",
        default=None,
        help="Provenance name stored in every cache payload; defaults to target-cache basename.",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--validate-workers", type=int, default=8)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--verify-bf",
        action="store_true",
        help="During validation, reread every direct incisor input and compare it with the packaged NPZ.",
    )
    parser.add_argument("--only-sessions", nargs="*", default=None, metavar="BUCKET/SESSION")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    classes = tuple(config.get("classes", ()))
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected class order: {classes}")
    return config


def iter_sessions(config: dict[str, Any]) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for split in SPLITS:
        for bucket, sessions in config.get(split, {}).items():
            for session in sessions:
                key = (str(bucket), str(session))
                if key in seen:
                    raise ValueError(f"Duplicate session across splits: {key[0]}/{key[1]}")
                seen.add(key)
                jobs.append({"split": split, "bucket": key[0], "session": key[1]})
    split_counts = {split: sum(job["split"] == split for job in jobs) for split in SPLITS}
    empty_splits = [split for split, count in split_counts.items() if count == 0]
    if empty_splits:
        raise ValueError(f"Config has empty dataset splits: {empty_splits}")
    return jobs


def expected_split_sessions(config: dict[str, Any]) -> dict[str, int]:
    return {
        split: sum(len(sessions) for sessions in config.get(split, {}).values())
        for split in SPLITS
    }


def parse_session_filter(values: list[str] | None) -> set[tuple[str, str]] | None:
    if not values:
        return None
    selected: set[tuple[str, str]] = set()
    for value in values:
        if "/" not in value:
            raise ValueError(f"Session must be BUCKET/SESSION, got {value}")
        bucket, session = value.split("/", 1)
        selected.add((bucket, session))
    return selected


def raw_path(root: Path, bucket: str, session: str) -> Path:
    return root / "raw_sessions/asd2" / bucket / f"{session}.pt"


def npz_path(root: Path, bucket: str, session: str) -> Path:
    return root / "raw_contour_npz/asd2" / bucket / f"{session}.npz"


def marker_path(root: Path, bucket: str, session: str) -> Path:
    return root / "markers" / bucket / f"{session}.json"


def atomic_json_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_npz_save(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as pack:
        return {key: np.array(pack[key], copy=True) for key in pack.files}


def validate_direct_contour(path: Path) -> np.ndarray:
    contour = np.load(path, allow_pickle=False)
    if contour.shape != (50, 2):
        raise ValueError(f"Expected (50,2) contour at {path}, got {contour.shape}")
    if contour.dtype != np.float32:
        raise TypeError(f"Expected float32 contour at {path}, got {contour.dtype}")
    if not np.isfinite(contour).all():
        raise ValueError(f"Non-finite contour at {path}")
    if np.array_equal(contour[0], contour[-1]):
        raise ValueError(f"Closed contour is not allowed: {path}")
    return contour


def frame_kind(value: float) -> tuple[str, int, int | None]:
    rounded = int(round(float(value)))
    if abs(float(value) - rounded) <= 1e-6:
        return "integer", rounded, None
    lower = int(np.floor(float(value)))
    if abs(float(value) - (lower + 0.5)) <= 1e-6:
        return "half", lower, lower + 1
    raise ValueError(f"Unsupported frame timestamp {value}; expected integer or N.5")


def required_frames(raw: dict[str, Any]) -> set[int]:
    result: set[int] = set()
    for frame_chunk in raw["frames"]:
        frame_array = np.asarray(frame_chunk)
        if frame_array.ndim != 2 or frame_array.shape[1] < 3:
            raise ValueError(f"Unexpected frame chunk shape {frame_array.shape}")
        for value in frame_array[:, 2]:
            _kind, lower, upper = frame_kind(float(value))
            result.add(lower)
            if upper is not None:
                result.add(upper)
    return result


def load_direct_incisors(
    incisor_root: Path,
    bucket: str,
    session: str,
    frames: Iterable[int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    contour_dir = incisor_root / bucket / session / "inference_contours"
    if not contour_dir.is_dir():
        raise FileNotFoundError(f"Missing incisor directory: {contour_dir}")
    loaded: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for frame in sorted(set(int(value) for value in frames)):
        lower = validate_direct_contour(contour_dir / f"{frame:04d}_lower-incisor.npy")
        upper = validate_direct_contour(contour_dir / f"{frame:04d}_upper-incisor.npy")
        loaded[frame] = (lower.reshape(100), upper.reshape(100))
    return loaded


def expected_incisor_row(
    frame_value: float,
    direct: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    kind, lower_frame, upper_frame = frame_kind(frame_value)
    if kind == "integer":
        return direct[lower_frame]
    lower_a, upper_a = direct[lower_frame]
    lower_b, upper_b = direct[int(upper_frame)]
    lower_mean = np.mean(np.stack((lower_a, lower_b)), axis=0, dtype=np.float32)
    upper_mean = np.mean(np.stack((upper_a, upper_b)), axis=0, dtype=np.float32)
    return lower_mean, upper_mean


def array_digest(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def marker_is_current(
    target_root: Path,
    bucket: str,
    session: str,
    variant_name: str,
) -> bool:
    marker = marker_path(target_root, bucket, session)
    target_raw = raw_path(target_root, bucket, session)
    target_npz = npz_path(target_root, bucket, session)
    if not marker.is_file() or not target_raw.is_file() or not target_npz.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        data.get("status") == "complete"
        and data.get("schema_version") == SCHEMA_VERSION
        and data.get("variant_name") == variant_name
        and data.get("output_raw_size") == target_raw.stat().st_size
        and data.get("output_npz_size") == target_npz.stat().st_size
    )


def build_session(job: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    source_root = Path(job["source_root"])
    target_root = Path(job["target_root"])
    incisor_root = Path(job["incisor_root"])
    variant_name = str(job["variant_name"])
    split, bucket, session = job["split"], job["bucket"], job["session"]
    if not job["rebuild"] and marker_is_current(
        target_root,
        bucket,
        session,
        variant_name,
    ):
        return {"split": split, "bucket": bucket, "session": session, "status": "skipped_complete"}

    source_raw_path = raw_path(source_root, bucket, session)
    source_npz_path = npz_path(source_root, bucket, session)
    if not source_raw_path.is_file() or not source_npz_path.is_file():
        raise FileNotFoundError(f"Missing source cache for {bucket}/{session}")
    payload = torch.load(source_raw_path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("raw"), dict):
        raise TypeError(f"Invalid raw session payload: {source_raw_path}")
    raw = payload["raw"]
    for key in ("features", "contours", "frames", "phonemes", "length_datas"):
        if key not in raw:
            raise KeyError(f"Missing raw key {key}: {source_raw_path}")
    source_arrays = load_npz_arrays(source_npz_path)
    frame_numbers = source_arrays["frame_numbers"]
    source_npz_contours = source_arrays["contours"]
    articulators = tuple(str(value) for value in source_arrays["articulators"].tolist())
    if articulators != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected NPZ class order for {bucket}/{session}: {articulators}")
    if frame_numbers.dtype != np.int32 or source_npz_contours.dtype != np.float32:
        raise TypeError(f"Unexpected source NPZ dtype for {bucket}/{session}")
    if source_npz_contours.shape != (len(frame_numbers), 11, 100):
        raise ValueError(f"Unexpected source NPZ contour shape {source_npz_contours.shape}")

    raw_required_frames = required_frames(raw)
    npz_frames = {int(value) for value in frame_numbers.tolist()}
    if raw_required_frames != npz_frames:
        missing = sorted(raw_required_frames - npz_frames)[:20]
        extra = sorted(npz_frames - raw_required_frames)[:20]
        raise ValueError(
            f"Raw/NPZ frame inventory mismatch for {bucket}/{session}: missing={missing} extra={extra}"
        )
    direct = load_direct_incisors(incisor_root, bucket, session, raw_required_frames)

    new_npz_contours = np.array(source_npz_contours, copy=True)
    for index, frame in enumerate(frame_numbers.tolist()):
        lower, upper = direct[int(frame)]
        new_npz_contours[index, 9] = lower
        new_npz_contours[index, 10] = upper
    if not np.array_equal(new_npz_contours[:, :9], source_npz_contours[:, :9]):
        raise AssertionError(f"Classes 0..8 changed while building {bucket}/{session}")

    new_raw_contours: list[np.ndarray] = []
    changed_rows = 0
    half_rows = 0
    integer_rows = 0
    for old_chunk, frame_chunk in zip(raw["contours"], raw["frames"]):
        old_chunk = np.asarray(old_chunk)
        frame_chunk = np.asarray(frame_chunk)
        if old_chunk.dtype != np.float32 or old_chunk.shape != (len(frame_chunk), 11, 100):
            raise ValueError(f"Unexpected raw contour chunk for {bucket}/{session}: {old_chunk.shape}")
        new_chunk = np.array(old_chunk, copy=True)
        for row, frame_value in enumerate(frame_chunk[:, 2]):
            kind, _lower_frame, _upper_frame = frame_kind(float(frame_value))
            lower, upper = expected_incisor_row(float(frame_value), direct)
            new_chunk[row, 9] = lower
            new_chunk[row, 10] = upper
            integer_rows += int(kind == "integer")
            half_rows += int(kind == "half")
            changed_rows += int(
                not np.array_equal(old_chunk[row, 9:11], new_chunk[row, 9:11])
            )
        if not np.array_equal(new_chunk[:, :9], old_chunk[:, :9]):
            raise AssertionError(f"Raw classes 0..8 changed while building {bucket}/{session}")
        if new_chunk.dtype != np.float32 or not np.isfinite(new_chunk).all():
            raise ValueError(f"Invalid output raw contour chunk for {bucket}/{session}")
        new_raw_contours.append(new_chunk)

    new_raw = dict(raw)
    new_raw["contours"] = new_raw_contours
    target_payload = dict(payload)
    target_payload["raw"] = new_raw
    target_payload["variant"] = {
        "schema_version": SCHEMA_VERSION,
        "name": variant_name,
        "source_cache": str(source_root.resolve()),
        "incisor_root": str(incisor_root.resolve()),
        "replaced_class_indices": [9, 10],
        "fractional_policy": "N.5=float32_mean(N,N+1)",
    }
    target_arrays = dict(source_arrays)
    target_arrays["contours"] = new_npz_contours
    target_arrays["incisor_source_root"] = np.asarray(str(incisor_root.resolve()))
    target_arrays["source_cache_npz"] = np.asarray(str(source_npz_path.resolve()))
    target_arrays["variant"] = np.asarray(variant_name)
    target_arrays["fractional_policy"] = np.asarray("N.5=float32_mean(N,N+1)")

    target_raw_path = raw_path(target_root, bucket, session)
    target_npz_path = npz_path(target_root, bucket, session)
    atomic_npz_save(target_npz_path, target_arrays)
    atomic_torch_save(target_raw_path, target_payload)
    marker = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "variant_name": variant_name,
        "split": split,
        "bucket": bucket,
        "session": session,
        "source_raw": str(source_raw_path.resolve()),
        "source_npz": str(source_npz_path.resolve()),
        "target_raw": str(target_raw_path.resolve()),
        "target_npz": str(target_npz_path.resolve()),
        "num_chunks": len(new_raw_contours),
        "num_integer_rows": integer_rows,
        "num_half_rows": half_rows,
        "num_changed_incisor_rows": changed_rows,
        "num_direct_frames": len(frame_numbers),
        "num_incisor_input_files": len(frame_numbers) * 2,
        "npz_incisor_sha256": array_digest(new_npz_contours[:, 9:11]),
        "output_raw_size": target_raw_path.stat().st_size,
        "output_npz_size": target_npz_path.stat().st_size,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json_save(marker_path(target_root, bucket, session), marker)
    return {**marker, "status": "built"}


def assert_array_lists_equal(
    source: list[Any],
    target: list[Any],
    label: str,
) -> None:
    if len(source) != len(target):
        raise AssertionError(f"{label} list length changed: {len(source)} != {len(target)}")
    for index, (source_item, target_item) in enumerate(zip(source, target)):
        if not np.array_equal(np.asarray(source_item), np.asarray(target_item)):
            raise AssertionError(f"{label}[{index}] changed")


def validate_session(job: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    source_root = Path(job["source_root"])
    target_root = Path(job["target_root"])
    incisor_root = Path(job["incisor_root"])
    split, bucket, session = job["split"], job["bucket"], job["session"]
    source_payload = torch.load(raw_path(source_root, bucket, session), map_location="cpu")
    target_payload = torch.load(raw_path(target_root, bucket, session), map_location="cpu")
    source_raw = source_payload["raw"]
    target_raw = target_payload["raw"]
    for key in ("features", "frames", "phonemes"):
        assert_array_lists_equal(source_raw[key], target_raw[key], key)
    if source_raw["length_datas"] != target_raw["length_datas"]:
        raise AssertionError(f"length_datas changed for {bucket}/{session}")
    if len(source_raw["contours"]) != len(target_raw["contours"]):
        raise AssertionError(f"Chunk count changed for {bucket}/{session}")

    source_arrays = load_npz_arrays(npz_path(source_root, bucket, session))
    target_arrays = load_npz_arrays(npz_path(target_root, bucket, session))
    if not np.array_equal(source_arrays["frame_numbers"], target_arrays["frame_numbers"]):
        raise AssertionError(f"NPZ frame_numbers changed for {bucket}/{session}")
    if not np.array_equal(source_arrays["contours"][:, :9], target_arrays["contours"][:, :9]):
        raise AssertionError(f"NPZ classes 0..8 changed for {bucket}/{session}")
    if target_arrays["contours"].dtype != np.float32 or not np.isfinite(target_arrays["contours"]).all():
        raise AssertionError(f"Invalid target NPZ contours for {bucket}/{session}")
    direct = {
        int(frame): (
            target_arrays["contours"][index, 9],
            target_arrays["contours"][index, 10],
        )
        for index, frame in enumerate(target_arrays["frame_numbers"].tolist())
    }
    if job["verify_bf"]:
        bf_direct = load_direct_incisors(incisor_root, bucket, session, direct)
        for frame in direct:
            if not np.array_equal(direct[frame][0], bf_direct[frame][0]):
                raise AssertionError(f"Lower incisor NPZ/BF mismatch at {bucket}/{session}/{frame}")
            if not np.array_equal(direct[frame][1], bf_direct[frame][1]):
                raise AssertionError(f"Upper incisor NPZ/BF mismatch at {bucket}/{session}/{frame}")

    integer_rows = 0
    half_rows = 0
    changed_rows = 0
    for chunk_index, (source_chunk, target_chunk, frame_chunk) in enumerate(
        zip(source_raw["contours"], target_raw["contours"], target_raw["frames"])
    ):
        source_chunk = np.asarray(source_chunk)
        target_chunk = np.asarray(target_chunk)
        frame_chunk = np.asarray(frame_chunk)
        if target_chunk.dtype != np.float32 or not np.isfinite(target_chunk).all():
            raise AssertionError(f"Invalid target raw chunk {bucket}/{session}/{chunk_index}")
        if not np.array_equal(source_chunk[:, :9], target_chunk[:, :9]):
            raise AssertionError(f"Raw classes 0..8 changed at {bucket}/{session}/{chunk_index}")
        for row, frame_value in enumerate(frame_chunk[:, 2]):
            kind, _lower_frame, _upper_frame = frame_kind(float(frame_value))
            expected_lower, expected_upper = expected_incisor_row(float(frame_value), direct)
            if not np.array_equal(target_chunk[row, 9], expected_lower):
                raise AssertionError(
                    f"Lower incisor formula mismatch at {bucket}/{session}/{frame_value}"
                )
            if not np.array_equal(target_chunk[row, 10], expected_upper):
                raise AssertionError(
                    f"Upper incisor formula mismatch at {bucket}/{session}/{frame_value}"
                )
            for contour in target_chunk[row, 9:11].reshape(2, 50, 2):
                if np.array_equal(contour[0], contour[-1]):
                    raise AssertionError(f"Closed incisor at {bucket}/{session}/{frame_value}")
            integer_rows += int(kind == "integer")
            half_rows += int(kind == "half")
            changed_rows += int(not np.array_equal(source_chunk[row, 9:11], target_chunk[row, 9:11]))
    if changed_rows <= 0:
        raise AssertionError(f"No incisor row changed for {bucket}/{session}")
    return {
        "status": "ok",
        "split": split,
        "bucket": bucket,
        "session": session,
        "num_chunks": len(target_raw["features"]),
        "num_integer_rows": integer_rows,
        "num_half_rows": half_rows,
        "num_changed_incisor_rows": changed_rows,
        "num_direct_frames": len(direct),
        "elapsed_seconds": time.time() - started,
    }


def run_parallel(
    function: Any,
    jobs: list[dict[str, Any]],
    workers: int,
    phase: str,
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError(f"{phase} workers must be >= 1")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(function, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                result = future.result()
            except Exception:
                print(
                    f"[{phase} FAILED {job['bucket']}/{job['session']}]\n{traceback.format_exc()}",
                    file=sys.stderr,
                    flush=True,
                )
                raise
            results.append(result)
            print(
                f"[{phase} {completed}/{len(jobs)}] {job['bucket']}/{job['session']} "
                f"status={result['status']} elapsed={result.get('elapsed_seconds', 0.0):.1f}s",
                flush=True,
            )
    return results


def summarize_validation(
    config_path: Path,
    source_root: Path,
    target_root: Path,
    incisor_root: Path,
    variant_name: str,
    results: list[dict[str, Any]],
    verify_bf: bool,
    elapsed: float,
    expected_sessions: dict[str, int],
) -> dict[str, Any]:
    split_sessions = {split: sum(row["split"] == split for row in results) for split in SPLITS}
    split_chunks = {
        split: sum(int(row["num_chunks"]) for row in results if row["split"] == split)
        for split in SPLITS
    }
    if split_sessions != expected_sessions:
        raise AssertionError(
            f"Final split session counts mismatch: expected={expected_sessions}, actual={split_sessions}"
        )
    summary = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "variant_name": variant_name,
        "config": str(config_path.resolve()),
        "source_cache": str(source_root.resolve()),
        "target_cache": str(target_root.resolve()),
        "incisor_root": str(incisor_root.resolve()),
        "num_sessions": len(results),
        "split_sessions": split_sessions,
        "split_chunks": split_chunks,
        "num_chunks": sum(int(row["num_chunks"]) for row in results),
        "num_integer_rows": sum(int(row["num_integer_rows"]) for row in results),
        "num_half_rows": sum(int(row["num_half_rows"]) for row in results),
        "num_changed_incisor_rows": sum(int(row["num_changed_incisor_rows"]) for row in results),
        "num_direct_frames": sum(int(row["num_direct_frames"]) for row in results),
        "verify_bf": verify_bf,
        "invariants": {
            "features_frames_phonemes_bit_identical": True,
            "classes_0_through_8_bit_identical": True,
            "only_classes_9_and_10_replaced": True,
            "fractional_rows_equal_float32_adjacent_mean": True,
            "all_contours_finite_float32_open_50x2": True,
        },
        "elapsed_seconds": elapsed,
        "sessions": sorted(results, key=lambda row: (row["split"], row["bucket"], row["session"])),
    }
    atomic_json_save(target_root / "validation_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    started = time.time()
    config_path = args.config.resolve()
    source_root = args.source_cache.resolve()
    target_root = args.target_cache.resolve()
    incisor_root = args.incisor_root.resolve()
    variant_name = str(args.variant_name or target_root.name)
    if source_root == target_root:
        raise ValueError("Source and target cache roots must be different")
    config = load_config(config_path)
    jobs = iter_sessions(config)
    expected_sessions = expected_split_sessions(config)
    selected = parse_session_filter(args.only_sessions)
    if selected is not None:
        jobs = [job for job in jobs if (job["bucket"], job["session"]) in selected]
        missing = selected - {(job["bucket"], job["session"]) for job in jobs}
        if missing:
            raise ValueError(f"Selected sessions are not in the config: {sorted(missing)}")
    runtime_jobs = [
        {
            **job,
            "source_root": str(source_root),
            "target_root": str(target_root),
            "incisor_root": str(incisor_root),
            "rebuild": bool(args.rebuild),
            "verify_bf": bool(args.verify_bf),
            "variant_name": variant_name,
        }
        for job in jobs
    ]
    print(
        json.dumps(
            {
                "phase": "start",
                "config": str(config_path),
                "source_cache": str(source_root),
                "target_cache": str(target_root),
                "incisor_root": str(incisor_root),
                "variant_name": variant_name,
                "sessions": len(runtime_jobs),
                "workers": args.workers,
                "validate_workers": args.validate_workers,
                "validate_only": args.validate_only,
                "verify_bf": args.verify_bf,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.validate_only:
        run_parallel(build_session, runtime_jobs, args.workers, "build")
    validation = run_parallel(validate_session, runtime_jobs, args.validate_workers, "validate")
    if selected is None:
        summary = summarize_validation(
            config_path,
            source_root,
            target_root,
            incisor_root,
            variant_name,
            validation,
            bool(args.verify_bf),
            time.time() - started,
            expected_sessions,
        )
        print(json.dumps({key: value for key, value in summary.items() if key != "sessions"}, indent=2), flush=True)
    else:
        print(
            json.dumps(
                {"status": "partial_ok", "num_sessions": len(validation), "elapsed_seconds": time.time() - started},
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
