#!/usr/bin/env python3
"""Audit a refreshed ASD2 train-global normalization against the epoch-211 reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


CLASSES = (
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
REQUIRED_KEYS = ("std_mfcc", "mean_mfcc", "std_contour", "mean_contour")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-incisor-std", type=float, default=0.1)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stats(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as pack:
        missing = [key for key in REQUIRED_KEYS if key not in pack.files]
        if missing:
            raise KeyError(f"Missing normalization arrays {missing}: {path}")
        return {key: np.array(pack[key], copy=True) for key in REQUIRED_KEYS}


def array_summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "all_finite": bool(np.isfinite(values).all()),
        "all_positive": bool((values > 0).all()),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    reference = load_stats(args.reference.resolve())
    candidate = load_stats(args.candidate.resolve())
    metadata_path = args.metadata.resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    expected_shapes = {
        "std_mfcc": (39,),
        "mean_mfcc": (39,),
        "std_contour": (11, 100),
        "mean_contour": (11, 100),
    }
    for key, shape in expected_shapes.items():
        if candidate[key].shape != shape or candidate[key].dtype != np.float32:
            raise AssertionError(
                f"Candidate {key} must be float32 {shape}, got {candidate[key].dtype} {candidate[key].shape}"
            )

    checks = {
        "mfcc_mean_bit_identical": bool(np.array_equal(reference["mean_mfcc"], candidate["mean_mfcc"])),
        "mfcc_std_bit_identical": bool(np.array_equal(reference["std_mfcc"], candidate["std_mfcc"])),
        "classes_0_through_8_mean_bit_identical": bool(
            np.array_equal(reference["mean_contour"][:9], candidate["mean_contour"][:9])
        ),
        "classes_0_through_8_std_bit_identical": bool(
            np.array_equal(reference["std_contour"][:9], candidate["std_contour"][:9])
        ),
        "candidate_contour_std_finite": bool(np.isfinite(candidate["std_contour"]).all()),
        "candidate_contour_std_strictly_positive": bool((candidate["std_contour"] > 0).all()),
        "candidate_mfcc_std_finite": bool(np.isfinite(candidate["std_mfcc"]).all()),
        "candidate_mfcc_std_strictly_positive": bool((candidate["std_mfcc"] > 0).all()),
        "metadata_train_only": metadata.get("normalization_fit_splits") == ["train_sequences"],
        "metadata_raw_positive": metadata.get("normalization_std_policy") == "raw_positive",
        "metadata_expected_train_sequences": metadata.get("fit_num_sequences") == 7504,
        "metadata_expected_train_frames": metadata.get("fit_num_frames") == 561463,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise AssertionError(f"Normalization invariants failed: {failed}")

    incisor_rows = []
    for index in (9, 10):
        std = candidate["std_contour"][index]
        row = {
            "class_index": index,
            "class_name": CLASSES[index],
            "mean": array_summary(candidate["mean_contour"][index]),
            "std": array_summary(std),
            "reference_mean_max_abs_delta": float(
                np.max(np.abs(candidate["mean_contour"][index] - reference["mean_contour"][index]))
            ),
            "reference_std_max_abs_delta": float(
                np.max(np.abs(std - reference["std_contour"][index]))
            ),
        }
        if float(std.min()) < float(args.minimum_incisor_std):
            raise AssertionError(
                f"{CLASSES[index]} minimum std {float(std.min()):.8g} is below gate "
                f"{args.minimum_incisor_std:.8g}"
            )
        incisor_rows.append(row)

    report = {
        "status": "complete",
        "reference": str(args.reference.resolve()),
        "reference_sha256": file_sha256(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": file_sha256(args.candidate.resolve()),
        "metadata": str(metadata_path),
        "metadata_sha256": file_sha256(metadata_path),
        "minimum_incisor_std_gate": float(args.minimum_incisor_std),
        "checks": checks,
        "candidate_mfcc_std": array_summary(candidate["std_mfcc"]),
        "candidate_contour_std": array_summary(candidate["std_contour"]),
        "incisors": incisor_rows,
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
