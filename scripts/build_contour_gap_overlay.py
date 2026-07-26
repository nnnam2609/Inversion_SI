#!/usr/bin/env python3
"""Create a small, auditable linear-interpolation overlay for missing contour frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-folder", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--fill-frames", type=int, nargs="+", required=True)
    parser.add_argument("--articulators", nargs="+", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_npy_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with tmp_path.open("wb") as stream:
        np.save(stream, array)
    os.replace(tmp_path, path)


def atomic_json_save(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, path)


def main() -> None:
    args = parse_args()
    source_folder = args.source_folder.resolve()
    output_folder = args.output_folder.resolve()
    if args.end_frame <= args.start_frame:
        raise ValueError("end-frame must be greater than start-frame")
    if any(frame <= args.start_frame or frame >= args.end_frame for frame in args.fill_frames):
        raise ValueError("Every fill frame must lie strictly between the endpoint frames")

    records = []
    for articulator in args.articulators:
        start_path = source_folder / f"{args.start_frame:04d}_{articulator}.npy"
        end_path = source_folder / f"{args.end_frame:04d}_{articulator}.npy"
        start = np.load(start_path)
        end = np.load(end_path)
        if start.shape != end.shape:
            raise ValueError(
                f"Endpoint shape mismatch for {articulator}: {start.shape} != {end.shape}"
            )
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            raise ValueError(f"Non-finite endpoint contour for {articulator}")
        for frame in args.fill_frames:
            alpha = (frame - args.start_frame) / (args.end_frame - args.start_frame)
            interpolated = (
                (1.0 - alpha) * start.astype(np.float32)
                + alpha * end.astype(np.float32)
            ).astype(np.float32)
            output_path = output_folder / f"{frame:04d}_{articulator}.npy"
            atomic_npy_save(output_path, interpolated)
            records.append(
                {
                    "articulator": articulator,
                    "frame": frame,
                    "alpha": alpha,
                    "output": str(output_path),
                    "output_sha256": sha256(output_path),
                    "start_source": str(start_path),
                    "start_sha256": sha256(start_path),
                    "end_source": str(end_path),
                    "end_sha256": sha256(end_path),
                }
            )

    manifest = {
        "method": "linear_interpolation_between_integer_contour_frames",
        "source_folder": str(source_folder),
        "output_folder": str(output_folder),
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "fill_frames": args.fill_frames,
        "articulators": args.articulators,
        "num_generated_files": len(records),
        "records": records,
    }
    manifest_path = output_folder / "INTERPOLATION_MANIFEST.json"
    atomic_json_save(manifest_path, manifest)
    print(
        f"generated={len(records)} output_folder={output_folder} "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
