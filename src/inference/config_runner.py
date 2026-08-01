#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.utils.config_validation import load_yaml_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cached session inference and render videos from a YAML config.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return load_yaml_config(path)


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def optional_args(flag: str, values: list[str] | None) -> list[str]:
    if not values:
        return []
    return [flag, *[str(value) for value in values]]


def render_args(render: dict[str, Any], output_path: Path) -> list[str]:
    args = [
        "--output",
        str(output_path),
        "--fps",
        str(render.get("fps", 12)),
        "--scale",
        str(render.get("scale", 4)),
        "--background",
        str(render.get("background", "gray")),
        "--dicom-index-method",
        str(render.get("dicom_index_method", "filename")),
        "--dicom-index-workers",
        str(render.get("dicom_index_workers", 24)),
        "--dicom-read-workers",
        str(render.get("dicom_read_workers", 8)),
        "--frame-offset",
        str(render.get("frame_offset", 0)),
    ]
    if render.get("max_frames") is not None:
        args.extend(["--max-frames", str(render["max_frames"])])
    if bool(render.get("integer_frames_only", False)):
        args.append("--integer-frames-only")
    mri_dicom_dir = resolve_path(render.get("mri_dicom_dir"))
    mri_npy_dir = resolve_path(render.get("mri_npy_dir"))
    if mri_dicom_dir is not None:
        args.extend(["--mri-dicom-dir", str(mri_dicom_dir)])
    if mri_npy_dir is not None:
        args.extend(["--mri-npy-dir", str(mri_npy_dir)])
    return args


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    train_config = resolve_path(config["train_config"])
    checkpoint = resolve_path(config["checkpoint"])
    output_dir = resolve_path(config["output_dir"])
    if train_config is None or checkpoint is None or output_dir is None:
        raise ValueError("train_config, checkpoint, and output_dir are required")

    target = config.get("target", {})
    speaker = str(target["speaker"])
    session = str(target["session"])
    split = str(target.get("split", "test_sequences"))
    device = str(config.get("device", "auto"))
    excluded = list(config.get("exclude_rmse_classes", []))
    render = dict(config.get("render", {}))
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_summaries: dict[str, Any] = {
        "config": str(args.config),
        "train_config": str(train_config),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "speaker": speaker,
        "session": session,
        "split": split,
    }

    python = sys.executable
    for mode_name in ("predict_only", "eval"):
        mode_cfg = dict(config.get(mode_name, {}))
        if not mode_cfg.get("enabled", True):
            continue
        mode_dir = output_dir / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)

        inference_command = [
            python,
            "-m",
            "src.inference.session_inference",
            "--config",
            str(train_config),
            "--checkpoint",
            str(checkpoint),
            "--speaker",
            speaker,
            "--session",
            session,
            "--split",
            split,
            "--output-dir",
            str(mode_dir),
            "--device",
            device,
            *optional_args("--exclude-rmse-classes", excluded),
        ]
        if mode_name == "predict_only":
            denorm_cache = resolve_path(mode_cfg.get("denorm_cache"))
            if denorm_cache is None:
                raise ValueError("predict_only.denorm_cache is required")
            inference_command.extend(["--prediction-only", "--denorm-cache", str(denorm_cache)])
        elif mode_cfg.get("prediction_denorm_cache") is not None:
            inference_command.extend(["--prediction-denorm-cache", str(resolve_path(mode_cfg["prediction_denorm_cache"]))])
        if bool(mode_cfg.get("write_contours", config.get("write_contours", False))):
            contour_dir = resolve_path(mode_cfg.get("contour_output_dir"))
            inference_command.append("--write-contours")
            inference_command.extend([
                "--contour-output-format",
                str(mode_cfg.get("contour_output_format", config.get("contour_output_format", "xy50"))),
            ])
            if contour_dir is not None:
                inference_command.extend(["--contour-output-dir", str(contour_dir)])
        run(inference_command)

        output_name = str(mode_cfg.get("video_name", f"{mode_name}.mp4"))
        video_path = output_dir / output_name
        mri_frame_cache = output_dir / f"{mode_name}_mri_frames.npz"
        dicom_index_cache = output_dir / "dicom_index.json"
        render_command = [
            python,
            "scripts/render_cached_compare_video.py",
            "--predictions",
            str(mode_dir / "cached_session_predictions.pt"),
            "--config",
            str(train_config),
            *render_args(render, video_path),
            "--dicom-index-cache",
            str(dicom_index_cache),
            "--mri-frame-cache",
            str(mri_frame_cache),
            *optional_args("--exclude-rmse-classes", excluded),
        ]
        run(render_command)
        render_summary_path = video_path.with_suffix(".json")
        render_summary = None
        if render_summary_path.exists():
            with render_summary_path.open("r", encoding="utf-8") as handle:
                render_summary = json.load(handle)
        mode_summaries[mode_name] = {
            "inference_dir": str(mode_dir),
            "predictions": str(mode_dir / "cached_session_predictions.pt"),
            "video": str(video_path),
            "render_summary": None if render_summary is None else str(render_summary_path),
            "render_metrics": render_summary,
            "summary": str(mode_dir / "summary.json"),
        }

    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(mode_summaries, handle, indent=2, sort_keys=True)
    print(json.dumps(mode_summaries, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
