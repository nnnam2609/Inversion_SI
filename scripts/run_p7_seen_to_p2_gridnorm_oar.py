#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_VENV_PYTHON = WORKSPACE_ROOT / "inversion" / ".venv" / "bin" / "python"
SEEN_CONFIG = REPO_ROOT / "config" / "train_config" / "asd1_p7_seen_trainvaltest_paper_st5_mfcc_500epoch.yaml"
UNSEEN_CONFIG = REPO_ROOT / "config" / "train_config" / "asd1_p7_seen_trainstats_p2_s1_unseen_eval_st5_mfcc.yaml"
OUTPUT_DIR = REPO_ROOT / "results" / "p7_seen_to_p2_unseen_gridnorm"
LOG_DIR = REPO_ROOT / "logs" / "p7_seen_to_p2_unseen_gridnorm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P7 seen-speaker training and P2 unseen grid-normalized evaluation.")
    parser.add_argument("--python", type=Path, default=DEFAULT_VENV_PYTHON)
    parser.add_argument("--seen-config", type=Path, default=SEEN_CONFIG)
    parser.add_argument("--unseen-config", type=Path, default=UNSEEN_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Reuse an existing checkpoint and skip training.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print commands without training/eval.")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def command_line(command: list[str]) -> str:
    return " ".join(str(item) for item in command)


def run(command: list[str], *, cwd: Path = REPO_ROOT, env: dict[str, str] | None = None, log_path: Path | None = None, dry_run: bool = False) -> None:
    print("+ " + command_line(command), flush=True)
    if dry_run:
        return
    if log_path is None:
        subprocess.run(command, cwd=cwd, env=env, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + command_line(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def project_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    pythonpath = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def build_split_cache_command(python: Path, config: Path) -> list[str]:
    code = (
        "import json, sys; "
        "from pathlib import Path; "
        "sys.path.insert(0, str(Path.cwd())); "
        "sys.path.insert(0, str(Path.cwd() / 'src')); "
        "from src.train.split_cache import ensure_split_caches; "
        "from src.utils.config_validation import load_yaml_config; "
        f"config = load_yaml_config(Path({str(config)!r})); "
        "meta = ensure_split_caches(config); "
        "print(json.dumps(meta, indent=2, sort_keys=True))"
    )
    return [str(python), "-c", code]


def newest_checkpoint(start_time: float, data_save: Path) -> Path:
    candidates = sorted(data_save.glob("mlruns/**/artifacts/best_model.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    fresh = [path for path in candidates if path.stat().st_mtime >= start_time - 2]
    if fresh:
        return fresh[0]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No best_model.pth found under {data_save / 'mlruns'}")


def inference_command(python: Path, config: Path, checkpoint: Path, speaker: int, session: int, output_dir: Path) -> list[str]:
    return [
        str(python),
        "-m",
        "src.inference.session_inference",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--speaker",
        str(speaker),
        "--session",
        str(session),
        "--split",
        "test_sequences",
        "--output-dir",
        str(output_dir),
        "--device",
        "cuda",
        "--write-contours",
    ]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def main() -> None:
    args = parse_args()
    python = args.python if args.python.is_absolute() else (Path.cwd() / args.python)
    seen_config = args.seen_config.resolve()
    unseen_config = args.unseen_config.resolve()
    output_dir = args.output_dir.resolve()
    log_dir = args.log_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    require_file(python, "venv python")
    require_file(seen_config, "seen config")
    require_file(unseen_config, "unseen config")
    vtln_dir = REPO_ROOT / "external" / "grid-transform" / "VTLN" / "data"
    for anchor in ("1640_P7_S2_F0829", "1617_P2_S9_F1478"):
        require_file(vtln_dir / f"{anchor}.png", f"{anchor} VTLN anchor image")
        require_file(vtln_dir / f"{anchor}.zip", f"{anchor} VTLN RoiSet")

    env = project_env()
    started_at = time.time()
    started_iso = datetime.now().isoformat(timespec="seconds")
    run_manifest: dict[str, Any] = {
        "started_at": started_iso,
        "repo_root": str(REPO_ROOT),
        "python": str(python),
        "seen_config": str(seen_config),
        "unseen_config": str(unseen_config),
        "output_dir": str(output_dir),
        "log_dir": str(log_dir),
        "train_sessions": "P7/S1-S12",
        "validation_sessions": "P7/S13-S14",
        "seen_test_sessions": "P7/S15-S16",
        "unseen_test_session": "P2/S1",
        "source_anchor": "1640_P7_S2_F0829",
        "target_anchor": "1617_P2_S9_F1478",
        "note": "P2 is never used for training; P2/S1 is evaluation only.",
        "commands": [],
    }

    cache_seen = build_split_cache_command(python, seen_config)
    cache_unseen = build_split_cache_command(python, unseen_config)
    run_manifest["commands"].append({"step": "build_seen_split_cache", "command": command_line(cache_seen)})
    run_manifest["commands"].append({"step": "build_unseen_split_cache", "command": command_line(cache_unseen)})
    run(cache_seen, env=env, log_path=log_dir / "01_build_seen_split_cache.log", dry_run=args.dry_run)
    run(cache_unseen, env=env, log_path=log_dir / "02_build_unseen_split_cache.log", dry_run=args.dry_run)

    if args.checkpoint is None:
        train_command = [str(python), "src/main_train.py", "--config", str(seen_config)]
        run_manifest["commands"].append({"step": "train_seen_p7", "command": command_line(train_command)})
        train_started = time.time()
        run(train_command, env=env, log_path=log_dir / "03_train_seen_p7.log", dry_run=args.dry_run)
        checkpoint = newest_checkpoint(train_started, REPO_ROOT)
    else:
        checkpoint = args.checkpoint.resolve()
        require_file(checkpoint, "checkpoint")
    run_manifest["checkpoint"] = str(checkpoint)

    seen_prediction_paths = []
    for session in (15, 16):
        session_output = output_dir / "seen_eval" / f"P7_S{session}" / "eval"
        command = inference_command(python, seen_config, checkpoint, 7, session, session_output)
        run_manifest["commands"].append({"step": f"seen_eval_P7_S{session}", "command": command_line(command)})
        run(command, env=env, log_path=log_dir / f"04_seen_eval_P7_S{session}.log", dry_run=args.dry_run)
        seen_prediction_paths.append(session_output / "cached_session_predictions.pt")

    unseen_output = output_dir / "unseen_baseline" / "eval"
    unseen_command = inference_command(python, unseen_config, checkpoint, 2, 1, unseen_output)
    run_manifest["commands"].append({"step": "unseen_eval_P2_S1", "command": command_line(unseen_command)})
    run(unseen_command, env=env, log_path=log_dir / "05_unseen_eval_P2_S1.log", dry_run=args.dry_run)
    unseen_prediction_path = unseen_output / "cached_session_predictions.pt"

    evaluate_command = [
        str(python),
        "scripts/evaluate_grid_normalization.py",
        "--seen-config",
        str(seen_config),
        "--unseen-config",
        str(unseen_config),
        "--checkpoint",
        str(checkpoint),
        "--unseen-predictions",
        str(unseen_prediction_path),
        "--seen-predictions",
        *[str(path) for path in seen_prediction_paths],
        "--output-dir",
        str(output_dir),
    ]
    run_manifest["commands"].append({"step": "gridnorm_evaluation", "command": command_line(evaluate_command)})
    run(evaluate_command, env=env, log_path=log_dir / "06_gridnorm_evaluation.log", dry_run=args.dry_run)

    summary_json = output_dir / "grid_transform_summary.json"
    if summary_json.exists():
        run_manifest["grid_transform_summary"] = load_json(summary_json)
    run_manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    run_manifest["elapsed_seconds"] = round(time.time() - started_at, 3)
    run_manifest["outputs"] = {
        "seen_train_summary": str(output_dir / "seen_train_summary.md"),
        "seen_test_metrics": str(output_dir / "seen_test_metrics.csv"),
        "unseen_baseline_metrics": str(output_dir / "unseen_baseline_metrics.csv"),
        "unseen_gridnorm_metrics": str(output_dir / "unseen_gridnorm_metrics.csv"),
        "per_phoneme_comparison": str(output_dir / "per_phoneme_comparison.csv"),
        "per_articulator_comparison": str(output_dir / "per_articulator_comparison.csv"),
        "grid_transform_summary": str(output_dir / "grid_transform_summary.json"),
        "summary": str(output_dir / "summary.md"),
    }
    manifest_path = output_dir / "job_summary.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "manifest": str(manifest_path), "checkpoint": str(checkpoint)}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
