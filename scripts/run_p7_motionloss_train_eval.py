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
DEFAULT_PYTHON = WORKSPACE_ROOT / "inversion" / ".venv" / "bin" / "python"
DEFAULT_TRAIN_CONFIG = (
    REPO_ROOT
    / "config"
    / "train_config"
    / "asd1_p7_trainval_p2_s1_test_st5_mfcc_500epoch_stdfloor01_motionloss.yaml"
)
DEFAULT_EVAL_CONFIG = (
    REPO_ROOT
    / "config"
    / "train_config"
    / "asd1_p7_trainval_p2_s1_test_st5_mfcc_stdfloor01_motionloss_eval.yaml"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "p7_trainval_p2_s1_test_motionloss"
DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "p7_trainval_p2_s1_test_motionloss"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train P7 motion-loss model and evaluate P2/S1.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Reuse checkpoint and skip training.")
    parser.add_argument(
        "--trained-checkpoint-name",
        default="best_model.pth",
        help="Artifact checkpoint to use after training, for example best_model.pth or last_model.pth.",
    )
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--target-util", type=float, default=0.80)
    parser.add_argument("--max-batch", type=int, default=4096)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command_line(command: list[str | Path]) -> str:
    return " ".join(str(part) for part in command)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def absolute_no_symlink_resolve(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.absolute()
    return (Path.cwd() / expanded).absolute()


def project_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    pythonpath = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def run(
    command: list[str | Path],
    *,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> None:
    print("+ " + command_line(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + command_line(command) + "\n")
        handle.flush()
        if dry_run:
            return
        subprocess.run([str(part) for part in command], cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def build_split_cache_command(python: Path, config: Path) -> list[str | Path]:
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
    return [python, "-c", code]


def newest_checkpoint(start_time: float, filename: str = "best_model.pth") -> Path:
    candidates = sorted(
        REPO_ROOT.glob(f"mlruns/**/artifacts/{filename}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    fresh = [path for path in candidates if path.stat().st_mtime >= start_time - 2]
    if fresh:
        return fresh[0]
    raise FileNotFoundError(f"No fresh {filename} found under {REPO_ROOT / 'mlruns'}")


def train_command(args: argparse.Namespace) -> list[str | Path]:
    return [
        args.python,
        "scripts/train_auto_batch.py",
        "--config",
        args.train_config,
        "--gpus",
        str(args.gpus),
        "--target-util",
        str(args.target_util),
        "--max-batch",
        str(args.max_batch),
        "--min-batch",
        str(args.min_batch),
    ]


def inference_command(python: Path, config: Path, checkpoint: Path, output_dir: Path) -> list[str | Path]:
    return [
        python,
        "-m",
        "src.inference.session_inference",
        "--config",
        config,
        "--checkpoint",
        checkpoint,
        "--speaker",
        "2",
        "--session",
        "1",
        "--split",
        "test_sequences",
        "--output-dir",
        output_dir,
        "--device",
        "cuda",
        "--write-contours",
    ]


def motion_command(python: Path, prediction_payload: Path, output_dir: Path) -> list[str | Path]:
    return [
        python,
        "scripts/diagnose_prediction_motion.py",
        "--prediction-payload",
        prediction_payload,
        "--output-json",
        output_dir / "motion_diagnostic.json",
        "--output-md",
        output_dir / "motion_diagnostic.md",
    ]


def main() -> None:
    args = parse_args()
    args.python = absolute_no_symlink_resolve(args.python)
    args.train_config = args.train_config.resolve()
    args.eval_config = args.eval_config.resolve()
    args.output_dir = args.output_dir.resolve()
    args.log_dir = args.log_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    require_file(args.python, "Python executable")
    require_file(args.train_config, "train config")
    require_file(args.eval_config, "eval config")

    env = project_env()
    started = time.time()
    manifest: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "python": str(args.python),
        "train_config": str(args.train_config),
        "eval_config": str(args.eval_config),
        "output_dir": str(args.output_dir),
        "log_dir": str(args.log_dir),
        "train_sessions": "P7/S1-S12",
        "valid_sessions": "P7/S13-S14",
        "test_session": "P2/S1",
        "normalization": "fit mean/std on train_sequences only, then apply to valid/test",
        "trained_checkpoint_name": args.trained_checkpoint_name,
        "commands": [],
    }

    cache_cmd = build_split_cache_command(args.python, args.train_config)
    manifest["commands"].append({"step": "build_split_cache", "command": command_line(cache_cmd)})
    run(cache_cmd, env=env, log_path=args.log_dir / "01_build_split_cache.log", dry_run=args.dry_run)

    if args.checkpoint is None:
        cmd = train_command(args)
        manifest["commands"].append({"step": "train_motionloss", "command": command_line(cmd)})
        train_started = time.time()
        run(cmd, env=env, log_path=args.log_dir / "02_train_motionloss.log", dry_run=args.dry_run)
        checkpoint = (
            Path("DRY_RUN_CHECKPOINT")
            if args.dry_run
            else newest_checkpoint(train_started, args.trained_checkpoint_name)
        )
    else:
        checkpoint = args.checkpoint.resolve()
        require_file(checkpoint, "checkpoint")
    manifest["checkpoint"] = str(checkpoint)

    eval_dir = args.output_dir / "p2_s1" / "eval"
    infer_cmd = inference_command(args.python, args.eval_config, checkpoint, eval_dir)
    manifest["commands"].append({"step": "infer_p2_s1", "command": command_line(infer_cmd)})
    run(infer_cmd, env=env, log_path=args.log_dir / "03_infer_p2_s1.log", dry_run=args.dry_run)

    prediction_payload = eval_dir / "cached_session_predictions.pt"
    diag_cmd = motion_command(args.python, prediction_payload, args.output_dir / "p2_s1")
    manifest["commands"].append({"step": "diagnose_motion", "command": command_line(diag_cmd)})
    run(diag_cmd, env=env, log_path=args.log_dir / "04_diagnose_motion.log", dry_run=args.dry_run)

    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["elapsed_seconds"] = round(time.time() - started, 3)
    manifest["outputs"] = {
        "prediction_payload": str(prediction_payload),
        "motion_json": str(args.output_dir / "p2_s1" / "motion_diagnostic.json"),
        "motion_md": str(args.output_dir / "p2_s1" / "motion_diagnostic.md"),
    }
    manifest_path = args.output_dir / "job_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "manifest": str(manifest_path), "checkpoint": str(checkpoint)}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
