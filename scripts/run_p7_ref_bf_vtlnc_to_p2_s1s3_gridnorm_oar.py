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
SEEN_CONFIG = REPO_ROOT / "config/train_config/asd1_p7_seen_trainvaltest_paper_st5_mfcc_500epoch.yaml"
UNSEEN_CONFIG = REPO_ROOT / "config/train_config/asd1_p7_seen_trainstats_p2_s1s3_unseen_eval_st5_mfcc.yaml"
CHECKPOINT = REPO_ROOT / "mlruns/633341852101316043/847a0621b38a47f5aa2cfaae8c8e7756/artifacts/best_model.pth"
VTLN_DIR = WORKSPACE_ROOT / "_downloads/grid-transform-vtln/vtln-data-v0.1.14/extracted/VTLN/data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results"
DEFAULT_LOG_DIR = REPO_ROOT / "logs/p7_ref_bf_vtlnc_to_p2_s1s3_gridnorm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P7->P2 S1-S3 inference and BF+VTLN-C grid-normalized evaluation.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--seen-config", type=Path, default=SEEN_CONFIG)
    parser.add_argument("--unseen-config", type=Path, default=UNSEEN_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--vtln-dir", type=Path, default=VTLN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command_line(command: list[str]) -> str:
    return " ".join(str(item) for item in command)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")


def project_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    pythonpath = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "external/grid-transform")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def run(command: list[str], *, env: dict[str, str], log_path: Path, dry_run: bool) -> None:
    print("+ " + command_line(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + command_line(command) + "\n")
        handle.flush()
        if not dry_run:
            subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


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


def inference_command(python: Path, config: Path, checkpoint: Path, session: int, output_dir: Path) -> list[str]:
    return [
        str(python),
        "-m",
        "src.inference.session_inference",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--speaker",
        "2",
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


def main() -> None:
    args = parse_args()
    python = args.python if args.python.is_absolute() else (Path.cwd() / args.python)
    seen_config = args.seen_config.resolve()
    unseen_config = args.unseen_config.resolve()
    checkpoint = args.checkpoint.resolve()
    vtln_dir = args.vtln_dir.resolve()
    started = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or (DEFAULT_OUTPUT_ROOT / f"p7_ref_bf_vtlnc_to_p2_s1s3_gridnorm_{started}")).resolve()
    log_dir = args.log_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    require_file(python, "Python executable")
    require_file(seen_config, "P7 seen config")
    require_file(unseen_config, "P2 S1-S3 unseen config")
    require_file(checkpoint, "P7 checkpoint")
    require_dir(vtln_dir, "VTLN v0.1.14 directory")
    for name in ("1640_P7_S2_F0829", "1617_P2_S9_F1478"):
        require_file(vtln_dir / f"{name}.png", f"{name} VTLN image")
        require_file(vtln_dir / f"{name}.zip", f"{name} VTLN contours")
    for path in (
        WORKSPACE_ROOT / "bf/inference/P7/S2/contours/0829_tongue.npy",
        WORKSPACE_ROOT / "bf/inference/P2/S9/contours/1478_tongue.npy",
    ):
        require_file(path, "BF anchor contour")

    env = project_env()
    started_at = time.time()
    manifest: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "python": str(python),
        "seen_config": str(seen_config),
        "unseen_config": str(unseen_config),
        "checkpoint": str(checkpoint),
        "vtln_dir": str(vtln_dir),
        "sessions": args.sessions,
        "output_dir": str(output_dir),
        "log_dir": str(log_dir),
        "commands": [],
        "policy": "No retraining; P7 checkpoint inference only, then BF+VTLN-C grid transform.",
    }

    cache_command = build_split_cache_command(python, unseen_config)
    manifest["commands"].append({"step": "build_unseen_split_cache", "command": command_line(cache_command)})
    run(cache_command, env=env, log_path=log_dir / "01_build_unseen_split_cache.log", dry_run=args.dry_run)

    prediction_paths = []
    for session in args.sessions:
        session_output = output_dir / "unseen_baseline" / f"P2_S{session}" / "eval"
        command = inference_command(python, unseen_config, checkpoint, session, session_output)
        manifest["commands"].append({"step": f"unseen_eval_P2_S{session}", "command": command_line(command)})
        run(command, env=env, log_path=log_dir / f"02_unseen_eval_P2_S{session}.log", dry_run=args.dry_run)
        prediction_paths.append(session_output / "cached_session_predictions.pt")

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
        *[str(path) for path in prediction_paths],
        "--output-dir",
        str(output_dir),
        "--anchor-source",
        "bf_vtln_c",
        "--source-anchor",
        "1640_P7_S2_F0829",
        "--target-anchor",
        "1617_P2_S9_F1478",
        "--source-speaker",
        "P7",
        "--source-session",
        "S2",
        "--source-frame",
        "0829",
        "--target-speaker-name",
        "P2",
        "--target-session-name",
        "S9",
        "--target-frame",
        "1478",
        "--target-speaker",
        "2",
        "--target-sessions",
        *[str(session) for session in args.sessions],
        "--vtln-dir",
        str(vtln_dir),
        "--prediction-space-size",
        "136",
        "--anchor-space-size",
        "136",
        "--overlay-count",
        "9",
    ]
    manifest["commands"].append({"step": "gridnorm_evaluation", "command": command_line(evaluate_command)})
    run(evaluate_command, env=env, log_path=log_dir / "03_gridnorm_evaluation.log", dry_run=args.dry_run)

    manifest["prediction_payloads"] = [str(path) for path in prediction_paths]
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["elapsed_seconds"] = round(time.time() - started_at, 3)
    manifest["outputs"] = {
        "summary": str(output_dir / "summary.md"),
        "session_metrics": str(output_dir / "session_metrics.csv"),
        "frame_metrics": str(output_dir / "frame_metrics.csv"),
        "per_articulator_comparison": str(output_dir / "per_articulator_comparison.csv"),
        "per_phoneme_comparison": str(output_dir / "per_phoneme_comparison.csv"),
        "grid_transform_summary": str(output_dir / "grid_transform_summary.json"),
    }
    manifest_path = output_dir / "job_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "manifest": str(manifest_path), "output_dir": str(output_dir)}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
