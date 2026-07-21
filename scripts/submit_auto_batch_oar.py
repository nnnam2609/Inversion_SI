#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_PYTHON = WORKSPACE_ROOT / "inversion" / ".venv" / "bin" / "python"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "auto_batch_oar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or submit an OAR job for Inversion_SI auto-batch training."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--target-util", type=float, default=0.80)
    parser.add_argument("--max-batch", type=int, default=4096)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--smoke-epochs", type=int, default=0)
    parser.add_argument("--walltime", default="02:00:00")
    parser.add_argument("--queue", default="production")
    parser.add_argument("--cluster", default=None, help="Optional OAR cluster constraint, e.g. gres or gruss.")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--output-config-dir", type=Path, default=None)
    parser.add_argument("--submit", action="store_true", help="Actually call oarsub. Without this, only prints the command.")
    return parser.parse_args()


def shell_join(parts: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def absolute_no_symlink_resolve(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.absolute()
    return (Path.cwd() / expanded).absolute()


def run_preflight(python: Path, config: Path) -> str:
    command = [
        str(python),
        "scripts/train_auto_batch.py",
        "--config",
        str(config),
        "--preflight-only",
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Auto-batch preflight failed before OAR submission.\n"
            f"command={shell_join(command)}\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )
    return result.stdout.strip()


def train_auto_batch_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python),
        "scripts/train_auto_batch.py",
        "--config",
        str(args.config.resolve()),
        "--gpus",
        str(args.gpus),
        "--target-util",
        str(args.target_util),
        "--max-batch",
        str(args.max_batch),
        "--min-batch",
        str(args.min_batch),
    ]
    if args.output_config_dir is not None:
        command.extend(["--output-config-dir", str(args.output_config_dir.resolve())])
    if args.smoke_epochs:
        command.extend(["--smoke-epochs", str(args.smoke_epochs)])
    return command


def write_job_script(path: Path, command: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shlex.quote(str(REPO_ROOT))}",
            "if ! type module >/dev/null 2>&1; then source /etc/profile; fi",
            "module purge",
            "module load cuda/12.1.1",
            "export PYTHONUNBUFFERED=1",
            'export PYTHONPATH="${PWD}:${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"',
            shell_join(command),
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def oar_command(args: argparse.Namespace, job_script: Path, stdout_path: Path, stderr_path: Path) -> list[str]:
    command = [
        "oarsub",
        "-q",
        args.queue,
    ]
    if args.cluster:
        command.extend(["-p", f"cluster='{args.cluster}'"])
    command.extend(
        [
            "-l",
            f"/host=1/gpu={args.gpus},walltime={args.walltime}",
            "-d",
            str(WORKSPACE_ROOT),
            "-O",
            str(stdout_path),
            "-E",
            str(stderr_path),
            str(job_script),
        ]
    )
    return command


def parse_oar_job_id(output: str) -> str | None:
    patterns = [
        r"\bOAR_JOB_ID\s*=\s*(\d+)\b",
        r"\bjob(?:\s+id)?\s*[:=]\s*(\d+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    args.python = absolute_no_symlink_resolve(args.python)
    require_file(args.config, "training config")
    require_file(args.python, "Python executable")
    if args.gpus < 1:
        raise ValueError("--gpus must be >= 1")

    started = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.log_root.resolve() / f"{args.config.stem}_{started}"
    stdout_path = run_dir / "%jobid%.out"
    stderr_path = run_dir / "%jobid%.err"
    job_script = run_dir / "job_auto_batch.sh"
    preflight_stdout = run_preflight(args.python, args.config)
    command = train_auto_batch_command(args)
    write_job_script(job_script, command)
    submit_command = oar_command(args, job_script, stdout_path, stderr_path)
    manifest: dict[str, Any] = {
        "status": "prepared",
        "submitted": False,
        "repo_root": str(REPO_ROOT),
        "config": str(args.config),
        "python": str(args.python),
        "gpus": args.gpus,
        "target_util": args.target_util,
        "max_batch": args.max_batch,
        "min_batch": args.min_batch,
        "smoke_epochs": args.smoke_epochs,
        "queue": args.queue,
        "cluster": args.cluster,
        "walltime": args.walltime,
        "run_dir": str(run_dir),
        "job_script": str(job_script),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "preflight_stdout": preflight_stdout,
        "train_auto_batch_command": shell_join(command),
        "oar_command": shell_join(submit_command),
    }
    if args.submit:
        result = subprocess.run(
            submit_command,
            cwd=WORKSPACE_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        manifest["submitted"] = result.returncode == 0
        manifest["oarsub_returncode"] = result.returncode
        manifest["oarsub_stdout"] = result.stdout
        manifest["oarsub_stderr"] = result.stderr
        manifest["job_id"] = parse_oar_job_id(result.stdout + "\n" + result.stderr)
        if result.returncode != 0:
            (run_dir / "submit_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            raise RuntimeError(f"oarsub failed: {result.stderr or result.stdout}")
    manifest_path = run_dir / "submit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
