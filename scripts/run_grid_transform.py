#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def default_python_bin(repo_root: Path) -> Path:
    shared_venv = repo_root.parent / "inversion" / ".venv"
    candidates = (
        shared_venv / "Scripts" / "python.exe",
        shared_venv / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    grid_transform_root = repo_root / "external" / "grid-transform"
    python_bin = os.environ.get("PYTHON_BIN", str(default_python_bin(repo_root)))

    if not argv:
        print("Usage: scripts/run_grid_transform.py <scripts/run/*.py> [args...]", file=sys.stderr)
        print(
            "Example: scripts/run_grid_transform.py run_create_speaker_grid.py --source vtln --speaker 1640_P7_S2_F0829",
            file=sys.stderr,
        )
        return 2

    run_script = Path(argv[0])
    script_args = argv[1:]
    if len(run_script.parts) == 1:
        run_script = grid_transform_root / "scripts" / "run" / run_script

    if not run_script.is_file():
        print(f"grid-transform run script not found: {run_script}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(grid_transform_root)
        if not existing_pythonpath
        else f"{grid_transform_root}{os.pathsep}{existing_pythonpath}"
    )

    os.chdir(repo_root)
    os.execvpe(python_bin, [python_bin, str(run_script), *script_args], env)
    return 127


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
