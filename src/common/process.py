"""Checked subprocess helpers used by rendering and orchestration."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


def run_checked(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one command without a shell and return captured text output."""

    return subprocess.run(
        [str(item) for item in command],
        cwd=None if cwd is None else str(cwd),
        env=None if env is None else dict(env),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def ffprobe_json(path: Path) -> dict:
    """Return ffprobe stream/format metadata for one media file."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing media file: {path}")
    completed = run_checked(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            path,
        ]
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected ffprobe payload for {path}")
    return payload
