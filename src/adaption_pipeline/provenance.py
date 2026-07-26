"""Git and runtime provenance without modifying external repositories."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .contracts import ContractError, ExternalRepoState


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode:
        raise ContractError(
            f"git {' '.join(args)} failed for {repo}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def capture_external_repo(name: str, path: Path) -> ExternalRepoState:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ContractError(f"External repository does not exist: {resolved}")
    head = _git(resolved, "rev-parse", "HEAD")
    dirty = bool(_git(resolved, "status", "--porcelain"))
    remote: Optional[str] = _git(
        resolved, "remote", "get-url", "origin", check=False
    ) or None
    return ExternalRepoState(
        name=name,
        path=str(resolved),
        head=head,
        dirty=dirty,
        remote=remote,
    )


def runtime_provenance() -> Dict[str, Any]:
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid(),
    }
