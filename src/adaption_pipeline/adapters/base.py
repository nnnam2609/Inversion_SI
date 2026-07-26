"""Base adapter for a separate Git repository."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from ..contracts import ContractError, ExternalRepoState
from ..provenance import capture_external_repo


class ExternalRepoAdapter:
    name = "external"

    def __init__(self, repo: Path):
        self.repo = repo.resolve()

    def state(self) -> ExternalRepoState:
        return capture_external_repo(self.name, self.repo)

    def environment(self) -> Dict[str, str]:
        environment = dict(os.environ)
        current = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            f"{self.repo}:{current}" if current else str(self.repo)
        )
        return environment

    def run(
        self,
        command: List[str],
        *,
        cwd: Optional[Path] = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        state = self.state()
        if not state.head:
            raise ContractError(f"Cannot resolve {self.name} Git revision")
        return subprocess.run(
            command,
            cwd=str((cwd or self.repo).resolve()),
            env=self.environment(),
            check=check,
            text=True,
        )
