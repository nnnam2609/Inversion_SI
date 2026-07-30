"""Stable boundary around the external anatomical-normalization repository."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .base import ExternalRepoAdapter
from ..contracts import CalibrationPair, ContractError


class GridTransformAdapter(ExternalRepoAdapter):
    name = "grid-transform"

    def validate(self) -> None:
        required = [self.repo / "grid_transform", self.repo / "scripts"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise ContractError(f"Invalid grid-transform checkout; missing {missing}")

    def build_command(
        self,
        *,
        entrypoint: str,
        calibration: CalibrationPair,
        output_dir: Path,
        extra_args: List[str],
    ) -> List[str]:
        calibration.validate()
        script = (self.repo / "scripts" / entrypoint).resolve()
        if not script.is_file():
            raise ContractError(f"Grid-transform entrypoint not found: {script}")
        return [
            "python",
            str(script),
            "--output-dir",
            str(output_dir.resolve()),
            *extra_args,
        ]

    def provenance(self) -> Dict[str, Any]:
        self.validate()
        return {"adapter": self.name, "repo": self.state()}
