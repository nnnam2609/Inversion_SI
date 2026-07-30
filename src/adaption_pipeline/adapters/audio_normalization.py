"""Stable boundary around the external audio-normalization repository."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .base import ExternalRepoAdapter
from ..contracts import ContractError


class AudioNormalizationAdapter(ExternalRepoAdapter):
    name = "audio-speaker-normalization"

    @property
    def project_root(self) -> Path:
        nested = self.repo / "audio-speaker-normalization"
        return nested if nested.is_dir() else self.repo

    def validate(self) -> None:
        required = [
            self.project_root / "audio_speaker_norm" / "audio_normalization.py",
            self.project_root / "run_audio_normalization.py",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ContractError(
                f"Invalid audio normalization checkout; missing {missing}"
            )

    def environment(self) -> Dict[str, str]:
        environment = super().environment()
        current = environment.get("PYTHONPATH")
        root = str(self.project_root)
        environment["PYTHONPATH"] = f"{root}:{current}" if current else root
        return environment

    def build_command(
        self, *, config: Path, output_dir: Path, extra_args: List[str]
    ) -> List[str]:
        self.validate()
        return [
            "python",
            str((self.project_root / "run_audio_normalization.py").resolve()),
            "--config",
            str(config.resolve()),
            "--output-dir",
            str(output_dir.resolve()),
            *extra_args,
        ]

    def provenance(self) -> Dict[str, Any]:
        self.validate()
        return {
            "adapter": self.name,
            "repo": self.state(),
            "project_root": str(self.project_root),
        }
