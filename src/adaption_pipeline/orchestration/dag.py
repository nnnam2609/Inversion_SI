"""Dependency graph with resumable stage completion records."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set

from ..contracts import ContractError, atomic_write_json, utc_now


FORBIDDEN_STAGE_KINDS = {"train", "training", "fit_model"}


@dataclass(frozen=True)
class Stage:
    name: str
    command: List[str]
    dependencies: List[str] = field(default_factory=list)
    kind: str = "cpu"
    outputs: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.name or not self.command:
            raise ContractError("DAG stage requires a name and command")
        if self.kind.lower() in FORBIDDEN_STAGE_KINDS:
            raise ContractError(
                f"Training stage {self.name!r} is forbidden in this inference pipeline"
            )


class Dag:
    def __init__(self, stages: Iterable[Stage], state_root: Path):
        stage_list = list(stages)
        self.stages: Dict[str, Stage] = {stage.name: stage for stage in stage_list}
        self.state_root = state_root.resolve()
        if len(self.stages) != len(stage_list):
            raise ContractError("Duplicate DAG stage name")
        for stage in self.stages.values():
            stage.validate()
            missing = set(stage.dependencies).difference(self.stages)
            if missing:
                raise ContractError(
                    f"Stage {stage.name} has missing dependencies {sorted(missing)}"
                )
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        visiting: Set[str] = set()
        visited: Set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ContractError(f"DAG cycle detected at {name}")
            if name in visited:
                return
            visiting.add(name)
            for dependency in self.stages[name].dependencies:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)

        for name in self.stages:
            visit(name)

    def _marker(self, stage: Stage) -> Path:
        return self.state_root / stage.name / "_SUCCESS"

    def complete(self, stage: Stage) -> bool:
        marker = self._marker(stage)
        return marker.is_file() and all(Path(output).exists() for output in stage.outputs)

    def _run_one(self, stage: Stage) -> None:
        if stage.kind.lower().startswith("gpu") and not os.environ.get("OAR_JOB_ID"):
            raise ContractError(
                f"GPU stage {stage.name!r} requires an active OAR allocation"
            )
        destination = self.state_root / stage.name
        destination.mkdir(parents=True, exist_ok=True)
        log_path = destination / "stage.log"
        environment = dict(os.environ)
        environment.update(
            {
                "ADAPTION_PIPELINE_STAGE": stage.name,
                "ADAPTION_PIPELINE_STATE_DIR": str(destination),
            }
        )
        started = utc_now()
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                stage.command,
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
            )
        record = {
            "stage": stage.name,
            "kind": stage.kind,
            "command": stage.command,
            "dependencies": stage.dependencies,
            "started_at": started,
            "finished_at": utc_now(),
            "returncode": result.returncode,
            "outputs": stage.outputs,
            "oar_job_id": os.environ.get("OAR_JOB_ID"),
        }
        atomic_write_json(destination / "stage_run.json", record)
        if result.returncode:
            raise RuntimeError(
                f"Stage {stage.name} failed with code {result.returncode}; "
                f"see {log_path}"
            )
        missing = [output for output in stage.outputs if not Path(output).exists()]
        if missing:
            raise ContractError(
                f"Stage {stage.name} succeeded but outputs are missing: {missing}"
            )
        self._marker(stage).write_text(f"{stage.name}\n", encoding="utf-8")

    def run(self, targets: Optional[Iterable[str]] = None, force: bool = False) -> None:
        requested = list(targets or self.stages)
        unknown = set(requested).difference(self.stages)
        if unknown:
            raise ContractError(f"Unknown DAG targets: {sorted(unknown)}")
        executed: Set[str] = set()

        def execute(name: str) -> None:
            if name in executed:
                return
            stage = self.stages[name]
            for dependency in stage.dependencies:
                execute(dependency)
            if force or not self.complete(stage):
                self._run_one(stage)
            executed.add(name)

        for target in requested:
            execute(target)


def load_dag(path: Path, state_root: Path) -> Dag:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("training_enabled") is not False:
        raise ContractError(
            "Inference/adaptation DAG must explicitly set training_enabled=false"
        )
    stages = [Stage(**item) for item in payload["stages"]]
    return Dag(stages, state_root)
