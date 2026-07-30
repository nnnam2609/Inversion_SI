"""Single public command line interface for the adaptation pipeline."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

from .adapters import AudioNormalizationAdapter, GridTransformAdapter
from .contracts import (
    ContractError,
    atomic_write_json,
    load_cohort,
    load_model_bundle,
)
from .domain import STRATEGIES
from .io import load_mapping
from .orchestration.dag import load_dag
from .provenance import runtime_provenance


def _load(path: Path) -> Dict[str, Any]:
    return load_mapping(path)


def validate_config(args: argparse.Namespace) -> None:
    payload = _load(args.config)
    models = [load_model_bundle(item) for item in payload["models"]]
    for model in models:
        model.validate(verify_files=not args.skip_file_hashes)
    cohort = load_cohort(payload["cohort"])
    cohort.validate()
    output = {
        "status": "valid",
        "models": [dataclasses.asdict(item) for item in models],
        "cohort": dataclasses.asdict(cohort),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


def capture_provenance(args: argparse.Namespace) -> None:
    grid = GridTransformAdapter(args.grid_repo)
    audio = AudioNormalizationAdapter(args.audio_repo)
    payload = {
        "runtime": runtime_provenance(),
        "external_repositories": {
            "grid_transform": dataclasses.asdict(grid.provenance()["repo"]),
            "audio_normalization": dataclasses.asdict(audio.provenance()["repo"]),
        },
        "audio_project_root": str(audio.project_root),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_dag(args: argparse.Namespace) -> None:
    dag = load_dag(args.dag, args.state_root)
    dag.run(targets=args.target or None, force=args.force)


def run_workflow(args: argparse.Namespace) -> int:
    module = importlib.import_module(args.workflow_module)
    return int(module.run(args) or 0)


def add_pipeline_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pipeline-config", type=Path, required=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-config")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--skip-file-hashes", action="store_true")
    validate.set_defaults(handler=validate_config)

    provenance = commands.add_parser("capture-provenance")
    provenance.add_argument("--grid-repo", type=Path, required=True)
    provenance.add_argument("--audio-repo", type=Path, required=True)
    provenance.add_argument("--output", type=Path, required=True)
    provenance.set_defaults(handler=capture_provenance)

    run = commands.add_parser("run-dag")
    run.add_argument("--dag", type=Path, required=True)
    run.add_argument("--state-root", type=Path, required=True)
    run.add_argument("--target", action="append")
    run.add_argument("--force", action="store_true")
    run.set_defaults(handler=run_dag)

    fit_audio = commands.add_parser(
        "fit-audio", help="Fit RMS/VTLN parameters and audio statistics"
    )
    add_pipeline_config(fit_audio)
    fit_audio.add_argument("--output-root", type=Path, required=True)
    fit_audio.add_argument("--gmm-components", type=int, default=64)
    fit_audio.add_argument("--gmm-max-iter", type=int, default=200)
    fit_audio.add_argument("--max-reference-frames", type=int, default=120_000)
    fit_audio.set_defaults(
        handler=run_workflow,
        workflow_module="src.adaption_pipeline.stages.fit_audio_normalization",
    )

    infer = commands.add_parser(
        "infer", help="Run one configured inference strategy"
    )
    add_pipeline_config(infer)
    infer.add_argument("--strategy", choices=STRATEGIES, required=True)
    infer.add_argument("--speaker", action="append", help="Limit to P#, repeatable")
    infer.add_argument("--output-root", type=Path, required=True)
    infer.add_argument("--device", default="cuda:0")
    infer.add_argument("--batch-size", type=int, default=128)
    infer.add_argument("--transform-frame-batch", type=int, default=256)
    infer.add_argument(
        "--audio-normalization",
        type=Path,
        help="audio_normalization.json; enables audio adaptation conditions",
    )
    infer.set_defaults(
        handler=run_workflow,
        workflow_module="src.adaption_pipeline.stages.infer_adapt",
    )

    audio_report = commands.add_parser(
        "report-audio", help="Report audio-only correlation before/after VTLN"
    )
    add_pipeline_config(audio_report)
    audio_report.add_argument("--audio-normalization", type=Path, required=True)
    audio_report.add_argument("--output-json", type=Path, required=True)
    audio_report.add_argument("--output-csv", type=Path, required=True)
    audio_report.set_defaults(
        handler=run_workflow,
        workflow_module="src.adaption_pipeline.stages.report_audio_correlation",
    )

    evaluate = commands.add_parser(
        "evaluate", help="Evaluate paired contour predictions"
    )
    add_pipeline_config(evaluate)
    evaluate.add_argument("--prediction-root", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.set_defaults(
        handler=run_workflow,
        workflow_module="src.adaption_pipeline.stages.evaluate",
    )

    anatomy = commands.add_parser(
        "render-anatomy", help="Render 2x4 anatomical diagnostic figures"
    )
    add_pipeline_config(anatomy)
    anatomy.add_argument("--output-root", type=Path, required=True)
    anatomy.add_argument("--speaker", action="append", help="Limit to P#, repeatable")
    anatomy.set_defaults(
        handler=run_workflow,
        workflow_module=(
            "src.adaption_pipeline.stages.render_anatomical_diagnostic"
        ),
    )

    videos = commands.add_parser(
        "render-videos", help="Render synchronized common video layouts"
    )
    videos.add_argument("--prediction-root", type=Path, required=True)
    videos.add_argument("--output-root", type=Path, required=True)
    videos.add_argument("--speaker", default="P1")
    videos.add_argument("--session", default="S16")
    videos.add_argument("--mri-workers", type=int, default=8)
    videos.add_argument(
        "--layout",
        action="append",
        choices=(
            "single",
            "pair",
            "global_four",
            "moving_four",
            "strategy_compare",
        ),
        help="Render selected layout; repeatable. Default renders all layouts.",
    )
    videos.set_defaults(
        handler=run_workflow,
        workflow_module="src.adaption_pipeline.stages.render_videos",
    )

    audit = commands.add_parser(
        "audit", help="Audit the complete non-training artifact graph"
    )
    add_pipeline_config(audit)
    audit.add_argument("--artifact-root", type=Path, required=True)
    audit.set_defaults(
        handler=run_workflow,
        workflow_module="src.adaption_pipeline.stages.audit_run",
    )
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        result = arguments.handler(arguments)
    except (ContractError, KeyError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
