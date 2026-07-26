"""Single public command router for the Inversion_SI repository."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence


@dataclass(frozen=True)
class Command:
    path: tuple[str, ...]
    module: str
    help: str
    passes_argv: bool = False


COMMANDS = (
    Command(("preprocess", "sessions"), "src.preprocessing.cli", "Build per-session caches."),
    Command(
        ("preprocess", "incisors"),
        "src.preprocessing.incisor_cache",
        "Refresh ASD2 incisor contours in session caches.",
    ),
    Command(
        ("preprocess", "splits"),
        "src.commands.build_split_caches",
        "Build validated train/validation/test split caches.",
    ),
    Command(
        ("preprocess", "vtln-cache"),
        "src.inference.vtln_cache",
        "Build inversion-frontend VTLN evaluation features.",
    ),
    Command(
        ("train", "model"),
        "src.main_train",
        "Run training from an explicit config.",
    ),
    Command(
        ("train", "auto-batch"),
        "src.orchestration.auto_batch",
        "Tune per-GPU batch size and run training.",
    ),
    Command(
        ("train", "submit"),
        "src.orchestration.oar",
        "Prepare or submit an OAR training job.",
    ),
    Command(
        ("infer", "session"),
        "src.inference.config_runner",
        "Run configured cached-session inference and rendering.",
    ),
    Command(
        ("infer", "dense"),
        "src.inference.dense_audio",
        "Infer direct integer-frame contours from dense audio.",
    ),
    Command(
        ("render", "cached"),
        "src.rendering.cached_compare",
        "Render a cached prediction comparison.",
    ),
    Command(
        ("render", "session"),
        "src.rendering.gridnorm_session",
        "Render and audit a grid-normalized session.",
    ),
    Command(
        ("audit", "configs"),
        "src.commands.audit_normalization_configs",
        "Audit normalization configuration contracts.",
    ),
    Command(
        ("audit", "splits"),
        "src.commands.audit_split_cache_normalization",
        "Audit existing split-cache normalization.",
    ),
    Command(
        ("audit", "motion"),
        "src.commands.diagnose_prediction_motion",
        "Diagnose temporal motion in prediction packs.",
    ),
    Command(
        ("adapt",),
        "src.adaption_pipeline.cli",
        "Run the modular ASD2-to-ASD1 adaptation pipeline.",
        passes_argv=True,
    ),
    Command(
        ("grid-transform",),
        "src.orchestration.grid_transform",
        "Run a command from the separate grid-transform repository.",
        passes_argv=True,
    ),
)


def _usage(prefix: tuple[str, ...] = ()) -> str:
    rows = [
        command
        for command in COMMANDS
        if command.path[: len(prefix)] == prefix and command.path != prefix
    ]
    if prefix:
        heading = f"Usage: scripts/inversion_si.py {' '.join(prefix)} <command> [args...]"
    else:
        heading = "Usage: scripts/inversion_si.py <command> [args...]"
    lines = [heading, "", "Commands:"]
    for command in rows:
        suffix = command.path[len(prefix) :]
        lines.append(f"  {' '.join(suffix):24s} {command.help}")
    if not rows:
        lines.append("  (no matching commands)")
    return "\n".join(lines)


@contextmanager
def _temporary_argv(program: str, arguments: Sequence[str]) -> Iterator[None]:
    previous = sys.argv
    sys.argv = [program, *arguments]
    try:
        yield
    finally:
        sys.argv = previous


def _run(command: Command, arguments: list[str]) -> int:
    module = importlib.import_module(command.module)
    entrypoint: Callable = getattr(module, "main")
    if command.passes_argv:
        result = entrypoint(arguments)
    else:
        with _temporary_argv(" ".join(command.path), arguments):
            result = entrypoint()
    return int(result or 0)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments == ["--help"] or arguments == ["-h"]:
        print(_usage())
        return 0

    for command in sorted(COMMANDS, key=lambda item: len(item.path), reverse=True):
        if tuple(arguments[: len(command.path)]) == command.path:
            remainder = arguments[len(command.path) :]
            return _run(command, remainder)

    prefix = tuple(item for item in arguments if not item.startswith("-"))
    for size in range(len(prefix), 0, -1):
        candidate = prefix[:size]
        if any(command.path[:size] == candidate for command in COMMANDS):
            print(_usage(candidate))
            return 0 if any(flag in arguments for flag in ("-h", "--help")) else 2

    print(f"Unknown command: {' '.join(arguments)}", file=sys.stderr)
    print(_usage(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
