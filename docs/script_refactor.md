# Script refactor

Updated: 2026-07-26

## Result

The public script surface was reduced from 62 top-level files (52 Python and
10 shell files, about 30,000 lines) to one 18-line entrypoint:
`scripts/inversion_si.py`.

The entrypoint only routes commands. Maintained implementations now live in
domain packages:

| Package | Responsibility |
|---|---|
| `src/common` | Atomic artifacts, strict frame identity, contour metrics, phonemes, subprocesses, resource reporting |
| `src/preprocessing` | Session, contour, incisor, VTLN, and split-cache preparation |
| `src/train` | Split assembly and model training |
| `src/inference` | Cached-session and dense-audio inference |
| `src/rendering` | General cached/session video rendering |
| `src/commands` | Config, split, and prediction-motion audits |
| `src/orchestration` | Auto-batch, OAR job construction, and external command launch |
| `src/adaption_pipeline` | Versioned ASD2-to-ASD1 stages and resumable file DAG |

This removes command-path coupling: callers depend on one stable command tree,
while Python code imports domain modules directly.

## Adaptation boundary

The adaptation pipeline uses a modular-monolith core plus a file-based DAG.
Each stage has a versioned input/output contract, so VTLN, inference,
anatomical transformation, metrics, or rendering can be replaced independently.

The two projects below remain independent Git repositories and were not copied
or modified:

- `external/grid-transform`
- `external/audio-speaker-normalization`

Stable adapters in `src/adaption_pipeline/adapters/` isolate their APIs and
record repository provenance. Ten validated historical experiment modules are
retained under `src/adaption_pipeline/legacy/` as a frozen internal
compatibility core. They are not public commands, and maintained stage modules
do not reach into their nested implementation details.

## Removed material and recovery

Duplicated job wrappers, one-off publication scripts, historical diagnostics,
and obsolete command entrypoints were removed from the working tree. They
remain recoverable from Git:

- Preservation commit for previously untracked experiment scripts:
  `a2dfb16`
- Earlier tracked command implementations: repository history before the
  refactor commit

Generated caches, checkpoints, results, datasets, and the two external
repositories were outside the cleanup scope and remain untouched.

Old command-path compatibility wrappers are intentionally not kept. Use:

```bash
../inversion/.venv/bin/python scripts/inversion_si.py --help
```

## Validation gates

The refactor is accepted only when:

- `scripts/` contains exactly `inversion_si.py`;
- every public command routes into a domain module;
- no maintained module imports from `scripts/`;
- exact ordered speaker/session/frame pairing remains enforced;
- an AST audit reports zero exact duplicated function bodies under `src/`;
- `python -m compileall` passes;
- the full unit regression suite passes.

No training, inference, or GPU evaluation is part of this source-only refactor.
