# ASD2-to-ASD1 adaptation pipeline

## Boundary rule

The parent repository owns experiment configuration, orchestration, validation,
evaluation, reports, and rendering. `external/grid-transform` and
`external/audio-speaker-normalization` remain separate Git repositories. Thin
adapters call their public entry points and record their exact Git HEAD and
dirty state in every run. No external implementation is copied into the parent
package.

## Why the implementation combines option 2 and option 3

The scientific code is a modular monolith (option 2): contracts, inference,
normalization, evaluation, and rendering live in one parent package. This
keeps refactoring and local testing simple, avoids network/service overhead,
and makes cross-module types explicit. Its trade-off is that modules still
share one Python environment and release boundary, so the artifact contracts
and import boundaries must be enforced in tests.

The execution layer is a file-based DAG (option 3): each stage has declared
dependencies, commands, outputs, logs, and `_SUCCESS` markers. A failed run can
resume at the failed stage and independent branches can later be scheduled
separately through OAR. Its trade-off is more configuration and artifact
bookkeeping. The DAG deliberately stays thin so it does not duplicate
scientific logic from the modular package.

There is exactly one repository-wide public executable:
`scripts/inversion_si.py`. Adaptation commands are namespaced under `adapt`,
for example `scripts/inversion_si.py adapt evaluate`. Files under
`src/adaption_pipeline/stages/` are reusable library modules; they do not parse
their own command lines or act as independent scripts. Shared strategy names,
conditions, array keys, labels, and primary-result rules live in `domain.py`;
shared atomic JSON/CSV helpers live in `contracts.py` and `io.py`.

```text
scripts/
  inversion_si.py                # the only public executable
src/
  cli.py                         # repository-wide command router
src/adaption_pipeline/
  cli.py                         # all command definitions
  domain.py                      # names, conditions, array keys, labels
  contracts.py                   # versioned data contracts and validation
  io.py                          # shared atomic file I/O
  runtime.py                     # one bridge for legacy/external imports
  strategies.py                  # interchangeable inference implementations
  adapters/                      # boundaries around the two external repos
  stages/                        # reusable workflow functions, one concern each
  orchestration/dag.py           # dependencies, logs, resume markers
config/adaption_pipeline/
  asd2_to_asd1_global_and_moving_average.yaml
  full_nontraining_dag.json
```

To replace one implementation, preserve its input/output contract and change
only that workflow module or adapter. Add a CLI command only for a genuinely
new user operation; helper functions belong in the relevant module, `domain`,
or `io`, not in another top-level script.

The two external repositories remain independent Git checkouts. Thin adapters
give the parent package stable interfaces while preserving separate histories
and replacement freedom. The cost is that every run must pin and audit their
Git HEAD and dirty state.

```text
scripts/
└── inversion_si.py                # only public executable

src/adaption_pipeline/
├── cli.py                         # all command definitions
├── domain.py                      # strategy/condition vocabulary
├── io.py                          # shared mapping and CSV I/O
├── contracts.py                   # versioned data validation
├── metrics.py                     # P2CP, RMSE, correlation
├── strategies.py                  # interchangeable inference strategies
├── adapters/                      # stable boundaries to external repos
├── stages/
│   ├── fit_audio_normalization.py
│   ├── report_audio_correlation.py
│   ├── infer_adapt.py
│   ├── evaluate.py
│   ├── render_anatomical_diagnostic.py
│   ├── render_videos.py
│   └── audit_run.py
└── orchestration/dag.py           # dependency/resume layer only
```

Stage modules remain separate where the input/output contract differs. This
allows, for example, replacing VTLN without editing contour evaluation, or
replacing the video renderer without touching inference. Boilerplate that did
not represent a scientific boundary—argument parsing, repeated CSV writing,
repeated strategy/condition constants, and unused artifact-store code—has been
removed or centralized.

Every stage consumes and produces a versioned artifact contract. A scientific
implementation can therefore be replaced as long as its adapter emits the same
schema. DAG completion additionally requires declared outputs and a stage
`_SUCCESS` marker; final audit checks checkpoint hashes, strategy capabilities,
external Git states, frame identities, and output manifests.

## Modules and graph

The modular monolith is under `src/adaption_pipeline/`. The small file-based DAG
runner only coordinates these modules:

```text
model lock + cohort lock + external Git lock
                    |
        +-----------+-----------+
        |                       |
 raw-audio inference      normalized-audio inference
        |                       |
        +--------+--------------+
                 |
        anatomical calibration (/u/)
                 |
          affine -> affine+TPS
                 |
     original / anatomical / audio / both
                 |
 strict paired contour metrics + audio-only correlation change
                 |
        tables / figures / videos
```

Training is intentionally absent from this graph. The strategy is named
`moving_average`. Its capability metadata still declares
`uses_target_labels=true`, `uses_target_statistics=true`,
`causal=false`, and `blind_inference_compatible=false`; the name does not hide
those scientific constraints.

The DAG config is
`config/adaption_pipeline/full_nontraining_dag.json`. It explicitly declares
`training_enabled=false`; any training stage kind is rejected, and GPU
inference stages refuse to run without an active OAR allocation.

The model bundle also declares its output coordinate space. Global outputs are
in reference-speaker coordinates and may pass through the ASD2-to-ASD1
anatomical transform. Moving-average outputs restored with target contour
centres are already target-native; applying the same transform again is saved
only as a clearly marked double-adaptation diagnostic and is excluded from
primary comparisons.

## Comparison gates

- Sessions are compared only after exact count and `session_order` validation.
- Frames are compared only after exact ordered frame-key validation.
- No stage may silently intersect speakers, sessions, or frames.
- P2CP is calculated between predicted and target contours within each MRI
  frame. The default is symmetric point-to-curve distance in millimetres.
- Coordinate RMSE remains a separately named secondary metric.
- Audio/VTLN reports are independent of contour prediction. They use
  target-to-ASD2-reference Pearson correlation of 39-D MFCC centroids and
  include the value before VTLN, after VTLN, signed change (`after - before`),
  signed percentage change, and direction. Positive means correlation
  increased; negative means it decreased. The exact same number and order of
  selected sessions is required across target speakers. P2CP and contour RMSE
  are never inputs to this report.

## Common visual formats

- One condition: `1x1`.
- Two conditions or two inference strategies: synchronized `1x2`.
- The four adaptation cases (original, anatomical, audio, both): `2x2`.
- Full strategy comparison: `2x4`; one row per strategy and the four conditions
  as columns. Titles use `Global` and `Moving average`.
- Anatomical diagnostic image: `2x4` (eight panels). Column 1 contains source
  and target MRI with annotations; column 2 contains selected landmarks and
  legends for shared, affine-only, and TPS-only landmarks; column 3 contains
  the two grids; column 4 contains affine and affine+TPS results. Annotation,
  affine, and affine+TPS errors in millimetres are printed in their panels.

All video panels use identical frame IDs, identical duration, 50 fps, and one
original audio track. A comparison renderer fails instead of padding or
truncating a mismatched panel.
