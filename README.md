# Inversion_SI

Cache-first articulatory inversion pipeline for ASD1/ASD2 single-task 5 experiments.

The code is organized around one public CLI and replaceable domain modules:

1. Build reusable per-session preprocessing and split caches.
2. Train/evaluate or run inference from explicit YAML configs.
3. Adapt ASD2 predictions to ASD1 with independent audio and anatomy stages.
4. Evaluate and render from versioned artifacts.

Large generated files are intentionally excluded from git. Keep `cache/`, `repro/`, `logs/`, `results/`, `mlruns/`, `.pt`, `.npz`, and `.zip` local.

## Repository Layout

```text
config/
  preprocess_config/     Per-session cache build configs
  train_config/          Train/eval split and normalization configs
  inference_config/      Single-session inference configs
scripts/
  inversion_si.py        The only public command-line entrypoint
src/
  cli.py                 Repository-wide command router
  common/                Shared artifact, frame, contour, phoneme, process helpers
  commands/              Read-only audits and split-cache commands
  preprocessing/         Session, incisor, and contour cache logic
  train/                 Split assembly and training
  inference/             Config-driven and dense-audio inference
  rendering/             Maintained video renderers
  orchestration/         Auto-batch, OAR, and external command launchers
  adaption_pipeline/     Modular ASD2-to-ASD1 stages and file-based DAG
  model/                 Baseline model
  utils/                 Lower-level numerical and rendering utilities
```

Run `scripts/inversion_si.py --help` for the complete command tree. Files below
`src/` are importable modules, not independent executables. The internally
retained adaptation compatibility core lives under
`src/adaption_pipeline/legacy/` and is reachable only through stable adapters.
See [the refactor inventory](docs/script_refactor.md) and
[the adaptation architecture](docs/adaption_pipeline_architecture.md).

## Environment

Use the inversion environment from the workspace:

```bash
cd /srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/Inversion_SI
../inversion/.venv/bin/python --version
```

On the Windows workspace, reuse the corresponding shared environment and add
the grid-transform runtime packages pinned by this repo:

```powershell
cd C:\Users\nhnguyen\PhD_A2A\Inversion_SI
..\inversion\.venv\Scripts\python.exe -m pip install imageio==2.37.2 roifile==2024.9.15 shapely==2.0.7 pydicom==2.4.4
..\inversion\.venv\Scripts\python.exe scripts\inversion_si.py grid-transform run_create_speaker_grid.py --help
```

The `grid-transform` command selects `Scripts/python.exe` on Windows and
`bin/python` on Linux. `PYTHON_BIN` still overrides that selection.

GPU training and GPU inference should run inside an OAR GPU allocation on Grid5000.
CPU-only config inspection and light preprocessing checks can run on the login node.

## Cache Layout

`session_cache_dir` is shared by preprocess and train configs.

Expected per-session cache files:

```text
cache/raw_sessions/<dataset>/<speaker_or_bucket>/<session>.pt
cache/raw_contour_npz/<dataset>/<speaker_or_bucket>/<session>.npz
```

Training assembles final split tensors into:

```text
repro/<run>/splits/train_sequences.pt
repro/<run>/splits/valid_sequences.pt
repro/<run>/splits/test_sequences.pt
repro/<run>/splits/session_cache_validation.json
```

No `bucket_parts` are required in the current pipeline.

## Preprocess

Preprocess configs only describe data/session information and where to write the reusable session cache.

Example:

```bash
PYTHONUNBUFFERED=1 ../inversion/.venv/bin/python scripts/inversion_si.py preprocess sessions \
  --config config/preprocess_config/asd1_11contour_sessions.yaml \
  --max-workers 5
```

Useful options:

```bash
--only-sessions P2/S1 P2/S2
--rebuild
--rebuild-contour-packs
--fail-fast
```

Each run writes a JSON report into `session_cache_dir` and prints a summary with declared, built, failed, and missing session counts.

## Training

Training configs define split membership, cache paths, and normalization behavior.

Important fields:

```yaml
session_cache_dir: /.../Inversion_SI/cache
split_cache_dir: /.../Inversion_SI/repro/<run>/splits
normalization_mode: train_global
normalization_fit_split: train_sequences
skip_bad_session_cache: true
```

Normalization policy:

- `normalization_mode: train_global` is the default unseen-speaker path: fit normalization on `normalization_fit_split` (default `train_sequences`) only, then apply the same stats to validation/test. Use this for ASD1 experiments such as P7 train/validation with P2 test-only.
- `normalization_mode: all_splits_global` is explicit opt-in for fitting normalization on train+validation+test together.
- Contour normalization uses a project minimum `normalization_contour_std_floor: 0.1`. Normal runs may keep or raise this floor; lower values are rejected unless `allow_low_contour_std_floor_diagnostic: true` is set for a diagnostic run. This avoids stale near-zero contour std values that make de-normalized predictions look almost static.
- Split cache metadata records the std-floor source (`default`, `normalization_contour_std_floor`, or legacy `contour_std_floor`) so old configs remain traceable.
- Inference validates the contour std floor for any split cache used to de-normalize predictions. If an old cache fails this check, rebuild the split cache with the current normalization policy.
- Cached-prediction render scripts record prediction-motion diagnostics in their summaries but do not block video rendering based on those diagnostics.
- Audio VTLN for final inversion RMSE/video must use `scripts/inversion_si.py preprocess vtln-cache`, which re-extracts MFCC through the Inversion_SI frontend and chunking. The legacy exported-NPZ override is no longer a public command.
- Session inference rejects configs containing the legacy `audio_vtln_feature_npz` key before loading the model, so stale exported-NPZ audio VTLN configs cannot create new under-moving prediction payloads by accident.
- Quick config audit:
  `PYTHONUNBUFFERED=1 ../inversion/.venv/bin/python scripts/inversion_si.py audit configs config/train_config/<file>.yaml`.
  The audit fails on low contour std-floor settings and legacy exported-NPZ audio VTLN configs; add `--allow-legacy-audio-vtln` only when inventorying old diagnostic configs.
- Quick split-cache audit:
  `PYTHONUNBUFFERED=1 ../inversion/.venv/bin/python scripts/inversion_si.py audit splits config/train_config/<file>.yaml --splits train_sequences test_sequences`.
  The audit loads split `.pt` files on CPU and fails if cached contour `std` is below the configured floor, which is the stale-cache failure mode that can make prediction contours look under-moving.

Launch training inside OAR:

```bash
PYTHONUNBUFFERED=1 ../inversion/.venv/bin/python scripts/inversion_si.py train model \
  --config config/train_config/asd1_11contour_trainnorm_p1val_p2test_paper_st5_mfcc_500epoch.yaml
```

Auto-batch helper:

```bash
PYTHONUNBUFFERED=1 ../inversion/.venv/bin/python scripts/inversion_si.py train auto-batch \
  --config config/train_config/asd1_11contour_trainnorm_p1val_p2test_paper_st5_mfcc_500epoch.yaml \
  --gpus 4 \
  --target-util 0.80
```

Prepare an OAR auto-batch job without submitting:

```bash
PYTHONUNBUFFERED=1 ../inversion/.venv/bin/python scripts/inversion_si.py train submit \
  --config config/train_config/asd1_11contour_trainnorm_p1val_p2test_paper_st5_mfcc_500epoch.yaml \
  --python ../inversion/.venv/bin/python \
  --gpus 1 \
  --cluster gres \
  --walltime 02:00:00
```

Pass `--submit` only when you intend to call `oarsub`. The helper runs the
CPU-safe split-cache preflight first, writes `job_auto_batch.sh` and
`submit_manifest.json`, and keeps the venv Python symlink path intact instead
of resolving it to the system Python.

After a successful test pass, training writes a compact result bundle to:

```text
results/<folder_save>/<run_name>/
```

The bundle includes `training_summary.json`, `training_summary.md`, and links/copies for key artifacts such as `best_model.pth`, `config.yaml`, `datasets.txt`, and RMSE reports.

## Inference

Inference is config-driven and targets one cached session.

Example:

```bash
PYTHONUNBUFFERED=1 ../inversion/.venv/bin/python scripts/inversion_si.py infer session \
  --config config/inference_config/asd1_11contour_trainnorm_p2_s1_video.yaml
```

The inference config contains:

```yaml
train_config: config/train_config/<train>.yaml
checkpoint: mlruns/<experiment>/<run>/artifacts/best_model.pth
output_dir: results/<run>/<session_inference>
target:
  speaker: 2
  session: 1
  split: test_sequences
write_contours: true
contour_output_format: xy50
```

Outputs include:

```text
output_dir/run_summary.json
output_dir/<mode>/summary.json
output_dir/<mode>/cached_session_predictions.pt
output_dir/<mode>/predicted_contours/*.npy
output_dir/eval_pred_vs_gt_mri.mp4
```

Predicted contours are averaged across overlapping sequence predictions per frame and articulator.

Render a target-label-only MRI video with all configured contours, without
prediction overlays or RMSE:

```bash
PYTHONUNBUFFERED=1 ../inversion/.venv/bin/python scripts/inversion_si.py render session \
  --predictions results/<run>/p7_s15/eval/cached_session_predictions.pt \
  --config config/train_config/<p7-config>.yaml \
  --output-dir results/<run>/p7_s15_ground_truth_video \
  --speaker 7 --session 15 --speaker-name P7 --session-name S15 \
  --mri-dicom-dir /path/to/P7/DCM_2D/S15 \
  --audio /path/to/DENOISED_SOUND_P7_S15.wav \
  --ground-truth-contour-dir /path/to/P7/S15/complete-contours \
  --timeline-step 1.0 \
  --ground-truth-only
```

Ground-truth-only rendering uses integer MRI frames only:

- The prediction payload is used only to select the session timeline range.
- Every integer frame is loaded from `--ground-truth-contour-dir`.
- Frame `.5` is not rendered and no contour is interpolated.
- If a configured contour file is missing or invalid, that contour is left
  empty and its class name is shown in the frame's `missing:` text.
- The renderer never holds or reuses a contour from another frame.
- `frame_metrics.csv` records the exact source and missing contour classes for
  every rendered frame.

Use `--prediction-only --prediction-contour-dir <dir> --timeline-step 1.0` to
render existing model contour files with the same integer-only, no-hold, and
missing-text policy. The MRI and audio paths default from `--speaker` and
`--session`; the renderer records the resolved MRI directory in `summary.json`
and refuses mismatched numeric/display session names. MRI frame caches also
record their DICOM source directory and are never reused for another speaker or
session.

For a strictly label-free prediction render, also pass an explicit integer
range:

```bash
PYTHONUNBUFFERED=1 ../inversion/.venv/bin/python scripts/inversion_si.py render session \
  --config config/train_config/<p7-config>.yaml \
  --output-dir results/<run>/p7_s15_prediction_video \
  --speaker 7 --session 15 --speaker-name P7 --session-name S15 \
  --prediction-only --prediction-contour-dir results/<run>/predicted_contours \
  --frame-min 143 --frame-max 1606 --timeline-step 1.0
```

With `--frame-min/--frame-max`, the renderer does not load a cached
prediction/label payload. It reads only the requested contour files, MRI
frames, audio, and class/config metadata. Missing contour files stay empty and
are listed as `missing:`; they are never interpolated or held.

### Direct non-overlapping P7/S15 inference

Use the prediction-only entry point with the same temporal contract as the
classic inversion pipeline: TextGrid tier-0 intervals define independent
speech sequences, each sequence is split into non-overlapping chunks no longer
than the training `sequence_length`, and every chunk is forwarded with its
actual length. The TextGrid supplies boundaries/silence only; target contours
are never loaded.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONPATH=.:src \
  ../inversion/.venv/bin/python scripts/inversion_si.py infer dense \
  --config config/train_config/<p7-config>.yaml \
  --checkpoint /path/to/best_model.pth \
  --audio /path/to/DENOISED_SOUND_P7_S15.wav \
  --textgrid /path/to/TEXT_ALIGNMENT_P7_S15.textgrid \
  --output-dir results/<run>/p7_s15_direct_chunks \
  --speaker 7 --session 15 \
  --frame-min 143 --frame-max 1606 \
  --inference-mode legacy_interval_chunks \
  --window-size 80 --batch-size 1 --device cuda
```

The script selects one MFCC nearest each integer MRI-frame center inside the
selected speech intervals, applies the training split's global normalization,
forwards every max-80 chunk independently, and concatenates the outputs
directly. With `--batch-size 1`, one model call corresponds to one chunk and
the final short chunk uses its real length. There is no overlap averaging,
interpolation, `.5` frame, or contour hold. Frames outside the selected
intervals get no contour files and the label-free renderer reports them as
`missing:`.

`full_sequence` and `overlapping_windows` remain diagnostic modes. The former
changes the recurrent context far beyond the training length; the latter
creates contributor-change seams by averaging overlapping BiLSTM windows.

## Grid Transform Submodule

`grid-transform` is included as a Git submodule at:

```text
external/grid-transform/
```

Clone/update it with:

```bash
git submodule update --init --recursive
```

The submodule contains the reusable `grid_transform/` Python package, bundled VTLN reference data under `VTLN/data/`, and the canonical command wrappers under `scripts/run/`.

Use the local helper to run a wrapper from this repo with the workspace inversion environment and the right `PYTHONPATH`:

```bash
scripts/inversion_si.py grid-transform run_create_speaker_grid.py --help
scripts/inversion_si.py grid-transform run_create_speaker_grid.py --source vtln --speaker 1640_P7_S2_F0829
```

Current environment note:

- `external/grid-transform/pyproject.toml` declares Python `>=3.10`.
- The shared workspace env `../inversion/.venv` is currently Python `3.9.2`, so the submodule is used through `PYTHONPATH` instead of editable install.
- Runtime dependencies needed by the submodule and not already present in the inversion env are pinned in `requirements.txt`: `imageio`, `roifile`, and `shapely`.

## Current Reference Configs

`config/train_config/` intentionally contains only five canonical training
setups:

```text
asd1_11contour_trainnorm_p1val_p2test_paper_st5_mfcc_500epoch.yaml
asd1_p7_only_train_global_pooledraw_st5_mfcc_500epoch_fixedbs10_2gpu_20260723.yaml
asd1_p7_seen_trainvaltest_paper_st5_mfcc_500epoch_stdfloor01.yaml
asd2_11contour_sofiane153_s25_bfincisor_train_global_st5_mfcc_500epoch.yaml
asd2_11contour_vtln_lowerrepairv2_upperlegacypos_20260721_train_global_rawstd_st5_mfcc_500epoch_fixedbs10_4gpu.yaml
```

Evaluation-only YAML belongs in `config/inference_config/`; preprocessing
overrides belong in `config/preprocess_config/`.

## Git Hygiene

Before pushing, stage source/config/script/docs only:

```bash
git add .gitignore README.md workflow.md config scripts src docs
git status --short
```

Do not stage generated outputs:

```text
cache/
repro/
logs/
results/
mlruns/
normalization_values/
*.pt
*.npz
*.zip
```
