# Inversion_SI

Cache-first articulatory inversion pipeline for ASD1/ASD2 single-task 5 experiments.

The code is organized around three explicit steps:

1. Build reusable per-session preprocessing cache.
2. Train/evaluate from cached sessions and assemble final split tensors.
3. Run single-session inference from a YAML config, optionally rendering video and exporting predicted contours.

Large generated files are intentionally excluded from git. Keep `cache/`, `repro/`, `logs/`, `results/`, `mlruns/`, `.pt`, `.npz`, and `.zip` local.

## Repository Layout

```text
config/
  preprocess_config/     Per-session cache build configs
  train_config/          Train/eval split and normalization configs
  inference_config/      Single-session inference configs
scripts/
  preprocess_sessions.py Build per-session .pt and .npz cache
  train_auto_batch.py    Tune batch size then launch training
  infer_session.py       Run one configured inference target
  render_cached_compare_video.py
src/
  preprocessing/         Session cache and contour loading logic
  train/                 Split assembly and training logic
  inference/             Config-driven inference logic
  model/                 Baseline model
  utils/                 Dataset, metrics, DDP, and YAML helpers
```

## Environment

Use the inversion environment from the workspace:

```bash
cd /srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/Inversion_SI
../inversion/.venv/bin/python --version
```

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
PYTHONUNBUFFERED=1 ../inversion/.venv/bin/python scripts/preprocess_sessions.py \
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
- Audio VTLN for final inversion RMSE/video must use `scripts/build_inversion_frontend_vtln_eval_cache.py`, which re-extracts MFCC through the Inversion_SI frontend and chunking. The legacy exported-NPZ override script is diagnostic-only and is blocked by default.
- Session inference rejects configs containing the legacy `audio_vtln_feature_npz` key before loading the model, so stale exported-NPZ audio VTLN configs cannot create new under-moving prediction payloads by accident.
- Quick config audit:
  `PYTHONUNBUFFERED=1 ../inversion/.venv/bin/python scripts/audit_normalization_configs.py config/train_config/<file>.yaml`.
  The audit fails on low contour std-floor settings and legacy exported-NPZ audio VTLN configs; add `--allow-legacy-audio-vtln` only when inventorying old diagnostic configs.
- Quick split-cache audit:
  `PYTHONUNBUFFERED=1 ../inversion/.venv/bin/python scripts/audit_split_cache_normalization.py config/train_config/<file>.yaml --splits train_sequences test_sequences`.
  The audit loads split `.pt` files on CPU and fails if cached contour `std` is below the configured floor, which is the stale-cache failure mode that can make prediction contours look under-moving.

Launch training inside OAR:

```bash
PYTHONUNBUFFERED=1 ../inversion/.venv/bin/python src/main_train.py \
  --config config/train_config/asd1_11contour_trainnorm_p1val_p2test_paper_st5_mfcc_500epoch.yaml
```

Auto-batch helper:

```bash
PYTHONUNBUFFERED=1 ../inversion/.venv/bin/python scripts/train_auto_batch.py \
  --config config/train_config/asd1_11contour_trainnorm_p1val_p2test_paper_st5_mfcc_500epoch.yaml \
  --gpus 4 \
  --target-util 0.80
```

Prepare an OAR auto-batch job without submitting:

```bash
PYTHONUNBUFFERED=1 ../inversion/.venv/bin/python scripts/submit_auto_batch_oar.py \
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
PYTHONUNBUFFERED=1 ../inversion/.venv/bin/python scripts/infer_session.py \
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
PYTHONUNBUFFERED=1 ../inversion/.venv/bin/python scripts/render_gridnorm_session_video.py \
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

### Dense audio inference for every integer MRI frame

The regular session inference command reads a cached dataset split. That split
is built from non-silence TextGrid intervals and may therefore omit MRI frames
even though the model itself predicts contours from MFCC audio. Do not use its
sparse contour directory when a video must contain a fresh prediction at every
integer MRI frame.

Use the dense prediction-only entry point instead (on an OAR GPU allocation):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONPATH=.:src \
  ../inversion/.venv/bin/python scripts/infer_dense_audio_integer_contours.py \
  --config config/train_config/<p7-config>.yaml \
  --checkpoint /path/to/best_model.pth \
  --audio /path/to/DENOISED_SOUND_P7_S15.wav \
  --output-dir results/<run>/p7_s15_dense_audio_integer_inference \
  --speaker 7 --session 15 \
  --frame-min 143 --frame-max 1606 \
  --inference-mode full_sequence --device cuda
```

This path extracts the full audio MFCC sequence with the training frontend,
uses the training split's global normalization, and runs one direct model
forward over the entire selected MFCC sequence. It writes exactly one direct
prediction for every requested integer frame and each configured contour. It
never reads target contours, never uses a TextGrid or silence filter, never
creates `.5` frames, and never interpolates, averages windows, or holds a
contour. The command fails if any expected output is missing, non-finite, or if
the output directory contains stale unexpected contour files. The legacy
`--inference-mode overlapping_windows` path is diagnostic-only because changing
BiLSTM window contributors creates periodic contour jumps.

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
scripts/run_grid_transform.py run_create_speaker_grid.py --help
scripts/run_grid_transform.py run_create_speaker_grid.py --source vtln --speaker 1640_P7_S2_F0829
```

Current environment note:

- `external/grid-transform/pyproject.toml` declares Python `>=3.10`.
- The shared workspace env `../inversion/.venv` is currently Python `3.9.2`, so the submodule is used through `PYTHONPATH` instead of editable install.
- Runtime dependencies needed by the submodule and not already present in the inversion env are pinned in `requirements.txt`: `imageio`, `roifile`, and `shapely`.

## Current Reference Configs

ASD1:

```text
config/preprocess_config/asd1_11contour_sessions.yaml
config/train_config/asd1_11contour_trainnorm_p1val_p2test_paper_st5_mfcc_500epoch.yaml
config/inference_config/asd1_11contour_trainnorm_p2_s1_video.yaml
```

ASD2:

```text
config/preprocess_config/asd2_11contour_full_sessions.yaml
config/preprocess_config/asd2_11contour_original_incisor_only_sessions.yaml
config/train_config/asd2_11contour_full_preprocessed_paper_st5_mfcc_500epoch.yaml
config/train_config/asd2_11contour_original_incisor_only_paper_st5_mfcc_500epoch.yaml
```

## Git Hygiene

Before pushing, stage source/config/script/docs only:

```bash
git add .gitignore README.md config scripts src
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
