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
speaker_independent: true
skip_bad_session_cache: true
```

Normalization policy:

- `speaker_independent: true`: fit normalization on train only, then apply to validation/test. Use this for ASD1 speaker-independent experiments such as P1 validation and P2 test.
- `speaker_independent: false`: fit normalization on train+validation+test and apply to all splits. Use this for ASD2 same-speaker/session-bucket experiments.

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
