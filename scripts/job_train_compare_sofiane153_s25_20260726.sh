#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen"
REPO="${WORKSPACE}/Inversion_SI"
MOVING_CODE="${WORKSPACE}/Inversion_SI_sofiane153_s25_moving_20260726"
PYTHON="${WORKSPACE}/inversion/.venv/bin/python"
GLOBAL_CONFIG="${REPO}/config/train_config/asd2_11contour_sofiane153_s25_bfincisor_train_global_st5_mfcc_500epoch.yaml"
MOVING_CONFIG="${REPO}/repro/asd2_11contour_sofiane153_s25_bfincisor_moving_average_20260726/runtime_configs/asd2_11contour_sofiane153_s25_bfincisor_moving_average_fixedbs10_4gpu.yaml"
GLOBAL_SMOKE="${REPO}/repro/asd2_11contour_sofiane153_s25_comparison_20260726/smoke_configs/global_smoke1epoch.yaml"
MOVING_SMOKE="${REPO}/repro/asd2_11contour_sofiane153_s25_comparison_20260726/smoke_configs/moving_average_smoke1epoch.yaml"
GLOBAL_SPLITS="${REPO}/repro/asd2_11contour_sofiane153_s25_bfincisor_train_global_20260725/splits"
MOVING_SPLITS="${REPO}/repro/asd2_11contour_sofiane153_s25_bfincisor_moving_average_20260726/splits"
RESULT="${REPO}/results/asd2_sofiane153_s25_global_vs_moving_oracle_20260726"
STAGE_LOGS="${RESULT}/logs"
PAIRED="${RESULT}/paired_integer_rmse.npz"
PAIRED_AUDIT="${RESULT}/paired_integer_extraction_audit.json"
TABLE_DIR="${RESULT}/table"
MRI_DIR="/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/ArtSpeech_Database_2/1775/S37/NPY_MR_registered"
AUDIO="/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/ArtSpeech_Database_2/1775/S37/1775_S37.wav"
GROUND_TRUTH="${REPO}/cache_variants/asd2_11_bfincisor_sofiane153_s25_20260725/raw_contour_npz/asd2/1775/S37.npz"

if [[ -z "${OAR_JOB_ID:-}" ]]; then
  echo "This training workflow requires an OAR GPU allocation." >&2
  exit 2
fi

cd "${REPO}"
if ! type module >/dev/null 2>&1; then
  source /etc/profile
fi
module purge
module load cuda/12.1.1

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${REPO}/.cache/matplotlib"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "${STAGE_LOGS}" "${TABLE_DIR}" "${MPLCONFIGDIR}"

echo "workflow_start=$(date -Is) oar_job_id=${OAR_JOB_ID} host=$(hostname -f)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
GPU_COUNT="$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${GPU_COUNT}" != "4" ]]; then
  echo "Expected exactly 4 CUDA devices for effective batch size 40; found ${GPU_COUNT}." >&2
  exit 3
fi

for required in \
  "${GLOBAL_CONFIG}" "${MOVING_CONFIG}" "${GLOBAL_SMOKE}" "${MOVING_SMOKE}" \
  "${MOVING_CODE}/src/main_train.py" "${GROUND_TRUTH}" "${AUDIO}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required file: ${required}" >&2
    exit 4
  fi
done
for required_dir in "${GLOBAL_SPLITS}" "${MOVING_SPLITS}" "${MRI_DIR}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Missing required directory: ${required_dir}" >&2
    exit 4
  fi
done

echo "preflight_global_start=$(date -Is)"
PYTHONPATH="${REPO}:${REPO}/src" "${PYTHON}" "${REPO}/scripts/train_auto_batch.py" \
  --config "${GLOBAL_CONFIG}" --gpus 4 --preflight-only \
  2>&1 | tee "${STAGE_LOGS}/preflight_global.log"
echo "preflight_moving_start=$(date -Is)"
PYTHONPATH="${MOVING_CODE}:${MOVING_CODE}/src" "${PYTHON}" "${MOVING_CODE}/scripts/train_auto_batch.py" \
  --config "${MOVING_CONFIG}" --gpus 4 --preflight-only \
  2>&1 | tee "${STAGE_LOGS}/preflight_moving.log"

run_training() {
  local label="$1"
  local code_root="$2"
  local config="$3"
  local log_path="$4"
  echo "${label}_start=$(date -Is) code_root=${code_root} config=${config}"
  PYTHONPATH="${code_root}:${code_root}/src" "${PYTHON}" "${code_root}/src/main_train.py" \
    --config "${config}" 2>&1 | tee "${log_path}"
  grep -q "Training finished successfully" "${log_path}"
  echo "${label}_complete=$(date -Is)"
}

latest_summary() {
  local folder="$1"
  "${PYTHON}" - "${REPO}/results/${folder}" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
matches = sorted(root.glob("*/training_summary.json"), key=lambda p: p.stat().st_mtime)
if not matches:
    raise SystemExit(f"No training_summary.json under {root}")
print(matches[-1].resolve())
PY
}

json_value() {
  local json_path="$1"
  local key="$2"
  "${PYTHON}" - "${json_path}" "${key}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload[sys.argv[2]])
PY
}

run_training "smoke_global" "${REPO}" "${GLOBAL_SMOKE}" "${STAGE_LOGS}/smoke_global.log"
GLOBAL_SMOKE_SUMMARY="$(latest_summary "asd2_11contour_sofiane153_s25_bfincisor_train_global_st5_mfcc_500epoch_smoke1epoch")"
test -f "$(json_value "${GLOBAL_SMOKE_SUMMARY}" best_model)"

run_training "smoke_moving" "${MOVING_CODE}" "${MOVING_SMOKE}" "${STAGE_LOGS}/smoke_moving.log"
MOVING_SMOKE_SUMMARY="$(latest_summary "asd2_11contour_sofiane153_s25_bfincisor_moving_average_500epoch_smoke1epoch")"
test -f "$(json_value "${MOVING_SMOKE_SUMMARY}" best_model)"
echo "both_smokes_passed=$(date -Is)"

run_training "full_global" "${REPO}" "${GLOBAL_CONFIG}" "${STAGE_LOGS}/full_global.log"
GLOBAL_SUMMARY="$(latest_summary "asd2_11contour_sofiane153_s25_bfincisor_train_global_st5_mfcc_500epoch")"
GLOBAL_CHECKPOINT="$(json_value "${GLOBAL_SUMMARY}" best_model)"
test -f "${GLOBAL_CHECKPOINT}"

run_training "full_moving" "${MOVING_CODE}" "${MOVING_CONFIG}" "${STAGE_LOGS}/full_moving.log"
MOVING_SUMMARY="$(latest_summary "asd2_11contour_sofiane153_s25_bfincisor_moving_average_500epoch")"
MOVING_CHECKPOINT="$(json_value "${MOVING_SUMMARY}" best_model)"
test -f "${MOVING_CHECKPOINT}"

{
  echo "global_summary=${GLOBAL_SUMMARY}"
  echo "global_checkpoint=${GLOBAL_CHECKPOINT}"
  echo "moving_summary=${MOVING_SUMMARY}"
  echo "moving_checkpoint=${MOVING_CHECKPOINT}"
} > "${RESULT}/resolved_training_artifacts.txt"

echo "paired_evaluation_start=$(date -Is)"
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH="${MOVING_CODE}:${MOVING_CODE}/src" "${PYTHON}" \
  "${REPO}/scripts/extract_paired_oracle_rmse.py" \
  --branch-worktree "${MOVING_CODE}" \
  --config "${MOVING_CONFIG}" \
  --global-cache "${GLOBAL_SPLITS}/test_sequences.pt" \
  --global-checkpoint "${GLOBAL_CHECKPOINT}" \
  --moving-cache "${MOVING_SPLITS}/test_sequences.pt" \
  --moving-checkpoint "${MOVING_CHECKPOINT}" \
  --output "${PAIRED}" \
  --audit "${PAIRED_AUDIT}" \
  --batch-size 64 --device cuda:0 --mm-per-pixel 1.62 \
  2>&1 | tee "${STAGE_LOGS}/paired_evaluation.log"

"${PYTHON}" "${REPO}/scripts/generate_normalization_comparison.py" \
  --paired-npz "${PAIRED}" \
  --extraction-audit "${PAIRED_AUDIT}" \
  --global-summary "${GLOBAL_SUMMARY}" \
  --moving-summary "${MOVING_SUMMARY}" \
  --output-dir "${TABLE_DIR}" \
  2>&1 | tee "${STAGE_LOGS}/table_generation.log"

echo "s37_inference_start=$(date -Is)"
GLOBAL_INFERENCE="${RESULT}/inference/global_1775_s37"
MOVING_INFERENCE="${RESULT}/inference/moving_oracle_1775_s37"
PYTHONPATH="${REPO}:${REPO}/src" "${PYTHON}" \
  "${REPO}/scripts/infer_cached_session_centered.py" \
  --code-root "${REPO}" --config "${GLOBAL_CONFIG}" \
  --checkpoint "${GLOBAL_CHECKPOINT}" \
  --cache "${GLOBAL_SPLITS}/train_sequences.pt" \
  --speaker 1775 --session 37 --center-key mean \
  --output-dir "${GLOBAL_INFERENCE}" --device cuda:0 \
  2>&1 | tee "${STAGE_LOGS}/inference_global_s37.log"
PYTHONPATH="${MOVING_CODE}:${MOVING_CODE}/src" "${PYTHON}" \
  "${REPO}/scripts/infer_cached_session_centered.py" \
  --code-root "${MOVING_CODE}" --config "${MOVING_CONFIG}" \
  --checkpoint "${MOVING_CHECKPOINT}" \
  --cache "${MOVING_SPLITS}/train_sequences.pt" \
  --speaker 1775 --session 37 --center-key sofiane_moving_average \
  --output-dir "${MOVING_INFERENCE}" --device cuda:0 \
  2>&1 | tee "${STAGE_LOGS}/inference_moving_s37.log"

GLOBAL_RENDER="${RESULT}/videos/global"
MOVING_RENDER="${RESULT}/videos/moving_oracle"
"${PYTHON}" "${REPO}/scripts/render_gridnorm_session_video.py" \
  --config "${GLOBAL_CONFIG}" --output-dir "${GLOBAL_RENDER}" \
  --speaker 1775 --session 37 --speaker-name P1775 --session-name S37 \
  --mri-npy-dir "${MRI_DIR}" --audio "${AUDIO}" \
  --ground-truth-prediction-compare --ground-truth-contour-pack "${GROUND_TRUTH}" \
  --prediction-contour-dir "${GLOBAL_INFERENCE}/predicted_contours" \
  --frame-min 1 --frame-max 4000 --timeline-step 1.0 --ms-image 20.0 \
  --prediction-model-label "Global" --scale 4 --remove-silent-after-audio \
  2>&1 | tee "${STAGE_LOGS}/render_global_s37.log"
"${PYTHON}" "${REPO}/scripts/render_gridnorm_session_video.py" \
  --config "${MOVING_CONFIG}" --output-dir "${MOVING_RENDER}" \
  --speaker 1775 --session 37 --speaker-name P1775 --session-name S37 \
  --mri-npy-dir "${MRI_DIR}" --audio "${AUDIO}" \
  --ground-truth-prediction-compare --ground-truth-contour-pack "${GROUND_TRUTH}" \
  --prediction-contour-dir "${MOVING_INFERENCE}/predicted_contours" \
  --frame-min 1 --frame-max 4000 --timeline-step 1.0 --ms-image 20.0 \
  --prediction-model-label "Moving average oracle" --scale 4 --remove-silent-after-audio \
  2>&1 | tee "${STAGE_LOGS}/render_moving_s37.log"

GLOBAL_VIDEO="${GLOBAL_RENDER}/prediction/p1775_s37_global_prediction_11contour_audio.mp4"
MOVING_VIDEO="${MOVING_RENDER}/prediction/p1775_s37_moving_average_oracle_prediction_11contour_audio.mp4"
SIDE_VIDEO="${RESULT}/videos/p1775_s37_global_vs_moving_average_oracle_side_by_side_audio.mp4"
test -f "${GLOBAL_VIDEO}"
test -f "${MOVING_VIDEO}"
ffmpeg -hide_banner -loglevel error -y \
  -i "${GLOBAL_VIDEO}" -i "${MOVING_VIDEO}" \
  -filter_complex "[0:v:0][1:v:0]hstack=inputs=2[v]" \
  -map "[v]" -map "0:a:0?" -c:v libx264 -preset medium -crf 18 \
  -c:a aac -b:a 192k -shortest "${SIDE_VIDEO}"

"${PYTHON}" - \
  "${GLOBAL_RENDER}/summary.json" "${MOVING_RENDER}/summary.json" \
  "${GLOBAL_VIDEO}" "${MOVING_VIDEO}" "${SIDE_VIDEO}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

for summary_path in map(Path, sys.argv[1:3]):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["fps"] == 50.0
    assert summary["rendered_fractional_frame_count"] == 0
    assert summary["num_timeline_frames"] == 4000
    assert len(summary["outputs"]) == 1
    assert summary["outputs"][0]["num_frames"] == 4000
    assert summary["outputs"][0]["num_held_frames"] == 0
    assert summary["outputs"][0]["audio_attached"] is True

for video in map(Path, sys.argv[3:]):
    def probe(entry, selector="v:0"):
        return subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", selector,
                "-show_entries", f"stream={entry}",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video),
            ],
            text=True,
        ).strip()
    assert probe("avg_frame_rate") == "50/1", (video, probe("avg_frame_rate"))
    assert probe("nb_frames") == "4000", (video, probe("nb_frames"))
    assert probe("codec_type", "a:0") == "audio"
PY

export GLOBAL_SUMMARY MOVING_SUMMARY GLOBAL_CHECKPOINT MOVING_CHECKPOINT
export GLOBAL_VIDEO MOVING_VIDEO SIDE_VIDEO RESULT
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

result = Path(os.environ["RESULT"])
artifacts = {}
for key in (
    "GLOBAL_SUMMARY", "MOVING_SUMMARY", "GLOBAL_CHECKPOINT",
    "MOVING_CHECKPOINT", "GLOBAL_VIDEO", "MOVING_VIDEO", "SIDE_VIDEO",
):
    path = Path(os.environ[key]).resolve()
    artifacts[key.lower()] = {"path": str(path), "sha256": sha256(path)}
manifest = {
    "status": "complete",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "oar_job_id": os.environ["OAR_JOB_ID"],
    "operation": "two smoke trainings, two full trainings, paired evaluation, and S37 inference videos",
    "environment": "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/inversion/.venv",
    "gpu_count": 4,
    "effective_batch_size": 40,
    "seed": 42,
    "split_counts": {"train": 122, "validation": 14, "test": 17},
    "moving_average_warning": "Exact-Sofiane oracle uses target-contour moving-average statistics.",
    "artifacts": artifacts,
}
(result / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
PY

echo "workflow_complete=$(date -Is) result=${RESULT}"
