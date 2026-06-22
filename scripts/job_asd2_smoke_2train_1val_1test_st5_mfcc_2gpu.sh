#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${WORKSPACE_ROOT}/inversion/.venv"
CONFIG_PATH="${REPO_ROOT}/config/train_config/asd2_smoke_2train_1val_1test_st5_mfcc_2gpu.yaml"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="${REPO_ROOT}/logs/asd2_smoke_2train_1val_1test_st5_mfcc_2gpu/${RUN_ID}"
PREPROCESS_LOG="${RUN_LOG_DIR}/preprocess.log"
TRAIN_LOG="${RUN_LOG_DIR}/train.log"
SUMMARY_LOG="${RUN_LOG_DIR}/summary.log"

mkdir -p "${RUN_LOG_DIR}"
cd "${REPO_ROOT}"

if ! type module >/dev/null 2>&1; then
  source /etc/profile
fi

module purge || true
module load cuda/12.1.1 || true

source "${ENV_ROOT}/bin/activate"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${REPO_ROOT}/.cache/matplotlib"
export XDG_CACHE_HOME="${REPO_ROOT}/.cache"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

log_summary() {
  echo "$*" | tee -a "${SUMMARY_LOG}"
}

log_summary "run_id=${RUN_ID}"
log_summary "started_at=$(date -Is)"
log_summary "host=$(hostname)"
log_summary "OAR_JOB_ID=${OAR_JOB_ID:-none}"
log_summary "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log_summary "repo=${REPO_ROOT}"
log_summary "config=${CONFIG_PATH}"
log_summary "env=${ENV_ROOT}"

python - <<'PY' | tee -a "${SUMMARY_LOG}"
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        print(f"gpu_{idx}", torch.cuda.get_device_name(idx))
PY

preprocess_start=$(date +%s)
log_summary "preprocess_started_at=$(date -Is)"
python scripts/prepare_asd2_full_single_task5_cache.py \
  --config "${CONFIG_PATH}" \
  --splits train_sequences valid_sequences test_sequences \
  --max-workers 4 \
  --rebuild-parts \
  --rebuild-assembled \
  --progress-every 10 \
  2>&1 | tee "${PREPROCESS_LOG}"
preprocess_end=$(date +%s)
log_summary "preprocess_finished_at=$(date -Is)"
log_summary "preprocess_elapsed_seconds=$((preprocess_end - preprocess_start))"
log_summary "preprocess_log=${PREPROCESS_LOG}"

train_start=$(date +%s)
log_summary "train_started_at=$(date -Is)"
python src/main_train.py --config "${CONFIG_PATH}" 2>&1 | tee "${TRAIN_LOG}"
train_end=$(date +%s)
log_summary "train_finished_at=$(date -Is)"
log_summary "train_elapsed_seconds=$((train_end - train_start))"
log_summary "train_log=${TRAIN_LOG}"
log_summary "finished_at=$(date -Is)"
