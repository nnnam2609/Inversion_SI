#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/Inversion_SI"
PYTHON_BIN="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/inversion/.venv/bin/python"
CONFIG="${REPO_ROOT}/config/train_config/asd2_11contour_vtln20260719_train_global_rawstd_st5_mfcc_500epoch.yaml"
TRAIN_GPUS="${TRAIN_GPUS:-1}"
TARGET_UTIL="${TARGET_UTIL:-0.80}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
TRAIN_SAMPLES=7504
MAX_BATCH="${MAX_BATCH:-$(((TRAIN_SAMPLES + TRAIN_GPUS - 1) / TRAIN_GPUS))}"
OUTPUT_CONFIG_DIR="${REPO_ROOT}/repro/asd2_11contour_vtln20260719_train_global/auto_batch_configs/${OAR_JOB_ID:-unknown}"

if [[ -z "${OAR_JOB_ID:-}" ]]; then
  echo "GPU training requires an OAR allocation (missing OAR_JOB_ID)." >&2
  exit 2
fi
if (( TRAIN_GPUS < 1 )); then
  echo "TRAIN_GPUS must be >= 1" >&2
  exit 2
fi

cd "${REPO_ROOT}"
if ! type module >/dev/null 2>&1; then
  source /etc/profile
fi
module purge
module load cuda/12.1.1

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${REPO_ROOT}/.cache/matplotlib"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "${OUTPUT_CONFIG_DIR}" "${MPLCONFIGDIR}"

echo "training_job_start oar_job_id=${OAR_JOB_ID} host=$(hostname -f) gpus=${TRAIN_GPUS} max_batch=${MAX_BATCH} target_util=${TARGET_UTIL} smoke_epochs=${SMOKE_EPOCHS} config=${CONFIG}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

"${PYTHON_BIN}" scripts/train_auto_batch.py \
  --config "${CONFIG}" \
  --gpus "${TRAIN_GPUS}" \
  --target-util "${TARGET_UTIL}" \
  --min-batch 1 \
  --max-batch "${MAX_BATCH}" \
  --smoke-epochs "${SMOKE_EPOCHS}" \
  --output-config-dir "${OUTPUT_CONFIG_DIR}"
