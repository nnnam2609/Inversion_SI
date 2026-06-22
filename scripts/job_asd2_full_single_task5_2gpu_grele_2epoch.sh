#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${WORKSPACE_ROOT}/inversion/.venv"
CONFIG_PATH="${REPO_ROOT}/config/train_config/asd2_full_single_task5_2gpu_grele_2epoch.yaml"
RUN_LOG_DIR="${REPO_ROOT}/logs/asd2_full_single_task5_mfcc_2gpu_grele_2epoch"
RUN_CAPTURE="${RUN_LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

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

echo "host=$(hostname)"
echo "OAR_JOB_ID=${OAR_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "config=${CONFIG_PATH}"
python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        print(f"gpu_{idx}", torch.cuda.get_device_name(idx))
PY

python src/main_train.py --config "${CONFIG_PATH}" 2>&1 | tee "${RUN_CAPTURE}"
echo "run_capture=${RUN_CAPTURE}"
