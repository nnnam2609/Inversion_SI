#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen"
RESULT_ROOT="${WORKSPACE_ROOT}/Inversion_SI/results/asd2_fixedbs10_selected_10speakers_adaptation_tables_20260724"
LOG_DIR="${WORKSPACE_ROOT}/Inversion_SI/logs/asd2_fixedbs10_p7_extension_20260724"
LOG_PATH="${LOG_DIR}/infer_and_tables_oar${OAR_JOB_ID:-unset}.log"

mkdir -p "${LOG_DIR}" "${RESULT_ROOT}"
exec > >(tee "${LOG_PATH}") 2>&1

if [[ -z "${OAR_JOB_ID:-}" ]]; then
    echo "This inference job must run inside OAR" >&2
    exit 1
fi

cd "${WORKSPACE_ROOT}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

date --iso-8601=seconds
hostname
echo "OAR_JOB_ID=${OAR_JOB_ID}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

inversion/.venv/bin/python \
    Inversion_SI/scripts/run_fixedbs10_p7_asd2_to_asd1_extension.py

inversion/.venv/bin/python \
    Inversion_SI/scripts/generate_fixedbs10_adaptation_tables_10speakers.py

date --iso-8601=seconds
