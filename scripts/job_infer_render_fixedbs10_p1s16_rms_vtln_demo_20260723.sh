#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen"
LOG_DIR="${WORKSPACE_ROOT}/Inversion_SI/logs/asd2_fixedbs10_p1s16_rms_vtln_demo_20260723"
LOG_PATH="${LOG_DIR}/infer_render_oar6792053.log"

mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_PATH}") 2>&1

cd "${WORKSPACE_ROOT}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

if [[ "${OAR_JOB_ID:-}" != "6792053" ]]; then
    echo "Expected active OAR allocation 6792053, got ${OAR_JOB_ID:-unset}" >&2
    exit 1
fi

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

inversion/.venv/bin/python \
    Inversion_SI/scripts/infer_fixedbs10_p1s16_rms_vtln_exact_u.py

inversion/.venv/bin/python \
    Inversion_SI/scripts/render_asd2_to_asd1_grid_adaptation_demo.py \
    --force

date --iso-8601=seconds
