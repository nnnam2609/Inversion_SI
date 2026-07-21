#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/Inversion_SI"
PYTHON_BIN="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/inversion/.venv/bin/python"
LOG_ROOT="${REPO_ROOT}/logs/asd2_11_vtln_cache_20260719"
RUN_LOG="${RUN_LOG:-${LOG_ROOT}/split_build.log}"
STATUS_FILE="${STATUS_FILE:-${LOG_ROOT}/split_build.status}"
CONFIG="config/train_config/asd2_11contour_vtln20260719_train_global_rawstd_st5_mfcc_500epoch.yaml"

if [[ -z "${OAR_JOB_ID:-}" ]]; then
  echo "This split-cache build must run on an allocated compute node (missing OAR_JOB_ID)." >&2
  exit 2
fi

cd "${REPO_ROOT}"
mkdir -p "${LOG_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

set +e
"${PYTHON_BIN}" scripts/build_split_caches.py --config "${CONFIG}" >"${RUN_LOG}" 2>&1
status=$?
set -e
printf 'EXIT:%s\n' "${status}" >"${STATUS_FILE}"
exit "${status}"
