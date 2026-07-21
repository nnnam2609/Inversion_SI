#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <training-parent-pid>" >&2
  exit 2
fi

repo_root="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/Inversion_SI"
training_pid="$1"
training_log="${repo_root}/logs/asd2_fixedbs10_4gpu_20260721/train_oar6786593_4gpu.log"
comparison_output="${repo_root}/repro/asd2_11contour_vtln_lowerrepairv2_upperlegacypos_20260721_fixedbs10_4gpu/integer_only_batch1150_vs_batch10_comparison.json"

cd "${repo_root}"

while kill -0 "${training_pid}" 2>/dev/null; do
  sleep 30
done

if ! awk '
  /full_training_start/ { full = 1; next }
  full && /Training finished successfully/ { complete = 1 }
  END { exit complete ? 0 : 1 }
' "${training_log}"; then
  echo "Full training did not report successful completion; comparison is blocked." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

../inversion/.venv/bin/python scripts/evaluate_asd2_retrain_integer_only.py \
  --config config/train_config/asd2_11contour_vtln_lowerrepairv2_upperlegacypos_20260721_train_global_rawstd_st5_mfcc_500epoch_fixedbs10_4gpu.yaml \
  --test-cache repro/asd2_11contour_vtln_lowerrepairv2_upperlegacypos_20260721_train_global/splits/test_sequences.pt \
  --target-normalization repro/asd2_11contour_vtln_lowerrepairv2_upperlegacypos_20260721_train_global/splits/normalization_stats.npz \
  --reference-checkpoint results/asd2_11contour_vtln_lowerrepairv2_upperlegacypos_20260721_train_global_rawstd_st5_mfcc_500epoch/single_task5_asd2_11contour_vtln_lowerrepairv2_upperlegacypos_20260721_train_global_rawstd_st5_mfcc_500epoch_11_articulators_ac_e_ll_p_spm_t_ul_vf_tc_li_ui/best_model.pth \
  --reference-normalization repro/asd2_11contour_vtln_lowerrepairv2_upperlegacypos_20260721_train_global/splits/normalization_stats.npz \
  --reference-name current_incisor_batch1150_per_gpu \
  --candidate-checkpoint mlruns/846774538033499469/39e2314f4d02443a8e065ccdfc04bcb3/artifacts/best_model.pth \
  --candidate-normalization repro/asd2_11contour_vtln_lowerrepairv2_upperlegacypos_20260721_train_global/splits/normalization_stats.npz \
  --candidate-name current_incisor_batch10_per_gpu \
  --output "${comparison_output}" \
  --batch-size 64 \
  --mm-per-pixel 1.62 \
  --device cuda:0
