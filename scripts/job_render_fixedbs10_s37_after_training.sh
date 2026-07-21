#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <training-parent-pid>" >&2
  exit 2
fi

repo_root="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/Inversion_SI"
training_pid="$1"
training_log="${repo_root}/logs/asd2_fixedbs10_4gpu_20260721/train_oar6786593_4gpu.log"
train_config="${repo_root}/config/train_config/asd2_11contour_vtln_lowerrepairv2_upperlegacypos_20260721_train_global_rawstd_st5_mfcc_500epoch_fixedbs10_4gpu.yaml"
checkpoint="${repo_root}/mlruns/846774538033499469/39e2314f4d02443a8e065ccdfc04bcb3/artifacts/best_model.pth"
train_norm="${repo_root}/repro/asd2_11contour_vtln_lowerrepairv2_upperlegacypos_20260721_train_global/splits/train_sequences.pt"
result_root="${repo_root}/results/asd2_fixedbs10_best_test_1775_s37_inference_video_20260721"
inference_dir="${result_root}/eval"
render_dir="${result_root}/continuous_50fps_audio"
mri_dir="/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/ArtSpeech_Database_2/1775/S37/NPY_MR_registered"
audio_path="/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/ArtSpeech_Database_2/1775/S37/1775_S37.wav"
video_path="${render_dir}/prediction/p1775_s37_asd2_fixedbs10_best_prediction_11contour_audio.mp4"
summary_path="${render_dir}/summary.json"

cd "${repo_root}"

while kill -0 "${training_pid}" 2>/dev/null; do
  sleep 30
done

if ! awk '
  /full_training_start/ { full = 1; next }
  full && /Training finished successfully/ { complete = 1 }
  END { exit complete ? 0 : 1 }
' "${training_log}"; then
  echo "Full training did not report successful completion; video generation is blocked." >&2
  exit 1
fi

if [[ ! -f "${checkpoint}" ]]; then
  echo "Missing actual best checkpoint: ${checkpoint}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

../inversion/.venv/bin/python -m src.inference.session_inference \
  --config "${train_config}" \
  --checkpoint "${checkpoint}" \
  --speaker 1775 \
  --session 37 \
  --split test_sequences \
  --output-dir "${inference_dir}" \
  --device cuda \
  --prediction-denorm-cache "${train_norm}" \
  --write-contours \
  --contour-output-format xy50

../inversion/.venv/bin/python scripts/render_gridnorm_session_video.py \
  --config "${train_config}" \
  --output-dir "${render_dir}" \
  --speaker 1775 \
  --session 37 \
  --speaker-name P1775 \
  --session-name S37 \
  --mri-npy-dir "${mri_dir}" \
  --audio "${audio_path}" \
  --prediction-only \
  --prediction-contour-dir "${inference_dir}/predicted_contours" \
  --frame-min 1 \
  --frame-max 4000 \
  --timeline-step 1.0 \
  --ms-image 20.0 \
  --prediction-model-label ASD2_fixedbs10_best \
  --scale 4

test "$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 "${video_path}")" = "50/1"
test "$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 "${video_path}")" = "4000"
test "$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_type -of default=noprint_wrappers=1:nokey=1 "${video_path}")" = "audio"
../inversion/.venv/bin/python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["fps"] == 50.0; assert p["rendered_fractional_frame_count"] == 0; assert p["outputs"][0]["num_held_frames"] == 0; assert p["outputs"][0]["rendered_fractional_frame_count"] == 0' "${summary_path}"

echo "fixedbs10_s37_video_complete video=${video_path} summary=${summary_path}"
