#!/usr/bin/env bash
set -euo pipefail

repo_root="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/Inversion_SI"
python_bin="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/inversion/.venv/bin/python"
config="${repo_root}/config/train_config/asd1_p7_only_train_global_pooledraw_st5_mfcc_500epoch_fixedbs10_2gpu_20260723.yaml"
repro_root="${repo_root}/repro/asd1_p7_only_train_global_pooledraw_fixedbs10_2gpu_20260723"
runtime_config_dir="${repro_root}/runtime_configs"
training_results_root="${repo_root}/results/asd1_p7_only_train_global_pooledraw_fixedbs10_2gpu_20260723"
result_root="${repo_root}/results/p7_only_pooledraw_bs10_best_p7_s15_compare_20260723"
inference_dir="${result_root}/legacy_interval_inference"
render_dir="${result_root}/continuous_native_fps_audio"
log_dir="${repo_root}/logs/asd1_p7_only_pooledraw_bs10_2gpu_20260723"
combined_log="${log_dir}/train_infer_render_oar${OAR_JOB_ID:-unknown}_2gpu.log"

mri_dir="/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/ArtSpeech_Database_1_raw/P7/DCM_2D/S15"
audio_path="/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/ArtSpeech_Database_1_raw/P7/OTHER/S15/DENOISED_SOUND_P7_S15.wav"
textgrid_path="/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/ArtSpeech_Database_1_raw/P7/OTHER/S15/TEXT_ALIGNMENT_P7_S15.textgrid"
ground_truth_pack="${repo_root}/cache/raw_contour_npz/asd1/P7/S15.npz"
normalization_stats="${repro_root}/splits/normalization_stats.npz"

if [[ -z "${OAR_JOB_ID:-}" ]]; then
  echo "GPU training/inference requires an OAR allocation (missing OAR_JOB_ID)." >&2
  exit 2
fi

cd "${repo_root}"
if ! type module >/dev/null 2>&1; then
  source /etc/profile
fi
module purge
module load cuda/12.1.1

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${repo_root}/.cache/matplotlib"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${runtime_config_dir}" "${result_root}" "${log_dir}" "${MPLCONFIGDIR}"
exec > >(tee -a "${combined_log}") 2>&1

echo "pipeline_start timestamp=$(date --iso-8601=seconds) oar_job_id=${OAR_JOB_ID} host=$(hostname -f)"
echo "operation=from_scratch_training_then_p7_s15_inference_and_render"
echo "config=${config} environment=${python_bin} result_root=${result_root}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

"${python_bin}" scripts/train_auto_batch.py \
  --config "${config}" \
  --gpus 2 \
  --target-util 0.80 \
  --min-batch 10 \
  --max-batch 10 \
  --smoke-epochs 1 \
  --output-config-dir "${runtime_config_dir}"

training_summary="$(${python_bin} - "${training_results_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summaries = sorted(root.glob("*/training_summary.json"), key=lambda path: path.stat().st_mtime)
if len(summaries) != 1:
    raise SystemExit(f"Expected exactly one full-run training summary under {root}, found {len(summaries)}")
summary = json.loads(summaries[0].read_text(encoding="utf-8"))
checkpoint = Path(summary["best_model"])
if not checkpoint.is_file():
    raise SystemExit(f"Missing best checkpoint: {checkpoint}")
print(summaries[0])
PY
)"
checkpoint="$(${python_bin} - "${training_summary}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["best_model"])
PY
)"

echo "full_training_complete timestamp=$(date --iso-8601=seconds) summary=${training_summary} checkpoint=${checkpoint}"

"${python_bin}" scripts/infer_dense_audio_integer_contours.py \
  --config "${config}" \
  --checkpoint "${checkpoint}" \
  --audio "${audio_path}" \
  --textgrid "${textgrid_path}" \
  --output-dir "${inference_dir}" \
  --speaker 7 \
  --session 15 \
  --frame-min 1 \
  --frame-max 2200 \
  --inference-mode legacy_interval_chunks \
  --window-size 80 \
  --batch-size 32 \
  --device cuda \
  --normalization-stats "${normalization_stats}"

"${python_bin}" scripts/render_gridnorm_session_video.py \
  --config "${config}" \
  --output-dir "${render_dir}" \
  --speaker 7 \
  --session 15 \
  --speaker-name P7 \
  --session-name S15 \
  --mri-dicom-dir "${mri_dir}" \
  --audio "${audio_path}" \
  --ground-truth-prediction-compare \
  --ground-truth-contour-pack "${ground_truth_pack}" \
  --prediction-contour-dir "${inference_dir}/predicted_contours" \
  --frame-min 1 \
  --frame-max 2200 \
  --timeline-step 1.0 \
  --ms-image 19.98 \
  --prediction-model-label P7_pooledraw_bs10_best \
  --scale 4 \
  --remove-silent-after-audio

video_path="${render_dir}/prediction/p7_s15_p7_pooledraw_bs10_best_prediction_11contour_audio.mp4"
summary_path="${render_dir}/summary.json"
test "$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 "${video_path}")" = "50000/999"
test "$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 "${video_path}")" = "2200"
test "$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "${video_path}")" = "h264"
test "$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "${video_path}")" = "aac"
"${python_bin}" - "${summary_path}" "${inference_dir}/summary.json" "${training_summary}" "${result_root}/run_manifest.json" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

render_path, inference_path, training_path, output_path = map(Path, sys.argv[1:])
render = json.loads(render_path.read_text(encoding="utf-8"))
inference = json.loads(inference_path.read_text(encoding="utf-8"))
training = json.loads(training_path.read_text(encoding="utf-8"))
assert render["fps_rational"] == "50000/999"
assert render["num_timeline_frames"] == 2200
assert render["rendered_fractional_frame_count"] == 0
assert render["outputs"][0]["num_frames"] == 2200
assert render["outputs"][0]["num_held_frames"] == 0
assert render["outputs"][0]["audio_attached"] is True
assert render["outputs"][0]["silent_video_removed"] is True
assert inference["inference_mode"] == "legacy_interval_chunks"
assert inference["uses_target_labels"] is False
assert inference["num_half_frames"] == 0
assert inference["prediction_coverage_min"] == 1
assert inference["prediction_coverage_max"] == 1
manifest = {
    "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "training": training,
    "inference": inference,
    "render": render,
}
output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
PY

echo "pipeline_complete timestamp=$(date --iso-8601=seconds) video=${video_path} manifest=${result_root}/run_manifest.json"
