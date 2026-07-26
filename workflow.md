# ASD2 training and ASD1 speaker adaptation

Updated: 2026-07-26

## Mục tiêu

Workflow hiện tại train một acoustic-to-articulatory inversion model trên ASD2,
sau đó giữ nguyên model để inference và adapt prediction sang từng speaker
ASD1.

```text
ASD2 audio + contours
        |
        v
train-global normalization
        |
        v
    BiLSTM
        |
        +---------------------------------------+
        |                                       |
        v                                       v
ASD1 target audio                       ASD1 target audio
        |                               RMS + VTLN -> ASD2
        v                                       |
raw ASD2-space prediction                      v
        |                               raw ASD2-space prediction
        +-------------------+-------------------+
                            |
                            v
                  affine + TPS to ASD1
                            |
                            v
                 target-speaker contours
```

“Adaptation” ở đây không phải fine-tuning:

- model weight và checkpoint không đổi;
- target ASD1 không được dùng để fit normalization;
- audio adaptation là RMS + VTLN;
- contour adaptation là một fixed affine + TPS transform cho mỗi target
  speaker/session.

## Training trên ASD2

### Dataset và preprocessing

Training dùng
[`ArtSpeech_Database_2`](../ARTSPEECH_DATABASES_IADI.md), với năm bucket
`1775`, `1777`, `1789`, `1791`, và `1804`. Các ID này là các filesystem bucket
của **một người thật**, không phải năm speaker độc lập.

| Split | Sessions | Vai trò |
|---|---:|---|
| Train | 85 | Fit model và toàn bộ normalization statistics |
| Validation | 27 | Model selection và early stopping |
| Test | 27 | Đánh giá checkpoint, không fit normalization |

Danh sách session đầy đủ nằm trong
[`fixed-BS10 training config`](config/train_config/asd2_11contour_vtln_lowerrepairv2_upperlegacypos_20260721_train_global_rawstd_st5_mfcc_500epoch_fixedbs10_4gpu.yaml).

Mỗi sample gồm:

- input: 39-D MFCC = 13 MFCC + delta + delta-delta;
- window 25 ms, hop 10 ms;
- sequence dài tối đa 80 acoustic frames;
- output: 11 contours;
- mỗi contour có 50 điểm `(x, y)`, tức 100 coordinates;
- tổng output mỗi frame là `(11, 100)`.

Contour cache được version hóa tại:

```text
cache_variants/asd2_11_vtln_lowerrepairv2_upperlegacypos_20260721
```

Classes 0–8 giữ source contour hiện có; lower incisor dùng continuity repair v2
và upper incisor dùng legacy-position replacement. Cache construction nằm ở
[`incisor_cache.py`](src/preprocessing/incisor_cache.py), exposed as
`scripts/inversion_si.py preprocess incisors`,
và
[`session_cache.py`](src/preprocessing/session_cache.py).

### Normalization

Workflow dùng:

```text
normalization_mode: train_global
normalization_fit_split: train_sequences
normalization_std_policy: raw_positive
```

`mean_mfcc`, `std_mfcc`, `mean_contour`, và `std_contour` chỉ được fit từ 85
ASD2 training sessions. Cùng một bộ statistics được dùng cho validation, test,
và toàn bộ ASD1 inference.

Phần fit và assemble split cache nằm ở
[`split_cache.py`](src/train/split_cache.py#L215). Normalization bundle:

```text
repro/asd2_11contour_vtln_lowerrepairv2_upperlegacypos_20260721_train_global/
  splits/normalization_stats.npz
```

Không được fit mean/std mới trên target ASD1 vì đó là target-statistics leakage.

### Model và checkpoint

Model trong [`baseline_5.py`](src/model/baseline_5.py#L14):

```text
(T, 39)
 -> Linear 300
 -> Linear 300
 -> BiLSTM 300 x 2 directions
 -> BiLSTM 300 x 2 directions
 -> Linear (11 x 100)
```

Training loop nằm ở
[`train_single.py`](src/train/train_single.py#L25) và dùng summed MSE trên 11
contours.

| Thuộc tính | Giá trị |
|---|---|
| Batch size | 10/GPU |
| GPUs | 4 RTX 2080 Ti |
| Effective batch size | 40 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.001 |
| Maximum epochs | 500 |
| Best human epoch | 31 |
| Early stop | human epoch 131 |
| MLflow run | `39e2314f4d02443a8e065ccdfc04bcb3` |

Checkpoint và run provenance được ghi trong
[`RUN_MANIFEST.md`](repro/asd2_11contour_vtln_lowerrepairv2_upperlegacypos_20260721_fixedbs10_4gpu/RUN_MANIFEST.md).

### Kết quả test ASD2

Đây là paired integer-only evaluation trên 2,473 test sequences và 93,342
integer rows. Có 90,683 fractional rows bị loại; fractional
saved/scored/rendered đều bằng 0.

| Metric | RMSE (mm) |
|---|---:|
| Sequence-coordinate | 2.4216 |
| Image-coordinate | 2.4541 |
| Point-coordinate | 2.7565 |
| Global-coordinate | 2.8637 |

Nguồn số liệu:
[`integer-only comparison JSON`](repro/asd2_11contour_vtln_lowerrepairv2_upperlegacypos_20260721_fixedbs10_4gpu/integer_only_batch1150_vs_batch10_comparison.json).
Không dùng automatic trainer summary cho kết luận cuối vì summary cũ bao gồm
fractional rows.

## Adapt sang ASD1

### Cohort

Workflow hiện đánh giá một selected session cho mỗi ASD1 speaker:

| Speaker/session | Frames | Exact TextGrid `/u/` reference |
|---|---:|---:|
| P1/S16 | 533 | F0763 |
| P2/S9 | 606 | F1424 |
| P3/S14 | 1,342 | F1168 |
| P4/S4 | 860 | F1179 |
| P5/S6 | 1,185 | F0813 |
| P6/S8 | 1,134 | F0597 |
| P7/S2 | 914 | F0500 |
| P8/S2 | 1,167 | F0298 |
| P9/S5 | 806 | F0259 |
| P10/S14 | 952 | F0478 |

P1–P9 là unseen speakers. P10 là cùng người thật với ASD2 và chỉ được giữ làm
same-person control; P10 phải bị loại khỏi unseen aggregate.

Speaker mapping được ghi tại
[`ARTSPEECH_MAPPING.md`](../ARTSPEECH_MAPPING.md).

### Baseline inference

ASD1 target audio được trích xuất bằng cùng MFCC frontend như lúc train, sau đó
normalize bằng ASD2 train statistics. Fixed checkpoint dự đoán 11 contours
trong coordinate/geometry space của ASD2:

```text
ASD1 audio
 -> MFCC39
 -> normalize bằng ASD2 train mean/std
 -> fixed ASD2 model
 -> denormalize bằng ASD2 train contour mean/std
 -> raw ASD2-space prediction
```

Không có ASD1 contour hoặc ASD1 mean/std nào đi vào neural network.
Ground-truth/pseudo-label ASD1 chỉ dùng để score kết quả.

### Contour adaptation

Source anatomical reference là ASD2
`1791/S14/F0499`, được chọn trực tiếp từ exact TextGrid `/u/`. Mỗi target dùng
exact `/u/` reference trong bảng trên.

Code xây transform:

- selected-frame workflow:
  [`infer_adapt.py`](src/adaption_pipeline/stages/infer_adapt.py);
- affine + TPS implementation:
  [`transfer.py`](external/grid-transform/grid_transform/transfer.py#L58);
- landmarks và fitting:
  [`transform_helpers.py`](external/grid-transform/grid_transform/transform_helpers.py#L88).

Transform gồm:

1. affine dùng axis landmarks `I1..I7`, `P1`, `C1..C6`;
2. zero-smoothing TPS thêm residual controls `M1` và `L6`;
3. một transform cố định được apply cho toàn bộ target session, không refit
   theo từng frame.

### Audio adaptation

Audio branch chuẩn hóa target audio trước model:

```text
ASD1 waveform
 -> RMS target 0.03
 -> fit VTLN alpha về 85 ASD2 training sessions
 -> Inversion_SI MFCC39
 -> ASD2 train-global normalization
 -> fixed ASD2 checkpoint
 -> affine + TPS
```

VTLN alpha được chọn từ target audio likelihood đối với ASD2 training-audio
reference; không dùng target contours. Model input vẫn được re-extract bằng
đúng `Inversion_SI` frontend.

Implementation liên quan:

- [`fit_audio_normalization.py`](src/adaption_pipeline/stages/fit_audio_normalization.py);
- [`infer_adapt.py`](src/adaption_pipeline/stages/infer_adapt.py);
- [`vtln_cache.py`](src/inference/vtln_cache.py), exposed as
  `scripts/inversion_si.py preprocess vtln-cache`.

Ba nhánh được so sánh:

| Tên trong bảng | Prediction |
|---|---|
| Baseline | Raw ASD2-space prediction |
| Contour | Raw prediction + affine + TPS |
| Contour + Audio | RMS + VTLN input prediction + affine + TPS |

Tất cả prediction, scoring và rendering chỉ giữ integer MRI frames; không
interpolate, không hold contour và không fine-tune.

## Kết quả adaptation trên ASD1

### Unseen aggregate: P1–P9

Các bảng dưới đây dùng 8,547 integer frames và bao gồm fixed calibration frame
để cùng presentation protocol với bảng P7 trước đó.

| Metric | Baseline | Contour | Contour + Audio |
|---|---:|---:|---:|
| All 11 contours RMSE (mm) | 12.928 | 9.680 | 9.795 |
| Change vs baseline | — | -3.248 (-25.13%) | -3.133 (-24.24%) |
| Without 3 laryngeal contours (mm) | 12.992 | 10.179 | 10.273 |
| Change vs baseline | — | -2.813 (-21.65%) | -2.720 (-20.93%) |

Kết luận aggregate:

- contour affine+TPS giảm all-11 RMSE 25.13%;
- contour+audio giảm 24.24% so với baseline;
- contour+audio kém contour-only khoảng 1.19%;
- audio normalization không tạo thêm gain tổng thể.

### Per-speaker all-11 RMSE

Mean ± sample SD được tính trên per-frame all-11 RMSE.

| Speaker | Baseline | Contour | Contour + Audio |
|---|---:|---:|---:|
| P1/S16 | 12.88 ± 0.73 | 11.43 ± 1.32 (↓11.30%) | 11.64 ± 1.07 (↓9.65%) |
| P2/S9 | 9.11 ± 0.65 | 11.45 ± 1.22 (↑25.62%) | 11.63 ± 1.10 (↑27.63%) |
| P3/S14 | 11.87 ± 0.62 | 11.73 ± 0.85 (↓1.19%) | 11.81 ± 0.85 (↓0.48%) |
| P4/S4 | 13.70 ± 0.91 | 8.59 ± 0.52 (↓37.33%) | 8.70 ± 0.48 (↓36.53%) |
| P5/S6 | 9.58 ± 0.53 | 8.85 ± 0.36 (↓7.63%) | 9.03 ± 0.38 (↓5.66%) |
| P6/S8 | 12.76 ± 1.07 | 9.65 ± 1.01 (↓24.34%) | 9.87 ± 1.05 (↓22.63%) |
| P7/S2 | 14.71 ± 0.87 | 8.64 ± 0.51 (↓41.27%) | 8.49 ± 0.45 (↓42.32%) |
| P8/S2 | 19.82 ± 1.53 | 9.57 ± 0.79 (↓51.72%) | 9.65 ± 0.80 (↓51.31%) |
| P9/S5 | 9.92 ± 0.64 | 7.54 ± 0.50 (↓23.93%) | 7.70 ± 0.57 (↓22.33%) |
| P10/S14† | 8.69 ± 0.63 | 8.23 ± 0.69 (↓5.30%) | 8.30 ± 0.76 (↓4.56%) |

† P10 là same-person control, không nằm trong unseen aggregate.

Contour adaptation cải thiện 8/9 unseen speakers. P2 là failure case rõ ràng:
affine+TPS làm RMSE tăng 25.62%. Audio chỉ tốt hơn contour-only trên P7; với
các target còn lại, audio branch làm kết quả xấu hơn nhẹ.

### Per-articulator unseen results

| Articulator | Baseline RMSE | Contour RMSE | Contour + Audio RMSE |
|---|---:|---:|---:|
| Arytenoid cartilage | 11.76 ± 3.91 | 9.10 ± 4.97 | 9.25 ± 5.07 |
| Epiglottis | 12.67 ± 5.48 | 6.54 ± 2.96 | 6.66 ± 2.93 |
| Lower lip | 9.21 ± 4.34 | 4.87 ± 1.81 | 4.95 ± 1.77 |
| Pharynx | 10.72 ± 3.07 | 7.91 ± 4.56 | 8.10 ± 4.56 |
| Soft palate midline | 11.55 ± 4.61 | 5.22 ± 1.71 | 5.28 ± 1.69 |
| Tongue | 10.85 ± 4.38 | 7.60 ± 2.76 | 7.79 ± 2.92 |
| Upper lip | 8.85 ± 5.04 | 5.80 ± 2.44 | 6.01 ± 2.39 |
| Vocal folds | 12.38 ± 5.19 | 8.10 ± 3.63 | 8.38 ± 3.53 |
| Thyroid cartilage | 11.76 ± 6.22 | 8.22 ± 3.81 | 8.50 ± 3.78 |
| Lower incisor | 19.41 ± 2.58 | 17.80 ± 1.55 | 17.79 ± 1.52 |
| Upper incisor | 15.80 ± 2.44 | 13.11 ± 1.30 | 13.14 ± 1.31 |

So với baseline, cả Contour và Contour+Audio đều significant cho 11/11
articulators sau Holm correction trong từng branch. Tuy nhiên incisors vẫn có
absolute error cao nhất, đặc biệt lower incisor.

## Canonical artifacts

| Nội dung | Link |
|---|---|
| Training config | [`fixed-BS10 YAML`](config/train_config/asd2_11contour_vtln_lowerrepairv2_upperlegacypos_20260721_train_global_rawstd_st5_mfcc_500epoch_fixedbs10_4gpu.yaml) |
| Training manifest | [`RUN_MANIFEST.md`](repro/asd2_11contour_vtln_lowerrepairv2_upperlegacypos_20260721_fixedbs10_4gpu/RUN_MANIFEST.md) |
| Integer-only ASD2 evaluation | [`comparison JSON`](repro/asd2_11contour_vtln_lowerrepairv2_upperlegacypos_20260721_fixedbs10_4gpu/integer_only_batch1150_vs_batch10_comparison.json) |
| Adaptation report | [`report.md`](results/asd2_fixedbs10_selected_10speakers_adaptation_tables_20260724/report.md) |
| Publication summary | [`01_adaptation_summary.md`](results/asd2_fixedbs10_selected_10speakers_adaptation_tables_20260724/publication_tables/01_adaptation_summary.md) |
| Per-articulator table | [`02_per_articulator_comparison.md`](results/asd2_fixedbs10_selected_10speakers_adaptation_tables_20260724/publication_tables/02_per_articulator_comparison.md) |
| Per-speaker table | [`03_per_speaker_comparison.md`](results/asd2_fixedbs10_selected_10speakers_adaptation_tables_20260724/publication_tables/03_per_speaker_comparison.md) |
| Full-precision metrics | [`metrics_full_precision.json`](results/asd2_fixedbs10_selected_10speakers_adaptation_tables_20260724/publication_tables/metrics_full_precision.json) |
| Table generation and audit | [`evaluate.py`](src/adaption_pipeline/stages/evaluate.py), [`audit_run.py`](src/adaption_pipeline/stages/audit_run.py) |

## Current interpretation

- ASD2 training itself đạt khoảng 2.45 mm image-coordinate RMSE trên ASD2 test.
- Cross-speaker raw ASD2 prediction trên ASD1 có domain/geometry gap rất lớn:
  12.93 mm trên unseen P1–P9.
- Affine+TPS giải quyết phần lớn geometry gap và giảm xuống 9.68 mm.
- RMS+VTLN hiện không cải thiện aggregate so với contour-only.
- Adaptation vẫn phụ thuộc vào một target `/u/` calibration frame cho mỗi
  speaker/session, nên đây không phải zero-shot adaptation.
- P2 và incisor errors cho thấy fixed transform chưa robust đồng đều giữa các
  speaker.
