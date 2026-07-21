# Inversion_SI Server Context

Last verified: 2026-07-17 on `grappe-3.nancy.grid5000.fr`, OAR job
`6781207`.

## Runtime and repository

- Server workspace:
  `/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen`
- Server repo: `<workspace>/Inversion_SI`
- GitHub: `git@github.com:nnnam2609/Inversion_SI.git`
- Verified upstream commit: `e8e5954` (`Add dense integer-frame inference and safe session rendering`)
- Shared runtime: `<workspace>/inversion/.venv/bin/python` (Python 3.9.2)
- `grappe-3` is a CPU node (2 x Xeon Gold 5218R, 40 physical cores, no GPU).
- The server checkout is a linked worktree whose gitdir lives under
  `<workspace>/inversion/.git/worktrees/inversion_single_task5_minimal`.

At verification time the server worktree had uncommitted changes in:

- `scripts/infer_dense_audio_integer_contours.py`
- `scripts/render_gridnorm_session_video.py`
- untracked `scripts/job_p7_s15_nonoverlapping_chunks_inference.sh`

Do not overwrite those files without reviewing the diff.

## P7 model and normalization provenance

- Config:
  `config/train_config/asd1_p7_seen_trainvaltest_paper_st5_mfcc_500epoch_stdfloor01.yaml`
- Checkpoint:
  `mlruns/151291977315070763/b0887daa569d4659858d4f2e6ecc2451/artifacts/best_model.pth`
- Training split: P7/S1-S12
- Validation split: P7/S13-S14
- Test split: P7/S15-S16
- Normalization mode: `train_global`
- Normalization fit split: `train_sequences`
- Fit population recorded by split-cache metadata: 756 chunks / 24,307
  frames from training only
- Training sequence length: 80
- MFCC shape: 39 = 13 MFCC + delta + delta-delta
- Contours: 11 articulators x 100 coordinates (50 xy points)
- Contour std floor: 0.1

Verified SHA256 values:

```text
config:        840e78d2d9798cc2dd7751942c3a5704d1d5475a9852d822bf88c576767ba5e0
normalization: 1b760fc41859a02ac6b1b6d022ff1d08da0c1047914c69f7c410ceb493d31b4c
checkpoint:    861699dfd7ba3bb94a43f9c7f1e81bb87f8292901c17cb6c085560adf598772a
```

The prediction-only path loads the WAV, TextGrid boundaries, P7 training
normalization, config, and checkpoint. It does not load S15 contour labels or
the S15 cached split tensor.

## Canonical P7/S15 inference contract

For the classic ST-5 BiLSTM, use:

1. TextGrid tier 0 only to identify speech intervals and silence gaps.
2. One MFCC nearest each integer MRI-frame center inside selected intervals.
3. Non-overlapping chunks no longer than 80 frames; interval boundaries reset
   recurrent context.
4. `batch_size=1`, so every chunk is a separate forward with its actual length.
5. Direct concatenation of chunk outputs.
6. P7 training-global MFCC normalization and P7 training-global contour
   denormalization.
7. No overlap averaging, interpolation, half frames, or held contours.
8. No output file for a frame without a direct prediction; the renderer must
   show `missing:` for all absent contour classes.

## Verified grappe-3 run

Re-run from the server `Inversion_SI` directory (use `--device cpu` on
`grappe-3`; use `cuda` only inside a GPU OAR allocation):

```bash
../inversion/.venv/bin/python scripts/infer_dense_audio_integer_contours.py \
  --config config/train_config/asd1_p7_seen_trainvaltest_paper_st5_mfcc_500epoch_stdfloor01.yaml \
  --checkpoint mlruns/151291977315070763/b0887daa569d4659858d4f2e6ecc2451/artifacts/best_model.pth \
  --audio /srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/ArtSpeech_Database_1_raw/P7/OTHER/S15/DENOISED_SOUND_P7_S15.wav \
  --textgrid /srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/ArtSpeech_Database_1_raw/P7/OTHER/S15/TEXT_ALIGNMENT_P7_S15.textgrid \
  --output-dir results/p7_s15_direct_chunks \
  --speaker 7 --session 15 --frame-min 143 --frame-max 1606 \
  --inference-mode legacy_interval_chunks \
  --window-size 80 --batch-size 1 --device cpu
```

Inference output:

```text
results/p7_s15_direct_nonoverlap_actual_length_grappe3_20260717_v2/
```

Video output:

```text
results/p7_s15_direct_nonoverlap_actual_length_grappe3_20260717_v2_video/
```

Logs:

```text
logs/p7_s15_direct_nonoverlap_actual_length_grappe3_20260717_v2/inference.log
logs/p7_s15_direct_nonoverlap_actual_length_grappe3_20260717_v2_video/render.log
```

Verified results:

- 70 independent model calls
- chunk lengths: 2 to 80
- 1,347 predicted integer frames
- 14,817 contour files = 1,347 x 11
- coverage exactly 1 for every forwarded feature
- 117 missing integer frames out of 1,464 requested
- 0 half frames and 0 held frames
- video: H.264 + AAC, 544x662, 1,464 frames, 50.05005 fps
- missing instances in video: 1,287 = 117 x 11
- renderer did not load a prediction/label payload (`prediction_payload_loaded: false`)
- frame 0199 shows `missing: all 11 contours`

CPU output on `grappe-3` agrees with the earlier GPU run of the same algorithm:

```text
coordinate RMSE: 6.3158e-07 px
maximum absolute difference: 7.6294e-06 px
```

## Superseded or diagnostic P7/S15 outputs

- The cached split prediction is sparse and omitted 504 requested integer
  frames because the split includes only speech-selected timestamps.
- The overlapping-window dense run is diagnostic only: changing window
  contributors produced periodic seams.
- The one-forward full-sequence run is diagnostic only: its recurrent context
  is much longer than the training sequence length and produced under-moving
  contours.
- Earlier videos with held contours, `.5` frames, or a P2 MRI background must
  not be used as the canonical P7/S15 result.
- The July 17 fixed-size dense non-overlapping run predicts all 1,464 frames,
  including silence. It is useful as a comparison, but the classic temporal
  contract is the TextGrid-interval run documented above.
