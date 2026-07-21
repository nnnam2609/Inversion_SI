<INSTRUCTIONS>
- NEVER generate or render fractional MRI frames, including frame `.5`, in any prediction, comparison, grid-normalization, or diagnostic video.
- NEVER interpolate two adjacent MRI images or contours to create a fractional video frame.
- Filter prediction/ground-truth arrays to integer-numbered frames before saving contours, computing metrics, writing reports, or rendering videos.
- Every video/report pipeline must validate and record that its fractional saved/scored/rendered frame count is zero.
</INSTRUCTIONS>
