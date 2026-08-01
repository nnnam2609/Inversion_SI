# ASD1 Alignment Policy

## Required setting

ASD1 starts at contour frame 0. The only valid preprocessing alignment is:

```yaml
dataset_type: asd1
added_frames: 0
skip_ms: 0
```

For ASD1, zero is also the code default when these offset fields are omitted.
The preprocessing loader rejects any ASD1 configuration with a non-zero
`added_frames` or `skip_ms` value.

ASD2 is different and retains its dataset-specific 20-frame offset. Do not
share one alignment setting between ASD1 and ASD2 preprocessing jobs.

## Why the former setting is invalid

Historical ASD1 configurations copied the ASD2 values `added_frames: 20` and
`skip_ms: 400`. ASD1 annotations already cover the beginning of the recording,
so applying that extra skip advances audio features by roughly 20 MRI frames
(`20 * 19.98 ms = 399.6 ms`) relative to the ASD1 contour frame identifiers.
This creates the observed audio/contour lag.

Consequently, ASD1 artifacts derived from the historical offset-20 raw cache
are legacy artifacts. They must not be used as a current training or evaluation
baseline. This includes raw session caches, assembled split caches, and model
results whose training data came from those caches.

## Cache migration rules

1. Archive the offset-20 raw cache, contour packs, split caches, and the old
   ASD1 configs before replacing them.
2. Replace `cache/raw_sessions/asd1` and `cache/raw_contour_npz/asd1` together.
   Never merge offset-0 files into the old directories because missing sessions
   could leave stale offset-20 files behind.
3. Delete or rebuild every active ASD1 split cache after the raw-cache swap.
4. Record the source config, session inventory, failures, hashes, and build date
   beside the replacement cache.
5. Keep historical offset-20 artifacts only inside the dated backup archive.

Migration performed on 2026-08-01 uses the locally verified full ASD1 offset-0
cache. It contains 158 of 160 declared sessions. `P4/S12` is unavailable because
its TextGrid is missing, and `P4/S16` has no contour files.
