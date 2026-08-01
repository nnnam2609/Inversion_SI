# ASD1 Offset-0 Server Migration — 2026-08-01

## Status

ASD1 frame-zero alignment is now the active and mandatory server default.
Legacy ASD1 `added_frames: 20` / `skip_ms: 400` caches are invalid because
they shift audio by approximately 399.6 ms relative to the ASD1 contour
timeline.

The enforcing code was installed from commit
`3682ed79bbb70d2fb3512a6b87664e4ef1b1fdf0` in server worktree:

`/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/Inversion_SI_asd1_offset0_20260801`

## Legacy backup

All identified active offset-20 ASD1 raw caches, contour packs, split caches,
three legacy ASD1 configs, and the anatomy-branch ASD1 raw-cache variant were
stored in one CRC-verified ZIP archive:

`/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/nhanguyen/asd1_offset20_backup_20260801_pre_offset0.zip`

- ZIP entries: 553
- Uncompressed payload: 12,917,037,012 bytes
- SHA-256: `2c923fd7fc93bf4956e56ac27b6d0f0b2edabe93ddb10e239c4668e4042da59a`
- `unzip -tq`: no errors

The 16 legacy active ASD1 split-cache directories and old shared raw-cache
directories were removed after the replacement passed content and load tests.
They are recoverable from this ZIP.

## Active offset-0 cache

Canonical reusable caches:

- `Inversion_SI/cache/raw_sessions/asd1`
- `Inversion_SI/cache/raw_contour_npz/asd1`

Full validation split cache and build evidence:

- `Inversion_SI/repro/asd1_offset0_default_20260801/splits`

Inventory:

- 158 raw session `.pt` files
- 158 contour-pack `.npz` files
- 158 of 160 declared sessions
- missing `P4/S12`: TextGrid unavailable
- missing `P4/S16`: contour directory empty
- train: 7,249 sequences, shape `(7249, 80, 39)`
- validation: 1,274 sequences, shape `(1274, 80, 39)`
- test: 908 sequences, shape `(908, 80, 39)`

Aggregate identity over relative path, size, and per-file SHA-256 for the 326
uploaded raw/NPZ/evidence/split files:

`1cf8cb7e7e9e44a08a6373e4a3a12e1e3b11c1f8bb1510f7ea295b3d74028759`

The local source and active server cache produced the same aggregate identity.
The active P7/S15 raw file has SHA-256
`0e57ea4e4bdf4dce2990560cb3c5812186f0cb1631ac8744d148b09177057673`;
the archived offset-20 version has SHA-256
`f8172c64565e292b55358182051df24c63a78a27cd4fc79fecefc7226033b926`.

The anatomy-branch ASD1 variant was rebuilt as 84 hard links to the verified
offset-0 canonical raw sessions for P1, P3, P5, P6, P7, and P9, sessions S1–S14.

## Usage rule

Do not restore an archived offset-20 split cache into an active experiment.
All new ASD1 raw and split caches must be produced with frame-zero alignment.
ASD2 retains its separate dataset-specific 20-frame preprocessing offset.
