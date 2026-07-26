from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.utils.mri_rendering import (
    dicom_filename_sort_key,
    load_or_build_mri_cache,
    mri_for_frame,
    needed_integer_frames,
    normalize_mri_frame,
)


class MriRenderingTests(unittest.TestCase):
    def test_dicom_cache_is_not_reused_for_a_different_source_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "P2" / "DCM_2D" / "S1"
            source_b = root / "P7" / "DCM_2D" / "S15"
            source_a.mkdir(parents=True)
            source_b.mkdir(parents=True)
            cache_path = root / "mri_frames_cache.npz"
            np.savez(
                cache_path,
                frame_numbers=np.array([199], dtype=np.int32),
                images=np.zeros((1, 2, 2), dtype=np.uint8),
                source_dir=np.array(str(source_a.resolve())),
            )

            with self.assertRaisesRegex(RuntimeError, "Missing DICOM InstanceNumber"):
                load_or_build_mri_cache(
                    source_b,
                    dicom_index={},
                    frame_numbers=[199],
                    cache_path=cache_path,
                    workers=1,
                )

    def test_dicom_cache_is_reused_for_the_same_source_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "P7" / "DCM_2D" / "S15"
            source.mkdir(parents=True)
            cache_path = Path(directory) / "mri_frames_cache.npz"
            expected = np.full((2, 2), 17, dtype=np.uint8)
            np.savez(
                cache_path,
                frame_numbers=np.array([199], dtype=np.int32),
                images=expected[None, ...],
                source_dir=np.array(str(source.resolve())),
            )

            cache = load_or_build_mri_cache(
                source,
                dicom_index={},
                frame_numbers=[199],
                cache_path=cache_path,
                workers=1,
            )

        np.testing.assert_array_equal(cache[199], expected)

    def test_dicom_filename_sort_key_uses_timestamp_then_suffix(self) -> None:
        names = ["IMG_2020010100000010", "IMG_2020010100000002", "abc"]

        self.assertEqual(sorted(names, key=dicom_filename_sort_key), ["IMG_2020010100000002", "IMG_2020010100000010", "abc"])

    def test_needed_integer_frames_applies_offset(self) -> None:
        items = [(1.0, 0, 0), (2.0, 0, 0)]

        self.assertEqual(needed_integer_frames(items, frame_offset=10), [11, 12])

    def test_needed_integer_frames_rejects_half_frames(self) -> None:
        with self.assertRaisesRegex(ValueError, "NEVER render fractional"):
            needed_integer_frames([(2.5, 0, 0)], frame_offset=10)

    def test_normalize_mri_frame_returns_uint8_range(self) -> None:
        image = np.asarray([[0, 5], [10, 15]], dtype=np.uint16)

        normalized = normalize_mri_frame(image)

        self.assertEqual(normalized.dtype, np.uint8)
        self.assertEqual(normalized.shape, image.shape)
        self.assertGreater(int(normalized.max()), int(normalized.min()))

    def test_mri_for_frame_rejects_half_frame_with_offset(self) -> None:
        cache = {
            11: np.zeros((2, 2), dtype=np.uint8),
            12: np.full((2, 2), 10, dtype=np.uint8),
        }

        with self.assertRaisesRegex(ValueError, "NEVER render fractional"):
            mri_for_frame(1.5, frame_offset=10, cache=cache)


if __name__ == "__main__":
    unittest.main()
