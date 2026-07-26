"""Regression tests for the preprocessing domain."""

from __future__ import annotations

# --- Consolidated from test_build_asd2_vtln_incisor_cache.py ---

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.incisor_cache import build_session, validate_session  # noqa: E402


class BuildAsd2VtlnIncisorCacheTests(unittest.TestCase):
    def test_labels_only_refresh_preserves_payload_and_averages_half_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            target = root / "target"
            incisors = root / "bf"
            source_raw = source / "raw_sessions/asd2/1775/S6.pt"
            source_npz = source / "raw_contour_npz/asd2/1775/S6.npz"
            contour_dir = incisors / "1775/S6/inference_contours"
            source_raw.parent.mkdir(parents=True)
            source_npz.parent.mkdir(parents=True)
            contour_dir.mkdir(parents=True)

            features = np.arange(3 * 39, dtype=np.float32).reshape(3, 39)
            frames = np.asarray([[1775, 6, 1.0], [1775, 6, 1.5], [1775, 6, 2.0]], dtype=np.float64)
            phonemes = np.zeros((3, 1, 44), dtype=np.float64)
            old_contours = np.zeros((3, 11, 100), dtype=np.float32)
            old_contours[:, :9] = np.arange(9 * 100, dtype=np.float32).reshape(1, 9, 100)
            payload = {
                "split": "train_sequences",
                "dataset_type": "asd2",
                "bucket": "1775",
                "session": "S6",
                "num_chunks": 1,
                "raw": {
                    "features": [features],
                    "contours": [old_contours],
                    "frames": [frames],
                    "phonemes": [phonemes],
                    "length_datas": [1],
                },
            }
            torch.save(payload, source_raw)
            source_pack = np.zeros((2, 11, 100), dtype=np.float32)
            source_pack[:, :9] = old_contours[:2, :9]
            np.savez(
                source_npz,
                frame_numbers=np.asarray([1, 2], dtype=np.int32),
                contours=source_pack,
                articulators=np.asarray(
                    [
                        "arytenoid-cartilage",
                        "epiglottis",
                        "lower-lip",
                        "pharynx",
                        "soft-palate-midline",
                        "tongue",
                        "upper-lip",
                        "vocal-folds",
                        "thyroid-cartilage",
                        "lower-incisor",
                        "upper-incisor",
                    ]
                ),
                source_folder=np.asarray("synthetic"),
            )
            direct = {}
            for frame in (1, 2):
                lower = np.column_stack(
                    (
                        np.arange(50, dtype=np.float32) + frame,
                        np.arange(50, dtype=np.float32) + 100 + frame,
                    )
                ).astype(np.float32)
                upper = (lower + np.float32(10)).astype(np.float32)
                np.save(contour_dir / f"{frame:04d}_lower-incisor.npy", lower)
                np.save(contour_dir / f"{frame:04d}_upper-incisor.npy", upper)
                direct[frame] = (lower.reshape(100), upper.reshape(100))

            job = {
                "split": "train_sequences",
                "bucket": "1775",
                "session": "S6",
                "source_root": str(source),
                "target_root": str(target),
                "incisor_root": str(incisors),
                "variant_name": "synthetic-test",
                "rebuild": False,
                "verify_bf": True,
            }
            build_result = build_session(job)
            validation_result = validate_session(job)

            self.assertEqual(build_result["status"], "built")
            self.assertEqual(validation_result["status"], "ok")
            target_payload = torch.load(target / "raw_sessions/asd2/1775/S6.pt", map_location="cpu")
            new_raw = target_payload["raw"]
            np.testing.assert_array_equal(new_raw["features"][0], features)
            np.testing.assert_array_equal(new_raw["frames"][0], frames)
            np.testing.assert_array_equal(new_raw["phonemes"][0], phonemes)
            np.testing.assert_array_equal(new_raw["contours"][0][:, :9], old_contours[:, :9])
            np.testing.assert_array_equal(new_raw["contours"][0][0, 9], direct[1][0])
            np.testing.assert_array_equal(new_raw["contours"][0][2, 10], direct[2][1])
            expected_half = np.mean(
                np.stack((direct[1][0], direct[2][0])), axis=0, dtype=np.float32
            )
            np.testing.assert_array_equal(new_raw["contours"][0][1, 9], expected_half)
            marker = json.loads((target / "markers/1775/S6.json").read_text(encoding="utf-8"))
            self.assertEqual(marker["num_half_rows"], 1)
            self.assertEqual(marker["num_integer_rows"], 2)
