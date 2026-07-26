from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.utils.audio_vtln import reject_legacy_audio_vtln_config
from src.utils.normalization import (
    DEFAULT_CONTOUR_STD_FLOOR,
    LOW_CONTOUR_STD_FLOOR_DIAGNOSTIC_KEY,
    RAW_POSITIVE_NORMALIZATION_STD_POLICY,
    apply_std_floor,
    describe_split_denorm,
    load_cache_metadata,
    load_validated_split_cache_state,
    normalization_floor_metadata,
    normalization_std_floors,
    validate_raw_positive_std,
    validate_contour_std_floor,
)
from src.inference.session_inference import load_config, load_reference_denorm


class NormalizationGuardTests(unittest.TestCase):
    def test_raw_positive_policy_disables_floors_without_changing_default(self) -> None:
        config = {"normalization_std_policy": RAW_POSITIVE_NORMALIZATION_STD_POLICY}
        metadata = normalization_floor_metadata(config)
        self.assertEqual(metadata["normalization_std_policy"], "raw_positive")
        self.assertIsNone(metadata["normalization_contour_std_floor"])
        self.assertIsNone(metadata["normalization_mfcc_std_floor"])
        with self.assertRaisesRegex(ValueError, "has no std floors"):
            normalization_std_floors(config)
        self.assertEqual(normalization_std_floors({}), (DEFAULT_CONTOUR_STD_FLOOR, 1e-8))

    def test_raw_positive_policy_accepts_positive_cache_and_rejects_zero(self) -> None:
        config = {"normalization_std_policy": RAW_POSITIVE_NORMALIZATION_STD_POLICY}
        summary = validate_contour_std_floor(
            {"std": torch.full((1, 1, 2, 4), 1e-6)},
            config,
            "raw_positive_ok.pt",
            {"normalization_std_policy": "raw_positive"},
        )
        self.assertTrue(summary["cache_contour_std_raw_positive_ok"])
        self.assertIsNone(summary["normalization_contour_std_floor"])
        with self.assertRaisesRegex(RuntimeError, "finite and strictly positive"):
            validate_contour_std_floor(
                {"std": torch.tensor([[[[0.0, 1.0]]]])},
                config,
                "raw_positive_bad.pt",
                {},
            )

    def test_raw_positive_fitted_std_error_reports_contour_coordinate(self) -> None:
        values = np.ones((2, 4), dtype=np.float32)
        values[1, 3] = 0.0
        with self.assertRaisesRegex(ValueError, "class_name.*upper.*coordinate_index.*3"):
            validate_raw_positive_std(
                values,
                "contour",
                classes=["lower", "upper"],
                output_layer=4,
            )

    def test_default_and_override_std_floors(self) -> None:
        self.assertEqual(normalization_std_floors({}), (DEFAULT_CONTOUR_STD_FLOOR, 1e-8))
        self.assertEqual(
            normalization_std_floors(
                {
                    "normalization_contour_std_floor": 0.25,
                    "normalization_mfcc_std_floor": 0.01,
                }
            ),
            (0.25, 0.01),
        )

    def test_std_floor_metadata_tracks_source_keys(self) -> None:
        default_metadata = normalization_floor_metadata({})
        self.assertEqual(default_metadata["normalization_contour_std_floor"], DEFAULT_CONTOUR_STD_FLOOR)
        self.assertEqual(default_metadata["normalization_contour_std_floor_source"], "default")
        self.assertFalse(default_metadata["normalization_used_legacy_std_floor_key"])

        explicit_metadata = normalization_floor_metadata({"normalization_contour_std_floor": 0.2})
        self.assertEqual(explicit_metadata["normalization_contour_std_floor"], 0.2)
        self.assertEqual(
            explicit_metadata["normalization_contour_std_floor_source"],
            "normalization_contour_std_floor",
        )
        self.assertFalse(explicit_metadata["normalization_used_legacy_std_floor_key"])

        legacy_metadata = normalization_floor_metadata({"contour_std_floor": 0.3})
        self.assertEqual(legacy_metadata["normalization_contour_std_floor"], 0.3)
        self.assertEqual(legacy_metadata["normalization_contour_std_floor_source"], "contour_std_floor")
        self.assertTrue(legacy_metadata["normalization_used_legacy_std_floor_key"])
        self.assertFalse(legacy_metadata["normalization_low_contour_std_floor_diagnostic"])

    def test_rejects_non_positive_std_floors(self) -> None:
        with self.assertRaisesRegex(ValueError, "normalization_contour_std_floor"):
            normalization_std_floors({"normalization_contour_std_floor": 0.0})
        with self.assertRaisesRegex(ValueError, "normalization_mfcc_std_floor"):
            normalization_std_floors({"normalization_mfcc_std_floor": -1.0})

    def test_rejects_low_contour_std_floor_unless_diagnostic(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be >= 0.1"):
            normalization_std_floors({"normalization_contour_std_floor": 0.01})
        self.assertEqual(
            normalization_std_floors(
                {
                    "normalization_contour_std_floor": 0.01,
                    LOW_CONTOUR_STD_FLOOR_DIAGNOSTIC_KEY: True,
                }
            ),
            (0.01, 1e-8),
        )
        metadata = normalization_floor_metadata(
            {
                "normalization_contour_std_floor": 0.01,
                LOW_CONTOUR_STD_FLOOR_DIAGNOSTIC_KEY: True,
            }
        )
        self.assertTrue(metadata["normalization_low_contour_std_floor_diagnostic"])

    def test_apply_std_floor_preserves_dtype_unless_requested(self) -> None:
        values = np.asarray([0.0, 0.05, 0.2], dtype=np.float64)

        floored = apply_std_floor(values, 0.1)
        floored_float32 = apply_std_floor(values, 0.1, np.float32)

        np.testing.assert_allclose(floored, np.asarray([0.1, 0.1, 0.2]))
        self.assertEqual(floored.dtype, np.float64)
        np.testing.assert_allclose(floored_float32, np.asarray([0.1, 0.1, 0.2], dtype=np.float32))
        self.assertEqual(floored_float32.dtype, np.float32)

    def test_validate_contour_std_floor_accepts_current_cache_shape(self) -> None:
        state = {"std": torch.full((2, 1, 11, 100), 0.1)}
        summary = validate_contour_std_floor(
            state,
            {"normalization_contour_std_floor": 0.1},
            "synthetic_ok.pt",
            {"normalization_contour_std_floor": 0.1},
        )
        self.assertTrue(summary["cache_contour_std_floor_ok"])
        self.assertAlmostEqual(summary["cache_contour_std_min"], 0.1, places=6)

    def test_validate_contour_std_floor_rejects_stale_low_std_cache(self) -> None:
        state = {"std": torch.full((1, 1, 11, 100), 1e-8)}
        with self.assertRaisesRegex(RuntimeError, "below the configured normalization floor"):
            validate_contour_std_floor(
                state,
                {"normalization_contour_std_floor": 0.1},
                "synthetic_bad.pt",
                {},
            )

    def test_load_validated_split_cache_state_enforces_floor_and_required_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_sequences.pt"
            torch.save(
                {
                    "std": torch.full((1, 1, 11, 100), 0.1),
                    "mean": torch.zeros((1, 1, 11, 100)),
                },
                cache_path,
            )
            metadata_path = Path(tmpdir) / "split_cache_metadata.json"
            metadata_path.write_text(
                json.dumps({"normalization_contour_std_floor": 0.1}),
                encoding="utf-8",
            )

            state, summary = load_validated_split_cache_state(
                cache_path,
                {"normalization_contour_std_floor": 0.1},
                required_keys=("std", "mean"),
            )

            self.assertIn("std", state)
            self.assertTrue(summary["cache_contour_std_floor_ok"])
            self.assertEqual(summary["cache_metadata"], str(metadata_path))

            with self.assertRaisesRegex(KeyError, "missing required keys"):
                load_validated_split_cache_state(
                    cache_path,
                    {"normalization_contour_std_floor": 0.1},
                    required_keys=("features",),
                )

    def test_load_validated_split_cache_state_rejects_stale_low_std_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_sequences.pt"
            torch.save({"std": torch.full((1, 1, 11, 100), 1e-8)}, cache_path)

            with self.assertRaisesRegex(RuntimeError, "below the configured normalization floor"):
                load_validated_split_cache_state(
                    cache_path,
                    {"normalization_contour_std_floor": 0.1},
                )

    def test_load_cache_metadata_and_describe_train_global_denorm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_sequences.pt"
            cache_path.write_bytes(b"placeholder")
            metadata_path = Path(tmpdir) / "split_cache_metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "normalization_mode": "train_global",
                        "normalization_fit_splits": ["train_sequences"],
                    }
                ),
                encoding="utf-8",
            )

            metadata = load_cache_metadata(cache_path)
            denorm = describe_split_denorm(cache_path, {"normalization_mode": "fallback"})

        self.assertEqual(metadata["normalization_mode"], "train_global")
        self.assertEqual(metadata["metadata_path"], str(metadata_path))
        self.assertEqual(denorm["denorm_method"], "train_global_std_mean_from_cache")
        self.assertFalse(denorm["uses_target_std_mean"])
        self.assertEqual(denorm["std_mean_source_split"], "train_sequences")

    def test_describe_split_denorm_defaults_to_split_stats_when_metadata_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_sequences.pt"
            cache_path.write_bytes(b"placeholder")

            denorm = describe_split_denorm(cache_path, {"normalization_mode": "speaker_dependent"})

        self.assertEqual(denorm["denorm_method"], "split_cache_std_mean")
        self.assertTrue(denorm["uses_target_std_mean"])
        self.assertEqual(denorm["std_mean_source_split"], "test_sequences")
        self.assertEqual(denorm["normalization_mode"], "speaker_dependent")

    def test_rejects_legacy_audio_vtln_config(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "legacy audio_vtln_feature_npz"):
            reject_legacy_audio_vtln_config(
                {"audio_vtln_feature_npz": "vtln_mfcc39.npz"},
                Path("legacy.yaml"),
            )
        reject_legacy_audio_vtln_config(
            {"inversion_frontend_vtln_cache_metadata": "metadata.json"},
            Path("corrected.yaml"),
        )

    def test_session_inference_config_loader_rejects_legacy_audio_vtln_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_path = Path(tmpdir) / "legacy.yaml"
            legacy_path.write_text("audio_vtln_feature_npz: vtln_mfcc39.npz\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "legacy audio_vtln_feature_npz"):
                load_config(legacy_path)

            corrected_path = Path(tmpdir) / "corrected.yaml"
            corrected_path.write_text(
                "inversion_frontend_vtln_cache_metadata: metadata.json\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_config(corrected_path)["inversion_frontend_vtln_cache_metadata"],
                "metadata.json",
            )

    def test_load_reference_denorm_validates_reference_cache_std_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "train_sequences.pt"
            torch.save(
                {
                    "std": torch.full((2, 1, 11, 100), 0.1),
                    "mean": torch.zeros((2, 1, 11, 100)),
                },
                cache_path,
            )
            std, mean, metadata = load_reference_denorm(
                cache_path,
                torch.device("cpu"),
                {"normalization_contour_std_floor": 0.1},
            )
        self.assertEqual(tuple(std.shape), (1, 1, 11, 100))
        self.assertEqual(tuple(mean.shape), (1, 1, 11, 100))
        self.assertTrue(metadata["cache_contour_std_floor_ok"])

    def test_load_reference_denorm_rejects_stale_reference_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "train_sequences.pt"
            torch.save(
                {
                    "std": torch.full((1, 1, 11, 100), 1e-8),
                    "mean": torch.zeros((1, 1, 11, 100)),
                },
                cache_path,
            )
            with self.assertRaisesRegex(RuntimeError, "below the configured normalization floor"):
                load_reference_denorm(
                    cache_path,
                    torch.device("cpu"),
                    {"normalization_contour_std_floor": 0.1},
                )


if __name__ == "__main__":
    unittest.main()
