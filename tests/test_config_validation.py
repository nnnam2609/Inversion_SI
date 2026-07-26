from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import importlib
from pathlib import Path
from unittest import mock

from src.utils.config_validation import load_yaml_config, validate_runtime_config
from src.utils.read_yaml import read_datas


REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigValidationTests(unittest.TestCase):
    def test_config_validation_imports_from_src_and_legacy_utils_paths(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        try:
            src_module = importlib.import_module("src.utils.config_validation")
            legacy_module = importlib.import_module("utils.config_validation")
        finally:
            sys.path = [item for item in sys.path if item != str(REPO_ROOT / "src")]

        self.assertTrue(hasattr(src_module, "load_yaml_config"))
        self.assertTrue(hasattr(legacy_module, "load_yaml_config"))

    def test_validate_runtime_config_reports_floor_metadata(self) -> None:
        metadata = validate_runtime_config(
            {"normalization_contour_std_floor": 0.2},
            Path("config.yaml"),
        )

        self.assertEqual(metadata["normalization_contour_std_floor"], 0.2)
        self.assertEqual(metadata["normalization_contour_std_floor_source"], "normalization_contour_std_floor")
        self.assertFalse(metadata["allow_legacy_audio_vtln"])
        self.assertFalse(metadata["legacy_audio_vtln"])
        self.assertIsNone(metadata["legacy_audio_vtln_feature_npz"])

    def test_load_yaml_config_rejects_legacy_audio_vtln_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.yaml"
            path.write_text("audio_vtln_feature_npz: vtln_mfcc39.npz\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "legacy audio_vtln_feature_npz"):
                load_yaml_config(path)

            loaded = load_yaml_config(path, allow_legacy_audio_vtln=True)
            self.assertEqual(loaded["audio_vtln_feature_npz"], "vtln_mfcc39.npz")

    def test_audit_script_counts_legacy_audio_vtln_even_when_strict_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.yaml"
            path.write_text("audio_vtln_feature_npz: vtln_mfcc39.npz\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/inversion_si.py"),
                    "audit",
                    "configs",
                    str(path),
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn('"num_errors": 1', result.stdout)
        self.assertIn('"num_legacy_audio_vtln": 1', result.stdout)
        self.assertIn('"legacy_audio_vtln": true', result.stdout)

    def test_training_read_datas_uses_validated_config_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "bad_train.yaml"
            config_path.write_text("normalization_contour_std_floor: 0.01\n", encoding="utf-8")

            with mock.patch.object(sys, "argv", ["main_train.py", "--config", str(config_path)]):
                with self.assertRaisesRegex(ValueError, "must be >= 0.1"):
                    read_datas()

    def test_audit_script_exits_nonzero_for_bad_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.yaml"
            path.write_text("normalization_contour_std_floor: 0.01\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/inversion_si.py"),
                    "audit",
                    "configs",
                    str(path),
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn('"num_errors": 1', result.stdout)
        self.assertIn("must be >= 0.1", result.stdout)

    def test_audit_script_allows_legacy_audio_vtln_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.yaml"
            path.write_text("audio_vtln_feature_npz: vtln_mfcc39.npz\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/inversion_si.py"),
                    "audit",
                    "configs",
                    "--allow-legacy-audio-vtln",
                    str(path),
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"num_errors": 0', result.stdout)
        self.assertIn('"num_legacy_audio_vtln": 1', result.stdout)
        self.assertIn('"legacy_audio_vtln": true', result.stdout)


if __name__ == "__main__":
    unittest.main()
