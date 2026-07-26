from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/inversion_si.py"


class AuditSplitCacheNormalizationTests(unittest.TestCase):
    def write_config(self, root: Path, cache_dir: Path) -> Path:
        config_path = root / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    f"split_cache_dir: {cache_dir}",
                    "normalization_contour_std_floor: 0.1",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return config_path

    def write_cache(self, cache_dir: Path, split_file: str, std_value: float) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "std": torch.full((1, 1, 11, 100), std_value),
                "mean": torch.zeros((1, 1, 11, 100)),
            },
            cache_dir / split_file,
        )

    def run_audit(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "audit", "splits", *args],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_valid_split_cache_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "splits"
            config_path = self.write_config(root, cache_dir)
            self.write_cache(cache_dir, "test_sequences.pt", 0.1)

            result = self.run_audit(str(config_path), "--splits", "test_sequences")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"num_ok": 1', result.stdout)
        self.assertIn('"cache_contour_std_floor_ok": true', result.stdout)

    def test_stale_low_std_split_cache_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "splits"
            config_path = self.write_config(root, cache_dir)
            self.write_cache(cache_dir, "test_sequences.pt", 1e-8)

            result = self.run_audit(str(config_path), "--splits", "test_sequences")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn('"num_errors": 1', result.stdout)
        self.assertIn("below the configured normalization floor", result.stdout)

    def test_missing_split_is_strict_by_default_but_can_be_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "splits"
            cache_dir.mkdir()
            config_path = self.write_config(root, cache_dir)

            strict = self.run_audit(str(config_path), "--splits", "test_sequences")
            allowed = self.run_audit(
                str(config_path),
                "--splits",
                "test_sequences",
                "--allow-missing",
            )

        self.assertNotEqual(strict.returncode, 0)
        self.assertIn('"num_errors": 1', strict.stdout)
        self.assertEqual(allowed.returncode, 0, allowed.stderr)
        self.assertIn('"num_missing_allowed": 1', allowed.stdout)


if __name__ == "__main__":
    unittest.main()
