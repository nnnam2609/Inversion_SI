from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
SCRIPT = REPO_ROOT / "scripts/inversion_si.py"

from src.orchestration.auto_batch import auto_batch_suffix, smoke_runtime_config  # noqa: E402


def write_config(path: Path, split_cache_dir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f"split_cache_dir: {split_cache_dir}",
                "normalization_contour_std_floor: 0.1",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_split_cache(path: Path, std_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": torch.zeros((1, 2, 3), dtype=torch.float32),
            "labels": torch.zeros((1, 2, 1, 2), dtype=torch.float32),
            "frames": torch.zeros((1, 2, 3), dtype=torch.float32),
            "phonemes": torch.zeros((1, 2, 4), dtype=torch.float32),
            "std": torch.full((1, 1, 1, 2), std_value, dtype=torch.float32),
            "mean": torch.zeros((1, 1, 1, 2), dtype=torch.float32),
            "mean_datas": torch.zeros((1, 2), dtype=torch.float32),
            "length_datas": [2],
            "sequences_length": [2],
        },
        path,
    )


class TrainAutoBatchPreflightTests(unittest.TestCase):
    def test_smoke_config_changes_only_run_identity_and_smoke_controls(self) -> None:
        runtime = {
            "experiment_name": "experiment",
            "folder_save": "folder",
            "model": "model",
            "tag": "tag",
            "n_epochs": 500,
            "save_every": 10,
            "patience": 10,
            "batch_size": 128,
            "session_cache_dir": "/cache/sessions",
            "split_cache_dir": "/cache/splits",
        }
        smoke = smoke_runtime_config(runtime, 1)
        self.assertEqual(smoke["n_epochs"], 1)
        self.assertEqual(smoke["save_every"], 1)
        self.assertEqual(smoke["patience"], 1)
        self.assertEqual(smoke["batch_size"], 128)
        self.assertEqual(smoke["session_cache_dir"], runtime["session_cache_dir"])
        self.assertEqual(smoke["split_cache_dir"], runtime["split_cache_dir"])
        self.assertTrue(smoke["smoke_only"])
        self.assertEqual(runtime["n_epochs"], 500)

    def test_auto_batch_suffix_uses_requested_gpu_count(self) -> None:
        self.assertEqual(auto_batch_suffix(128, 1), "auto80_bs128_1gpu")
        self.assertEqual(auto_batch_suffix(256, 4), "auto80_bs256_4gpu")

    def run_preflight(self, config_path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "train",
                "auto-batch",
                "--config",
                str(config_path),
                "--preflight-only",
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_preflight_only_allows_missing_split_cache_without_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            write_config(config_path, root / "splits")

            result = self.run_preflight(config_path)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"num_missing": 3', result.stdout)
        self.assertIn('"status": "missing"', result.stdout)

    def test_preflight_only_accepts_existing_current_split_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "splits"
            config_path = root / "config.yaml"
            write_config(config_path, split_dir)
            write_split_cache(split_dir / "train_sequences.pt", 0.1)

            result = self.run_preflight(config_path)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"num_ok": 1', result.stdout)
        self.assertIn('"cache_contour_std_floor_ok": true', result.stdout)

    def test_preflight_only_rejects_stale_existing_split_cache_before_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "splits"
            config_path = root / "config.yaml"
            write_config(config_path, split_dir)
            write_split_cache(split_dir / "train_sequences.pt", 1e-8)

            result = self.run_preflight(config_path)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("below the configured normalization floor", result.stderr)


if __name__ == "__main__":
    unittest.main()
