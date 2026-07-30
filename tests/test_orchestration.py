"""Regression tests for the orchestration domain."""

from __future__ import annotations

# --- Consolidated from test_public_cli.py ---

import contextlib
import io
import unittest

from src import cli
from src.orchestration import grid_transform


class PublicCliTest(unittest.TestCase):
    def test_root_help_lists_the_single_command_tree(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = cli.main(["--help"])
        self.assertEqual(status, 0)
        self.assertIn("preprocess sessions", output.getvalue())
        self.assertIn("adapt", output.getvalue())

    def test_grid_transform_help_does_not_resolve_help_as_a_script(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = grid_transform.main(["--help"])
        self.assertEqual(status, 0)
        self.assertIn("grid-transform", output.getvalue())

# --- Consolidated from test_submit_auto_batch_oar.py ---

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
SCRIPT = REPO_ROOT / "scripts/inversion_si.py"
PYTHON = REPO_ROOT.parent / "inversion" / ".venv" / "bin" / "python"

from src.orchestration.oar import parse_oar_job_id  # noqa: E402


class SubmitAutoBatchOarTests(unittest.TestCase):
    def test_parse_oar_job_id_accepts_common_oarsub_outputs(self) -> None:
        self.assertEqual(parse_oar_job_id("OAR_JOB_ID=1234567\n"), "1234567")
        self.assertEqual(parse_oar_job_id("Job id: 7654321\n"), "7654321")
        self.assertIsNone(parse_oar_job_id("reservation queued\n"))

    def test_prepare_only_generates_manifest_and_job_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"split_cache_dir: {root / 'splits'}",
                        "normalization_contour_std_floor: 0.1",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "train",
                    "submit",
                    "--config",
                    str(config_path),
                    "--python",
                    str(PYTHON),
                    "--log-root",
                    str(root / "logs"),
                    "--gpus",
                    "1",
                    "--cluster",
                    "gres",
                    "--walltime",
                    "00:30:00",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            manifest_path = Path(payload["manifest"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            job_script = Path(manifest["job_script"])

            self.assertFalse(manifest["submitted"])
            self.assertTrue(job_script.exists())
            self.assertIn("split_cache_preflight", manifest["preflight_stdout"])
            self.assertIn("-p", manifest["oar_command"])
            self.assertIn("gres", manifest["oar_command"])
            self.assertIn("/host=1/gpu=1,walltime=00:30:00", manifest["oar_command"])
            self.assertIn(
                "scripts/inversion_si.py train auto-batch",
                job_script.read_text(encoding="utf-8"),
            )

# --- Consolidated from test_train_auto_batch_preflight.py ---

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
