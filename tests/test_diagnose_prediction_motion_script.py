from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_payload(static_prediction: bool) -> dict[str, torch.Tensor]:
    labels = torch.tensor(
        [
            [
                [[0.0, 0.0, 1.0, 1.0]],
                [[1.0, 1.0, 2.0, 2.0]],
                [[2.0, 2.0, 3.0, 3.0]],
            ]
        ],
        dtype=torch.float32,
    )
    predicted = labels[:, :1].repeat(1, labels.shape[1], 1, 1) if static_prediction else labels + 0.25
    return {
        "predicted_raw": predicted,
        "labels_raw": labels,
        "frames": torch.tensor([[[2.0, 1.0, 1.0], [2.0, 1.0, 2.0], [2.0, 1.0, 3.0]]], dtype=torch.float32),
        "lengths": torch.tensor([3]),
    }


class DiagnosePredictionMotionScriptTests(unittest.TestCase):
    def run_script(self, payload: dict[str, torch.Tensor], *extra_args: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            payload_path = Path(tmpdir) / "payload.pt"
            torch.save(payload, payload_path)
            return subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/inversion_si.py"),
                    "audit",
                    "motion",
                    "--prediction-payload",
                    str(payload_path),
                    *extra_args,
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

    def test_fail_on_static_exits_nonzero_for_frozen_payload(self) -> None:
        result = self.run_script(make_payload(static_prediction=True), "--fail-on-static")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Prediction payload appears frozen", result.stderr)

    def test_fail_on_static_accepts_moving_payload(self) -> None:
        result = self.run_script(make_payload(static_prediction=False), "--fail-on-static")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"is_frozen_prediction": false', result.stdout)

    def test_diagnostic_bypass_allows_static_payload(self) -> None:
        result = self.run_script(
            make_payload(static_prediction=True),
            "--fail-on-static",
            "--allow-static-prediction-diagnostic",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"is_frozen_prediction": true', result.stdout)


if __name__ == "__main__":
    unittest.main()
