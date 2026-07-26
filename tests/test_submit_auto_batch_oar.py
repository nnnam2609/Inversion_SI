from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
