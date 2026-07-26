import tempfile
import unittest
import json
import os
import sys
from pathlib import Path

import numpy as np

from src.adaption_pipeline.contracts import (
    CapabilityFlags,
    CohortManifest,
    ContractError,
    ModelBundle,
    SessionRecord,
    require_exact_session_order,
)
from src.adaption_pipeline.metrics import (
    correlation_change,
    correlation_increase,
    p2cp_mm,
    signed_correlation_change,
)
from src.adaption_pipeline.domain import MOVING_AVERAGE, STRATEGIES
from src.adaption_pipeline.orchestration.dag import Dag, Stage, load_dag


def session(speaker: str, speaker_order: int, session_order: int) -> SessionRecord:
    return SessionRecord(
        speaker=speaker,
        session=f"S{session_order + 1}",
        speaker_order=speaker_order,
        session_order=session_order,
        dataset="ASD1",
        role="target",
        reference_frame=10,
    )


class ContractTests(unittest.TestCase):
    def test_target_label_strategy_cannot_claim_blind_inference(self):
        flags = CapabilityFlags(
            uses_target_labels=True,
            uses_target_statistics=True,
            causal=False,
            blind_inference_compatible=True,
        )
        with self.assertRaises(ContractError):
            flags.validate()

    def test_target_native_model_must_declare_target_statistics(self):
        model = ModelBundle(
            strategy="bad",
            checkpoint="/missing",
            checkpoint_sha256="x",
            config="/missing",
            split_root="/missing",
            center_key="mean",
            capabilities=CapabilityFlags(
                uses_target_labels=False,
                uses_target_statistics=False,
                causal=True,
                blind_inference_compatible=True,
            ),
            output_coordinate_space="target_native_using_target_statistics",
        )
        with self.assertRaises(ContractError):
            model.validate(verify_files=False)

    def test_session_order_is_within_speaker(self):
        source = session("ASD2", 0, 0)
        cohort = CohortManifest(
            cohort_id="test",
            source_reference=source,
            sessions=[session("P1", 0, 0), session("P2", 1, 0)],
        )
        cohort.validate()

    def test_session_comparison_rejects_different_counts(self):
        with self.assertRaises(ContractError):
            require_exact_session_order(
                [session("P1", 0, 0)],
                [session("P2", 1, 0), session("P2", 1, 1)],
            )

    def test_symmetric_p2cp_is_per_frame(self):
        target = np.zeros((2, 1, 2, 2), dtype=np.float64)
        predicted = target.copy()
        predicted[1, :, :, 0] = 2.0
        result = p2cp_mm(predicted, target)
        self.assertEqual(result.shape, (2, 1))
        np.testing.assert_allclose(result[:, 0], [0.0, 2.0])

    def test_correlation_reduction_percent(self):
        result = correlation_change(0.8, 0.6)
        self.assertAlmostEqual(result["magnitude_reduction_percent"], 25.0)

    def test_target_reference_correlation_increase_percent(self):
        result = correlation_increase(0.5, 0.6)
        self.assertAlmostEqual(result["relative_increase_percent"], 20.0)

    def test_signed_audio_correlation_change_can_decrease(self):
        result = signed_correlation_change(0.8, 0.6)
        self.assertAlmostEqual(result["signed_change"], -0.2)
        self.assertAlmostEqual(result["signed_change_percent"], -25.0)
        self.assertEqual(result["direction"], "decreased")

    def test_signed_audio_correlation_change_can_increase(self):
        result = signed_correlation_change(0.5, 0.6)
        self.assertAlmostEqual(result["signed_change_percent"], 20.0)
        self.assertEqual(result["direction"], "increased")

    def test_dag_rejects_training(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ContractError):
                Dag(
                    [Stage(name="fit", command=["true"], kind="training")],
                    Path(directory),
                )

    def test_dag_config_must_disable_training(self):
        with tempfile.TemporaryDirectory() as directory:
            dag_path = Path(directory) / "dag.json"
            dag_path.write_text(
                json.dumps({"stages": [{"name": "x", "command": ["true"]}]}),
                encoding="utf-8",
            )
            with self.assertRaises(ContractError):
                load_dag(dag_path, Path(directory) / "state")

    def test_gpu_dag_stage_requires_oar(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.environ.pop("OAR_JOB_ID", None)
            try:
                dag = Dag(
                    [
                        Stage(
                            name="infer",
                            command=[sys.executable, "-c", "pass"],
                            kind="gpu_inference",
                        )
                    ],
                    Path(directory),
                )
                with self.assertRaises(ContractError):
                    dag.run()
            finally:
                if previous is not None:
                    os.environ["OAR_JOB_ID"] = previous

    def test_canonical_strategy_name_is_moving_average(self):
        self.assertEqual(MOVING_AVERAGE, "moving_average")
        self.assertEqual(STRATEGIES, ("global", "moving_average"))

    def test_workflow_modules_do_not_own_command_line_parsers(self):
        stage_root = Path(__file__).resolve().parents[1] / "src/adaption_pipeline/stages"
        for path in stage_root.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("def parse_args(", source, path.name)
            self.assertNotIn('if __name__ == "__main__"', source, path.name)

    def test_full_dag_uses_the_single_public_entrypoint(self):
        repository = Path(__file__).resolve().parents[1]
        dag_path = repository / "config/adaption_pipeline/full_nontraining_dag.json"
        with tempfile.TemporaryDirectory() as directory:
            dag = load_dag(dag_path, Path(directory))
        scientific = [
            stage
            for stage in dag.stages.values()
            if stage.name not in {"capture_provenance"}
        ]
        for stage in scientific:
            self.assertEqual(
                stage.command[1], "scripts/inversion_si.py", stage.name
            )
            self.assertEqual(stage.command[2], "adapt", stage.name)


if __name__ == "__main__":
    unittest.main()
