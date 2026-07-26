from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.utils.session_rendering import aggregate_state, build_timeline, frame_token


class SessionRenderingTests(unittest.TestCase):
    def test_frame_token_handles_integer_and_rejects_half_frames(self) -> None:
        self.assertEqual(frame_token(12.0), "0012")
        with self.assertRaisesRegex(ValueError, "NEVER render fractional"):
            frame_token(12.5)

    def test_aggregate_state_filters_session_and_averages_duplicate_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            phoneme_path = Path(tmpdir) / "phonemes.json"
            phoneme_path.write_text(json.dumps(["UNK", "i"]), encoding="utf-8")
            config = {"phonemesdir": str(phoneme_path)}
            state = {
                "predicted_raw": torch.tensor(
                    [
                        [
                            [[1.0, 3.0, 5.0, 7.0]],
                            [[2.0, 4.0, 6.0, 8.0]],
                        ],
                    ]
                ),
                "labels_raw": torch.tensor(
                    [
                        [
                            [[10.0, 12.0, 14.0, 16.0]],
                            [[20.0, 22.0, 24.0, 26.0]],
                        ],
                    ]
                ),
                "frames": torch.tensor([[[2.0, 1.0, 10.0], [2.0, 1.0, 10.0]]]),
                "lengths": torch.tensor([2]),
                "phonemes": torch.tensor([[[[0.0, 1.0]], [[0.0, 1.0]]]]),
            }

            rows = aggregate_state(state, config, speaker=2, session=1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["frame"], "0010")
        self.assertEqual(rows[0]["phoneme"], "i")
        np.testing.assert_allclose(rows[0]["predicted"], np.asarray([[1.5, 3.5, 5.5, 7.5]], dtype=np.float32))
        np.testing.assert_allclose(rows[0]["labels"], np.asarray([[15.0, 17.0, 19.0, 21.0]], dtype=np.float32))

    def test_build_timeline_rejects_half_frame_step(self) -> None:
        rows = [
            {"frame_number": 1.0, "frame": "0001", "predicted": np.zeros((1, 4)), "held": False},
            {"frame_number": 2.0, "frame": "0002", "predicted": np.ones((1, 4)), "held": False},
        ]

        with self.assertRaisesRegex(ValueError, "NEVER render fractional"):
            build_timeline(rows, step=0.5, max_frames=None)


if __name__ == "__main__":
    unittest.main()
