from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.common.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    load_mapping,
    sha256_file,
)
from src.common.frames import (
    frame_token,
    require_identical_frame_numbers,
    require_integer_frames,
)


class ArtifactHelpersTest(unittest.TestCase):
    def test_json_csv_and_hash_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "nested" / "payload.json"
            csv_path = root / "rows.csv"
            atomic_write_json(json_path, {"path": root, "values": (1, 2)})
            atomic_write_csv(csv_path, [{"speaker": "P1", "session": "S16"}])

            self.assertEqual(load_mapping(json_path)["values"], [1, 2])
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8"))["path"],
                str(root),
            )
            self.assertEqual(len(sha256_file(csv_path)), 64)

    def test_csv_rejects_non_rectangular_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "fields differ"):
                atomic_write_csv(
                    Path(directory) / "rows.csv",
                    [{"speaker": "P1"}, {"session": "S16"}],
                )


class IntegerFrameHelpersTest(unittest.TestCase):
    def test_integer_frames_and_tokens(self) -> None:
        require_integer_frames([1.0, 2.0, 3])
        self.assertEqual(frame_token(7.0), "0007")
        np.testing.assert_array_equal(
            require_identical_frame_numbers([1, 2], [1.0, 2.0]),
            np.asarray([1, 2]),
        )

    def test_fractional_frames_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Fractional"):
            require_integer_frames([1.5])


if __name__ == "__main__":
    unittest.main()
