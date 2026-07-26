from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import textgrid
import torch

from src.inference.dense_audio import (
    dense_feature_selection,
    infer_full_sequence,
    infer_nonoverlapping_chunks,
    infer_presegmented_sequences,
    integer_frame_feature_selection,
    legacy_interval_feature_selection,
    window_starts,
)


class EchoModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def forward(self, inputs, lengths):
        self.call_count += 1
        return inputs.reshape(inputs.shape[0], inputs.shape[1], 2, 2), None, None


class DenseAudioIntegerInferenceTests(unittest.TestCase):
    def test_full_sequence_uses_one_direct_model_forward(self) -> None:
        model = EchoModel()
        features = np.arange(7 * 4, dtype=np.float32).reshape(7, 4)

        predicted, coverage = infer_full_sequence(
            model,
            features,
            class_count=2,
            output_layer=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(model.call_count, 1)
        self.assertEqual(predicted.shape, (7, 2, 2))
        np.testing.assert_array_equal(predicted.reshape(7, 4), features)
        np.testing.assert_array_equal(coverage, np.ones(7, dtype=np.int32))

    def test_dense_feature_selection_covers_every_integer_mri_frame(self) -> None:
        config = {
            "added_frames": 20,
            "ms_image": 19.98,
            "hop_length_ratio": 10,
        }
        features = np.zeros((4396, 39), dtype=np.float32)

        selected, global_indices, first_by_frame = dense_feature_selection(
            features,
            sample_rate=16000,
            window_length_samples=400,
            config=config,
            frame_min=143,
            frame_max=1606,
        )

        self.assertEqual(selected.shape, (2925, 39))
        self.assertEqual(global_indices[0], 325)
        self.assertEqual(global_indices[-1], 3249)
        self.assertEqual(sorted(first_by_frame), list(range(143, 1607)))
        self.assertEqual(first_by_frame[143], 0)
        self.assertEqual(first_by_frame[199], 112)
        self.assertEqual(first_by_frame[1606], 2923)

    def test_integer_frame_selection_uses_one_center_nearest_mfcc_per_frame(self) -> None:
        config = {
            "added_frames": 20,
            "ms_image": 19.98,
            "hop_length_ratio": 10,
        }
        features = np.zeros((4396, 39), dtype=np.float32)

        selected, global_indices, local_by_frame = integer_frame_feature_selection(
            features,
            sample_rate=16000,
            window_length_samples=400,
            config=config,
            frame_min=143,
            frame_max=1606,
        )

        self.assertEqual(selected.shape, (1464, 39))
        self.assertEqual(global_indices[0], 325)
        self.assertEqual(global_indices[56], 437)
        self.assertEqual(global_indices[-1], 3248)
        self.assertEqual(local_by_frame[143], 0)
        self.assertEqual(local_by_frame[199], 56)
        self.assertEqual(local_by_frame[1606], 1463)
        self.assertTrue(np.all(np.diff(global_indices) > 0))

    def test_nonoverlapping_chunks_reset_context_and_copy_each_output_once(self) -> None:
        model = EchoModel()
        features = np.arange(7 * 4, dtype=np.float32).reshape(7, 4)

        predicted, coverage, starts = infer_nonoverlapping_chunks(
            model,
            features,
            chunk_size=3,
            batch_size=2,
            class_count=2,
            output_layer=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(starts, [0, 3, 6])
        self.assertEqual(model.call_count, 2)
        np.testing.assert_array_equal(predicted.reshape(7, 4), features)
        np.testing.assert_array_equal(coverage, np.ones(7, dtype=np.int32))

    def test_legacy_interval_selection_preserves_boundaries_and_skips_hash_silence(self) -> None:
        config = {
            "added_frames": 0,
            "ms_image": 20.0,
            "hop_length_ratio": 10,
        }
        features = np.zeros((30, 4), dtype=np.float32)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "session.TextGrid"
            grid = textgrid.TextGrid(minTime=0.0, maxTime=0.14)
            tier = textgrid.IntervalTier(name="words", minTime=0.0, maxTime=0.14)
            tier.add(0.0, 0.04, "word")
            tier.add(0.04, 0.10, "#")
            tier.add(0.10, 0.14, "word2")
            grid.append(tier)
            grid.write(str(path))

            selected, indices, local_by_frame, slices, metadata = legacy_interval_feature_selection(
                features,
                sample_rate=16000,
                window_length_samples=400,
                config=config,
                textgrid_path=path,
                frame_min=0,
                frame_max=7,
                chunk_size=2,
            )

        self.assertEqual(selected.shape, (6, 4))
        np.testing.assert_array_equal(indices, np.asarray([0, 2, 3, 10, 12, 13]))
        self.assertEqual(sorted(local_by_frame), [0, 1, 2, 5, 6, 7])
        self.assertEqual(slices, [(0, 2), (2, 3), (3, 5), (5, 6)])
        self.assertEqual(metadata["num_selected_textgrid_intervals"], 2)
        self.assertEqual(metadata["num_skipped_silence_intervals"], 1)

    def test_presegmented_sequences_keep_each_interval_independent(self) -> None:
        model = EchoModel()
        features = np.arange(7 * 4, dtype=np.float32).reshape(7, 4)

        predicted, coverage = infer_presegmented_sequences(
            model,
            features,
            sequence_slices=[(0, 2), (2, 7)],
            batch_size=1,
            class_count=2,
            output_layer=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(model.call_count, 2)
        np.testing.assert_array_equal(predicted.reshape(7, 4), features)
        np.testing.assert_array_equal(coverage, np.ones(7, dtype=np.int32))

    def test_overlapping_window_starts_cover_the_dense_sequence(self) -> None:
        starts = window_starts(length=2925, window_size=80, stride=40)
        coverage = np.zeros(2925, dtype=np.int32)
        for start in starts:
            coverage[start : start + 80] += 1

        self.assertEqual(len(starts), 73)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 2845)
        self.assertEqual(coverage.min(), 1)
        self.assertEqual(coverage.max(), 3)


if __name__ == "__main__":
    unittest.main()
