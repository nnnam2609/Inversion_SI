"""Integer-frame validation and exact timeline pairing."""

from __future__ import annotations

from typing import Iterable

import numpy as np


DEFAULT_INTEGER_TOLERANCE = 1e-8


def integer_frame_numbers(
    values: Iterable[float], *, tolerance: float = DEFAULT_INTEGER_TOLERANCE
) -> np.ndarray:
    """Return integer frame numbers or raise when any value is fractional."""

    frames = np.asarray(list(values), dtype=np.float64)
    if frames.ndim != 1:
        raise ValueError(f"Frame numbers must be one-dimensional, got {frames.shape}")
    if not np.isfinite(frames).all():
        raise ValueError("Frame numbers contain NaN or infinity")
    rounded = np.rint(frames)
    fractional = np.abs(frames - rounded) > tolerance
    if fractional.any():
        examples = frames[fractional][:10].tolist()
        raise ValueError(
            "Fractional frame numbers are forbidden; "
            f"count={int(fractional.sum())}, examples={examples}"
        )
    return rounded.astype(np.int64)


def require_integer_frames(
    values: Iterable[float], *, tolerance: float = DEFAULT_INTEGER_TOLERANCE
) -> None:
    """Validate the repository-wide integer-frame invariant."""

    integer_frame_numbers(values, tolerance=tolerance)


def require_identical_frame_numbers(
    left: Iterable[float], right: Iterable[float]
) -> np.ndarray:
    """Require two timelines to contain the same ordered integer frames."""

    left_frames = integer_frame_numbers(left)
    right_frames = integer_frame_numbers(right)
    if not np.array_equal(left_frames, right_frames):
        mismatch = next(
            (
                index
                for index, pair in enumerate(zip(left_frames, right_frames))
                if pair[0] != pair[1]
            ),
            min(len(left_frames), len(right_frames)),
        )
        raise ValueError(
            "Frame timeline mismatch at index "
            f"{mismatch}: left_count={len(left_frames)}, "
            f"right_count={len(right_frames)}"
        )
    return left_frames


def frame_token(value: float, *, width: int = 4) -> str:
    """Format a validated integer frame number with zero padding."""

    frame = integer_frame_numbers([value])[0]
    return f"{int(frame):0{width}d}"
