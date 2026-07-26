"""Strict pairing gates used before comparisons and correlation reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .contracts import ContractError, SessionRecord, require_exact_session_order


@dataclass(frozen=True)
class PairedFrame:
    session_order: int
    session_key: str
    frame_id: int


def require_identical_frame_keys(
    left: Sequence[PairedFrame], right: Sequence[PairedFrame]
) -> None:
    left_keys = [
        (item.session_order, item.session_key, item.frame_id) for item in left
    ]
    right_keys = [
        (item.session_order, item.session_key, item.frame_id) for item in right
    ]
    if left_keys != right_keys:
        first_difference = next(
            (
                index
                for index, pair in enumerate(zip(left_keys, right_keys))
                if pair[0] != pair[1]
            ),
            min(len(left_keys), len(right_keys)),
        )
        raise ContractError(
            "Paired comparison frame identity mismatch at index "
            f"{first_difference}; left_count={len(left_keys)}, "
            f"right_count={len(right_keys)}"
        )


def validate_speaker_session_matrix(
    sessions_by_speaker: Mapping[str, Iterable[SessionRecord]]
) -> List[Tuple[int, str]]:
    items = list(sessions_by_speaker.items())
    if not items:
        raise ContractError("No speakers supplied for comparison")
    reference_speaker, reference_sessions_iter = items[0]
    reference_sessions = list(reference_sessions_iter)
    for speaker, sessions_iter in items[1:]:
        sessions = list(sessions_iter)
        try:
            require_exact_session_order(reference_sessions, sessions)
        except ContractError as error:
            raise ContractError(
                f"Session pairing mismatch: {reference_speaker} versus {speaker}: {error}"
            ) from error
    return [(item.session_order, item.series_time or item.session) for item in reference_sessions]
