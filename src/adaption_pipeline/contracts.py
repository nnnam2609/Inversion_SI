"""Versioned contracts shared by independent pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.common.artifacts import atomic_write_json, sha256_file, utc_now


SCHEMA_VERSION = "1.0"


class ContractError(ValueError):
    """Raised when an artifact violates a pipeline contract."""


@dataclass(frozen=True)
class CapabilityFlags:
    uses_target_labels: bool
    uses_target_statistics: bool
    causal: bool
    blind_inference_compatible: bool

    def validate(self) -> None:
        if self.uses_target_labels and self.blind_inference_compatible:
            raise ContractError(
                "A model using target labels cannot be blind-inference compatible"
            )
        if self.uses_target_labels and not self.uses_target_statistics:
            raise ContractError(
                "A strategy using target labels must declare target-statistics use"
            )


@dataclass(frozen=True)
class ModelBundle:
    strategy: str
    checkpoint: str
    checkpoint_sha256: str
    config: str
    split_root: str
    center_key: str
    capabilities: CapabilityFlags
    output_coordinate_space: str
    run_id: Optional[str] = None
    code_root: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    def validate(self, verify_files: bool = True) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(f"Unsupported ModelBundle schema {self.schema_version}")
        if not self.strategy:
            raise ContractError("Model strategy cannot be empty")
        allowed_spaces = {
            "reference_native",
            "target_native_using_target_statistics",
        }
        if self.output_coordinate_space not in allowed_spaces:
            raise ContractError(
                f"Unknown model output coordinate space "
                f"{self.output_coordinate_space!r}; expected {sorted(allowed_spaces)}"
            )
        if (
            self.output_coordinate_space
            == "target_native_using_target_statistics"
            and not self.capabilities.uses_target_statistics
        ):
            raise ContractError(
                "target-native output requires target-statistics capability"
            )
        self.capabilities.validate()
        checkpoint = Path(self.checkpoint)
        if verify_files:
            if not checkpoint.is_file():
                raise ContractError(f"Missing checkpoint: {checkpoint}")
            actual = sha256_file(checkpoint)
            if actual != self.checkpoint_sha256:
                raise ContractError(
                    f"Checkpoint hash mismatch for {checkpoint}: "
                    f"{actual} != {self.checkpoint_sha256}"
                )
            if not Path(self.config).is_file():
                raise ContractError(f"Missing model config: {self.config}")
            if not Path(self.split_root).is_dir():
                raise ContractError(f"Missing split root: {self.split_root}")


@dataclass(frozen=True)
class SessionRecord:
    speaker: str
    session: str
    speaker_order: int
    session_order: int
    dataset: str
    role: str
    reference_frame: int
    reference_vowel: str = "u"
    raw_speaker: Optional[str] = None
    series_time: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.speaker}/{self.session}"

    def validate(self) -> None:
        if self.speaker_order < 0:
            raise ContractError(f"Negative speaker_order for {self.key}")
        if self.session_order < 0:
            raise ContractError(f"Negative session_order for {self.key}")
        if not self.speaker or not self.session:
            raise ContractError("Speaker and session are required")
        if self.reference_frame < 0:
            raise ContractError(f"Negative reference frame for {self.key}")
        if self.reference_vowel.strip("/").lower() != "u":
            raise ContractError(
                f"Current anatomical calibration requires /u/, got "
                f"{self.reference_vowel!r} for {self.key}"
            )


@dataclass(frozen=True)
class CohortManifest:
    cohort_id: str
    sessions: List[SessionRecord]
    source_reference: SessionRecord
    comparison_policy: str = "exact_session_order"
    schema_version: str = SCHEMA_VERSION

    def ordered_sessions(self) -> List[SessionRecord]:
        return sorted(
            self.sessions, key=lambda item: (item.speaker_order, item.session_order)
        )

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(f"Unsupported CohortManifest schema {self.schema_version}")
        if self.comparison_policy != "exact_session_order":
            raise ContractError(
                "Comparison policy must be exact_session_order; silent intersection "
                "is not allowed"
            )
        self.source_reference.validate()
        if not self.sessions:
            raise ContractError("Cohort has no target sessions")
        keys = set()
        order_pairs = set()
        orders_by_speaker: Dict[str, List[int]] = {}
        speaker_order_by_name: Dict[str, int] = {}
        for session in self.sessions:
            session.validate()
            if session.key in keys:
                raise ContractError(f"Duplicate cohort session: {session.key}")
            order_pair = (session.speaker_order, session.session_order)
            if order_pair in order_pairs:
                raise ContractError(
                    f"Duplicate speaker/session order pair: {order_pair}"
                )
            previous_speaker_order = speaker_order_by_name.setdefault(
                session.speaker, session.speaker_order
            )
            if previous_speaker_order != session.speaker_order:
                raise ContractError(
                    f"Speaker {session.speaker} has inconsistent speaker_order"
                )
            keys.add(session.key)
            order_pairs.add(order_pair)
            orders_by_speaker.setdefault(session.speaker, []).append(
                session.session_order
            )
        expected_speakers = list(range(len(speaker_order_by_name)))
        actual_speakers = sorted(speaker_order_by_name.values())
        if actual_speakers != expected_speakers:
            raise ContractError(
                f"speaker_order must be contiguous {expected_speakers}, "
                f"got {actual_speakers}"
            )
        for speaker, actual_orders in orders_by_speaker.items():
            actual_orders.sort()
            expected_orders = list(range(len(actual_orders)))
            if actual_orders != expected_orders:
                raise ContractError(
                    f"session_order for {speaker} must be contiguous "
                    f"{expected_orders}, got {actual_orders}"
                )


@dataclass(frozen=True)
class ExternalRepoState:
    name: str
    path: str
    head: str
    dirty: bool
    remote: Optional[str]
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class CalibrationPair:
    source_session: str
    source_frame: int
    target_session: str
    target_frame: int
    vowel: str
    affine_landmarks: List[str]
    tps_landmarks: List[str]
    annotation_provenance: Dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.vowel.strip("/").lower() != "u":
            raise ContractError("Calibration pair must use the preregistered /u/ vowel")
        if not self.affine_landmarks:
            raise ContractError("Affine landmarks cannot be empty")
        if not self.tps_landmarks:
            raise ContractError("TPS landmarks cannot be empty")


def require_exact_session_order(
    left: Iterable[SessionRecord], right: Iterable[SessionRecord]
) -> None:
    left_order = [item.session_order for item in left]
    right_order = [item.session_order for item in right]
    if left_order != right_order:
        raise ContractError(
            "Speaker comparisons require the same number and exact session order; "
            f"left={left_order}, right={right_order}"
        )


def load_model_bundle(data: Mapping[str, Any]) -> ModelBundle:
    capabilities = CapabilityFlags(**data["capabilities"])
    payload = dict(data)
    payload["capabilities"] = capabilities
    return ModelBundle(**payload)


def load_session(data: Mapping[str, Any]) -> SessionRecord:
    return SessionRecord(**dict(data))


def load_cohort(data: Mapping[str, Any]) -> CohortManifest:
    return CohortManifest(
        cohort_id=data["cohort_id"],
        sessions=[load_session(item) for item in data["sessions"]],
        source_reference=load_session(data["source_reference"]),
        comparison_policy=data.get("comparison_policy", "exact_session_order"),
        schema_version=data.get("schema_version", SCHEMA_VERSION),
    )
