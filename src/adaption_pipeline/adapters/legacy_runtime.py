"""Stable interfaces around the frozen adaptation compatibility core.

Stage modules depend on these adapters instead of reaching through nested
legacy modules. Replacing the compatibility implementation therefore requires
changes here, not across inference, evaluation and rendering stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..runtime import (
    activate_grid_transform_project,
    load_anatomy_core,
    load_inversion_audio_core,
)


@dataclass(frozen=True)
class TransformBundle:
    source: dict[str, Any]
    target: dict[str, Any]
    transform: dict[str, Any]
    diagnostics: dict[str, Any]


class LegacyInversionAdapter:
    """Inference/audio operations consumed by maintained pipeline stages."""

    def __init__(self) -> None:
        self._core = load_inversion_audio_core()

    def exact_asd1_audio_paths(self, speaker: int, session: int):
        return self._core.exact_asd1_audio_paths(speaker, session)

    def exact_asd2_audio_paths(self, bucket: str, session: str):
        return self._core.exact_asd2_audio_paths(bucket, session)

    def load_yaml_config(self, path: Path):
        return self._core.load_yaml_config(path)

    def load_normalization(self, path: Path):
        return self._core.load_normalization(path)

    def infer_session(self, *args: Any, **kwargs: Any):
        return self._core.infer_session(*args, **kwargs)

    def infer_with_features(self, *args: Any, **kwargs: Any):
        return self._core.infer_with_features(*args, **kwargs)

    def retain_integer_inferred(self, value: Any):
        return self._core.retain_integer_inferred(value)

    def asd1_audio_config(self, config: dict[str, Any]):
        return self._core.asd1_audio_config(config)

    def build_audio_normalized_chunks(self, *args: Any, **kwargs: Any):
        return self._core.build_audio_normalized_chunks(*args, **kwargs)


class LegacyAnatomyAdapter:
    """Geometry/rendering operations consumed by maintained pipeline stages."""

    def __init__(self) -> None:
        activate_grid_transform_project()
        self._core = load_anatomy_core()
        self._base = self._core.base

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(self._base.CLASSES)

    @property
    def panel_size(self) -> int:
        return int(self._base.PANEL_SIZE)

    @property
    def info_height(self) -> int:
        return int(self._base.INFO_HEIGHT)

    @property
    def separator(self) -> int:
        return int(self._base.SEPARATOR)

    @property
    def raw_root(self) -> Path:
        return Path(self._base.RAW_ROOT)

    def target_anchor(self, speaker: int, *, p7_anchor: str) -> str:
        if speaker == 7:
            return p7_anchor
        return str(self._base.asd2_core.target_spec(speaker).vtln_anchor)

    def exact_asd1_audio_paths(self, speaker: int, session: int):
        return self._base.asd2_core.exact_asd1_audio_paths(speaker, session)

    def build_transform_bundle(
        self,
        *,
        source_pack: Path,
        speaker: int,
        session: int,
        target_frame: int,
        target_anchor: str,
        source_frame: int = 499,
    ) -> TransformBundle:
        resolved_pack = source_pack.resolve()
        self._base.ASD2_SOURCE_PACK = resolved_pack
        self._base.prior_u.SOURCE_PACK = resolved_pack
        source = self._base.prior_u.load_source_reference(source_frame)
        target = self._base.prepare_frame(
            self._base.FrameSpec(
                f"P{speaker}",
                f"S{session}",
                f"{target_frame:04d}",
                target_anchor,
            ),
            self._base.asd2_core.DEFAULT_VTLN_DIR,
        )
        transform = self._base.build_two_step_transform(
            source["grid"], target["grid"]
        )
        diagnostics = self._base.prior_u.transform_diagnostics(
            transform, source["grid"], target["grid"]
        )
        return TransformBundle(
            source=source,
            target=target,
            transform=transform,
            diagnostics=diagnostics,
        )

    def select_exact_u_reference(self, **kwargs: Any):
        return self._base.select_exact_u_reference(**kwargs)

    def textgrid_u_mask(
        self, frames: np.ndarray, textgrid_path: Path, tier_index: int
    ) -> np.ndarray:
        return self._base.textgrid_u_mask(frames, textgrid_path, tier_index)

    def transform_contour_batch(
        self,
        contours: np.ndarray,
        transform: dict[str, Any],
        frame_batch: int,
    ):
        return self._base.transform_contour_batch(
            contours, transform, frame_batch
        )

    def apply_affine(
        self, transform: dict[str, Any], points: np.ndarray
    ) -> np.ndarray:
        from grid_transform.transform_helpers import apply_transform

        return np.asarray(
            apply_transform(transform["step1_affine"], points)
        )

    def draw_panel(
        self,
        image: np.ndarray,
        title: str,
        frame: int,
        predicted: np.ndarray,
        ground_truth: np.ndarray,
    ) -> np.ndarray:
        return self._base.draw_panel(
            image, title, frame, predicted, ground_truth
        )

    def write_original_audio_segments(
        self,
        audio_path: Path,
        frames: Sequence[int],
        output_wav: Path,
        output_csv: Path,
    ):
        return self._base.write_original_audio_segments(
            audio_path, frames, output_wav, output_csv
        )

    def audit_video(
        self, path: Path, frames: Sequence[int], source_audio: Path
    ):
        return self._base.audit_video(path, frames, source_audio)
