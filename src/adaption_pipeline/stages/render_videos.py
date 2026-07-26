"""Render synchronized 1-panel, 2-panel, 2x2, and 2x4 inference videos."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

from ..contracts import ContractError, atomic_write_json
from ..domain import (
    CONDITION_TITLES,
    GLOBAL,
    MOVING_AVERAGE,
    PREDICTION_ARRAY_KEYS,
    STRATEGIES,
    strategy_title,
)
from ..io import load_mapping
from ..adapters.legacy_runtime import LegacyAnatomyAdapter


from src.utils.mri_rendering import (  # noqa: E402
    build_filename_dicom_index,
    load_or_build_mri_cache,
)

anatomy_core = LegacyAnatomyAdapter()


FPS = 50
LAYOUTS = (
    "single",
    "pair",
    "global_four",
    "moving_four",
    "strategy_compare",
)
PANEL_SIZE = anatomy_core.panel_size
INFO_HEIGHT = anatomy_core.info_height
SEPARATOR = anatomy_core.separator
def load_pack(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "path": str(path.resolve()),
            "frames": np.asarray(payload["frame_numbers"], dtype=np.int32),
            "ground_truth": np.asarray(payload["ground_truth"], dtype=np.float32),
            "conditions": {
                condition: np.asarray(payload[key], dtype=np.float32)
                for condition, key in PREDICTION_ARRAY_KEYS.items()
            },
            "target_frame": int(payload["target_u_frame"]),
        }


def panel(
    image: np.ndarray,
    frame: int,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    title: str,
) -> np.ndarray:
    return anatomy_core.draw_panel(
        image, title, frame, predicted, ground_truth
    )


def compose(
    *,
    image: np.ndarray,
    frame: int,
    index: int,
    packs: Mapping[str, Mapping[str, Any]],
    layout: str,
) -> np.ndarray:
    specifications: List[Tuple[str, str, str]]
    if layout == "single":
        specifications = [(GLOBAL, "original", "Global | original")]
        rows, columns = 1, 1
    elif layout == "pair":
        specifications = [
            (GLOBAL, "original", "Global | original"),
            (GLOBAL, "anatomical", "Global | anatomical"),
        ]
        rows, columns = 1, 2
    elif layout == "global_four":
        specifications = [
            (GLOBAL, condition, f"Global | {CONDITION_TITLES[condition]}")
            for condition in PREDICTION_ARRAY_KEYS
        ]
        rows, columns = 2, 2
    elif layout == "moving_four":
        specifications = [
            (
                MOVING_AVERAGE,
                condition,
                (
                    f"Moving average | {CONDITION_TITLES[condition]}"
                    + (
                        " [DOUBLE TRANSFORM]"
                        if "anatomical" in condition
                        else ""
                    )
                ),
            )
            for condition in PREDICTION_ARRAY_KEYS
        ]
        rows, columns = 2, 2
    elif layout == "strategy_compare":
        specifications = []
        for strategy in STRATEGIES:
            prefix = strategy_title(strategy)
            for condition in PREDICTION_ARRAY_KEYS:
                warning = (
                    " [DOUBLE TRANSFORM]"
                    if strategy == MOVING_AVERAGE
                    and "anatomical" in condition
                    else ""
                )
                specifications.append(
                    (
                        strategy,
                        condition,
                        f"{prefix} | {CONDITION_TITLES[condition]}{warning}",
                    )
                )
        rows, columns = 2, 4
    else:
        raise ContractError(f"Unknown render layout {layout}")

    ground_truth = packs[GLOBAL]["ground_truth"][index]
    rendered = [
        panel(
            image,
            frame,
            packs[strategy]["conditions"][condition][index],
            ground_truth,
            title,
        )
        for strategy, condition, title in specifications
    ]
    panel_height = INFO_HEIGHT + PANEL_SIZE
    height = rows * panel_height + (rows - 1) * SEPARATOR
    width = columns * PANEL_SIZE + (columns - 1) * SEPARATOR
    canvas = np.full((height, width, 3), 8, dtype=np.uint8)
    for panel_index, rendered_panel in enumerate(rendered):
        row, column = divmod(panel_index, columns)
        y_value = row * (panel_height + SEPARATOR)
        x_value = column * (PANEL_SIZE + SEPARATOR)
        canvas[
            y_value : y_value + panel_height,
            x_value : x_value + PANEL_SIZE,
        ] = rendered_panel
    return canvas


def render_silent(
    *,
    layout: str,
    packs: Mapping[str, Mapping[str, Any]],
    mri: Mapping[int, np.ndarray],
    output: Path,
    qc_root: Path,
) -> List[str]:
    frames = packs[GLOBAL]["frames"]
    probe = compose(
        image=mri[int(frames[0])],
        frame=int(frames[0]),
        index=0,
        packs=packs,
        layout=layout,
    )
    height, width = probe.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(FPS),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output}")
    target_frame = packs[GLOBAL]["target_frame"]
    target_index = int(np.flatnonzero(frames == target_frame)[0])
    qc_indices = {0, len(frames) // 2, len(frames) - 1, target_index}
    qc_paths = []
    try:
        for index, value in enumerate(frames):
            frame = int(value)
            canvas = (
                probe
                if index == 0
                else compose(
                    image=mri[frame],
                    frame=frame,
                    index=index,
                    packs=packs,
                    layout=layout,
                )
            )
            writer.write(canvas)
            if index in qc_indices:
                qc_path = qc_root / f"{layout}_F{frame:04d}.png"
                qc_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(qc_path), canvas):
                    raise RuntimeError(f"Could not write {qc_path}")
                qc_paths.append(str(qc_path.resolve()))
    finally:
        writer.release()
    return qc_paths


def mux_audio(
    *,
    silent: Path,
    audio_wav: Path,
    output: Path,
) -> None:
    temporary = output.with_name(f".{output.stem}.muxing.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(silent),
            "-i",
            str(audio_wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-vsync",
            "0",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(temporary),
        ],
        check=True,
    )
    temporary.replace(output)


def run(args: Any) -> int:
    speaker_number = int(args.speaker.removeprefix("P"))
    session_number = int(args.session.removeprefix("S"))
    prediction_root = args.prediction_root.resolve()
    packs = {
        strategy: load_pack(
            prediction_root
            / strategy
            / args.speaker
            / args.session
            / "baseline_anatomical.npz"
        )
        for strategy in STRATEGIES
    }
    if not np.array_equal(
        packs[GLOBAL]["frames"], packs[MOVING_AVERAGE]["frames"]
    ):
        raise ContractError("Strategy video timelines differ")
    if not np.array_equal(
        packs[GLOBAL]["ground_truth"],
        packs[MOVING_AVERAGE]["ground_truth"],
    ):
        raise ContractError("Strategy video ground truths differ")
    frames = packs[GLOBAL]["frames"]
    output_root = args.output_root.resolve() / args.speaker / args.session
    dicom_dir = (
        anatomy_core.raw_root
        / args.speaker
        / "DCM_2D"
        / args.session
    )
    dicom_index, _metadata = build_filename_dicom_index(dicom_dir)
    mri = load_or_build_mri_cache(
        dicom_dir,
        dicom_index,
        [int(value) for value in frames],
        output_root / ".cache/evaluated_integer_mri_frames.npz",
        workers=args.mri_workers,
    )
    original_audio, _textgrid = anatomy_core.exact_asd1_audio_paths(
        speaker_number, session_number
    )
    segment_wav = output_root / "original_audio_evaluated_frame_segments.wav"
    audio_metadata = anatomy_core.write_original_audio_segments(
        original_audio,
        frames,
        segment_wav,
        output_root / "original_audio_evaluated_frame_segments.csv",
    )
    selected_layouts = tuple(args.layout or LAYOUTS)
    for layout in selected_layouts:
        output = output_root / f"{args.speaker.lower()}_{args.session.lower()}_{layout}_50fps.mp4"
        silent = output.with_name(f".{output.stem}.silent.mp4")
        qc_paths = render_silent(
            layout=layout,
            packs=packs,
            mri=mri,
            output=silent,
            qc_root=output_root / "qc",
        )
        mux_audio(silent=silent, audio_wav=segment_wav, output=output)
        silent.unlink(missing_ok=True)
        audit = anatomy_core.audit_video(output, frames, original_audio)
        audit.update(
            {
                "layout": layout,
                "synchronize_on": "exact_frame_id",
                "audio_track": "original target audio once",
                "uses_processed_audio_for_playback": False,
                "visual_qc_samples": qc_paths,
                "moving_average_anatomical_warning": (
                    "moving anatomical panels are target-native double transforms"
                    if layout in {"moving_four", "strategy_compare"}
                    else None
                ),
            }
        )
        atomic_write_json(output.with_suffix(".audit.json"), audit)
        print(
            f"DONE {layout}: {audit['video_frames']} frames, "
            f"{audit['video_duration_seconds']:.3f}s",
            flush=True,
        )
    rows = []
    for layout in LAYOUTS:
        audit_path = (
            output_root
            / f"{args.speaker.lower()}_{args.session.lower()}_{layout}_50fps.audit.json"
        )
        rows.append(load_mapping(audit_path))
    atomic_write_json(
        output_root / "video_manifest.json",
        {
            "status": "complete",
            "speaker": args.speaker,
            "session": args.session,
            "frame_count": int(len(frames)),
            "fps": FPS,
            "audio": audio_metadata,
            "videos": rows,
            "training_launched": False,
        },
    )
    return 0
