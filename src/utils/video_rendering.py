from __future__ import annotations

import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from matplotlib import colors as mcolors


MM_PER_PIXEL = 1.62


def mean_finite(values: list[float]) -> float:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def rmse_px(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def rgb_to_bgr255(color_name: str) -> tuple[int, int, int]:
    red, green, blue = mcolors.to_rgb(color_name)
    return int(blue * 255), int(green * 255), int(red * 255)


def scale_points(points: np.ndarray, scale: int) -> np.ndarray:
    coords = points.reshape(-1, 2).astype(np.float32) * float(scale)
    return np.round(coords).astype(np.int32)


def draw_dashed_polyline(
    canvas: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    dash_length: int = 12,
    gap_length: int = 16,
) -> None:
    pattern_pos = 0.0
    cycle = float(dash_length + gap_length)
    for start, end in zip(points[:-1], points[1:]):
        start_float = start.astype(np.float32)
        end_float = end.astype(np.float32)
        segment_length = float(np.linalg.norm(end_float - start_float))
        if segment_length == 0:
            continue
        direction = (end_float - start_float) / segment_length
        cursor = 0.0
        while cursor < segment_length:
            phase = pattern_pos % cycle
            remaining = segment_length - cursor
            if phase < dash_length:
                step = min(float(dash_length) - phase, remaining)
                seg_start = start_float + direction * cursor
                seg_end = start_float + direction * (cursor + step)
                cv2.line(
                    canvas,
                    tuple(np.round(seg_start).astype(int)),
                    tuple(np.round(seg_end).astype(int)),
                    color,
                    thickness,
                    lineType=cv2.LINE_AA,
                )
            else:
                step = min(cycle - phase, remaining)
            cursor += step
            pattern_pos += step


def attach_audio(
    silent_mp4: Path,
    audio_path: Path,
    output_mp4: Path,
    start_seconds: float,
    duration_seconds: float,
    output_fps: str | None = None,
) -> bool:
    if not audio_path.is_file():
        return False
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(silent_mp4),
        "-ss",
        f"{start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
    ]
    if output_fps is not None:
        command.extend(["-r", str(output_fps)])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-shortest",
            str(output_mp4),
        ]
    )
    subprocess.run(command, check=True)
    return True
