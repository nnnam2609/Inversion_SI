#!/usr/bin/env python3
"""Retitle the canonical P7/S15 side-by-side video without changing its data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path


FPS_NUM = 50_000
FPS_DEN = 999
FRAME_SECONDS = FPS_DEN / FPS_NUM
NUM_CONTOURS = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-video", type=Path, required=True)
    parser.add_argument("--right-video", type=Path, required=True)
    parser.add_argument("--left-metrics", type=Path, required=True)
    parser.add_argument("--right-metrics", type=Path, required=True)
    parser.add_argument("--selection-audit", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--left-label", default="global normalize")
    parser.add_argument("--right-label", default="moving average")
    return parser.parse_args()


def ass_time(seconds: float) -> str:
    centiseconds = max(0, int(seconds * 100.0 + 0.5))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def ass_escape(value: str) -> str:
    return (
        value.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_frames(audit_path: Path) -> list[int]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    ranges = audit["source_frame_ranges_inclusive"]
    frames = [
        frame
        for first, last in ranges
        for frame in range(int(first), int(last) + 1)
    ]
    if len(frames) != int(audit["num_frames"]):
        raise RuntimeError(
            f"Selection audit count mismatch: expanded={len(frames)} "
            f"declared={audit['num_frames']}"
        )
    return frames


def load_metrics(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {int(row["frame_number"]): row for row in csv.DictReader(handle)}
    if len(rows) != 2_200:
        raise RuntimeError(f"Expected 2,200 metric rows in {path}, found {len(rows)}")
    return rows


def metric_text(value: str) -> str:
    number = float(value)
    return "N/A" if not math.isfinite(number) else f"{number:.3f} mm"


def missing_names(row: dict[str, str], field: str) -> set[str]:
    return {value for value in row[field].split(";") if value}


def write_ass(
    path: Path,
    frames: list[int],
    metrics: dict[int, dict[str, str]],
    model_label: str,
) -> None:
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 544",
        "PlayResY: 662",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Info,Liberation Sans,17,&H00F5F5F5,&H00F5F5F5,&H00000000,"
        "&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,16,8,4,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text",
    ]
    for output_index, frame in enumerate(frames):
        row = metrics[frame]
        missing = missing_names(row, "missing_ground_truth_contours") | missing_names(
            row, "missing_prediction_contours"
        )
        paired = NUM_CONTOURS - len(missing)
        text = "\n".join(
            [
                f"P7/S15 frame {frame:04d} | ground truth vs {model_label}",
                f"RMSE: {metric_text(row['rmse_mm'])} | paired contours: "
                f"{paired}/{NUM_CONTOURS}",
                f"RMSE primary: {metric_text(row['primary_rmse_mm'])}",
                f"solid = ground truth | dashed = {model_label} prediction",
            ]
        )
        lines.append(
            "Dialogue: 0,{start},{end},Info,,0,0,0,,{text}".format(
                start=ass_time(output_index * FRAME_SECONDS),
                end=ass_time((output_index + 1) * FRAME_SECONDS),
                text=ass_escape(text),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def probe(path: Path) -> dict[str, object]:
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=index,codec_type,codec_name,width,height,avg_frame_rate,"
                "nb_frames,duration,sample_rate,channels",
                "-show_entries",
                "format=duration,size",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )


def escaped_filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", r"\\").replace("'", r"\'").replace(":", r"\:")


def main() -> None:
    args = parse_args()
    for path in (
        args.left_video,
        args.right_video,
        args.left_metrics,
        args.right_metrics,
        args.selection_audit,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    left_ass = args.output_video.with_name(f"{args.output_video.stem}_left.ass")
    right_ass = args.output_video.with_name(f"{args.output_video.stem}_right.ass")
    frames = selected_frames(args.selection_audit)
    if len(frames) != 955:
        raise RuntimeError(f"Expected 955 strict speech frames, found {len(frames)}")
    write_ass(left_ass, frames, load_metrics(args.left_metrics), args.left_label)
    write_ass(right_ass, frames, load_metrics(args.right_metrics), args.right_label)

    filter_complex = (
        "[0:v]drawbox=x=0:y=0:w=iw:h=92:color=0x0f0f0f:t=fill,"
        f"subtitles=filename='{escaped_filter_path(left_ass)}'[left];"
        "[1:v]drawbox=x=0:y=0:w=iw:h=92:color=0x0f0f0f:t=fill,"
        f"subtitles=filename='{escaped_filter_path(right_ass)}'[right];"
        "[left][right]hstack=inputs=2[v]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(args.left_video),
            "-i",
            str(args.right_video),
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "1:a:0",
            "-r",
            f"{FPS_NUM}/{FPS_DEN}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(args.output_video),
        ],
        check=True,
    )

    output_probe = probe(args.output_video)
    video_stream = next(
        stream for stream in output_probe["streams"] if stream["codec_type"] == "video"
    )
    audio_stream = next(
        stream for stream in output_probe["streams"] if stream["codec_type"] == "audio"
    )
    if (
        video_stream["codec_name"] != "h264"
        or video_stream["avg_frame_rate"] != "50000/999"
        or int(video_stream["nb_frames"]) != len(frames)
        or int(video_stream["width"]) != 1088
        or int(video_stream["height"]) != 662
        or audio_stream["codec_name"] != "aac"
    ):
        raise RuntimeError(output_probe)
    if abs(float(video_stream["duration"]) - len(frames) * FRAME_SECONDS) > 1e-6:
        raise RuntimeError(output_probe)
    if abs(float(audio_stream["duration"]) - float(video_stream["duration"])) > 0.002:
        raise RuntimeError(output_probe)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(args.output_video), "-f", "null", "-"],
        check=True,
    )
    print(
        json.dumps(
            {
                "definition_of_done_passed": True,
                "frames": len(frames),
                "left_label": args.left_label,
                "right_label": args.right_label,
                "output_video": str(args.output_video.resolve()),
                "output_sha256": sha256(args.output_video),
                "probe": output_probe,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
