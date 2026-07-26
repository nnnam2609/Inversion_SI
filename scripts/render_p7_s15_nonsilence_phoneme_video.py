#!/usr/bin/env python3
"""Create the P7/S15 speech-only comparison video with phoneme burn-in."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import textgrid


REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO / "results/p7_only_pooledraw_bs10_best_p7_s15_compare_20260723"
INFERENCE_ROOT = RESULT_ROOT / "legacy_interval_inference"
OUTPUT_DIR = RESULT_ROOT / "continuous_native_fps_audio/prediction"
SOURCE_VIDEO = (
    RESULT_ROOT
    / "dense11_full_render_tmp/prediction/"
    "p7_s15_p7_pooledraw_bs10_best_prediction_11contour_silent.mp4"
)
AUDIO = Path(
    "/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/"
    "ArtSpeech_Database_1_raw/P7/OTHER/S15/DENOISED_SOUND_P7_S15.wav"
)
TEXTGRID = Path(
    "/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/"
    "ArtSpeech_Database_1_raw/P7/OTHER/S15/TEXT_ALIGNMENT_P7_S15.textgrid"
)
GROUND_TRUTH_DIR = Path(
    "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/"
    "nhanguyen/bf/inference/P7/S15/contours.before-roi-import"
)
PREDICTION_DIR = INFERENCE_ROOT / "predicted_contours"
OUTPUT_VIDEO = (
    OUTPUT_DIR
    / "p7_s15_p7_pooledraw_bs10_best_prediction_11contour_nonsilence_phoneme_audio.mp4"
)
OUTPUT_ASS = OUTPUT_VIDEO.with_suffix(".ass")
OUTPUT_AUDIT = OUTPUT_VIDEO.with_suffix(".audit.json")

CLASSES = [
    "arytenoid-cartilage",
    "epiglottis",
    "lower-lip",
    "pharynx",
    "soft-palate-midline",
    "tongue",
    "upper-lip",
    "vocal-folds",
    "thyroid-cartilage",
    "lower-incisor",
    "upper-incisor",
]
FPS_NUM = 50_000
FPS_DEN = 999
FRAME_SECONDS = FPS_DEN / FPS_NUM
ADDED_FRAMES = 20
MS_IMAGE = 19.98
AUDIO_OFFSET_SECONDS = ADDED_FRAMES * MS_IMAGE / 1000.0


def interval_mark(tier: textgrid.IntervalTier, timestamp: float) -> str | None:
    for interval in tier.intervals:
        if interval.minTime <= timestamp < interval.maxTime:
            return str(interval.mark)
    if timestamp == tier.maxTime and tier.intervals:
        return str(tier.intervals[-1].mark)
    return None


def inclusive_ranges(frames: list[int]) -> list[tuple[int, int]]:
    if not frames:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        ranges.append((start, previous))
        start = previous = frame
    ranges.append((start, previous))
    return ranges


def ass_time(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100.0)))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def ass_escape(value: str) -> str:
    return value.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for path in (SOURCE_VIDEO, AUDIO, TEXTGRID, INFERENCE_ROOT / "frames.csv"):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (GROUND_TRUTH_DIR, PREDICTION_DIR):
        if not path.is_dir():
            raise FileNotFoundError(path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = textgrid.TextGrid.fromFile(str(TEXTGRID))
    if len(grid) < 2:
        raise RuntimeError(f"Expected word and phoneme tiers in {TEXTGRID}")
    word_tier = grid[0]
    phone_tier = grid[1]

    selected: list[dict[str, object]] = []
    with (INFERENCE_ROOT / "frames.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            mfcc_center_seconds = float(row["mfcc_center_seconds"])
            word = interval_mark(word_tier, mfcc_center_seconds)
            if word is None or not word.strip() or word.strip() == "#":
                continue
            phoneme = interval_mark(phone_tier, mfcc_center_seconds)
            if phoneme is None or not phoneme.strip() or phoneme.strip() == "#":
                raise RuntimeError(
                    f"Speech frame {frame} at {mfcc_center_seconds:.6f}s has no speech phoneme: "
                    f"word={word!r}, phoneme={phoneme!r}"
                )
            missing_ground_truth = [
                name
                for name in CLASSES
                if not (GROUND_TRUTH_DIR / f"{frame:04d}_{name}.npy").is_file()
            ]
            missing_prediction = [
                name
                for name in CLASSES
                if not (PREDICTION_DIR / f"{frame:04d}_{name}.npy").is_file()
            ]
            if missing_ground_truth or missing_prediction:
                raise RuntimeError(
                    f"Frame {frame} is not fully paired: "
                    f"missing_gt={missing_ground_truth}, missing_pred={missing_prediction}"
                )
            selected.append(
                {
                    "frame": frame,
                    "mfcc_center_seconds": mfcc_center_seconds,
                    "word": word,
                    "phoneme": phoneme,
                }
            )

    frames = [int(row["frame"]) for row in selected]
    if len(frames) != len(set(frames)) or frames != sorted(frames):
        raise RuntimeError("Selected speech frames must be unique and strictly increasing")
    ranges = inclusive_ranges(frames)
    if len(frames) != 955 or len(ranges) != 11:
        raise RuntimeError(f"Unexpected speech mask: frames={len(frames)}, ranges={ranges}")

    # The output is a concatenation of source frames. Compress equal adjacent
    # phoneme labels into ASS events on the concatenated output timeline.
    phone_runs: list[tuple[int, int, str]] = []
    run_start = 0
    run_phone = str(selected[0]["phoneme"])
    for output_index, row in enumerate(selected[1:], start=1):
        phone = str(row["phoneme"])
        if phone == run_phone:
            continue
        phone_runs.append((run_start, output_index, run_phone))
        run_start = output_index
        run_phone = phone
    phone_runs.append((run_start, len(selected), run_phone))

    ass_lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 544",
        "PlayResY: 662",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Phone,Liberation Sans,19,&H0000FFFF,&H0000FFFF,&H00000000,&H00000000,"
        "0,0,0,0,100,100,0,0,1,1,0,7,16,16,96,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for start_index, end_index, phone in phone_runs:
        ass_lines.append(
            "Dialogue: 0,{start},{end},Phone,,0,0,0,,phoneme: /{phone}/".format(
                start=ass_time(start_index * FRAME_SECONDS),
                end=ass_time(end_index * FRAME_SECONDS),
                phone=ass_escape(phone),
            )
        )
    OUTPUT_ASS.write_text("\n".join(ass_lines) + "\n", encoding="utf-8")

    filter_parts: list[str] = []
    concat_inputs: list[str] = []
    for index, (first, last) in enumerate(ranges):
        # Frame ids follow the zero-based MRI time used by the inversion
        # mapping: audio_time = added_frames*ms_image + frame*ms_image.
        audio_start = AUDIO_OFFSET_SECONDS + first * FRAME_SECONDS
        audio_end = AUDIO_OFFSET_SECONDS + (last + 1) * FRAME_SECONDS
        filter_parts.extend(
            [
                f"[0:v]trim=start_frame={first - 1}:end_frame={last},"
                f"setpts=PTS-STARTPTS[v{index}]",
                f"[1:a]atrim=start={audio_start:.9f}:end={audio_end:.9f},"
                f"asetpts=PTS-STARTPTS[a{index}]",
            ]
        )
        concat_inputs.append(f"[v{index}][a{index}]")
    filter_parts.append(
        f"{''.join(concat_inputs)}concat=n={len(ranges)}:v=1:a=1[vraw][aout]"
    )
    ass_filter_path = str(OUTPUT_ASS).replace("\\", r"\\").replace("'", r"\'")
    filter_parts.append(f"[vraw]subtitles=filename='{ass_filter_path}'[vout]")
    filter_complex = ";".join(filter_parts)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(SOURCE_VIDEO),
        "-i",
        str(AUDIO),
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-r",
        f"{FPS_NUM}/{FPS_DEN}",
        "-vsync",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-ar",
        "16000",
        "-movflags",
        "+faststart",
        str(OUTPUT_VIDEO),
    ]
    subprocess.run(command, check=True)

    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=index,codec_type,codec_name,avg_frame_rate,nb_frames,duration,sample_rate,channels",
                "-show_entries",
                "format=duration,size",
                "-of",
                "json",
                str(OUTPUT_VIDEO),
            ],
            text=True,
        )
    )
    video_stream = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    audio_stream = next(stream for stream in probe["streams"] if stream["codec_type"] == "audio")
    if video_stream["codec_name"] != "h264":
        raise RuntimeError(f"Unexpected video codec: {video_stream}")
    if video_stream["avg_frame_rate"] != f"{FPS_NUM}/{FPS_DEN}":
        raise RuntimeError(f"Unexpected frame rate: {video_stream}")
    if int(video_stream["nb_frames"]) != len(frames):
        raise RuntimeError(f"Unexpected frame count: {video_stream}")
    if audio_stream["codec_name"] != "aac":
        raise RuntimeError(f"Unexpected audio codec: {audio_stream}")
    expected_duration = len(frames) * FRAME_SECONDS
    video_duration = float(video_stream["duration"])
    audio_duration = float(audio_stream["duration"])
    if abs(video_duration - expected_duration) > 1e-6:
        raise RuntimeError((video_duration, expected_duration))
    if abs(audio_duration - video_duration) > 0.002:
        raise RuntimeError((audio_duration, video_duration))
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(OUTPUT_VIDEO), "-f", "null", "-"], check=True)

    audit = {
        "audio_offset_seconds": AUDIO_OFFSET_SECONDS,
        "audio_source": str(AUDIO),
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "definition_of_done_passed": True,
        "ffprobe": probe,
        "fps_rational": f"{FPS_NUM}/{FPS_DEN}",
        "ground_truth_dir": str(GROUND_TRUTH_DIR),
        "label_provenance": "dense BF inference pseudo-labels; not official manual annotation",
        "missing_ground_truth_instances": 0,
        "missing_prediction_instances": 0,
        "num_frames": len(frames),
        "num_merged_frame_ranges": len(ranges),
        "num_phoneme_runs": len(phone_runs),
        "phoneme_selection": "TextGrid tier 1 mark at each selected prediction MFCC center",
        "phoneme_subtitles": str(OUTPUT_ASS),
        "prediction_dir": str(PREDICTION_DIR),
        "silence_policy": "exclude tier-0 marks whose stripped value is empty or '#'; also require a nonempty non-# tier-1 phoneme",
        "source_frame_ranges_inclusive": ranges,
        "source_video": str(SOURCE_VIDEO),
        "video": str(OUTPUT_VIDEO),
        "video_sha256": sha256(OUTPUT_VIDEO),
    }
    OUTPUT_AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
