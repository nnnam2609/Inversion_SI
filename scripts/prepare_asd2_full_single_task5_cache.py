#!/usr/bin/env python3
"""Prepare exact ASD2 full-split dataset cache for single-task-5 training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import textgrid
import torch
import torchaudio
import yaml
from torch.nn.utils.rnn import pad_sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
(REPO_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from preprocessing.contours_preprocessing import Corpus_contours  # noqa: E402
from preprocessing.main_preprocessing import Corpus  # noqa: E402


SPLIT_FILES = {
    "train_sequences": "train_sequences.pt",
    "valid_sequences": "valid_sequences.pt",
    "test_sequences": "test_sequences.pt",
}


def dataset_type_for_sequence(config: Dict[str, Any], sequence: str) -> str:
    sequence_key = str(sequence)
    dataset_types = config.get("dataset_types", {})
    if sequence_key in dataset_types:
        return str(dataset_types[sequence_key]).lower()
    dataset_type = str(config.get("dataset_type", "asd2")).lower()
    if dataset_type == "mixed":
        return "asd1" if sequence_key.upper().startswith("P") else "asd2"
    return dataset_type


def raw_session_part_path(cache_dir: Path, config: Dict[str, Any], bucket: str, session: str) -> Path:
    dataset_type = dataset_type_for_sequence(config, bucket)
    return cache_dir / "raw_sessions" / dataset_type / str(bucket) / f"{session}.pt"


def raw_session_work_dir(cache_dir: Path, config: Dict[str, Any], bucket: str, session: str) -> Path:
    dataset_type = dataset_type_for_sequence(config, bucket)
    return cache_dir / "work" / "raw_sessions" / dataset_type / str(bucket) / str(session)


def contour_pack_path(cache_dir: Path, config: Dict[str, Any], bucket: str, session: str) -> Path:
    dataset_type = dataset_type_for_sequence(config, bucket)
    return cache_dir / "raw_contour_npz" / dataset_type / str(bucket) / f"{session}.npz"


def make_pseudo_textgrid(duration_seconds: float) -> textgrid.TextGrid:
    tg = textgrid.TextGrid(minTime=0.0, maxTime=duration_seconds)
    sentence_tier = textgrid.IntervalTier(name="sentences", minTime=0.0, maxTime=duration_seconds)
    sentence_tier.add(0.0, duration_seconds, "speech")
    phoneme_tier = textgrid.IntervalTier(name="phones", minTime=0.0, maxTime=duration_seconds)
    phoneme_tier.add(0.0, duration_seconds, "#")
    tg.append(sentence_tier)
    tg.append(phoneme_tier)
    return tg


def repair_textgrid_bounds_file(path: str, duration_seconds: float) -> Path:
    source_path = Path(path)
    text = source_path.read_text(errors="replace")
    values = [float(match) for match in re.findall(r"(?m)^\s*x(?:min|max)\s*=\s*([0-9.]+)", text)]
    repaired_max = max([duration_seconds, *values]) + 0.001
    digest = hashlib.sha1(f"{source_path}:{source_path.stat().st_mtime_ns}:{repaired_max}".encode()).hexdigest()[:16]
    output_path = REPO_ROOT / ".cache" / "repaired_textgrids" / f"{source_path.stem}_{digest}.TextGrid"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path

    lines = text.splitlines()
    repaired_lines = []
    inside_interval = False
    interval_xmin = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("intervals ["):
            inside_interval = True
            interval_xmin = None
            repaired_lines.append(line)
            continue
        if inside_interval and stripped.startswith("xmin ="):
            interval_xmin = float(stripped.split("=", 1)[1].strip())
            repaired_lines.append(line)
            continue
        if stripped.startswith("xmax ="):
            prefix = line.split("=", 1)[0] + "= "
            value = float(stripped.split("=", 1)[1].strip())
            if inside_interval:
                if interval_xmin is not None and value <= interval_xmin:
                    value = interval_xmin + 0.001
                repaired_lines.append(f"{prefix}{value:.6f}")
            else:
                repaired_lines.append(f"{prefix}{repaired_max:.6f}")
            continue
        repaired_lines.append(line)
        if inside_interval and stripped.startswith("text ="):
            inside_interval = False
            interval_xmin = None

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text("\n".join(repaired_lines) + "\n", encoding="utf-8")
    os.replace(tmp_path, output_path)
    return output_path


def load_textgrid_with_repair(path: str, duration_seconds: float, config: Dict[str, Any]) -> textgrid.TextGrid:
    try:
        return textgrid.TextGrid.fromFile(path)
    except Exception as exc:
        if config.get("repair_textgrid_bounds", False):
            repaired_path = repair_textgrid_bounds_file(path, duration_seconds)
            try:
                print(
                    f"TextGrid parse failed for {path}: {exc!r}; "
                    f"retrying repaired bounds file {repaired_path}",
                    flush=True,
                )
                return textgrid.TextGrid.fromFile(str(repaired_path))
            except Exception as repair_exc:
                print(
                    f"Repaired TextGrid parse also failed for {path}: {repair_exc!r}",
                    flush=True,
                )
        if config.get("pseudo_textgrid_on_parse_error", False):
            print(
                f"TextGrid parse failed for {path}: {exc!r}; "
                f"using pseudo TextGrid duration={duration_seconds:.3f}s",
                flush=True,
            )
            return make_pseudo_textgrid(duration_seconds)
        raise


def chunk_required_frames(image_numbers: Iterable[Any]) -> List[int]:
    frames = set()
    for image_number in image_numbers:
        int_image_number = int(image_number)
        if image_number == int_image_number:
            frames.add(int_image_number)
        elif isinstance(image_number, (float, np.floating)):
            rounded_down = int(np.floor(image_number))
            frames.add(rounded_down)
            frames.add(rounded_down + 1)
    return sorted(frames)


def missing_contour_paths(image_folder: Path, image_numbers: Iterable[Any], articulators: List[str]) -> List[Path]:
    missing = []
    for frame_number in chunk_required_frames(image_numbers):
        for articulator in articulators:
            contour_path = image_folder / f"{frame_number:04d}_{articulator}.npy"
            if not contour_path.exists():
                missing.append(contour_path)
    return missing


class RawContourSession(Corpus_contours):
    """Use Corpus helpers while saving raw, pre-normalization session chunks."""

    def __init__(
        self,
        config: Dict[str, Any],
        sequences: str,
        rank: int,
        contour_pack_path: str | None = None,
        rebuild_contour_pack: bool = False,
    ):
        Corpus.__init__(self, config, sequences, rank)
        self.contour_pack_path = Path(contour_pack_path) if contour_pack_path else None
        self.rebuild_contour_pack = rebuild_contour_pack

    def read_raw(self) -> Dict[str, Any]:
        all_features = []
        all_contours = []
        all_frames = []
        all_phonemes = []
        sequence_lengths = []

        for (_, audio_sessions), (_, tg_sessions), (_, contour_sessions) in zip(
            self.audio_files.items(),
            self.textgrid_files.items(),
            self.images_files.items(),
        ):
            for index, (audio_session, tg_session, contours_session) in enumerate(
                zip(audio_sessions, tg_sessions, contour_sessions)
            ):
                print(f"Processing raw session {index + 1}/{len(audio_sessions)}: {audio_session}", flush=True)
                if self.config["input_type"] == "mfcc":
                    import librosa

                    audio_signal, sample_rate = librosa.load(audio_session, sr=None)
                    features, window_length_samples, hop_length_samples = self.compute_mfcc(
                        audio_signal,
                        sample_rate,
                    )
                elif self.config["input_type"] == "cepstre":
                    audio_signal, sample_rate = torchaudio.load(audio_session, normalize=False)
                    features, window_length_samples, hop_length_samples = self.compute_cepstre(
                        audio_signal,
                        sample_rate,
                    )
                else:
                    raise ValueError(
                        "prepare_asd2_full_single_task5_cache.py only supports mfcc/cepstre "
                        f"for raw session caching, got input_type={self.config['input_type']}"
                    )

                duration_seconds = float(len(audio_signal) / sample_rate)
                tg = load_textgrid_with_repair(tg_session, duration_seconds, self.config)
                index_silence = self.detect_silence(features, tg, sample_rate)
                list_features, list_contour, _, list_phoneme_one_hot = self.detect_sentences(
                    features,
                    tg,
                    self.all_phonemes,
                    sample_rate,
                    window_length_samples,
                    hop_length_samples,
                    index_silence,
                )
                max_chunks = self.config.get("max_chunks_per_session")
                if max_chunks:
                    list_features = list_features[:max_chunks]
                    list_contour = list_contour[:max_chunks]
                    list_phoneme_one_hot = list_phoneme_one_hot[:max_chunks]
                print(f"Prepared {len(list_contour)} raw chunks for contour loading", flush=True)

                if self.config.get("skip_missing_contour_chunks", False):
                    image_folder = Path(contours_session)
                    keep_features = []
                    keep_contours = []
                    keep_phonemes = []
                    skipped_chunks = []
                    for chunk_idx, (feature, image_numbers, phoneme) in enumerate(
                        zip(list_features, list_contour, list_phoneme_one_hot)
                    ):
                        missing = missing_contour_paths(image_folder, image_numbers, self.config["classes"])
                        if missing:
                            skipped_chunks.append(
                                {
                                    "chunk_index": chunk_idx,
                                    "missing_count": len(missing),
                                    "first_missing": str(missing[0]),
                                }
                            )
                            continue
                        keep_features.append(feature)
                        keep_contours.append(image_numbers)
                        keep_phonemes.append(phoneme)
                    if skipped_chunks:
                        print(
                            f"Skipped {len(skipped_chunks)}/{len(list_contour)} chunks with missing contours; "
                            f"first_missing={skipped_chunks[0]['first_missing']}",
                            flush=True,
                        )
                    list_features = keep_features
                    list_contour = keep_contours
                    list_phoneme_one_hot = keep_phonemes
                    if not list_contour:
                        raise RuntimeError("No chunks remain after filtering missing contour files")

                if self.contour_pack_path:
                    contour_loaded = self.load_labels_from_npz_pack(contours_session, list_contour)
                else:
                    contour_loaded = self.load_labels(contours_session, list_contour)
                frame_paths = self.load_path(contours_session, list_contour)
                print(f"Loaded raw contours for {len(contour_loaded)} chunks", flush=True)

                all_features.extend(list_features)
                all_contours.extend(contour_loaded)
                all_frames.extend(frame_paths)
                all_phonemes.extend(list_phoneme_one_hot)
                sequence_lengths.append(len(contour_loaded))

        return {
            "features": all_features,
            "contours": all_contours,
            "frames": all_frames,
            "phonemes": all_phonemes,
            "length_datas": sequence_lengths,
        }

    def load_labels_from_npz_pack(self, image_folder: str, list_image: list) -> list:
        frame_numbers, contours = load_or_build_contour_npz_pack(
            image_folder=Path(image_folder),
            list_image=list_image,
            articulators=self.config["classes"],
            pack_path=self.contour_pack_path,
            rebuild=self.rebuild_contour_pack,
            file_workers=int(self.config.get("contour_file_workers", 1)),
        )
        frame_to_index = {int(frame): idx for idx, frame in enumerate(frame_numbers.tolist())}
        all_contours = []
        num_articulators = len(self.config["classes"])
        for image_numbers in list_image:
            sequence = np.zeros((len(image_numbers), num_articulators, 100), dtype=np.float32)
            for seq_idx, image_number in enumerate(image_numbers):
                int_image_number = int(image_number)
                if image_number == int_image_number:
                    sequence[seq_idx] = contours[frame_to_index[int_image_number]]
                elif isinstance(image_number, (float, np.floating)):
                    rounded_down = int(np.floor(image_number))
                    rounded_up = rounded_down + 1
                    sequence[seq_idx] = (
                        contours[frame_to_index[rounded_down]] + contours[frame_to_index[rounded_up]]
                    ) / 2.0
            all_contours.append(sequence)
        return all_contours


def required_contour_frames(list_image: list) -> List[int]:
    frames = set()
    for image_numbers in list_image:
        frames.update(chunk_required_frames(image_numbers))
    return sorted(frames)


def atomic_npz_save(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp_path, path)


def load_or_build_contour_npz_pack(
    image_folder: Path,
    list_image: list,
    articulators: List[str],
    pack_path: Path,
    rebuild: bool,
    file_workers: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    required_frames = required_contour_frames(list_image)
    if pack_path.exists() and not rebuild:
        with np.load(pack_path, allow_pickle=False) as pack:
            frame_numbers = pack["frame_numbers"]
            contours = pack["contours"]
        available = {int(frame) for frame in frame_numbers.tolist()}
        missing = [frame for frame in required_frames if frame not in available]
        if not missing:
            print(
                f"Loaded contour npz pack {pack_path} "
                f"with {len(frame_numbers)} frames for {len(required_frames)} requested frames",
                flush=True,
            )
            return frame_numbers, contours.astype(np.float32, copy=False)
        print(
            f"Contour npz pack {pack_path} is missing {len(missing)} requested frames; rebuilding",
            flush=True,
        )

    started_at = time.time()
    contours = np.zeros((len(required_frames), len(articulators), 100), dtype=np.float32)
    total_files = len(required_frames) * len(articulators)
    loaded_files = 0

    def load_one(item: Tuple[int, int, int, str]) -> Tuple[int, int, np.ndarray]:
        frame_idx, frame_number, art_idx, articulator = item
        contour_path = image_folder / f"{frame_number:04d}_{articulator}.npy"
        contour = np.load(contour_path).reshape(100).astype(np.float32, copy=False)
        return frame_idx, art_idx, contour

    load_items = [
        (frame_idx, frame_number, art_idx, articulator)
        for frame_idx, frame_number in enumerate(required_frames)
        for art_idx, articulator in enumerate(articulators)
    ]

    workers = max(1, int(file_workers))
    if workers == 1:
        loaded_iter = map(load_one, load_items)
    else:
        print(f"Using contour_file_workers={workers} for {pack_path}", flush=True)
        executor = ThreadPoolExecutor(max_workers=workers)
        loaded_iter = executor.map(load_one, load_items)

    try:
        for frame_idx, art_idx, contour in loaded_iter:
            contours[frame_idx, art_idx] = contour
            loaded_files += 1
            if loaded_files == 1 or loaded_files == total_files or loaded_files % 1000 == 0:
                elapsed = time.time() - started_at
                print(
                    f"Packing contour npz {pack_path}: {loaded_files}/{total_files} files "
                    f"in {elapsed:.1f}s",
                    flush=True,
                )
    finally:
        if workers != 1:
            executor.shutdown(wait=True)
    frame_numbers = np.asarray(required_frames, dtype=np.int32)
    atomic_npz_save(
        pack_path,
        frame_numbers=frame_numbers,
        contours=contours,
        articulators=np.asarray(articulators),
        source_folder=np.asarray(str(image_folder)),
    )
    print(
        f"Saved contour npz pack {pack_path} with {len(frame_numbers)} frames "
        f"in {time.time() - started_at:.1f}s",
        flush=True,
    )
    return frame_numbers, contours


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build dataset_cache_dir/*.pt for the ASD2 full single-task-5 config. "
            "Raw sessions are cached first, then assembled by top-level ASD2 "
            "bucket so normalization matches Corpus_contours while preprocessing "
            "can resume at session granularity."
        )
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config/train_config/asd2_full_single_task5_2gpu_grele_2epoch.yaml"),
        help="Training YAML to use as source of truth.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Output cache directory. Defaults to config['dataset_cache_dir'].",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLIT_FILES),
        choices=list(SPLIT_FILES),
        help="Splits to build and assemble.",
    )
    parser.add_argument("--max-workers", type=int, default=5, help="Parallel raw-session workers.")
    parser.add_argument("--rebuild-parts", action="store_true", help="Rebuild raw session part files.")
    parser.add_argument(
        "--contour-pack-format",
        choices=["none", "npz"],
        default="npz",
        help="Pack raw contour .npy files per session before building raw dataset parts.",
    )
    parser.add_argument(
        "--rebuild-contour-packs",
        action="store_true",
        help="Rebuild per-session contour packs even when they already exist.",
    )
    parser.add_argument(
        "--rebuild-assembled",
        action="store_true",
        help="Overwrite bucket and final split cache files even if they already exist.",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Build/resume only reusable per-session raw caches and skip final split assembly.",
    )
    parser.add_argument(
        "--skip-failed-sessions",
        action="store_true",
        help="Continue preprocessing when a session fails; missing sessions are skipped during assembly.",
    )
    parser.add_argument(
        "--failed-sessions-path",
        default=None,
        help="JSON path for failed/skipped sessions. Defaults to <cache-dir>/failed_sessions.json.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Contour chunk progress interval passed to the raw session loader.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config did not parse to a mapping: {path}")
    return config


def atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def build_raw_session(
    config: Dict[str, Any],
    split_key: str,
    bucket: str,
    session: str,
    part_path: str,
    work_dir: str,
    progress_every: int,
    contour_pack_path: str | None,
    rebuild_contour_pack: bool,
) -> Dict[str, Any]:
    started_at = time.time()
    part_file = Path(part_path)
    part_file.parent.mkdir(parents=True, exist_ok=True)
    worker_dir = Path(work_dir)
    worker_dir.mkdir(parents=True, exist_ok=True)
    (worker_dir / "normalization_values").mkdir(exist_ok=True)
    os.chdir(worker_dir)

    sub_config = copy.deepcopy(config)
    sub_config[split_key] = {bucket: [session]}
    sub_config["cache_dataset"] = False
    sub_config["load_labels_progress_every"] = progress_every
    sub_config.pop("assembled_dataset_cache_dir", None)

    print(f"[raw {split_key}/{bucket}/{session}] start -> {part_file}", flush=True)
    raw = RawContourSession(
        sub_config,
        split_key,
        rank=0,
        contour_pack_path=contour_pack_path,
        rebuild_contour_pack=rebuild_contour_pack,
    ).read_raw()
    payload = {
        "split": split_key,
        "dataset_type": dataset_type_for_sequence(config, bucket),
        "bucket": bucket,
        "session": session,
        "raw": raw,
        "num_chunks": len(raw["features"]),
        "built_seconds": time.time() - started_at,
    }
    atomic_torch_save(payload, part_file)
    print(
        f"[raw {split_key}/{bucket}/{session}] done: {payload['num_chunks']} chunks "
        f"in {payload['built_seconds']:.1f}s",
        flush=True,
    )
    return {
        "split": split_key,
        "dataset_type": dataset_type_for_sequence(config, bucket),
        "bucket": bucket,
        "session": session,
        "path": str(part_file),
        "num_chunks": payload["num_chunks"],
        "built_seconds": payload["built_seconds"],
    }


def iter_raw_jobs(
    config: Dict[str, Any],
    cache_dir: Path,
    splits: Iterable[str],
    progress_every: int,
) -> Iterable[Dict[str, Any]]:
    emitted = set()
    for split_key in splits:
        for bucket, sessions in config[split_key].items():
            for session in sessions:
                dataset_type = dataset_type_for_sequence(config, str(bucket))
                cache_key = (dataset_type, str(bucket), str(session))
                if cache_key in emitted:
                    continue
                emitted.add(cache_key)
                yield {
                    "config": config,
                    "split_key": split_key,
                    "bucket": str(bucket),
                    "session": str(session),
                    "part_path": str(raw_session_part_path(cache_dir, config, str(bucket), str(session))),
                    "work_dir": str(raw_session_work_dir(cache_dir, config, str(bucket), str(session))),
                    "progress_every": progress_every,
                    "contour_pack_path": (
                        str(contour_pack_path(cache_dir, config, str(bucket), str(session)))
                        if config.get("contour_pack_format", "npz") == "npz"
                        else None
                    ),
                    "rebuild_contour_pack": bool(config.get("rebuild_contour_packs", False)),
                }


def make_helper(config: Dict[str, Any], split_key: str, bucket: str) -> RawContourSession:
    sub_config = copy.deepcopy(config)
    first_session = next(iter(config[split_key][bucket]))
    sub_config[split_key] = {bucket: [first_session]}
    sub_config["cache_dataset"] = False
    return RawContourSession(sub_config, split_key, rank=0)


def assemble_bucket(config: Dict[str, Any], cache_dir: Path, split_key: str, bucket: str, rebuild: bool) -> Dict[str, Any]:
    bucket_path = cache_dir / "bucket_parts" / split_key / f"{bucket}.pt"
    if bucket_path.exists() and not rebuild:
        return torch.load(bucket_path, map_location="cpu")["state"]

    features = []
    contours = []
    frames = []
    phonemes = []
    length_datas = []
    for session in config[split_key][bucket]:
        raw_path = raw_session_part_path(cache_dir, config, str(bucket), str(session))
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw session cache: {raw_path}")
        raw = torch.load(raw_path, map_location="cpu")["raw"]
        features.extend(raw["features"])
        contours.extend(raw["contours"])
        frames.extend(raw["frames"])
        phonemes.extend(raw["phonemes"])
        length_datas.extend([int(x) for x in raw["length_datas"]])

    helper = make_helper(config, split_key, bucket)
    std_mfcc, mean_mfcc, std_contour, mean_contour, moving_average = helper.calculate_norm(
        str(bucket),
        features,
        contours,
    )

    std_contour_np = np.stack(std_contour)
    mean_contour_np = np.stack(mean_contour)
    std_items = [np.tile(std_contour_np, (1, 1, 1)) for _ in contours]
    mean_items = [np.tile(mean_contour_np, (1, 1, 1)) for _ in contours]
    std_for_norm = std_contour[None, :, :]

    norm_features = []
    norm_contours = []
    for idx, (feature, contour) in enumerate(zip(features, contours)):
        norm_features.append(helper.normalize_inputs(feature, std_mfcc, mean_mfcc))
        norm_contours.append(helper.normalize_labels(contour, std_for_norm, moving_average[idx]))

    tensors_inputs = [torch.tensor(seq, dtype=torch.float32) for seq in norm_features]
    tensors_outputs = [torch.tensor(seq, dtype=torch.float32) for seq in norm_contours]
    tensors_frames = [torch.tensor(seq, dtype=torch.float32) for seq in frames]
    tensors_phonemes = [torch.tensor(seq, dtype=torch.float32) for seq in phonemes]
    tensors_std = [torch.tensor(seq, dtype=torch.float32) for seq in std_items]
    tensors_mean = [torch.tensor(seq, dtype=torch.float32) for seq in mean_items]

    target_sequence_length = int(config["sequence_length"])
    padded_inputs = pad_time_dim(pad_sequence(tensors_inputs, batch_first=True), target_sequence_length)
    padded_labels = pad_time_dim(pad_sequence(tensors_outputs, batch_first=True), target_sequence_length)
    padded_frames = pad_time_dim(pad_sequence(tensors_frames, batch_first=True), target_sequence_length)
    padded_phonemes = pad_time_dim(pad_sequence(tensors_phonemes, batch_first=True), target_sequence_length)
    sample_count = len(tensors_outputs)
    state = {
        "features": padded_inputs,
        "labels": padded_labels.view(
            sample_count,
            target_sequence_length,
            len(config["classes"]),
            config["output_layer"],
        ),
        "frames": padded_frames,
        "phonemes": padded_phonemes,
        "std": pad_sequence(tensors_std, batch_first=True),
        "mean": pad_sequence(tensors_mean, batch_first=True),
        "mean_datas": recompute_mean_datas(
            padded_labels.view(
                sample_count,
                config["sequence_length"],
                len(config["classes"]),
                config["output_layer"],
            ),
            [int(x.shape[0]) for x in tensors_inputs],
        ),
        "length_datas": length_datas,
        "sequences_length": [int(x.shape[0]) for x in tensors_inputs],
    }
    payload = {
        "split": split_key,
        "bucket": bucket,
        "sessions": list(config[split_key][bucket]),
        "state": state,
        "num_samples": int(state["features"].shape[0]),
        "feature_shape": list(state["features"].shape),
        "label_shape": list(state["labels"].shape),
    }
    bucket_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload, bucket_path)
    print(
        f"[bucket {split_key}/{bucket}] assembled {payload['num_samples']} samples -> {bucket_path}",
        flush=True,
    )
    return state


def recompute_mean_datas(labels: torch.Tensor, lengths: List[int]) -> torch.Tensor:
    if labels.ndim != 4:
        raise ValueError(f"Expected labels [N,T,A,P], got shape {tuple(labels.shape)}")
    total = torch.zeros(labels.shape[2], labels.shape[3], dtype=torch.float32)
    total_frames = 0
    for idx, length in enumerate(lengths):
        length_int = int(length)
        if length_int <= 0:
            continue
        total += labels[idx, :length_int].sum(dim=0)
        total_frames += length_int
    if total_frames == 0:
        raise ValueError("Cannot recompute mean_datas with zero total frames")
    return total / total_frames


def pad_time_dim(tensor: torch.Tensor, target_length: int, dim: int = 1) -> torch.Tensor:
    current_length = int(tensor.shape[dim])
    if current_length == target_length:
        return tensor
    if current_length > target_length:
        index = [slice(None)] * tensor.ndim
        index[dim] = slice(0, target_length)
        return tensor[tuple(index)].contiguous()

    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_length - current_length
    padding = torch.zeros(*pad_shape, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=dim)


def filter_config_to_available_raw_sessions(
    config: Dict[str, Any],
    cache_dir: Path,
    splits: Iterable[str],
    skip_missing: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    filtered = copy.deepcopy(config)
    skipped = []
    for split_key in splits:
        filtered_split = {}
        for bucket, sessions in config[split_key].items():
            kept_sessions = []
            for session in sessions:
                raw_path = raw_session_part_path(cache_dir, config, str(bucket), str(session))
                if raw_path.exists():
                    kept_sessions.append(session)
                    continue
                skipped_item = {
                    "split": split_key,
                    "dataset_type": dataset_type_for_sequence(config, str(bucket)),
                    "bucket": str(bucket),
                    "session": str(session),
                    "path": str(raw_path),
                    "reason": "missing_raw_session_cache",
                }
                if not skip_missing:
                    raise FileNotFoundError(f"Missing raw session cache: {raw_path}")
                skipped.append(skipped_item)
            if kept_sessions:
                filtered_split[bucket] = kept_sessions
        filtered[split_key] = filtered_split
    return filtered, skipped


def assemble_split(config: Dict[str, Any], cache_dir: Path, split_key: str, rebuild: bool) -> Dict[str, Any]:
    output_path = cache_dir / SPLIT_FILES[split_key]
    if output_path.exists() and not rebuild:
        state = torch.load(output_path, map_location="cpu")
        return {
            "split": split_key,
            "path": str(output_path),
            "status": "existing",
            "num_samples": int(state["features"].shape[0]),
            "feature_shape": list(state["features"].shape),
            "label_shape": list(state["labels"].shape),
        }
    if not config.get(split_key):
        raise ValueError(f"No available sessions to assemble for split {split_key}")

    states = [
        assemble_bucket(config, cache_dir, split_key, str(bucket), rebuild)
        for bucket in config[split_key]
    ]
    assembled = {
        "features": torch.cat([state["features"].float() for state in states], dim=0),
        "labels": torch.cat([state["labels"].float() for state in states], dim=0),
        "frames": torch.cat([state["frames"].float() for state in states], dim=0),
        "phonemes": torch.cat([state["phonemes"].float() for state in states], dim=0),
        "std": torch.cat([state["std"].float() for state in states], dim=0),
        "mean": torch.cat([state["mean"].float() for state in states], dim=0),
        "length_datas": [],
        "sequences_length": [],
    }
    for state in states:
        assembled["length_datas"].extend([int(x) for x in state["length_datas"]])
        assembled["sequences_length"].extend([int(x) for x in state["sequences_length"]])
    assembled["mean_datas"] = recompute_mean_datas(
        assembled["labels"],
        assembled["sequences_length"],
    )

    atomic_torch_save(assembled, output_path)
    return {
        "split": split_key,
        "path": str(output_path),
        "status": "built",
        "num_samples": int(assembled["features"].shape[0]),
        "feature_shape": list(assembled["features"].shape),
        "label_shape": list(assembled["labels"].shape),
    }


def validate_cache(config: Dict[str, Any], cache_dir: Path, splits: Iterable[str]) -> Dict[str, Any]:
    validation = {}
    for split_key in splits:
        path = cache_dir / SPLIT_FILES[split_key]
        state = torch.load(path, map_location="cpu")
        features = state["features"]
        labels = state["labels"]
        if features.ndim != 3 or features.shape[2] != int(config["input_layer"]):
            raise ValueError(f"{split_key}: bad features shape {tuple(features.shape)}")
        expected_label_shape = (
            int(config["sequence_length"]),
            len(config["classes"]),
            int(config["output_layer"]),
        )
        if labels.shape[1:] != expected_label_shape:
            raise ValueError(f"{split_key}: bad labels shape {tuple(labels.shape)}")
        if len(state["sequences_length"]) != labels.shape[0]:
            raise ValueError(f"{split_key}: sequences_length count mismatch")
        validation[split_key] = {
            "features": list(features.shape),
            "labels": list(labels.shape),
            "std": list(state["std"].shape),
            "mean": list(state["mean"].shape),
            "num_samples": int(labels.shape[0]),
            "num_sessions": sum(len(v) for v in config[split_key].values()),
        }
    return validation


def write_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    config["contour_pack_format"] = args.contour_pack_format
    config["rebuild_contour_packs"] = args.rebuild_contour_packs
    cache_dir = Path(args.cache_dir or config["dataset_cache_dir"]).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "normalization_values").mkdir(parents=True, exist_ok=True)
    failed_sessions_path = Path(args.failed_sessions_path or cache_dir / "failed_sessions.json").resolve()

    started_at = time.time()
    raw_jobs = list(iter_raw_jobs(config, cache_dir, args.splits, args.progress_every))
    to_build = [
        job for job in raw_jobs
        if args.rebuild_parts or not Path(job["part_path"]).exists()
    ]

    print(f"config={config_path}", flush=True)
    print(f"cache_dir={cache_dir}", flush=True)
    print(f"splits={','.join(args.splits)}", flush=True)
    print(f"raw_session_jobs={len(raw_jobs)} to_build={len(to_build)}", flush=True)

    raw_results = []
    failed_sessions = []
    if to_build:
        workers = max(1, min(args.max_workers, len(to_build)))
        print(f"building raw session parts with max_workers={workers}", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_job = {
                executor.submit(build_raw_session, **job): job
                for job in to_build
            }
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    raw_results.append(future.result())
                except Exception as exc:
                    failure = {
                        "split": job["split_key"],
                        "dataset_type": dataset_type_for_sequence(config, job["bucket"]),
                        "bucket": job["bucket"],
                        "session": job["session"],
                        "part_path": job["part_path"],
                        "error": repr(exc),
                    }
                    failed_sessions.append(failure)
                    print(
                        f"[raw {job['split_key']}/{job['bucket']}/{job['session']}] failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if not args.skip_failed_sessions:
                        write_metadata(
                            failed_sessions_path,
                            {
                                "config": str(config_path),
                                "cache_dir": str(cache_dir),
                                "failed_sessions": failed_sessions,
                                "elapsed_seconds": time.time() - started_at,
                            },
                        )
                        raise
    else:
        print("all raw session caches already exist", flush=True)

    available_config, skipped_missing = filter_config_to_available_raw_sessions(
        config,
        cache_dir,
        args.splits,
        skip_missing=args.skip_failed_sessions,
    )
    all_skipped = failed_sessions + skipped_missing
    if all_skipped:
        write_metadata(
            failed_sessions_path,
            {
                "config": str(config_path),
                "cache_dir": str(cache_dir),
                "failed_or_skipped_sessions": all_skipped,
                "elapsed_seconds": time.time() - started_at,
            },
        )
        print(f"wrote failed/skipped session report: {failed_sessions_path}", flush=True)

    assemble_results = []
    validation = {}
    if args.raw_only:
        print("raw_only=true; skipping bucket/final split assembly", flush=True)
    else:
        assemble_results = [
            assemble_split(available_config, cache_dir, split_key, args.rebuild_assembled)
            for split_key in args.splits
        ]
        validation = validate_cache(available_config, cache_dir, args.splits)
    metadata = {
        "config": str(config_path),
        "cache_dir": str(cache_dir),
        "splits": args.splits,
        "raw_session_cache_layout": "raw_sessions/<dataset_type>/<bucket>/<session>.pt",
        "classes": config["classes"],
        "input_type": config["input_type"],
        "input_layer": config["input_layer"],
        "context_window": config["context_window"],
        "sequence_length": config["sequence_length"],
        "output_layer": config["output_layer"],
        "contour_pack_format": args.contour_pack_format,
        "raw_only": args.raw_only,
        "raw_results": sorted(raw_results, key=lambda x: (x["split"], x["bucket"], x["session"])),
        "failed_or_skipped_sessions": all_skipped,
        "assemble_results": assemble_results,
        "validation": validation,
        "elapsed_seconds": time.time() - started_at,
    }
    write_metadata(cache_dir / "metadata_asd2_full_single_task5.json", metadata)
    print(json.dumps(metadata["validation"], indent=2, sort_keys=True), flush=True)
    print(f"elapsed_seconds={metadata['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
