#!/usr/bin/env python3
"""Prepare exact ASD2 full-split dataset cache for single-task-5 training."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

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


class RawContourSession(Corpus_contours):
    """Use Corpus helpers while saving raw, pre-normalization session chunks."""

    def __init__(self, config: Dict[str, Any], sequences: str, rank: int):
        Corpus.__init__(self, config, sequences, rank)

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

                tg = textgrid.TextGrid.fromFile(tg_session)
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
        "--rebuild-assembled",
        action="store_true",
        help="Overwrite bucket and final split cache files even if they already exist.",
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
    raw = RawContourSession(sub_config, split_key, rank=0).read_raw()
    payload = {
        "split": split_key,
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
    for split_key in splits:
        for bucket, sessions in config[split_key].items():
            for session in sessions:
                yield {
                    "config": config,
                    "split_key": split_key,
                    "bucket": str(bucket),
                    "session": str(session),
                    "part_path": str(cache_dir / "raw_parts" / split_key / str(bucket) / f"{session}.pt"),
                    "work_dir": str(cache_dir / "work" / "raw" / split_key / str(bucket) / str(session)),
                    "progress_every": progress_every,
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
        raw_path = cache_dir / "raw_parts" / split_key / str(bucket) / f"{session}.pt"
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

    padded_labels = pad_sequence(tensors_outputs, batch_first=True)
    sample_count = len(tensors_outputs)
    state = {
        "features": pad_sequence(tensors_inputs, batch_first=True),
        "labels": padded_labels.view(
            sample_count,
            config["sequence_length"],
            len(config["classes"]),
            config["output_layer"],
        ),
        "frames": pad_sequence(tensors_frames, batch_first=True),
        "phonemes": pad_sequence(tensors_phonemes, batch_first=True),
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
    cache_dir = Path(args.cache_dir or config["dataset_cache_dir"]).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

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
                    print(
                        f"[raw {job['split_key']}/{job['bucket']}/{job['session']}] failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise
    else:
        print("all raw session caches already exist", flush=True)

    assemble_results = [
        assemble_split(config, cache_dir, split_key, args.rebuild_assembled)
        for split_key in args.splits
    ]
    validation = validate_cache(config, cache_dir, args.splits)
    metadata = {
        "config": str(config_path),
        "cache_dir": str(cache_dir),
        "splits": args.splits,
        "classes": config["classes"],
        "input_type": config["input_type"],
        "input_layer": config["input_layer"],
        "context_window": config["context_window"],
        "sequence_length": config["sequence_length"],
        "output_layer": config["output_layer"],
        "raw_results": sorted(raw_results, key=lambda x: (x["split"], x["bucket"], x["session"])),
        "assemble_results": assemble_results,
        "validation": validation,
        "elapsed_seconds": time.time() - started_at,
    }
    write_metadata(cache_dir / "metadata_asd2_full_single_task5.json", metadata)
    print(json.dumps(metadata["validation"], indent=2, sort_keys=True), flush=True)
    print(f"elapsed_seconds={metadata['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
