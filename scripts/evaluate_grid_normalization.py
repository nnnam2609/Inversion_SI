#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
GRID_ROOT = REPO_ROOT / "external" / "grid-transform"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(GRID_ROOT))

from grid_transform.annotation_projection import build_resize_affine, transform_reference_contours  # noqa: E402
from grid_transform.io import _load_roi_contours_from_zip  # noqa: E402
from grid_transform.io import load_frame_vtln  # noqa: E402
from grid_transform.transfer import build_two_step_transform  # noqa: E402
from grid_transform.transform_helpers import compute_metrics, extract_true_landmarks, map_landmarks  # noqa: E402
from grid_transform.vt import build_grid, visualize_grid  # noqa: E402
from src.utils.config_validation import load_yaml_config  # noqa: E402
from src.utils.video_rendering import MM_PER_PIXEL  # noqa: E402


RAW_ROOT = Path(
    "/srv/storage/talc@storage4.nancy/multispeech/corpus/speech_production/iadi/"
    "ArtSpeech_Database_1_raw"
)
BF_ROOT = WORKSPACE_ROOT / "bf" / "inference"
DEFAULT_VTLN_RELEASE_DIR = WORKSPACE_ROOT / "_downloads/grid-transform-vtln/vtln-data-v0.1.14/extracted/VTLN/data"
BF_TO_GRID_LABEL = {
    "upper-incisor": "incisior-hard-palate",
    "lower-incisor": "mandible-incisior",
    "lower-lip": "lower-lip",
    "pharynx": "pharynx",
    "soft-palate-midline": "soft-palate-midline",
    "tongue": "tongue",
    "upper-lip": "upper-lip",
}
PRIMARY_CLASSES = {
    "tongue",
    "pharynx",
    "soft-palate-midline",
    "upper-incisor",
    "lower-incisor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare unseen baseline vs P7->P2 grid-normalized predictions.")
    parser.add_argument("--seen-config", type=Path, required=True)
    parser.add_argument("--unseen-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--unseen-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--seen-predictions", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-anchor", default="1640_P7_S2_F0829")
    parser.add_argument("--target-anchor", default="1617_P2_S9_F1478")
    parser.add_argument("--vtln-dir", type=Path, default=DEFAULT_VTLN_RELEASE_DIR if DEFAULT_VTLN_RELEASE_DIR.is_dir() else GRID_ROOT / "VTLN" / "data")
    parser.add_argument("--source-speaker", default="P7")
    parser.add_argument("--source-session", default="S2")
    parser.add_argument("--source-frame", default="0829")
    parser.add_argument("--target-speaker-name", default="P2")
    parser.add_argument("--target-session-name", default="S9")
    parser.add_argument("--target-frame", default="1478")
    parser.add_argument("--anchor-source", choices=("vtln", "bf_vtln_c"), default="vtln")
    parser.add_argument("--target-speaker", type=int, default=2)
    parser.add_argument("--target-sessions", type=int, nargs="*", default=[1])
    parser.add_argument("--prediction-space-size", type=float, default=136.0)
    parser.add_argument("--anchor-space-size", type=float, default=480.0)
    parser.add_argument("--overlay-count", type=int, default=9)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return load_yaml_config(path)


def load_phonemes(config: dict[str, Any]) -> list[str]:
    with Path(config["phonemesdir"]).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def decode_phoneme(row: torch.Tensor, phonemes: list[str]) -> str:
    vector = row.detach().cpu().numpy()
    if vector.size == 0 or np.allclose(vector, 0):
        return "UNK"
    return str(phonemes[int(vector.argmax())])


def frame_token(value: float) -> str:
    rounded = int(round(value))
    if abs(value - rounded) < 1e-4:
        return f"{rounded:04d}"
    return f"{int(math.floor(value)):04d}p{int(round((value - math.floor(value)) * 10)):01d}"


def load_anchor_grid(anchor: str, vtln_dir: Path):
    image, contours = load_frame_vtln(anchor, vtln_dir, validate_triplet_bundle=True)
    return build_grid(image, contours, frame_number=0)


def dicom_filename_sort_key(name: str) -> tuple[int, str, int, str]:
    import re

    match = re.search(r"(\d{14})(\d+)$", name)
    if match is None:
        return (1, name, 0, name)
    return (0, match.group(1), int(match.group(2)), name)


def normalize_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = image[..., 1]
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, [1.0, 99.5])
    if hi <= lo:
        lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)
    return np.round(np.clip((image - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


def build_dicom_index(raw_session_dir: Path) -> dict[int, str]:
    names = [name for name in os.listdir(raw_session_dir) if not name.startswith(".")]
    names = sorted(names, key=dicom_filename_sort_key)
    return {index: name for index, name in enumerate(names, start=1)}


def load_asd1_mri(speaker: str, session: str, frame: str, target_shape: tuple[int, int]) -> np.ndarray:
    try:
        import pydicom
    except ImportError:
        return np.zeros(target_shape, dtype=np.uint8)

    raw_session_dir = RAW_ROOT / speaker / "DCM_2D" / session
    if not raw_session_dir.is_dir():
        return np.zeros(target_shape, dtype=np.uint8)
    index = build_dicom_index(raw_session_dir)
    name = index[int(frame)]
    image = normalize_image(pydicom.dcmread(str(raw_session_dir / name), force=True).pixel_array)
    if image.shape[:2] != target_shape:
        image = np.asarray(
            Image.fromarray(image).resize((target_shape[1], target_shape[0]), resample=Image.BILINEAR),
            dtype=np.uint8,
        )
    return image


def load_bf_grid_contours(speaker: str, session: str, frame: str, bf_root: Path = BF_ROOT) -> dict[str, np.ndarray]:
    contours_dir = bf_root / speaker / session / "contours"
    contours: dict[str, np.ndarray] = {}
    for bf_label, grid_label in BF_TO_GRID_LABEL.items():
        path = contours_dir / f"{frame}_{bf_label}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing BF contour for grid anchor: {path}")
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected contour shaped (N,2), got {arr.shape}: {path}")
        contours[grid_label] = arr.astype(float)
    return contours


def load_vtln_c_contours(reference: str, vtln_dir: Path, target_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    image_path = vtln_dir / f"{reference}.png"
    zip_path = vtln_dir / f"{reference}.zip"
    image = np.asarray(Image.open(image_path))
    contours_480 = _load_roi_contours_from_zip(zip_path, image_name=reference)
    c_contours = {label: contours_480[label] for label in ("c1", "c2", "c3", "c4", "c5", "c6")}
    affine = build_resize_affine(image.shape[:2], target_shape)
    return transform_reference_contours(c_contours, affine)


def load_bf_vtln_c_anchor_grid(
    *,
    speaker: str,
    session: str,
    frame: str,
    reference: str,
    vtln_dir: Path,
    output_dir: Path,
    output_name: str,
):
    image = load_asd1_mri(speaker, session, frame, (136, 136))
    contours = load_bf_grid_contours(speaker, session, frame)
    contours.update(load_vtln_c_contours(reference, vtln_dir, image.shape[:2]))
    grid = build_grid(image, contours, frame_number=int(frame))
    fig = visualize_grid(grid, figsize=(9, 9), show_contours=True, show_landmarks=True, show_labels=True)
    path = output_dir / "grid_overlays" / output_name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return grid, path, contours


def load_source_target_grids(args: argparse.Namespace, output_dir: Path):
    if args.anchor_source == "vtln":
        source_grid = load_anchor_grid(args.source_anchor, args.vtln_dir)
        target_grid = load_anchor_grid(args.target_anchor, args.vtln_dir)
        return source_grid, target_grid, None, None, {}, {}
    source_grid, source_path, source_contours = load_bf_vtln_c_anchor_grid(
        speaker=args.source_speaker,
        session=args.source_session,
        frame=args.source_frame,
        reference=args.source_anchor,
        vtln_dir=args.vtln_dir,
        output_dir=output_dir,
        output_name=f"{args.source_speaker}_{args.source_session}_{args.source_frame}_source_grid_bf_vtlnc.png",
    )
    target_grid, target_path, target_contours = load_bf_vtln_c_anchor_grid(
        speaker=args.target_speaker_name,
        session=args.target_session_name,
        frame=args.target_frame,
        reference=args.target_anchor,
        vtln_dir=args.vtln_dir,
        output_dir=output_dir,
        output_name=f"{args.target_speaker_name}_{args.target_session_name}_{args.target_frame}_target_grid_bf_vtlnc.png",
    )
    return source_grid, target_grid, str(source_path), str(target_path), source_contours, target_contours


def build_scaled_mapping(source_grid, target_grid, prediction_space_size: float, anchor_space_size: float):
    transform = build_two_step_transform(source_grid, target_grid)
    scale = anchor_space_size / prediction_space_size

    def mapping(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        original_shape = pts.shape
        pts2 = pts.reshape(-1, 2)
        mapped = transform["apply_two_step"](pts2 * scale) / scale
        return np.asarray(mapped, dtype=np.float32).reshape(original_shape)

    return transform, mapping


def aggregate_payload(
    payload_path: Path,
    config: dict[str, Any],
    *,
    speaker: int | None = None,
    session: int | None = None,
    sessions: set[int] | None = None,
) -> list[dict[str, Any]]:
    payload = torch.load(payload_path, map_location="cpu")
    if "labels_raw" not in payload:
        raise KeyError(f"Prediction payload must contain labels_raw for metric comparison: {payload_path}")
    phonemes = load_phonemes(config)
    classes = list(config["classes"])
    predicted = payload["predicted_raw"].float()
    labels = payload["labels_raw"].float()
    frames = payload["frames"]
    phoneme_vectors = payload["phonemes"]
    lengths = payload["lengths"]
    accum: dict[tuple[int, int, str], dict[str, Any]] = {}

    for seq_idx in range(predicted.shape[0]):
        length = int(lengths[seq_idx].item())
        for offset in range(length):
            frame = frames[seq_idx, offset]
            spk = int(round(float(frame[0])))
            ses = int(round(float(frame[1])))
            if speaker is not None and spk != speaker:
                continue
            if session is not None and ses != session:
                continue
            if sessions is not None and ses not in sessions:
                continue
            token = frame_token(float(frame[2]))
            key = (spk, ses, token)
            item = accum.setdefault(
                key,
                {
                    "speaker": spk,
                    "session": ses,
                    "frame": token,
                    "phonemes": [],
                    "predicted": [],
                    "labels": [],
                },
            )
            item["phonemes"].append(decode_phoneme(phoneme_vectors[seq_idx, offset, 0], phonemes))
            item["predicted"].append(predicted[seq_idx, offset].numpy())
            item["labels"].append(labels[seq_idx, offset].numpy())

    rows = []
    for item in accum.values():
        phoneme_counts = defaultdict(int)
        for phoneme in item["phonemes"]:
            phoneme_counts[phoneme] += 1
        rows.append(
            {
                "speaker": item["speaker"],
                "session": item["session"],
                "frame": item["frame"],
                "phoneme": max(phoneme_counts, key=phoneme_counts.get),
                "predicted": np.mean(np.stack(item["predicted"], axis=0), axis=0),
                "labels": np.mean(np.stack(item["labels"], axis=0), axis=0),
                "classes": classes,
            }
        )
    return sorted(rows, key=lambda row: (row["speaker"], row["session"], row["frame"]))


def aggregate_payloads(
    payload_paths: list[Path],
    config: dict[str, Any],
    *,
    speaker: int | None = None,
    sessions: set[int] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int, str]] = set()
    for path in payload_paths:
        for row in aggregate_payload(path, config, speaker=speaker, sessions=sessions):
            key = (row["speaker"], row["session"], row["frame"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rows.append(row)
    return sorted(rows, key=lambda row: (row["speaker"], row["session"], row["frame"]))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def metric_rows(rows: list[dict[str, Any]], prediction_key: str, class_names: list[str]) -> list[dict[str, Any]]:
    output = []
    primary_indices = [idx for idx, name in enumerate(class_names) if name in PRIMARY_CLASSES]
    for row in rows:
        pred = row[prediction_key]
        labels = row["labels"]
        all_rmse = rmse(pred, labels)
        primary_rmse = rmse(pred[primary_indices], labels[primary_indices]) if primary_indices else float("nan")
        output.append(
            {
                "speaker": row["speaker"],
                "session": row["session"],
                "frame": row["frame"],
                "phoneme": row["phoneme"],
                "overall_rmse_px": all_rmse,
                "overall_rmse_mm": all_rmse * MM_PER_PIXEL,
                "primary_rmse_px": primary_rmse,
                "primary_rmse_mm": primary_rmse * MM_PER_PIXEL,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_or_nan(values: list[float]) -> float:
    values = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return float(np.mean(values)) if values else float("nan")


def comparison_tables(rows: list[dict[str, Any]], class_names: list[str]):
    by_phoneme = defaultdict(lambda: {"baseline": [], "gridnorm": [], "primary_b": [], "primary_g": [], "n": 0})
    by_class = defaultdict(lambda: {"baseline": [], "gridnorm": [], "n": 0})
    primary_indices = [idx for idx, name in enumerate(class_names) if name in PRIMARY_CLASSES]
    for row in rows:
        b = row["predicted"]
        g = row["gridnorm_predicted"]
        y = row["labels"]
        phoneme = row["phoneme"]
        by_phoneme[phoneme]["baseline"].append(rmse(b, y))
        by_phoneme[phoneme]["gridnorm"].append(rmse(g, y))
        by_phoneme[phoneme]["primary_b"].append(rmse(b[primary_indices], y[primary_indices]))
        by_phoneme[phoneme]["primary_g"].append(rmse(g[primary_indices], y[primary_indices]))
        by_phoneme[phoneme]["n"] += 1
        for idx, name in enumerate(class_names):
            by_class[name]["baseline"].append(rmse(b[idx], y[idx]))
            by_class[name]["gridnorm"].append(rmse(g[idx], y[idx]))
            by_class[name]["n"] += 1

    phoneme_rows = []
    for phoneme, values in sorted(by_phoneme.items()):
        b = mean_or_nan(values["baseline"])
        g = mean_or_nan(values["gridnorm"])
        bp = mean_or_nan(values["primary_b"])
        gp = mean_or_nan(values["primary_g"])
        phoneme_rows.append(
            {
                "phoneme": phoneme,
                "num_frames": values["n"],
                "baseline_rmse_px": b,
                "gridnorm_rmse_px": g,
                "delta_px": g - b,
                "baseline_primary_rmse_px": bp,
                "gridnorm_primary_rmse_px": gp,
                "primary_delta_px": gp - bp,
            }
        )

    class_rows = []
    for name, values in sorted(by_class.items()):
        b = mean_or_nan(values["baseline"])
        g = mean_or_nan(values["gridnorm"])
        class_rows.append(
            {
                "class": name,
                "num_frames": values["n"],
                "is_primary": name in PRIMARY_CLASSES,
                "baseline_rmse_px": b,
                "gridnorm_rmse_px": g,
                "delta_px": g - b,
                "baseline_rmse_mm": b * MM_PER_PIXEL,
                "gridnorm_rmse_mm": g * MM_PER_PIXEL,
                "delta_mm": (g - b) * MM_PER_PIXEL,
            }
        )
    return phoneme_rows, class_rows


def combined_frame_rows(rows: list[dict[str, Any]], class_names: list[str]) -> list[dict[str, Any]]:
    primary_indices = [idx for idx, name in enumerate(class_names) if name in PRIMARY_CLASSES]
    output = []
    for row in rows:
        baseline = rmse(row["predicted"], row["labels"])
        gridnorm = rmse(row["gridnorm_predicted"], row["labels"])
        baseline_primary = rmse(row["predicted"][primary_indices], row["labels"][primary_indices])
        gridnorm_primary = rmse(row["gridnorm_predicted"][primary_indices], row["labels"][primary_indices])
        output.append(
            {
                "speaker": row["speaker"],
                "session": row["session"],
                "frame": row["frame"],
                "phoneme": row["phoneme"],
                "baseline_rmse_px": baseline,
                "gridnorm_rmse_px": gridnorm,
                "delta_px": gridnorm - baseline,
                "baseline_rmse_mm": baseline * MM_PER_PIXEL,
                "gridnorm_rmse_mm": gridnorm * MM_PER_PIXEL,
                "delta_mm": (gridnorm - baseline) * MM_PER_PIXEL,
                "baseline_primary_rmse_px": baseline_primary,
                "gridnorm_primary_rmse_px": gridnorm_primary,
                "primary_delta_px": gridnorm_primary - baseline_primary,
                "baseline_primary_rmse_mm": baseline_primary * MM_PER_PIXEL,
                "gridnorm_primary_rmse_mm": gridnorm_primary * MM_PER_PIXEL,
                "primary_delta_mm": (gridnorm_primary - baseline_primary) * MM_PER_PIXEL,
            }
        )
    return output


def session_metric_rows(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        grouped[(int(row["speaker"]), int(row["session"]))].append(row)
    output = []
    for (speaker, session), rows in sorted(grouped.items()):
        baseline = mean_or_nan([row["baseline_rmse_px"] for row in rows])
        gridnorm = mean_or_nan([row["gridnorm_rmse_px"] for row in rows])
        baseline_primary = mean_or_nan([row["baseline_primary_rmse_px"] for row in rows])
        gridnorm_primary = mean_or_nan([row["gridnorm_primary_rmse_px"] for row in rows])
        output.append(
            {
                "speaker": speaker,
                "session": session,
                "num_frames": len(rows),
                "baseline_rmse_px": baseline,
                "gridnorm_rmse_px": gridnorm,
                "delta_px": gridnorm - baseline,
                "baseline_rmse_mm": baseline * MM_PER_PIXEL,
                "gridnorm_rmse_mm": gridnorm * MM_PER_PIXEL,
                "delta_mm": (gridnorm - baseline) * MM_PER_PIXEL,
                "baseline_primary_rmse_px": baseline_primary,
                "gridnorm_primary_rmse_px": gridnorm_primary,
                "primary_delta_px": gridnorm_primary - baseline_primary,
                "baseline_primary_rmse_mm": baseline_primary * MM_PER_PIXEL,
                "gridnorm_primary_rmse_mm": gridnorm_primary * MM_PER_PIXEL,
                "primary_delta_mm": (gridnorm_primary - baseline_primary) * MM_PER_PIXEL,
            }
        )
    return output


def transformed_coord_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"num_values": 0}
    points = np.concatenate([row["gridnorm_predicted"].reshape(-1, 2) for row in rows], axis=0)
    finite = np.isfinite(points).all(axis=1)
    out = finite & ((points[:, 0] < 0) | (points[:, 0] > 136) | (points[:, 1] < 0) | (points[:, 1] > 136))
    return {
        "num_points": int(points.shape[0]),
        "num_nonfinite_points": int((~finite).sum()),
        "num_out_of_0_136_points": int(out.sum()),
        "fraction_out_of_0_136_points": float(out.mean()) if len(out) else 0.0,
        "min_xy": [float(np.nanmin(points[:, 0])), float(np.nanmin(points[:, 1]))],
        "max_xy": [float(np.nanmax(points[:, 0])), float(np.nanmax(points[:, 1]))],
    }


def plot_overlay(path: Path, row: dict[str, Any], class_names: list[str], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 136)
    ax.set_ylim(136, 0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    primary = [idx for idx, name in enumerate(class_names) if name in PRIMARY_CLASSES]
    for idx in primary:
        name = class_names[idx]
        label = row["labels"][idx].reshape(50, 2)
        baseline = row["predicted"][idx].reshape(50, 2)
        gridnorm = row["gridnorm_predicted"][idx].reshape(50, 2)
        ax.plot(label[:, 0], label[:, 1], color="black", linewidth=1.4, alpha=0.75)
        ax.plot(baseline[:, 0], baseline[:, 1], color="#d62728", linewidth=1.0, alpha=0.75)
        ax.plot(gridnorm[:, 0], gridnorm[:, 1], color="#2ca02c", linewidth=1.0, alpha=0.75)
        anchor = label[len(label) // 2]
        ax.text(anchor[0], anchor[1], name, fontsize=6, color="black")
    ax.plot([], [], color="black", label="P2 label")
    ax.plot([], [], color="#d62728", label="baseline")
    ax.plot([], [], color="#2ca02c", label="gridnorm")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_overlays(output_dir: Path, rows: list[dict[str, Any]], class_names: list[str], count: int) -> list[str]:
    if not rows or count <= 0:
        return []
    scored = []
    primary_indices = [idx for idx, name in enumerate(class_names) if name in PRIMARY_CLASSES]
    for row in rows:
        b = rmse(row["predicted"][primary_indices], row["labels"][primary_indices])
        g = rmse(row["gridnorm_predicted"][primary_indices], row["labels"][primary_indices])
        scored.append((g - b, row))
    scored.sort(key=lambda item: item[0])
    picks = []
    indices = [0, len(scored) // 2, len(scored) - 1]
    extra_needed = max(0, count - len(indices))
    if extra_needed:
        step = max(1, len(scored) // max(1, extra_needed))
        indices.extend(range(0, len(scored), step))
    seen = set()
    paths = []
    for idx in indices:
        idx = min(max(idx, 0), len(scored) - 1)
        delta, row = scored[idx]
        key = (row["speaker"], row["session"], row["frame"])
        if key in seen:
            continue
        seen.add(key)
        path = output_dir / "sample_overlays" / f"p{row['speaker']}_s{row['session']}_{row['frame']}_delta_{delta:+.3f}.png"
        title = f"P{row['speaker']}/S{row['session']} frame {row['frame']} phoneme={row['phoneme']} primary_delta={delta:+.3f}px"
        plot_overlay(path, row, class_names, title)
        paths.append(str(path))
        if len(paths) >= count:
            break
    return paths


def write_session_overlays(output_dir: Path, rows: list[dict[str, Any]], class_names: list[str]) -> list[str]:
    grouped: dict[tuple[int, int], list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    primary_indices = [idx for idx, name in enumerate(class_names) if name in PRIMARY_CLASSES]
    for row in rows:
        b = rmse(row["predicted"][primary_indices], row["labels"][primary_indices])
        g = rmse(row["gridnorm_predicted"][primary_indices], row["labels"][primary_indices])
        grouped[(int(row["speaker"]), int(row["session"]))].append((g - b, row))

    paths = []
    for (speaker, session), scored in sorted(grouped.items()):
        scored.sort(key=lambda item: item[0])
        pick_indices = [0, len(scored) // 2, len(scored) - 1]
        pick_names = ["best", "median", "worst"]
        seen = set()
        for pick_name, idx in zip(pick_names, pick_indices):
            delta, row = scored[idx]
            key = (row["frame"], pick_name)
            if key in seen:
                continue
            seen.add(key)
            path = output_dir / "session_overlays" / f"p{speaker}_s{session}_{pick_name}_{row['frame']}_primary_delta_{delta:+.3f}.png"
            title = f"P{speaker}/S{session} {pick_name} frame {row['frame']} phoneme={row['phoneme']} primary_delta={delta:+.3f}px"
            plot_overlay(path, row, class_names, title)
            paths.append(str(path))
    return paths


def write_delta_histograms(output_dir: Path, frame_rows: list[dict[str, Any]]) -> list[str]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in frame_rows:
        grouped[(int(row["speaker"]), int(row["session"]))].append(float(row["primary_delta_px"]))

    paths = []
    for (speaker, session), deltas in sorted(grouped.items()):
        path = output_dir / "delta_histograms" / f"p{speaker}_s{session}_primary_delta_hist.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
        ax.hist(deltas, bins=40, color="#4c78a8", alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=1.2)
        ax.set_title(f"P{speaker}/S{session} primary RMSE delta (gridnorm - baseline)")
        ax.set_xlabel("delta px")
        ax.set_ylabel("frames")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_config = load_yaml(args.seen_config)
    unseen_config = load_yaml(args.unseen_config)
    class_names = list(unseen_config["classes"])

    source_grid, target_grid, source_grid_png, target_grid_png, source_contours, target_contours = load_source_target_grids(args, output_dir)
    transform, mapping = build_scaled_mapping(source_grid, target_grid, args.prediction_space_size, args.anchor_space_size)
    source_landmarks = extract_true_landmarks(source_grid)
    target_landmarks = extract_true_landmarks(target_grid)
    mapped_landmarks = map_landmarks(transform["apply_two_step"], source_landmarks)
    grid_metrics = compute_metrics(mapped_landmarks, target_landmarks)

    target_sessions = set(args.target_sessions) if args.target_sessions else None
    unseen_rows = aggregate_payloads(args.unseen_predictions, unseen_config, speaker=args.target_speaker, sessions=target_sessions)
    for row in unseen_rows:
        row["gridnorm_predicted"] = mapping(row["predicted"].reshape(len(class_names), 50, 2)).reshape(len(class_names), 100)

    seen_metric_rows = []
    for path in args.seen_predictions:
        for row in aggregate_payload(path, seen_config):
            pred = row["predicted"]
            labels = row["labels"]
            primary_indices = [idx for idx, name in enumerate(class_names) if name in PRIMARY_CLASSES]
            seen_metric_rows.append(
                {
                    "speaker": row["speaker"],
                    "session": row["session"],
                    "frame": row["frame"],
                    "phoneme": row["phoneme"],
                    "overall_rmse_px": rmse(pred, labels),
                    "overall_rmse_mm": rmse(pred, labels) * MM_PER_PIXEL,
                    "primary_rmse_px": rmse(pred[primary_indices], labels[primary_indices]),
                    "primary_rmse_mm": rmse(pred[primary_indices], labels[primary_indices]) * MM_PER_PIXEL,
                }
            )

    baseline_rows = metric_rows(unseen_rows, "predicted", class_names)
    gridnorm_rows = metric_rows(unseen_rows, "gridnorm_predicted", class_names)
    frame_rows = combined_frame_rows(unseen_rows, class_names)
    session_rows = session_metric_rows(frame_rows)
    phoneme_rows, class_rows = comparison_tables(unseen_rows, class_names)
    overlay_paths = write_overlays(output_dir, unseen_rows, class_names, args.overlay_count)
    session_overlay_paths = write_session_overlays(output_dir, unseen_rows, class_names)
    histogram_paths = write_delta_histograms(output_dir, frame_rows)

    metric_fields = ["speaker", "session", "frame", "phoneme", "overall_rmse_px", "overall_rmse_mm", "primary_rmse_px", "primary_rmse_mm"]
    write_csv(output_dir / "seen_test_metrics.csv", seen_metric_rows, metric_fields)
    write_csv(output_dir / "unseen_baseline_metrics.csv", baseline_rows, metric_fields)
    write_csv(output_dir / "unseen_gridnorm_metrics.csv", gridnorm_rows, metric_fields)
    frame_fields = [
        "speaker", "session", "frame", "phoneme",
        "baseline_rmse_px", "gridnorm_rmse_px", "delta_px",
        "baseline_rmse_mm", "gridnorm_rmse_mm", "delta_mm",
        "baseline_primary_rmse_px", "gridnorm_primary_rmse_px", "primary_delta_px",
        "baseline_primary_rmse_mm", "gridnorm_primary_rmse_mm", "primary_delta_mm",
    ]
    session_fields = [
        "speaker", "session", "num_frames",
        "baseline_rmse_px", "gridnorm_rmse_px", "delta_px",
        "baseline_rmse_mm", "gridnorm_rmse_mm", "delta_mm",
        "baseline_primary_rmse_px", "gridnorm_primary_rmse_px", "primary_delta_px",
        "baseline_primary_rmse_mm", "gridnorm_primary_rmse_mm", "primary_delta_mm",
    ]
    write_csv(output_dir / "frame_metrics.csv", frame_rows, frame_fields)
    write_csv(output_dir / "session_metrics.csv", session_rows, session_fields)
    write_csv(
        output_dir / "per_phoneme_comparison.csv",
        phoneme_rows,
        ["phoneme", "num_frames", "baseline_rmse_px", "gridnorm_rmse_px", "delta_px", "baseline_primary_rmse_px", "gridnorm_primary_rmse_px", "primary_delta_px"],
    )
    write_csv(
        output_dir / "per_articulator_comparison.csv",
        class_rows,
        ["class", "num_frames", "is_primary", "baseline_rmse_px", "gridnorm_rmse_px", "delta_px", "baseline_rmse_mm", "gridnorm_rmse_mm", "delta_mm"],
    )

    baseline_overall = mean_or_nan([row["overall_rmse_px"] for row in baseline_rows])
    gridnorm_overall = mean_or_nan([row["overall_rmse_px"] for row in gridnorm_rows])
    baseline_primary = mean_or_nan([row["primary_rmse_px"] for row in baseline_rows])
    gridnorm_primary = mean_or_nan([row["primary_rmse_px"] for row in gridnorm_rows])

    grid_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "anchor_source": args.anchor_source,
        "source_anchor": args.source_anchor,
        "target_anchor": args.target_anchor,
        "source_reference": {
            "speaker": args.source_speaker,
            "session": args.source_session,
            "frame": args.source_frame,
            "grid_png": source_grid_png,
            "contour_point_counts": {name: int(len(points)) for name, points in source_contours.items()},
        },
        "target_reference": {
            "speaker": args.target_speaker_name,
            "session": args.target_session_name,
            "frame": args.target_frame,
            "grid_png": target_grid_png,
            "contour_point_counts": {name: int(len(points)) for name, points in target_contours.items()},
        },
        "vtln_dir": str(args.vtln_dir),
        "bf_root": str(BF_ROOT),
        "prediction_space_size": args.prediction_space_size,
        "anchor_space_size": args.anchor_space_size,
        "step1_labels": transform["step1_labels"],
        "step2_labels": transform["step2_labels"],
        "grid_metrics_anchor_space_px": grid_metrics,
        "transformed_coordinate_summary": transformed_coord_summary(unseen_rows),
        "uses_c1_to_c6_static_anchor_landmarks": True,
        "c1_to_c6_predicted_by_inversion": False,
        "one_image_per_speaker_policy": "one BF+VTLN-C anchor image per speaker is used only for static morphology normalization",
    }
    (output_dir / "grid_transform_summary.json").write_text(json.dumps(grid_summary, indent=2, sort_keys=True), encoding="utf-8")

    train_summary = {
        "checkpoint": str(args.checkpoint),
        "seen_config": str(args.seen_config),
        "unseen_config": str(args.unseen_config),
        "seen_prediction_payloads": [str(path) for path in args.seen_predictions],
        "unseen_prediction_payloads": [str(path) for path in args.unseen_predictions],
        "target_speaker": args.target_speaker,
        "target_sessions": sorted(target_sessions) if target_sessions else None,
        "seen_num_frames": len(seen_metric_rows),
        "unseen_num_frames": len(unseen_rows),
        "seen_overall_rmse_px": mean_or_nan([row["overall_rmse_px"] for row in seen_metric_rows]),
        "seen_primary_rmse_px": mean_or_nan([row["primary_rmse_px"] for row in seen_metric_rows]),
    }
    with (output_dir / "seen_train_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# P7 Seen-Speaker Training Summary\n\n")
        for key, value in train_summary.items():
            handle.write(f"- {key}: `{value}`\n")

    accepted = (gridnorm_overall < baseline_overall) or (gridnorm_primary < baseline_primary)
    with (output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# P7 Reference BF+VTLN-C to P2 S1-S3 Grid Normalization\n\n")
        handle.write("## Policy Notes\n\n")
        handle.write("- `C1..C6` are used by grid-transform as static speaker anatomy anchors.\n")
        handle.write("- `C1..C6` are not predicted by inversion.\n")
        handle.write("- Reference grids use BF inference dynamic contours plus VTLN `C1..C6`; P2 labels are used only for RMSE.\n")
        handle.write("- One image per speaker is used only for morphology normalization, not phoneme-specific dynamic alignment.\n\n")
        handle.write("## Metrics\n\n")
        handle.write(f"- unseen baseline overall RMSE: `{baseline_overall:.6f}` px / `{baseline_overall * MM_PER_PIXEL:.6f}` mm\n")
        handle.write(f"- unseen gridnorm overall RMSE: `{gridnorm_overall:.6f}` px / `{gridnorm_overall * MM_PER_PIXEL:.6f}` mm\n")
        handle.write(f"- unseen overall delta: `{gridnorm_overall - baseline_overall:.6f}` px / `{(gridnorm_overall - baseline_overall) * MM_PER_PIXEL:.6f}` mm\n")
        handle.write(f"- unseen baseline primary RMSE: `{baseline_primary:.6f}` px / `{baseline_primary * MM_PER_PIXEL:.6f}` mm\n")
        handle.write(f"- unseen gridnorm primary RMSE: `{gridnorm_primary:.6f}` px / `{gridnorm_primary * MM_PER_PIXEL:.6f}` mm\n")
        handle.write(f"- unseen primary delta: `{gridnorm_primary - baseline_primary:.6f}` px / `{(gridnorm_primary - baseline_primary) * MM_PER_PIXEL:.6f}` mm\n")
        handle.write(f"- acceptance met: `{accepted}`\n\n")
        handle.write("## Session Metrics\n\n")
        handle.write("| Session | Frames | Baseline overall px | Gridnorm overall px | Delta px | Baseline primary px | Gridnorm primary px | Primary delta px |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in session_rows:
            handle.write(
                f"| P{row['speaker']}/S{row['session']} | {row['num_frames']} | "
                f"{row['baseline_rmse_px']:.6f} | {row['gridnorm_rmse_px']:.6f} | {row['delta_px']:.6f} | "
                f"{row['baseline_primary_rmse_px']:.6f} | {row['gridnorm_primary_rmse_px']:.6f} | {row['primary_delta_px']:.6f} |\n"
            )
        handle.write("\n")
        handle.write("## Files\n\n")
        for path in (
            "seen_train_summary.md",
            "seen_test_metrics.csv",
            "session_metrics.csv",
            "frame_metrics.csv",
            "unseen_baseline_metrics.csv",
            "unseen_gridnorm_metrics.csv",
            "per_phoneme_comparison.csv",
            "per_articulator_comparison.csv",
            "grid_transform_summary.json",
        ):
            handle.write(f"- `{path}`\n")
        for path in (source_grid_png, target_grid_png):
            if path:
                handle.write(f"- `{Path(path).relative_to(output_dir)}`\n")
        for path in overlay_paths:
            handle.write(f"- `{Path(path).relative_to(output_dir)}`\n")
        for path in session_overlay_paths:
            handle.write(f"- `{Path(path).relative_to(output_dir)}`\n")
        for path in histogram_paths:
            handle.write(f"- `{Path(path).relative_to(output_dir)}`\n")

    print(json.dumps({
        "output_dir": str(output_dir),
        "baseline_overall_rmse_px": baseline_overall,
        "gridnorm_overall_rmse_px": gridnorm_overall,
        "baseline_primary_rmse_px": baseline_primary,
        "gridnorm_primary_rmse_px": gridnorm_primary,
        "accepted": accepted,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
