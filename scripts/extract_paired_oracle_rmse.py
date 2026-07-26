#!/usr/bin/env python3
"""Extract aligned integer-row RMSE for global and Exact-Sofiane oracle models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-worktree", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--global-cache", type=Path, required=True)
    parser.add_argument("--global-checkpoint", type=Path, required=True)
    parser.add_argument("--moving-cache", type=Path, required=True)
    parser.add_argument("--moving-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mm-per-pixel", type=float, default=1.62)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_evaluator(worktree: Path):
    script = worktree / "scripts" / "evaluate_sofiane_exact_vs_global.py"
    spec = importlib.util.spec_from_file_location("sofiane_evaluator", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect(
    *,
    evaluator,
    config: dict[str, Any],
    state: dict[str, Any],
    checkpoint: Path,
    center_key: str,
    device: torch.device,
    batch_size: int,
    mm_per_pixel: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    model, epoch = evaluator.load_model(config, checkpoint, device)
    lengths = np.asarray(state["sequences_length"], dtype=np.int64)
    frame_ids: list[np.ndarray] = []
    rmse: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(lengths), batch_size):
            stop = min(start + batch_size, len(lengths))
            batch_lengths = torch.as_tensor(
                lengths[start:stop], dtype=torch.long, device=device
            )
            predicted, _, _ = model(
                state["features"][start:stop].to(device), batch_lengths
            )
            predicted_np = predicted.detach().cpu().numpy()
            labels = state["labels"][start:stop, : predicted_np.shape[1]].numpy()
            std = state["std"][start:stop].numpy()
            center = state[center_key][start:stop].numpy()
            predicted_raw = predicted_np * std + center
            target_raw = labels * std + center
            for local_index, length in enumerate(lengths[start:stop]):
                valid_length = min(int(length), predicted_np.shape[1])
                frames = state["frames"][
                    start + local_index, :valid_length
                ].numpy()
                integer = np.isclose(
                    frames[:, 2], np.rint(frames[:, 2]), atol=1e-6, rtol=0.0
                )
                if not np.any(integer):
                    continue
                error = (
                    predicted_raw[local_index, :valid_length][integer]
                    - target_raw[local_index, :valid_length][integer]
                )
                values = np.sqrt(
                    np.mean(np.square(error.astype(np.float64)), axis=2)
                )
                frame_ids.append(np.rint(frames[integer]).astype(np.int64))
                rmse.append((values * mm_per_pixel).astype(np.float32))
    del model
    torch.cuda.empty_cache()
    return np.concatenate(frame_ids), np.concatenate(rmse), epoch


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Paired extraction requires an OAR GPU")
    evaluator = load_evaluator(args.branch_worktree.resolve())
    config = evaluator.load_config(args.config.resolve())
    global_state = torch.load(args.global_cache.resolve(), map_location="cpu")
    moving_state = torch.load(args.moving_cache.resolve(), map_location="cpu")
    pair_audit = evaluator.validate_cache_pair(global_state, moving_state)

    global_ids, global_rmse, global_epoch = collect(
        evaluator=evaluator,
        config=config,
        state=global_state,
        checkpoint=args.global_checkpoint.resolve(),
        center_key="mean",
        device=device,
        batch_size=args.batch_size,
        mm_per_pixel=args.mm_per_pixel,
    )
    moving_ids, moving_rmse, moving_epoch = collect(
        evaluator=evaluator,
        config=config,
        state=moving_state,
        checkpoint=args.moving_checkpoint.resolve(),
        center_key="sofiane_moving_average",
        device=device,
        batch_size=args.batch_size,
        mm_per_pixel=args.mm_per_pixel,
    )
    if not np.array_equal(global_ids, moving_ids):
        raise ValueError("Global and moving-average frame identities are not aligned")
    if global_rmse.shape != moving_rmse.shape:
        raise ValueError(f"RMSE shape mismatch: {global_rmse.shape} != {moving_rmse.shape}")
    if global_rmse.ndim != 2 or global_rmse.shape[1] != len(evaluator.CLASSES):
        raise ValueError(f"Unexpected RMSE shape: {global_rmse.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            frame_ids=global_ids,
            global_rmse_mm=global_rmse,
            moving_average_oracle_rmse_mm=moving_rmse,
            classes=np.asarray(evaluator.CLASSES),
        )
    os.replace(temporary, args.output)
    audit = {
        "status": "complete",
        "operation": "paired GPU evaluation; no training",
        "oar_job_id": os.environ.get("OAR_JOB_ID"),
        "hostname": os.uname().nodename,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "branch_worktree": str(args.branch_worktree.resolve()),
        "config": str(args.config.resolve()),
        "global_cache": str(args.global_cache.resolve()),
        "moving_cache": str(args.moving_cache.resolve()),
        "global_checkpoint": str(args.global_checkpoint.resolve()),
        "moving_checkpoint": str(args.moving_checkpoint.resolve()),
        "global_checkpoint_sha256": sha256(args.global_checkpoint.resolve()),
        "moving_checkpoint_sha256": sha256(args.moving_checkpoint.resolve()),
        "global_best_human_epoch": global_epoch + 1,
        "moving_best_human_epoch": moving_epoch + 1,
        "moving_prediction_center": "per-chunk 60-chunk moving average",
        "moving_uses_target_statistics_for_prediction_center": True,
        "pair_audit": pair_audit,
        "integer_rows": int(global_rmse.shape[0]),
        "articulators": list(evaluator.CLASSES),
        "rmse_shape": list(global_rmse.shape),
        "mm_per_pixel": args.mm_per_pixel,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output.resolve()),
        "runtime_seconds": time.time() - started,
    }
    atomic_json(args.audit.resolve(), audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
