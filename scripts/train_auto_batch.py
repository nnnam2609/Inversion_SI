#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.model.baseline_5 import BaselineModel  # noqa: E402
from src.utils import metrics  # noqa: E402
from src.utils.config_validation import load_yaml_config  # noqa: E402
from src.utils.normalization import TRAINING_SPLIT_CACHE_KEYS, load_validated_split_cache_state  # noqa: E402
from src.utils.split_cache_overrides import SPLIT_FILES, split_cache_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune per-GPU batch size, then launch training.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--target-util", type=float, default=0.80)
    parser.add_argument("--max-batch", type=int, default=4096)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--output-config-dir", type=Path, default=None)
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=0,
        help="Run a separate from-scratch smoke training before the full run; 0 disables it.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate config and any existing split caches, then exit before CUDA batch tuning.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return load_yaml_config(path)


def save_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    os.replace(tmp_path, path)


def resolve_output_config_dir(config: dict[str, Any], requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    split_cache = Path(config["split_cache_dir"]).resolve()
    repro_root = split_cache.parent
    return repro_root / "auto_batch_configs"


def auto_batch_suffix(selected_batch_size: int, gpus: int) -> str:
    return f"auto80_bs{int(selected_batch_size)}_{int(gpus)}gpu"


def smoke_runtime_config(runtime_config: dict[str, Any], smoke_epochs: int) -> dict[str, Any]:
    if smoke_epochs < 1:
        raise ValueError("smoke_epochs must be >= 1")
    smoke = dict(runtime_config)
    suffix = f"smoke{int(smoke_epochs)}epoch"
    for key in ("experiment_name", "folder_save", "model", "tag"):
        smoke[key] = f"{runtime_config[key]}_{suffix}"
    smoke["n_epochs"] = int(smoke_epochs)
    smoke["save_every"] = 1
    smoke["patience"] = max(1, min(int(runtime_config.get("patience", 1)), smoke_epochs))
    smoke["smoke_parent_experiment"] = runtime_config["experiment_name"]
    smoke["smoke_only"] = True
    return smoke


def run_training(config_path: Path, env: dict[str, str]) -> None:
    command = [
        sys.executable,
        "src/main_train.py",
        "--config",
        str(config_path),
    ]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def audit_existing_split_caches(config: dict[str, Any]) -> dict[str, Any]:
    try:
        base_dir = split_cache_dir(config).resolve()
    except KeyError as exc:
        return {
            "status": "skipped",
            "reason": str(exc),
            "rows": [],
        }
    rows: list[dict[str, Any]] = []
    for split_key, filename in SPLIT_FILES.items():
        cache_path = base_dir / filename
        row: dict[str, Any] = {
            "split": split_key,
            "cache_path": str(cache_path),
        }
        if not cache_path.exists():
            row["status"] = "missing"
            rows.append(row)
            continue
        _state, floor_summary = load_validated_split_cache_state(
            cache_path,
            config,
            required_keys=TRAINING_SPLIT_CACHE_KEYS,
        )
        row.update({"status": "ok", **floor_summary})
        rows.append(row)
    return {
        "status": "ok",
        "split_cache_dir": str(base_dir),
        "num_ok": sum(1 for row in rows if row["status"] == "ok"),
        "num_missing": sum(1 for row in rows if row["status"] == "missing"),
        "rows": rows,
    }


def clear_cuda() -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def make_model(config: dict[str, Any], device: torch.device) -> BaselineModel:
    return BaselineModel(
        config["input_layer"],
        config["hidden_layer"],
        config["num_layers"],
        config["output_layer"],
        len(config["classes"]),
        config["nbr_phonemes"],
        config["phonemesdir"],
    ).to(device)


def try_batch(config: dict[str, Any], batch_size: int, device: torch.device) -> tuple[bool, int, str | None]:
    clear_cuda()
    try:
        model = make_model(config, device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config.get("learning_rate", 0.001)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
        sequence_length = int(config["sequence_length"])
        input_layer = int(config["input_layer"])
        output_layer = int(config["output_layer"])
        n_classes = len(config["classes"])
        x = torch.randn(batch_size, sequence_length, input_layer, device=device)
        y = torch.randn(batch_size, sequence_length, n_classes, output_layer, device=device)
        std = torch.ones(batch_size, 1, n_classes, output_layer, device=device)
        mean = torch.zeros(batch_size, 1, n_classes, output_layer, device=device)
        lengths = torch.full((batch_size,), sequence_length, dtype=torch.long, device=device)
        pred, _, _ = model(x, lengths)
        y = y[:, : pred.shape[1]]
        # Match TrainSingle.train_batch, including metric-only allocations that
        # remain live before the actual MSE backward pass. A plain MSE probe can
        # substantially overestimate the safe batch for criterion_pearson.
        loss = metrics.loss_mse(y, pred)
        loss_rmse = metrics.loss_rmse(y, pred)
        y_raw = (y * std) + mean
        pred_raw = (pred * std) + mean
        loss_rmse_raw = metrics.loss_rmse(y_raw, pred_raw)
        loss_pearson = metrics.pearson_correlation(y, pred)
        loss_criterion = metrics.criterion_both(
            y,
            pred,
            90,
            True,
            int(device.index or 0),
        )
        # Force the same scalar materialization performed by TrainSingle.
        _ = (
            loss_rmse.item(),
            loss_rmse_raw.item(),
            loss_pearson.item(),
            loss_criterion.item(),
        )
        loss.backward()
        optimizer.step()
        peak = int(torch.cuda.max_memory_allocated(device))
        del (
            model,
            optimizer,
            x,
            y,
            std,
            mean,
            lengths,
            pred,
            loss,
            loss_rmse,
            y_raw,
            pred_raw,
            loss_rmse_raw,
            loss_pearson,
            loss_criterion,
        )
        clear_cuda()
        return True, peak, None
    except torch.cuda.OutOfMemoryError as exc:
        clear_cuda()
        return False, int(torch.cuda.max_memory_allocated(device)), repr(exc)


def tune_batch_size(config: dict[str, Any], target_util: float, min_batch: int, max_batch: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; auto batch tuning must run inside a GPU OAR allocation.")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    target_bytes = int(props.total_memory * target_util)
    low = 0
    high = max(min_batch, 1)
    observations = []
    while high <= max_batch:
        ok, peak, error = try_batch(config, high, device)
        observations.append({"batch_size": high, "ok": ok, "peak_bytes": peak, "error": error})
        if not ok or peak >= target_bytes:
            break
        low = high
        high *= 2
    high = min(high, max_batch)
    best = max(low, min_batch)
    left = max(best + 1, min_batch)
    right = high
    while left <= right:
        mid = (left + right) // 2
        ok, peak, error = try_batch(config, mid, device)
        observations.append({"batch_size": mid, "ok": ok, "peak_bytes": peak, "error": error})
        if ok and peak <= target_bytes:
            best = mid
            left = mid + 1
        else:
            right = mid - 1
    ok, peak, error = try_batch(config, best, device)
    observations.append({"batch_size": best, "ok": ok, "peak_bytes": peak, "error": error, "selected": True})
    if not ok:
        raise RuntimeError(f"Selected batch size unexpectedly failed: {best}: {error}")
    return {
        "selected_batch_size": best,
        "selected_peak_bytes": peak,
        "target_bytes": target_bytes,
        "target_util": target_util,
        "gpu_name": props.name,
        "gpu_total_bytes": int(props.total_memory),
        "observations": observations,
    }


def main() -> None:
    args = parse_args()
    if args.smoke_epochs < 0:
        raise ValueError("--smoke-epochs must be >= 0")
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    split_cache_preflight = audit_existing_split_caches(config)
    print("split_cache_preflight " + json.dumps(split_cache_preflight, sort_keys=True), flush=True)
    if args.preflight_only:
        return
    tuning = tune_batch_size(config, args.target_util, args.min_batch, args.max_batch)
    runtime_config = dict(config)
    runtime_config["batch_size"] = int(tuning["selected_batch_size"])
    runtime_config["auto_batch_tuning"] = tuning
    runtime_config["split_cache_preflight"] = split_cache_preflight
    suffix = auto_batch_suffix(tuning["selected_batch_size"], args.gpus)
    runtime_config["run_name_suffix"] = suffix
    output_dir = resolve_output_config_dir(runtime_config, args.output_config_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    runtime_config_path = output_dir / f"{config_path.stem}_{suffix}_{timestamp}.yaml"
    save_yaml(runtime_config, runtime_config_path)
    report_path = runtime_config_path.with_suffix(".json")
    report_path.write_text(json.dumps(tuning, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "auto_batch_selected "
        f"batch_size={tuning['selected_batch_size']} "
        f"peak_gib={tuning['selected_peak_bytes'] / (1024 ** 3):.2f} "
        f"target_gib={tuning['target_bytes'] / (1024 ** 3):.2f} "
        f"runtime_config={runtime_config_path}",
        flush=True,
    )
    if args.dry_run:
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(args.gpus))
    env["PYTHONUNBUFFERED"] = "1"
    if args.smoke_epochs:
        smoke_config = smoke_runtime_config(runtime_config, args.smoke_epochs)
        smoke_config_path = output_dir / f"{runtime_config_path.stem}_smoke{args.smoke_epochs}epoch.yaml"
        save_yaml(smoke_config, smoke_config_path)
        print(
            f"smoke_training_start epochs={args.smoke_epochs} config={smoke_config_path}",
            flush=True,
        )
        run_training(smoke_config_path, env)
        print(f"smoke_training_passed config={smoke_config_path}", flush=True)
    print(f"full_training_start from_scratch=true config={runtime_config_path}", flush=True)
    run_training(runtime_config_path, env)


if __name__ == "__main__":
    main()
