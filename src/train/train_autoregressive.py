"""Single-GPU trainer for one-frame-reference autoregressive inversion."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset

from src.model.autoregressive_contour import AnchoredAutoregressiveContour
from src.utils.motion_metrics import (
    autoregressive_loss,
    train_velocity_rms,
    velocity_class_weights,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def load_motion_ar_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    base = config.pop("base_config", None)
    if base:
        base_path = Path(base)
        if not base_path.is_absolute():
            candidate = (path.parent / base_path).resolve()
            base_path = candidate if candidate.exists() else (REPOSITORY_ROOT / base_path).resolve()
        merged = load_motion_ar_config(base_path)
        merged.update(config)
        config = merged
    return config


def resolve_local_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path.resolve()
    normalized = str(value).replace("\\", "/")
    marker = "/Inversion_SI/"
    if marker in normalized:
        candidate = REPOSITORY_ROOT / normalized.split(marker, 1)[1]
        if candidate.exists():
            return candidate.resolve()
    candidate = REPOSITORY_ROOT / path
    if candidate.exists():
        return candidate.resolve()
    return path


def split_cache_path(config: dict[str, Any], split: str) -> Path:
    return resolve_local_path(config["split_cache_dir"]) / f"{split}.pt"


def load_split_state(config: dict[str, Any], split: str) -> dict[str, Any]:
    path = split_cache_path(config, split)
    if not path.exists():
        raise FileNotFoundError(f"Stage-0 split cache is missing: {path}")
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


class CachedSplitDataset(Dataset):
    def __init__(self, state: dict[str, Any]):
        self.state = state

    def __len__(self) -> int:
        return int(self.state["features"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "features": self.state["features"][index],
            "labels": self.state["labels"][index],
            "frames": self.state["frames"][index],
            "std": self.state["std"][index],
            "mean": self.state["mean"][index],
            "sequences_length": int(self.state["sequences_length"][index]),
        }


def make_loader(
    state: dict[str, Any],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    indices: list[int] | None = None,
) -> DataLoader:
    dataset: Dataset = CachedSplitDataset(state)
    if indices is not None:
        dataset = Subset(dataset, indices)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def teacher_forcing_ratio(config: dict[str, Any], epoch: int) -> float:
    schedule = config.get("teacher_forcing", {})
    start = float(schedule.get("start_ratio", 1.0))
    end = float(schedule.get("end_ratio", 0.0))
    warmup = int(schedule.get("warmup_epochs", 5))
    decay = int(schedule.get("decay_epochs", 20))
    if epoch < warmup:
        return start
    if decay <= 0 or epoch >= warmup + decay:
        return end
    progress = (epoch - warmup + 1) / decay
    return start + progress * (end - start)


def _optimizer(
    model: AnchoredAutoregressiveContour, config: dict[str, Any]
) -> torch.optim.Optimizer:
    audio_parameters = list(model.audio_encoder.parameters())
    audio_ids = {id(parameter) for parameter in audio_parameters}
    decoder_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in audio_ids
    ]
    return torch.optim.Adam(
        [
            {
                "params": decoder_parameters,
                "lr": float(config.get("decoder_learning_rate", 1e-3)),
            },
            {
                "params": audio_parameters,
                "lr": float(config.get("audio_encoder_learning_rate", 1e-4)),
            },
        ],
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


def _set_audio_frozen(model: AnchoredAutoregressiveContour, frozen: bool) -> None:
    for parameter in model.audio_encoder.parameters():
        parameter.requires_grad_(not frozen)


def _run_epoch(
    model: AnchoredAutoregressiveContour,
    loader: DataLoader,
    *,
    device: torch.device,
    velocity_weight: float,
    class_weights: torch.Tensor,
    forcing_ratio: float,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
    seed: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "position_mse": 0.0,
        "velocity_mse": 0.0,
        "anchor_embedding_norm": 0.0,
        "history_embedding_norm": 0.0,
        "audio_embedding_norm": 0.0,
        "decoder_state_norm": 0.0,
        "gradient_norm": 0.0,
    }
    batches = 0
    diagnostic_batches = 0
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            audio = batch["features"].to(device, non_blocking=True)
            target = batch["labels"].to(device, non_blocking=True)
            lengths = batch["sequences_length"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction, diagnostics = model(
                audio,
                lengths,
                target[:, 0],
                targets=target if forcing_ratio > 0 else None,
                teacher_forcing_ratio=forcing_ratio,
                sampling_mode="per_sequence",
                generator=generator,
                collect_diagnostics=batch_index == 0,
            )
            losses = autoregressive_loss(
                prediction,
                target,
                lengths,
                velocity_weight=velocity_weight,
                class_weights=class_weights,
            )
            gradient_norm = 0.0
            if training:
                losses["loss"].backward()
                finite_gradients = all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                )
                if not finite_gradients:
                    raise FloatingPointError("Non-finite gradient detected")
                gradient_norm = float(
                    clip_grad_norm_(model.parameters(), float(gradient_clip_norm)).detach().cpu()
                )
                optimizer.step()
            for key in ("loss", "position_mse", "velocity_mse"):
                totals[key] += float(losses[key].detach().cpu())
            for key, value in asdict(diagnostics).items():
                totals[key] += float(value)
            if batch_index == 0:
                diagnostic_batches += 1
            totals["gradient_norm"] += gradient_norm
            batches += 1
    if not batches:
        raise RuntimeError("Training/evaluation loader produced no batches")
    result = {
        key: value / batches
        for key, value in totals.items()
        if key not in {
            "anchor_embedding_norm",
            "history_embedding_norm",
            "audio_embedding_norm",
            "decoder_state_norm",
        }
    }
    for key in (
        "anchor_embedding_norm",
        "history_embedding_norm",
        "audio_embedding_norm",
        "decoder_state_norm",
    ):
        result[key] = totals[key] / max(1, diagnostic_batches)
    return result


def save_checkpoint(
    path: Path,
    model: AnchoredAutoregressiveContour,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    validation_loss: float,
    config: dict[str, Any],
    class_weights: torch.Tensor,
    velocity_rms: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "validation_loss": float(validation_loss),
            "config": config,
            "class_weights": class_weights.cpu(),
            "train_velocity_rms": velocity_rms.cpu(),
        },
        path,
    )


def train_from_config(
    config: dict[str, Any],
    *,
    output_dir: str | Path,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Train on the declared train split and select only on validation."""
    seed = int(config.get("seed", 42))
    set_deterministic_seed(seed)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_state = load_split_state(config, "train_sequences")
    valid_state = load_split_state(config, "valid_sequences")

    expected_dimension = len(config["classes"]) * int(config["output_layer"])
    actual_dimension = int(train_state["labels"].shape[-2] * train_state["labels"].shape[-1])
    if expected_dimension != actual_dimension:
        raise RuntimeError(
            f"Configured contour dimension {expected_dimension} != cache dimension {actual_dimension}"
        )
    velocity_rms = train_velocity_rms(
        train_state["labels"], train_state["sequences_length"]
    )
    weighting_mode = str(config.get("velocity_class_weighting", "uniform"))
    class_weights = velocity_class_weights(
        velocity_rms,
        weighting_mode,
        mobile_threshold_ratio=float(config.get("mobile_threshold_ratio", 0.25)),
    ).to(device)
    statistics = {
        "classes": list(config["classes"]),
        "velocity_rms_normalized": velocity_rms.tolist(),
        "velocity_class_weighting": weighting_mode,
        "velocity_class_weights": class_weights.detach().cpu().tolist(),
    }
    (output_dir / "train_motion_statistics.json").write_text(
        json.dumps(statistics, indent=2), encoding="utf-8"
    )

    model = AnchoredAutoregressiveContour.from_config(config).to(device)
    baseline_checkpoint = config.get("audio_encoder_checkpoint")
    load_report = None
    if baseline_checkpoint:
        checkpoint = resolve_local_path(baseline_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Audio encoder checkpoint is missing: {checkpoint}")
        load_report = model.audio_encoder.load_baseline_checkpoint(str(checkpoint))
    optimizer = _optimizer(model, config)
    resume_report = None
    start_epoch = 0
    best_loss = float("inf")
    best_epoch = -1
    selection_reference_path: Path | None = None
    resume_checkpoint = config.get("resume_training_checkpoint")
    if resume_checkpoint:
        resume_path = resolve_local_path(resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint is missing: {resume_path}")
        resume_state = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(resume_state["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        start_epoch = int(resume_state["epoch"]) + 1
        selection_reference = config.get("selection_reference_checkpoint")
        if selection_reference:
            selection_reference_path = resolve_local_path(selection_reference)
            if not selection_reference_path.exists():
                raise FileNotFoundError(
                    f"Selection-reference checkpoint is missing: "
                    f"{selection_reference_path}"
                )
            selection_state = torch.load(
                selection_reference_path,
                map_location="cpu",
                weights_only=False,
            )
            reference_model = AnchoredAutoregressiveContour.from_config(config)
            reference_model.load_state_dict(
                selection_state["model_state_dict"],
                strict=True,
            )
            best_epoch = int(selection_state["epoch"])
            best_loss = float(selection_state["validation_loss"])
            del reference_model, selection_state
        else:
            best_epoch = int(resume_state["epoch"])
            best_loss = float(resume_state["validation_loss"])
        resume_report = {
            "checkpoint": str(resume_path),
            "checkpoint_epoch": int(resume_state["epoch"]),
            "checkpoint_validation_loss": float(resume_state["validation_loss"]),
            "selection_reference_checkpoint": (
                str(selection_reference_path) if selection_reference_path else str(resume_path)
            ),
            "selection_reference_epoch": best_epoch,
            "selection_reference_validation_loss": best_loss,
            "optimizer_state_loaded": True,
            "strict_model_state_loaded": True,
        }
        del resume_state

    batch_size = int(config.get("batch_size", 10))
    train_indices = None
    maximum = config.get("max_train_sequences")
    if maximum is not None and int(maximum) < len(train_state["sequences_length"]):
        generator = np.random.default_rng(seed)
        train_indices = sorted(
            generator.choice(
                len(train_state["sequences_length"]), size=int(maximum), replace=False
            ).tolist()
        )
    train_loader = make_loader(
        train_state,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        indices=train_indices,
    )
    valid_loader = make_loader(
        valid_state,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    freeze_epochs = int(config.get("freeze_audio_encoder_epochs", 5))
    velocity_weight = float(config.get("velocity_loss_weight", 0.0))
    gradient_clip = float(config.get("gradient_clip_norm", 1.0))
    if selection_reference_path and bool(
        config.get("reevaluate_selection_reference", False)
    ):
        selection_state = torch.load(
            selection_reference_path,
            map_location="cpu",
            weights_only=False,
        )
        reference_model = AnchoredAutoregressiveContour.from_config(config).to(device)
        reference_model.load_state_dict(
            selection_state["model_state_dict"],
            strict=True,
        )
        reference_metrics = _run_epoch(
            reference_model,
            valid_loader,
            device=device,
            velocity_weight=velocity_weight,
            class_weights=class_weights,
            forcing_ratio=0.0,
            optimizer=None,
            gradient_clip_norm=gradient_clip,
            seed=seed,
        )
        stored_loss = best_loss
        best_loss = float(reference_metrics["loss"])
        resume_report["selection_reference_stored_validation_loss"] = stored_loss
        resume_report["selection_reference_validation_loss"] = best_loss
        resume_report["selection_reference_reevaluated"] = True
        resume_report["selection_reference_validation_metrics"] = reference_metrics
        del reference_model, selection_state
    if resume_checkpoint and config.get("additional_training_epochs") is not None:
        end_epoch = start_epoch + int(config["additional_training_epochs"])
    else:
        end_epoch = int(config.get("n_epochs", 30))
    if end_epoch <= start_epoch:
        raise ValueError(
            f"Training end epoch {end_epoch} must be greater than start epoch {start_epoch}"
        )
    patience = int(config.get("patience", 10))
    history: list[dict[str, Any]] = []
    stale = 0
    started = time.time()
    if resume_checkpoint:
        if selection_reference_path:
            shutil.copy2(selection_reference_path, output_dir / "best_model.pth")
        else:
            save_checkpoint(
                output_dir / "best_model.pth",
                model,
                optimizer,
                epoch=best_epoch,
                validation_loss=best_loss,
                config=config,
                class_weights=class_weights,
                velocity_rms=velocity_rms,
            )
    for epoch in range(start_epoch, end_epoch):
        frozen = epoch < freeze_epochs
        _set_audio_frozen(model, frozen)
        ratio = teacher_forcing_ratio(config, epoch)
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            velocity_weight=velocity_weight,
            class_weights=class_weights,
            forcing_ratio=ratio,
            optimizer=optimizer,
            gradient_clip_norm=gradient_clip,
            seed=seed * 1000 + epoch,
        )
        validation_metrics = _run_epoch(
            model,
            valid_loader,
            device=device,
            velocity_weight=velocity_weight,
            class_weights=class_weights,
            forcing_ratio=0.0,
            optimizer=None,
            gradient_clip_norm=gradient_clip,
            seed=seed,
        )
        row = {
            "epoch": epoch,
            "teacher_forcing_ratio": ratio,
            "audio_encoder_frozen": frozen,
            "train": train_metrics,
            "validation_free_running": validation_metrics,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        current = validation_metrics["loss"]
        if current < best_loss:
            best_loss = current
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                output_dir / "best_model.pth",
                model,
                optimizer,
                epoch=epoch,
                validation_loss=current,
                config=config,
                class_weights=class_weights,
                velocity_rms=velocity_rms,
            )
        else:
            stale += 1
        save_checkpoint(
            output_dir / "last_model.pth",
            model,
            optimizer,
            epoch=epoch,
            validation_loss=current,
            config=config,
            class_weights=class_weights,
            velocity_rms=velocity_rms,
        )
        if stale >= patience:
            break

    summary = {
        "seed": seed,
        "device": str(device),
        "num_train_sequences": len(train_loader.dataset),
        "num_validation_sequences": len(valid_loader.dataset),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "elapsed_seconds": time.time() - started,
        "audio_checkpoint_load_report": load_report,
        "resume_checkpoint_load_report": resume_report,
        "start_epoch": start_epoch,
        "end_epoch_exclusive": end_epoch,
        "num_epochs_completed": len(history),
        "config_sha256": hashlib.sha256(
            yaml.safe_dump(config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def overfit_cached_batch(
    config: dict[str, Any],
    *,
    free_running: bool,
    steps: int,
    batch_size: int,
    device_name: str | None = None,
) -> dict[str, Any]:
    """S3/S4 optimization test on real cached normalized tensors."""
    seed = int(config.get("seed", 42))
    set_deterministic_seed(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = load_split_state(config, "train_sequences")
    count = 1 if free_running else batch_size
    batch = next(
        iter(make_loader(state, batch_size=count, shuffle=False, seed=seed, indices=list(range(count))))
    )
    audio = batch["features"].to(device)
    target = batch["labels"].to(device)
    lengths = batch["sequences_length"].to(device)
    model = AnchoredAutoregressiveContour.from_config(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    class_weights = torch.ones(len(config["classes"]), device=device)
    first: dict[str, float] | None = None
    last: dict[str, float] = {}
    checkpoints = {0, max(0, steps // 4), max(0, steps // 2), max(0, steps - 1)}
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    for step in range(steps):
        if free_running:
            progress = step / max(1, int(steps * 0.7))
            ratio = max(0.0, 1.0 - progress)
        else:
            ratio = 1.0
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(
            audio,
            lengths,
            target[:, 0],
            targets=target if ratio > 0 else None,
            teacher_forcing_ratio=ratio,
            sampling_mode="per_sequence",
            generator=generator,
        )
        losses = autoregressive_loss(
            prediction,
            target,
            lengths,
            velocity_weight=0.5,
            class_weights=class_weights,
        )
        losses["loss"].backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last = {key: float(value.detach().cpu()) for key, value in losses.items()}
        if first is None:
            first = dict(last)
        if step in checkpoints:
            print(json.dumps({"step": step, "ratio": ratio, **last}), flush=True)
    result = {
        "gate": "S4" if free_running else "S3",
        "steps": steps,
        "batch_size": count,
        "first": first,
        "last": last,
        "position_reduction": 1.0 - last["position_mse"] / first["position_mse"],
        "velocity_reduction": 1.0 - last["velocity_mse"] / first["velocity_mse"],
        "final_teacher_forcing_ratio": ratio,
    }
    return result
