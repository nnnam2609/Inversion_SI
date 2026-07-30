"""Lightweight process and accelerator resource diagnostics."""

from __future__ import annotations

import os

import psutil
import torch


def log_process_memory(stage: str = "") -> None:
    """Print resident RAM plus current and peak CUDA allocations in MiB."""

    process = psutil.Process(os.getpid())
    ram = process.memory_info().rss / (1024**2)
    vram = torch.cuda.memory_allocated() / (1024**2)
    max_vram = torch.cuda.max_memory_allocated() / (1024**2)
    print(
        f"[{stage}] RAM: {ram:.2f} MiB | "
        f"VRAM allocated: {vram:.2f} MiB | VRAM peak: {max_vram:.2f} MiB",
        flush=True,
    )
