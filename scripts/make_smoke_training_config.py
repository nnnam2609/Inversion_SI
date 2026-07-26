#!/usr/bin/env python3
"""Create a one-epoch training config without changing the full config."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suffix", default="smoke1epoch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.input.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Config is not a mapping: {args.input}")
    for key in ("experiment_name", "folder_save", "model", "tag"):
        config[key] = f"{config[key]}_{args.suffix}"
    config.update(
        {
            "n_epochs": 1,
            "save_every": 1,
            "patience": 1,
            "smoke_only": True,
            "smoke_parent_config": str(args.input.resolve()),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    os.replace(temporary, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
