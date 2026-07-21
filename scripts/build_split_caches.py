#!/usr/bin/env python3
"""Build and report train-global split caches for one training config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.train.split_cache import ensure_split_caches  # noqa: E402
from src.utils.config_validation import load_yaml_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml_config(config_path)
    metadata = ensure_split_caches(config)
    print(json.dumps({"status": "ok", "config": str(config_path), **metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
