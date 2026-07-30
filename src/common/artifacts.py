"""Small, dependency-light helpers for versioned workflow artifacts."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp suitable for manifests."""

    return datetime.now(timezone.utc).isoformat()


def local_now() -> str:
    """Return the historical local-time manifest timestamp format."""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Hash one file without reading it entirely into memory."""

    path = require_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def to_primitive(value: Any) -> Any:
    """Convert dataclasses, paths, mappings and tuples to JSON-safe values."""

    if dataclasses.is_dataclass(value):
        return {
            str(key): to_primitive(item)
            for key, item in dataclasses.asdict(value).items()
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_primitive(item) for item in value]
    return value


def _atomic_write(path: Path, payload: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically in the destination filesystem."""

    _atomic_write(path, text)


def atomic_write_json(
    path: Path, data: Any, *, allow_nan: bool = False
) -> None:
    """Write strict JSON atomically in the destination filesystem."""

    payload = json.dumps(
        to_primitive(data),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=allow_nan,
    )
    _atomic_write(path, payload + "\n")


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a non-empty, rectangular sequence of mappings atomically."""

    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames = list(rows[0])
    expected = set(fieldnames)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(
                f"CSV row {index} fields differ from the first row: "
                f"expected={fieldnames}, actual={list(row)}"
            )
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_mapping(path: Path) -> dict[str, Any]:
    """Read a JSON or YAML file and require a mapping at its root."""

    path = require_file(path)
    text = path.read_text(encoding="utf-8")
    payload = (
        yaml.safe_load(text)
        if path.suffix.lower() in {".yaml", ".yml"}
        else json.loads(text)
    )
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def require_file(path: Path) -> Path:
    """Resolve and validate one required file."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing required file: {resolved}")
    return resolved


def require_directory(path: Path) -> Path:
    """Resolve and validate one required directory."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Missing required directory: {resolved}")
    return resolved
