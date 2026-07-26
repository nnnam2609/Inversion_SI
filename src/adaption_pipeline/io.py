"""Small atomic I/O helpers shared by pipeline workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .contracts import ContractError
from src.common.artifacts import atomic_write_csv, load_mapping as _load_mapping


def load_mapping(path: Path) -> Dict[str, Any]:
    try:
        return _load_mapping(path)
    except (FileNotFoundError, ValueError) as error:
        raise ContractError(str(error)) from error


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    try:
        atomic_write_csv(path, rows)
    except ValueError as error:
        raise ContractError(str(error)) from error
