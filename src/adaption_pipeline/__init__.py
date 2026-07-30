"""Composable ASD2-to-ASD1 adaptation pipeline.

The package owns contracts, reusable workflow modules, and orchestration.
Scientific implementations in separate repositories are accessed through
adapters, so an implementation can be replaced without changing downstream
artifact interfaces.
"""

from .contracts import (
    CapabilityFlags,
    CohortManifest,
    ModelBundle,
    SessionRecord,
)

__all__ = [
    "CapabilityFlags",
    "CohortManifest",
    "ModelBundle",
    "SessionRecord",
]

__version__ = "0.1.0"
