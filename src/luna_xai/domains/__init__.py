"""
LUNA XAI — Domains Subpackage
=============================

This subpackage defines the data domain abstractions used by the LUNA
framework to bridge raw user data, preprocessing pipelines, and model
prediction interfaces.

Domains are responsible for:
----------------------------
- Executing feature preprocessing and transformations (via DataInterface)
- Managing hybrid encodings for structured data (numeric + categorical)
- Providing access to neighborhood search structures (e.g., HNSW)
- Exposing a unified prediction API used by all LUNA explainers and methods

Current Implementations
-----------------------
- **TabularDomain**:
    Hybrid tabular domain supporting categorical encodings (e.g., CANE),
    numerical scaling, HNSW indexing, and model-space ↔ data-space
    transformations.

Notes
-----
All domains register themselves automatically into the global LUNA registry
under the namespace ``domain.<name>`` upon import.

Importing this package triggers:
- Domain class availability
- Auto-registration into the global component registry
"""

from luna_xai.domains.tabular_domain import TabularDomain
from luna_xai.core import registry  # ensures registry availability

__all__ = [
    "TabularDomain",
]
