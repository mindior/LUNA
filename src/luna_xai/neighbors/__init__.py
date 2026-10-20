"""
Neighbors Subpackage for LUNA XAI
=================================

Provides all neighbor-search and retrieval utilities used in
Dual-Space Analysis (DSA) and other local explainability methods.

Components
----------
- **HNSWBuilder** — Builds and loads HNSWlib indices for high-dimensional
  approximate nearest neighbor search.
- **HNSWManager** — Handles neighbor retrieval and distance queries.
- **HybridNeighborManager** — Combines CANE-IDF preprocessing with HNSW-based
  retrieval for mixed (numeric + categorical) tabular data.

Notes
-----
These components are **infrastructure-level tools** — they are *not* part of
the LUNA registry, since they do not represent explainability strategies
but rather enable efficient neighborhood-based sensitivity analysis.

Dependencies
------------
Requires the `hnswlib` library. If not installed, a warning is issued and
neighbor-based methods will be unavailable.
"""

from luna_xai.neighbors.hnsw_builder import HNSWBuilder
from luna_xai.neighbors.hnsw_manager import HNSWManager
from luna_xai.neighbors.hybrid_neighbor_manager import HybridNeighborManager

__all__ = [
    "HNSWBuilder",
    "HNSWManager",
    "HybridNeighborManager",
]

# ---------------------------------------------------------------------
# Optional runtime dependency check
# ---------------------------------------------------------------------
try:
    import hnswlib  # noqa: F401
except ImportError:
    import warnings
    warnings.warn(
        "⚠️ hnswlib not found. Please install it with `pip install hnswlib` "
        "to enable neighbor-based explainability in LUNA XAI.",
        category=ImportWarning,
        stacklevel=2,
    )
