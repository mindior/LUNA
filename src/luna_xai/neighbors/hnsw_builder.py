"""
Enhanced HNSWBuilder
====================

Builds and persists an HNSW index with metadata.

- build_index(X)       → returns the hnswlib.Index and stores metadata internally
- save_index("path")   → saves path.bin + path.meta.json
- load_index("path")   → loads index + metadata

This is the canonical single API used by both:
- TabularDomain
- TabularDomainManager
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Any

import hnswlib
import numpy as np

from luna_xai.utils.logging_utils import get_luna_logger
from luna_xai.utils.debug_mixin import DebugMixin

# Module-level logger for HNSW index build/load operations.
# Application code is expected to configure handlers so that
# these logs are written to a log file (e.g., via a FileHandler attached to
# the "luna_xai" logger). If you prefer console output instead, you can
# configure the "luna_xai" logger with a StreamHandler or via
# `configure_luna_logging`.
LOGGER = get_luna_logger("neighbors.hnsw.builder")


class HNSWBuilder(DebugMixin):
    """
    Builds and persists an HNSW index with metadata.

    Metadata stored:
        - space
        - dim
        - ef_construction
        - M
        - random_seed
        - num_elements
    """

    def __init__(
        self,
        space: str = "l2",
        ef_construction: int = 100,
        M: int = 16,
        random_seed: int = 42,
        debug: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        space : {"l2", "cosine", ...}, default="l2"
            Distance space used by hnswlib.
        ef_construction : int, default=100
            Construction-time `ef` parameter controlling accuracy vs speed.
        M : int, default=16
            Maximum number of outgoing connections in the graph.
        random_seed : int, default=42
            Seed used internally by hnswlib during index construction.
        debug : bool, default=False
            If True, enables debug-level logging via DebugMixin.
        """
        self.space = space
        self.ef_construction = ef_construction
        self.M = M
        self.random_seed = random_seed

        self.index: Optional[hnswlib.Index] = None
        self.num_elements: Optional[int] = None
        self.dim: Optional[int] = None

        # Instance-level logger and debug flag
        self._logger = LOGGER
        self._debug_enabled = bool(debug)

    # ------------------------------------------------------------------
    def build_index(self, X: Any) -> hnswlib.Index:
        """
        Build dense float32 HNSW index with capacity = 2 × dataset size.

        Parameters
        ----------
        X : array-like or sparse matrix
            Feature matrix to index.

        Returns
        -------
        index : hnswlib.Index
            Trained HNSW index.

        Side-effects
        ------------
        - self.index
        - self.dim
        - self.num_elements
        """
        # Convert to dense float32
        if hasattr(X, "toarray"):
            self._debug("Converting sparse matrix to dense array for HNSW index.")
            X_dense = X.toarray().astype(np.float32)
        else:
            X_dense = np.asarray(X, dtype=np.float32)

        num_elements, dim = X_dense.shape
        self.num_elements = num_elements
        self.dim = dim

        LOGGER.info(
            "Building HNSW index (n_samples=%d, dim=%d, space=%s, ef_construction=%d, M=%d, seed=%d).",
            num_elements,
            dim,
            self.space,
            self.ef_construction,
            self.M,
            self.random_seed,
        )

        self._debug(
            f"Initializing hnswlib.Index(space='{self.space}', dim={dim}) "
            f"with max_elements={num_elements * 2}."
        )

        index = hnswlib.Index(space=self.space, dim=dim)
        index.init_index(
            max_elements=num_elements * 2,
            ef_construction=self.ef_construction,
            M=self.M,
            random_seed=self.random_seed,
        )
        index.add_items(X_dense, ids=np.arange(num_elements))
        index.set_ef(100)

        self.index = index
        LOGGER.info("HNSW index successfully built.")
        return index

    # ------------------------------------------------------------------
    def save_index(self, path: str) -> None:
        """
        Save index and metadata:

            path.bin
            path.meta.json
        """
        if self.index is None:
            raise RuntimeError("Cannot save — index is not initialized.")

        base = Path(path)
        bin_path = base.with_suffix(".bin")
        meta_path = base.with_suffix(".meta.json")

        self._debug(f"Saving HNSW index to '{bin_path}'.")
        self._debug(f"Saving HNSW metadata to '{meta_path}'.")

        # Save raw index
        self.index.save_index(str(bin_path))

        # Save metadata
        metadata = {
            "space": self.space,
            "dim": self.dim,
            "num_elements": self.num_elements,
            "ef_construction": self.ef_construction,
            "M": self.M,
            "random_seed": self.random_seed,
        }

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        LOGGER.info("Saved HNSW index to '%s'.", str(bin_path))
        LOGGER.info("Saved HNSW metadata to '%s'.", str(meta_path))

    # ------------------------------------------------------------------
    def load_index(self, path: str) -> hnswlib.Index:
        """
        Load index and metadata:

            path.bin
            path.meta.json

        Parameters
        ----------
        path : str
            Base path without extension (same used in save_index).

        Returns
        -------
        index : hnswlib.Index
            Loaded HNSW index ready for querying.
        """
        base = Path(path)
        bin_path = base.with_suffix(".bin")
        meta_path = base.with_suffix(".meta.json")

        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {meta_path}")

        self._debug(f"Loading HNSW metadata from '{meta_path}'.")
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Restore parameters
        self.space = metadata["space"]
        self.dim = metadata["dim"]
        self.num_elements = metadata["num_elements"]
        self.ef_construction = metadata["ef_construction"]
        self.M = metadata["M"]
        self.random_seed = metadata["random_seed"]

        LOGGER.info(
            "Loading HNSW index from '%s' with metadata (space=%s, dim=%s, n_elements=%s).",
            str(bin_path),
            self.space,
            str(self.dim),
            str(self.num_elements),
        )

        self._debug(
            f"Instantiating hnswlib.Index(space='{self.space}', dim={self.dim}) "
            f"for loading from '{bin_path}'."
        )

        index = hnswlib.Index(space=self.space, dim=self.dim)
        index.load_index(str(bin_path))
        index.set_ef(100)

        self.index = index

        LOGGER.info("Loaded HNSW index from '%s'.", str(bin_path))
        LOGGER.info("Loaded HNSW metadata from '%s'.", str(meta_path))

        return index
