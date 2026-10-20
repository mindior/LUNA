from __future__ import annotations

from typing import Any, Optional, Tuple, Protocol

import numpy as np
import pandas as pd

from luna_xai.utils.dataframe import ensure_dataframe
from luna_xai.utils.logging_utils import get_luna_logger
from luna_xai.utils.debug_mixin import DebugMixin


class HNSWIndexProtocol(Protocol):
    """Minimal protocol for the HNSW index used by HNSWManager."""

    def knn_query(self, data: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def get_current_count(self) -> int:
        ...

    def set_ef(self, ef: int) -> None:
        ...


class HNSWManager(DebugMixin):
    """
    Provides neighbor retrieval with exact re-ranking.

    This manager never mutates the underlying HNSW index; the query instance
    is *not* inserted in the index. Approximate neighbors are obtained from
    the index and then re-ranked using exact Euclidean distances in the same
    feature space used to build the index.

    Parameters
    ----------
    index :
        Trained HNSW index implementing :class:`HNSWIndexProtocol`.
    data_ref : pd.DataFrame
        Original dataset corresponding to indexed vectors. The row order
        must match the ids used when building the index.
    preprocessor : optional
        Transformer with a ``transform(X)`` method that maps raw tabular
        data into the numeric feature space used to build the index.
    ef_search : int, default 100
        Value passed to ``index.set_ef`` before each query.
    debug : bool, default False
        If True, enables lightweight debug messages through ``_debug()``.
    """

    def __init__(
        self,
        index: HNSWIndexProtocol,
        data_ref: pd.DataFrame,
        preprocessor: Optional[Any] = None,
        ef_search: int = 100,
        debug: bool = False,
    ) -> None:
        required_attrs = ("knn_query", "get_current_count", "set_ef")
        missing = [name for name in required_attrs if not hasattr(index, name)]
        if missing:
            raise TypeError(
                "Provided index does not implement the required HNSW interface "
                f"(missing attributes: {', '.join(missing)})."
            )

        self.index = index
        self.data_ref = data_ref
        self.preprocessor = preprocessor
        self.ef_search = int(ef_search)

        # Logger for HNSWManager operations.
        # Handlers (file vs console) are configured at application level.
        self._logger = get_luna_logger("neighbors.hnsw.manager")
        self._debug_enabled = bool(debug)

    # ------------------------------------------------------------------
    def find_neighbors(
        self,
        instance: pd.Series | pd.DataFrame,
        n_neighbors: int = 7,
        k_factor: int = 10,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Finds the nearest neighbors with exact distance re-ranking.

        The method queries the HNSW index for a candidate pool of size
        ``k_factor * n_neighbors`` (clipped to the index size), computes
        exact Euclidean distances in the transformed feature space and
        returns the top ``n_neighbors`` neighbors *excluding* the query
        instance itself (any candidate at distance 0 is discarded).

        Parameters
        ----------
        instance : pd.Series or pd.DataFrame
            Instance to be explained / queried. If a Series is provided,
            it is treated as a single-row DataFrame.
        n_neighbors : int, default 7
            Number of neighbors to return.
        k_factor : int, default 10
            Oversampling factor for the candidate pool retrieved from the
            HNSW index before exact re-ranking.

        Returns
        -------
        neighbors_df : pd.DataFrame
            Subset of ``data_ref`` containing the nearest neighbors,
            with an extra ``"distance"`` column (float32) holding the
            exact Euclidean distances to ``instance``.
        distances : np.ndarray
            One-dimensional array with the exact distances corresponding
            to the returned neighbors, sorted in ascending order.

        Raises
        ------
        TypeError
            If ``instance`` is not a Series/DataFrame or if the transformed
            data is not purely numeric.
        ValueError
            If ``n_neighbors`` or ``k_factor`` are not positive.
        RuntimeError
            If no valid neighbor candidates can be found.
        """
        if not isinstance(instance, (pd.Series, pd.DataFrame)):
            raise TypeError(
                "HNSWManager.find_neighbors expects a pandas Series or DataFrame "
                f"as 'instance', got {type(instance)!r}."
            )

        if n_neighbors <= 0:
            raise ValueError("Parameter 'n_neighbors' must be a positive integer.")
        if k_factor <= 0:
            raise ValueError("Parameter 'k_factor' must be a positive integer.")

        if isinstance(instance, pd.Series):
            instance_df = ensure_dataframe(instance)
        else:
            instance_df = instance.copy()

        n_indexed = int(self.index.get_current_count())
        if n_indexed == 0:
            raise RuntimeError("HNSW index is empty; cannot retrieve neighbors.")

        # Transform instance to the same space as HNSW
        inst_vec = self._transform(instance_df)  # shape: (1, n_features)

        # Candidate pool size: oversample + leave room for potential self-match
        k_prime = min(max(k_factor * (n_neighbors + 1), n_neighbors + 1), n_indexed)
        self._debug(
            f"Querying HNSW with k_prime={k_prime}, n_neighbors={n_neighbors}, "
            f"k_factor={k_factor}."
        )

        # Configure search ef (does not modify index contents)
        self.index.set_ef(max(self.ef_search, k_prime))

        # Approximate search
        labels_approx, _ = self.index.knn_query(inst_vec, k=k_prime)
        cand_idxs = labels_approx[0].tolist()

        if not cand_idxs:
            raise RuntimeError(
                "HNSWManager could not obtain any candidate neighbors from the index."
            )

        # Compute exact Euclidean distances in the transformed feature space
        candidates = self.data_ref.iloc[cand_idxs].reset_index(drop=True)
        cand_vecs = self._transform(candidates)  # shape: (len(cand_idxs), n_features)

        dists_exact = np.linalg.norm(cand_vecs - inst_vec, axis=1).astype(np.float32)

        # Exclude the query itself (and any perfect duplicates) by discarding distance == 0
        non_self_mask = dists_exact > 0.0
        if not np.any(non_self_mask):
            raise RuntimeError(
                "All retrieved candidates are identical to the query instance; "
                "cannot form a neighbor set excluding the instance itself."
            )

        cand_idxs = [idx for idx, keep in zip(cand_idxs, non_self_mask) if keep]
        dists_exact = dists_exact[non_self_mask]

        # Sort by real distance and keep the k nearest
        order = np.argsort(dists_exact)
        order = order[: min(n_neighbors, order.size)]
        final_idxs = [cand_idxs[i] for i in order]

        neighbors_df = self.data_ref.iloc[final_idxs].copy().reset_index(drop=True)
        final_dists = dists_exact[order]

        neighbors_df["distance"] = final_dists

        self._debug(f"Selected {len(neighbors_df)} neighbors.")
        return neighbors_df, final_dists

    # ------------------------------------------------------------------
    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms tabular data into a dense float32 feature matrix.

        This helper centralizes the logic of applying an optional preprocessor,
        converting sparse matrices to dense and enforcing a numeric float32
        representation suitable for distance computations and the HNSW index.

        Parameters
        ----------
        X : pd.DataFrame
            Input data in tabular form.

        Returns
        -------
        np.ndarray
            Dense, C-contiguous ``float32`` matrix.

        Raises
        ------
        TypeError
            If the transformed data cannot be converted to numeric float32.
        """
        if self.preprocessor is not None:
            X_trans: Any = self.preprocessor.transform(X)
        else:
            X_trans = X.values

        # Convert sparse -> dense, if applicable
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()

        try:
            arr = np.asarray(X_trans, dtype=np.float32, order="C")
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "HNSWManager requires numeric data (float32) after preprocessing; "
                "received non-numeric values."
            ) from exc

        return arr
