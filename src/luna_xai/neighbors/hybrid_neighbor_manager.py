from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Protocol

import numpy as np
import pandas as pd

from luna_xai.utils.dataframe import ensure_dataframe
from luna_xai.utils.logging_utils import get_luna_logger
from luna_xai.utils.debug_mixin import DebugMixin


class HNSWIndexProtocol(Protocol):
    """Minimal protocol for the HNSW index used by HybridNeighborManager."""

    def knn_query(self, data: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def get_current_count(self) -> int:
        ...

    def set_ef(self, ef: int) -> None:
        ...


class HybridNeighborManager(DebugMixin):
    """
    Manages nearest neighbor search using hybrid preprocessing
    (e.g. CANE + scaling) and HNSW-based approximate search with
    exact Euclidean re-ranking.

    This manager does **not** modify the underlying HNSW index.

    Parameters
    ----------
    preprocessor_hnsw : object
        Preprocessor for the HNSW space (e.g., HybridPreprocessor) implementing
        a ``transform(X)`` method that outputs numeric features.
    preprocessor_model : object, optional
        Preprocessor for model input space. Stored for API symmetry with other
        components but not used directly by this class.
    index : HNSWIndexProtocol, optional
        Prebuilt HNSW index operating on the same numeric space produced by
        ``preprocessor_hnsw`` and ``data_base_hnsw``.
    data_base_hnsw : pd.DataFrame, optional
        Preprocessed numeric data used to build the index. Row ordering must
        match the ids used when inserting items into the HNSW index.
    data_base_original : pd.DataFrame, optional
        Original raw data (categorical + numeric). Used to return neighbors in
        the original feature space.
    ef_search : int, default 100
        Value passed to ``index.set_ef`` before each query.
    debug : bool, default False
        If True, enables lightweight debug messages through ``_debug()``.
    """

    def __init__(
        self,
        preprocessor_hnsw: Any,
        preprocessor_model: Any | None = None,
        index: HNSWIndexProtocol | None = None,
        data_base_hnsw: pd.DataFrame | None = None,
        data_base_original: pd.DataFrame | None = None,
        ef_search: int = 100,
        debug: bool = False,
    ) -> None:
        if not hasattr(preprocessor_hnsw, "transform"):
            raise TypeError(
                "HybridNeighborManager requires 'preprocessor_hnsw' to implement "
                "a 'transform(X)' method."
            )

        # Index can be configured later, but if provided it must support the minimal protocol.
        if index is not None:
            required_attrs = ("knn_query", "get_current_count", "set_ef")
            missing = [name for name in required_attrs if not hasattr(index, name)]
            if missing:
                raise TypeError(
                    "Provided index does not implement the required HNSW interface "
                    f"(missing attributes: {', '.join(missing)})."
                )

        self.preprocessor_hnsw = preprocessor_hnsw
        self.preprocessor_model = preprocessor_model
        self.index = index
        self.data_base_hnsw = data_base_hnsw
        self.data_base_original = data_base_original
        self.ef_search = int(ef_search)

        # Logger for hybrid neighbor operations. Handlers are configured at
        # application level (typically to log into a file). To log to console
        # instead, attach a StreamHandler to the "luna_xai" logger or call
        # `configure_luna_logging` accordingly.
        self._logger = get_luna_logger("neighbors.hybrid.manager")
        self._debug_enabled = bool(debug)

    # ------------------------------------------------------------------
    def find_neighbors(
        self,
        instance: pd.Series | pd.DataFrame,
        n_neighbors: int = 7,
        k_factor: int = 10,
    ) -> pd.DataFrame:
        """
        Retrieve the nearest neighbors with Euclidean refinement,
        **without** modifying the HNSW index.

        The method:
        1. Transforms the query instance into the HNSW feature space.
        2. Queries the HNSW index for an oversampled candidate pool of
           size ``k_factor * n_neighbors`` (clipped to the index size).
        3. Computes exact Euclidean distances in the same HNSW space,
           using the preprocessed base data.
        4. Sorts by exact distance and returns the top ``n_neighbors``
           neighbors in the **original** feature space.

        Parameters
        ----------
        instance : pd.Series or pd.DataFrame
            Query instance in the original feature space. If a Series is
            provided, it is treated as a single-row DataFrame.
        n_neighbors : int, default 7
            Number of neighbors to return.
        k_factor : int, default 10
            Oversampling factor for approximate search before re-ranking.

        Returns
        -------
        neighbors : pd.DataFrame
            DataFrame containing neighbors in the original feature space plus
            a ``"distance"`` column (float32) with exact Euclidean distances.

        Raises
        ------
        TypeError
            If ``instance`` is not a pandas Series/DataFrame or if the HNSW
            data cannot be converted to numeric float32.
        ValueError
            If ``n_neighbors`` or ``k_factor`` are not positive.
        RuntimeError
            If the index is not built, base data is missing, or no valid
            neighbor candidates can be found.
        """
        if not isinstance(instance, (pd.Series, pd.DataFrame)):
            raise TypeError(
                "HybridNeighborManager.find_neighbors expects a pandas Series or "
                f"DataFrame as 'instance', got {type(instance)!r}."
            )

        if n_neighbors <= 0:
            raise ValueError("'n_neighbors' must be a positive integer.")
        if k_factor <= 0:
            raise ValueError("'k_factor' must be a positive integer.")

        if self.index is None:
            raise RuntimeError("HybridNeighborManager: HNSW index not built (index is None).")

        if self.data_base_hnsw is None or self.data_base_original is None:
            raise RuntimeError(
                "HybridNeighborManager requires both 'data_base_hnsw' and "
                "'data_base_original' to be set."
            )

        # Normalize input instance
        if isinstance(instance, pd.Series):
            instance_df = ensure_dataframe(instance)
        else:
            instance_df = instance.copy()

        # --- Transform query to HNSW numeric space
        inst_vec = self._transform_to_hnsw(instance_df)  # shape: (1, n_features)

        # Number of candidates to retrieve
        current_count = int(self.index.get_current_count())
        if current_count == 0:
            raise RuntimeError("HNSW index is empty; cannot retrieve neighbors.")

        k_prime = min(max(n_neighbors * k_factor, n_neighbors + 1), current_count)
        self._debug(
            f"Querying HNSW with k_prime={k_prime}, n_neighbors={n_neighbors}, "
            f"k_factor={k_factor}."
        )

        # Query without inserting the instance into the index
        self.index.set_ef(max(self.ef_search, k_prime))
        labels_approx, _ = self.index.knn_query(inst_vec, k=k_prime)

        cand_idxs = labels_approx[0].tolist()
        if not cand_idxs:
            raise RuntimeError("HybridNeighborManager: no valid neighbor candidates found.")

        # Candidate vectors in HNSW space (preprocessed base)
        base_hnsw_subset = self.data_base_hnsw.iloc[cand_idxs]
        cand_vecs = self._ensure_float32_matrix(base_hnsw_subset)
        dists_exact = np.linalg.norm(cand_vecs - inst_vec, axis=1).astype(np.float32)

        # Exclude potential self-matches (distance == 0)
        non_self_mask = dists_exact > 0.0
        if not np.any(non_self_mask):
            raise RuntimeError(
                "All retrieved candidates are identical to the query instance; "
                "cannot form a neighbor set excluding the instance itself."
            )

        cand_idxs = [idx for idx, keep in zip(cand_idxs, non_self_mask) if keep]
        dists_exact = dists_exact[non_self_mask]

        # Select top-n neighbors
        order = np.argsort(dists_exact)
        order = order[: min(n_neighbors, order.size)]
        final_idxs = [cand_idxs[i] for i in order]
        final_dists = dists_exact[order]

        # Return neighbors from ORIGINAL dataset
        neighbors = self.data_base_original.iloc[final_idxs].copy().reset_index(drop=True)
        neighbors["distance"] = final_dists

        self._debug(f"Selected {len(neighbors)} neighbors.")
        return neighbors

    # ------------------------------------------------------------------
    def prepare_neighborhood(
        self,
        instance: pd.Series | pd.DataFrame,
        n_neighbors: int = 7,
        n_synthetic: int = 1000,
        k_factor: int = 3,
        **kwargs: Any,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Build a local analysis neighborhood consisting of:
          - The original instance
          - Real nearest neighbors (refined)
          - Synthetic samples drawn from the neighbor distribution

        The synthetic samples are generated via
        :func:`luna_xai.utils.sampling.create_synthetic_samples`.

        Parameters
        ----------
        instance : pd.Series or pd.DataFrame
            Query instance in the original feature space.
        n_neighbors : int, default 7
            Number of real neighbors to retrieve.
        n_synthetic : int, default 1000
            Number of synthetic samples to generate. If zero, no synthetic
            samples are added.
        k_factor : int, default 3
            Oversampling factor forwarded to :meth:`find_neighbors`.
        **kwargs :
            Additional keyword arguments forwarded to
            ``create_synthetic_samples``. The following special keys
            are recognized:

            - ``method`` : str, default "lhs"
            - ``seed`` : int or None, default None

        Returns
        -------
        combined : pd.DataFrame
            DataFrame containing the original instance, real neighbors
            and synthetic samples.
        context : dict
            Metadata describing the neighborhood, including:

            - ``n_neighbors`` : int
            - ``n_values`` : int (number of synthetic samples)
            - ``k_factor`` : int

        Raises
        ------
        ValueError
            If ``n_neighbors`` or ``n_synthetic`` are negative.
        """
        if n_neighbors <= 0:
            raise ValueError("'n_neighbors' must be a positive integer.")
        if n_synthetic < 0:
            raise ValueError("'n_synthetic' must be a non-negative integer.")
        if k_factor <= 0:
            raise ValueError("'k_factor' must be a positive integer.")

        from luna_xai.utils.sampling import create_synthetic_samples

        # Special kwargs for the sampler (backwards compatible)
        method = kwargs.pop("method", "lhs")
        seed: Optional[int] = kwargs.pop("seed", None)

        self._debug(
            f"Preparing neighborhood with n_neighbors={n_neighbors}, "
            f"n_synthetic={n_synthetic}, k_factor={k_factor}, "
            f"method={method}, seed={seed}."
        )

        neighbors = self.find_neighbors(
            instance=instance,
            n_neighbors=n_neighbors,
            k_factor=k_factor,
        )

        # Generate synthetic samples (optional)
        if n_synthetic > 0:
            synthetic = create_synthetic_samples(
                neighbors,
                n_samples=n_synthetic,
                method=method,
                seed=seed,
                **kwargs,
            )
        else:
            synthetic = neighbors.iloc[0:0].copy()  # empty frame with same columns

        # Normalize original instance shape
        if isinstance(instance, pd.Series):
            instance_df = ensure_dataframe(instance)
        else:
            instance_df = instance.copy()

        combined = pd.concat(
            [instance_df, neighbors, synthetic],
            ignore_index=True,
        ).reset_index(drop=True)

        context: Dict[str, Any] = {
            "n_neighbors": n_neighbors,
            "n_values": n_synthetic,
            "k_factor": k_factor,
        }
        return combined, context

    # ------------------------------------------------------------------
    def _transform_to_hnsw(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply hybrid-space preprocessing (e.g. CANE + scaling) and
        ensure a dense float32 output suitable for HNSW and distance
        computations.

        Parameters
        ----------
        X : pd.DataFrame
            Input data in the original tabular space.

        Returns
        -------
        np.ndarray
            Dense, C-contiguous float32 feature matrix.

        Raises
        ------
        TypeError
            If the transformed data cannot be converted to numeric float32.
        """
        X_trans = self.preprocessor_hnsw.transform(X)
        return self._ensure_float32_matrix(X_trans)

    # ------------------------------------------------------------------
    def _ensure_float32_matrix(self, X: Any) -> np.ndarray:
        """
        Ensure that the input is converted to a dense float32 matrix.

        Parameters
        ----------
        X : array-like or sparse-like
            Input data, possibly sparse.

        Returns
        -------
        np.ndarray
            Dense, C-contiguous float32 matrix.

        Raises
        ------
        TypeError
            If the data cannot be converted to numeric float32.
        """
        if hasattr(X, "toarray"):
            X = X.toarray()

        try:
            arr = np.asarray(X, dtype=np.float32, order="C")
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "HybridNeighborManager requires numeric data (float32) after "
                "preprocessing; received non-numeric values."
            ) from exc

        return arr

    # ------------------------------------------------------------------
    def make_context(
        self,
        instance: pd.Series | pd.DataFrame,
        preds: np.ndarray,
        n_neighbors: int,
        n_values: int,
    ) -> Dict[str, Any]:
        """
        Build a standardized LUNA visualization context.

        Parameters
        ----------
        instance : pd.Series or pd.DataFrame
            Instance that was explained.
        preds : np.ndarray
            Model predictions for the instance (first element is used).
        n_neighbors : int
            Number of real neighbors used in the neighborhood.
        n_values : int
            Number of synthetic values/samples used in the neighborhood.

        Returns
        -------
        dict
            Context dictionary with keys:

            - ``instance``
            - ``instance_prediction``
            - ``n_neighbors``
            - ``n_values``
        """
        instance_pred = preds[0] if len(preds) > 0 else None
        return {
            "instance": instance,
            "instance_prediction": instance_pred,
            "n_neighbors": n_neighbors,
            "n_values": n_values,
        }
