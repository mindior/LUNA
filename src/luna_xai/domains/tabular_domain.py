# FILE: luna_xai/domains/tabular_domain.py

from __future__ import annotations

from typing import Any, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from luna_xai.core.base_domain import BaseDomain  # ajuste o path se for diferente
from luna_xai.data import DataInterface
from luna_xai.neighbors.hnsw_builder import HNSWBuilder
from luna_xai.neighbors.hybrid_neighbor_manager import HybridNeighborManager
from luna_xai.utils.dataframe import ensure_dataframe
from luna_xai.utils.preprocessing import HybridPreprocessor
from luna_xai.utils.logging_utils import get_luna_logger


class TabularDomain(BaseDomain):
    """
    Minimal domain implementation for tabular data.

    Responsibilities
    ----------------
    - Prepare instances in model-space using DataInterface
    - Run predictions with safe numeric handling
    - Build a hybrid-space (CANE + scaling) representation
    - Construct and expose an HNSW-based neighbor finder
    """

    def __init__(
        self,
        model: Any,
        data_interface: DataInterface,
        categorical_features: list[str],
        use_cane: bool,
        debug: bool = False,
        output_dir: str = ".",
    ) -> None:
        # Initialize BaseDomain: define generic _logger (“domain”) + _debug_enabled
        super().__init__(debug=debug)

        # Especializa logger para este domínio
        self._logger = get_luna_logger("domain.tabular")

        self.model = model
        self.data_interface = data_interface
        self.categorical_features = categorical_features or []
        self.use_cane = use_cane
        self.output_dir = output_dir

        # Populated by build_index() or TabularDomainManager.load()
        self.hybrid_preprocessor: Optional[HybridPreprocessor] = None
        self.index: Any = None
        self.hnsw_builder: Optional[HNSWBuilder] = None
        self.neighbor_manager: Optional[HybridNeighborManager] = None
        self.neighbor_finder: Optional[HybridNeighborManager] = None

        self.data_original: Optional[pd.DataFrame] = None
        self.data_hnsw: Optional[pd.DataFrame] = None

    # -------------------------------------------------------------
    # Model-space preparation
    # -------------------------------------------------------------
    def prepare(self, instance: Any) -> Any:
        """
        Convert a raw instance to model-space representation using DataInterface.
        """
        df = ensure_dataframe(instance)
        return self.data_interface.transform_input(df)

    # -------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------
    def predict(self, X: Any) -> np.ndarray:
        """
        Run prediction in model-space with strict numeric enforcement.

        Steps
        -----
        1. Transform X via DataInterface
        2. Convert sparse → dense (if needed)
        3. Ensure numeric dtype
        4. Cast to float32 and predict
        5. Optionally inverse-transform labels via DataInterface
        """
        if self.data_interface is None or self.data_interface.preprocessor is None:
            raise RuntimeError("DataInterface.preprocessor is not defined.")

        self._debug("TabularDomain.predict(): transforming input for prediction.")
        X_proc = self.data_interface.transform_input(X)

        # Sparse → dense
        if hasattr(X_proc, "toarray"):
            self._debug("TabularDomain.predict(): converting sparse matrix to dense array.")
            X_proc = X_proc.toarray()

        arr = np.asarray(X_proc)

        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(
                f"TabularDomain.predict(): model-space data must be numeric. "
                f"Received dtype={arr.dtype}. Check your preprocessing pipeline."
            )

        # float32 for performance and consistency
        X_proc = np.asarray(arr, dtype=np.float32, order="C")
        self._debug(
            f"TabularDomain.predict(): calling model.predict() on shape={X_proc.shape}, dtype={X_proc.dtype}."
        )
        preds = self.model.predict(X_proc)

        # Use the DataInterface helper to handle inverse_transform output
        try:
            preds = self.data_interface.inverse_transform_output(preds)
        except Exception:
            # Maintain tolerant behavior; any problems here should not break the rating.
            pass

        return preds

    # -------------------------------------------------------------
    # Hybrid preprocessing + index construction
    # -------------------------------------------------------------
    def build_index(self, X_train: pd.DataFrame) -> None:
        """
        Build the hybrid representation and construct the HNSW index.

        This is the default path when the user does **not** use
        TabularDomainManager and wants everything built on the fly.
        """
        self._debug("TabularDomain.build_index(): building hybrid-space representation.")

        if not isinstance(X_train, pd.DataFrame):
            raise TypeError(
                "TabularDomain.build_index expects X_train to be a pandas.DataFrame."
            )

        # Hybrid encoder (CANE + scaler)
        self.hybrid_preprocessor = HybridPreprocessor(
            categorical_features=self.categorical_features,
            use_cane=self.use_cane,
            verbose=self._debug_enabled,
        )

        X_hnsw, _, _ = self.hybrid_preprocessor.fit_transform(X_train)

        self.data_hnsw = X_hnsw.reset_index(drop=True)
        self.data_original = X_train.reset_index(drop=True)

        # HNSWBuilder with unified API: index = build_index(X)
        self.hnsw_builder = HNSWBuilder(space="l2")
        self.index = self.hnsw_builder.build_index(self.data_hnsw)
        dim = self.hnsw_builder.dim

        self._debug(
            f"TabularDomain.build_index(): HNSW index built with "
            f"{len(X_train)} samples (dim={dim})."
        )

        # Neighbor manager wired with both hybrid and original spaces
        self.neighbor_manager = HybridNeighborManager(
            preprocessor_hnsw=self.hybrid_preprocessor,
            preprocessor_model=self.data_interface.preprocessor,
            index=self.index,
            data_base_hnsw=self.data_hnsw,
            data_base_original=self.data_original,
        )

        self.neighbor_finder = self.neighbor_manager

    # -------------------------------------------------------------
    # Neighborhood preparation (for explainers)
    # -------------------------------------------------------------
    def prepare_neighborhood(
        self,
        instance: Any,
        n_neighbors: int = 7,
        n_synthetic: int = 1000,
        **kwargs: Any,
    ) -> Tuple[Any, Dict]:
        """
        Prepare a neighborhood around a given instance for sensitivity analysis.

        This delegates to the underlying HybridNeighborManager, if available.

        Returns
        -------
        Tuple[Any, Dict]
            (neighborhood_data, context_dict)

        Raises
        ------
        RuntimeError
            If the neighbor manager has not been initialized.
        NotImplementedError
            If the HybridNeighborManager does not implement a compatible API.
        """
        if self.neighbor_manager is None:
            raise RuntimeError(
                "Neighbor manager not initialized. "
                "Call build_index() or use TabularDomainManager.load() first."
            )

        # We delegate defensively: we only call if the method exists.
        # Adjust the method name if HybridNeighborManager has a different API.
        if hasattr(self.neighbor_manager, "prepare_neighborhood"):
            self._debug(
                f"TabularDomain.prepare_neighborhood(): delegating to "
                f"HybridNeighborManager.prepare_neighborhood(n_neighbors={n_neighbors}, "
                f"n_synthetic={n_synthetic})."
            )
            return self.neighbor_manager.prepare_neighborhood(  # type: ignore[no-any-return]
                instance,
                n_neighbors=n_neighbors,
                n_synthetic=n_synthetic,
                **kwargs,
            )

        raise NotImplementedError(
            "HybridNeighborManager does not implement 'prepare_neighborhood'. "
            "Update TabularDomain.prepare_neighborhood() to match its API."
        )

    # -------------------------------------------------------------
    # Neighbor accessor
    # -------------------------------------------------------------
    def get_neighbor_finder(self) -> HybridNeighborManager:
        """
        Expose the internal neighbor manager (HybridNeighborManager).

        Raises
        ------
        RuntimeError
            If the neighbor_finder has not been initialized yet.
        """
        if self.neighbor_finder is None:
            raise RuntimeError("Neighbor finder not initialized.")
        return self.neighbor_finder
