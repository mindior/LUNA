from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from luna_xai.utils.logging_utils import get_luna_logger
from luna_xai.utils.debug_mixin import DebugMixin


class BaseDomain(DebugMixin, ABC):
    """
    Abstract base class defining the core interface for all data domains
    (e.g., tabular, image, text).

    This class establishes the minimal contract required by the LUNA XAI
    framework to ensure consistency across different domain implementations.
    Subclasses must provide preprocessing logic, prediction handling, and
    optional neighborhood-based reasoning.

    Core Responsibilities
    ---------------------
    - Transform raw input instances into model-consumable representations.
    - Produce predictions in model space.
    - Optionally manage neighborhood structures for sensitivity methods
      such as DSA.
    """

    def __init__(self, debug: bool = False) -> None:
        """
        Initialize the base domain with an optional debug flag.

        Parameters
        ----------
        debug : bool, default False
            If True, enables debug-level logging for this domain.
        """
        # Base logger for all domain implementations.
        # By default, logging is configured to write to a file; if you prefer
        # console output, configure the handler accordingly in your application
        # or via `configure_luna_logging` (see logging_utils).
        self._logger = get_luna_logger("domain")
        self._debug_enabled: bool = bool(debug)

    # ------------------------------------------------------------------
    @abstractmethod
    def prepare(self, instance: Any) -> Any:
        """
        Prepare a single raw instance so it can be consumed by the predictive model.

        Notes
        -----
        Each domain type is responsible for defining appropriate preprocessing
        transformations, such as:
        - Tabular: scaling, encoding, or imputation
        - Image: resizing, normalization, or channel reordering
        - Text: tokenization, embedding, or sequence construction

        Parameters
        ----------
        instance : Any
            Raw input instance.

        Returns
        -------
        Any
            Processed representation suitable for model inference.
        """
        raise NotImplementedError("Subclasses must implement 'prepare()'.")

    # ------------------------------------------------------------------
    @abstractmethod
    def predict(self, X: Any) -> Any:
        """
        Generate model predictions given preprocessed input.

        Notes
        -----
        This method abstracts the prediction mechanism to ensure consistency
        across domains. Subclasses may internally handle:
        - Direct calls to underlying ML models
        - Additional transformations needed before or after prediction

        Parameters
        ----------
        X : Any
            Preprocessed input ready for model consumption.

        Returns
        -------
        Any
            Model prediction(s).
        """
        raise NotImplementedError("Subclasses must implement 'predict()'.")

    # ------------------------------------------------------------------
    def build_index(self, X_train: Any) -> None:
        """
        Optionally build a neighborhood index (e.g., HNSW, FAISS) for the dataset.

        Notes
        -----
        This method is only relevant for domains supporting neighborhood-based
        reasoning. If not applicable, subclasses are not required to override it.

        Parameters
        ----------
        X_train : Any
            Training dataset used to construct the index.

        Raises
        ------
        NotImplementedError
            If the domain does not support index construction.
        """
        raise NotImplementedError(
            "This domain does not support index building."
        )

    # ------------------------------------------------------------------
    def prepare_neighborhood(
        self,
        instance: Any,
        n_neighbors: int = 7,
        n_synthetic: int = 1000,
        **kwargs: Any,
    ) -> Tuple[Any, Dict]:
        """
        Prepare a neighborhood around a given instance for sensitivity analysis.

        Notes
        -----
        Domains supporting hybrid neighborhood reasoning (e.g., tabular) should
        override this method. The neighborhood may include a combination of real
        and synthetic samples.

        Parameters
        ----------
        instance : Any
            Target instance for which the neighborhood will be constructed.
        n_neighbors : int, default=7
            Number of real nearest neighbors to retrieve.
        n_synthetic : int, default=1000
            Number of synthetic samples to generate.
        **kwargs
            Domain-specific parameters.

        Returns
        -------
        Tuple[Any, Dict]
            A tuple containing:
            - Neighborhood dataset
            - Auxiliary metadata (e.g., encoders, scalers)

        Raises
        ------
        NotImplementedError
            If the domain does not support neighborhood preparation.
        """
        raise NotImplementedError(
            "Neighborhood preparation is not implemented for this domain."
        )

    # ------------------------------------------------------------------
    def get_neighbor_finder(self) -> Any:
        """
        Optionally return the internal neighbor search structure.

        Notes
        -----
        Methods such as DSA rely on efficient nearest-neighbor queries. Domains
        without such mechanisms should not override this method.

        Returns
        -------
        Any
            Underlying neighbor finder implementation.

        Raises
        ------
        NotImplementedError
            If the domain does not expose a neighbor search component.
        """
        raise NotImplementedError(
            "This domain does not expose a neighbor finder."
        )
