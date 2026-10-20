"""
BaseMethod
==========

Defines the abstract interface for all sensitivity-analysis or
perturbation-based explanation methods in LUNA XAI.

Each method operates on data prepared by the Domain (e.g., a hybrid
neighborhood around an instance) and produces a structured mapping of
feature-level sensitivities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional

from luna_xai.utils.logging_utils import get_luna_logger
from luna_xai.utils.debug_mixin import DebugMixin


class BaseMethod(DebugMixin, ABC):
    """
    Abstract base class for all sensitivity analysis methods.

    Responsibilities
    -----------------
    - Consume neighborhood data produced by the Domain.
    - Apply perturbations, evaluations, or sampling strategies.
    - Produce structured sensitivity mappings for consumption by a Metric.

    Notes
    -----
    Subclasses are expected to implement domain-specific perturbation logic,
    ensuring compatibility with the model interface exposed by the Domain.
    """

    def __init__(self, debug: bool = False) -> None:
        """
        Initialize the base method with an optional debug flag.

        Parameters
        ----------
        debug : bool, default False
            If True, enables debug-level logging for this method instance.
        """
        # Logger for all method implementations.
        # By default, handlers should be configured to log into a file.
        # If you prefer console output instead, you can configure the
        # "luna_xai" logger accordingly (e.g. via `configure_luna_logging`)
        # or attach a StreamHandler manually in your application or demos.
        self._logger = get_luna_logger("method")
        self._debug_enabled: bool = bool(debug)

    # ------------------------------------------------------------------
    @abstractmethod
    def generate(
        self,
        domain: Any,
        data: Any,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Mapping[str, Mapping[Any, list]]:
        """
        Generate structured sensitivity mappings from precomputed neighborhood data.

        This method orchestrates the core mechanism of a sensitivity analysis
        technique. Implementations may perturb features individually, evaluate
        model responses under constrained variations, or extract local stability
        patterns.

        Parameters
        ----------
        domain : BaseDomain
            Domain instance providing access to preprocessing utilities and
            model prediction methods.
        data : Any
            Hybrid neighborhood dataset or structure produced by the domain.
        context : dict, optional
            Additional metadata such as distances, original instance details, or
            domain-specific auxiliary information.
        **kwargs : Any
            Method-specific configuration parameters.

        Returns
        -------
        Mapping[str, Mapping[Any, list]]
            Nested mapping describing feature-level sensitivities.
            Example structure:

            {
                "feature_1": {
                    value_a: [pred_1, pred_2, ...],
                    value_b: [pred_1, pred_2, ...],
                },
                "feature_2": {
                    value_x: [...],
                    ...
                },
            }
        """
        raise NotImplementedError(
            "Subclasses must implement 'generate()' to produce sensitivity mappings."
        )
