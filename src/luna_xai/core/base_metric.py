"""
BaseMetric
==========

Defines the abstract interface for all metrics in LUNA XAI.

Each metric receives structured sensitivity data (produced by a Method)
and converts it into quantitative measures of feature relevance or model
response strength — typically returned as a MetricResult object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from luna_xai.core.results import MetricResult
from luna_xai.utils.logging_utils import get_luna_logger
from luna_xai.utils.debug_mixin import DebugMixin


class BaseMetric(DebugMixin, ABC):
    """
    Abstract base class for all metrics that quantify local sensitivities
    or feature importances.

    Responsibilities
    -----------------
    - Consume structured sensitivity data produced by a BaseMethod.
    - Quantify how feature perturbations influence model predictions.
    - Return a standardized MetricResult instance.

    Notes
    -----
    Subclasses should implement mathematically consistent scoring mechanisms
    that match the assumptions of the explanation method.
    """

    def __init__(self, debug: bool = False) -> None:
        """
        Initialize the base metric with an optional debug flag.

        Parameters
        ----------
        debug : bool, default False
            If True, enables debug-level logging for this metric instance.
        """
        # Logger for all metric implementations.
        # By default, handlers should be configured to write into a log file.
        # If you prefer console output instead, configure the "luna_xai" logger
        # accordingly (e.g. via `configure_luna_logging`) or attach a
        # StreamHandler manually in your application or demos.
        self._logger = get_luna_logger("metric")
        self._debug_enabled: bool = bool(debug)

   # ------------------------------------------------------------------
    @abstractmethod
    def compute(
        self,
        sensitivity_data: Mapping[str, Mapping[Any, list]],
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute feature importance or response magnitudes from sensitivity data.

        Parameters
        ----------
        sensitivity_data : Mapping[str, Mapping[Any, list]]
            Nested mapping structure produced by a BaseMethod. Example:

            {
                "feature_1": {value_a: [preds...], value_b: [preds...]},
                "feature_2": {value_x: [...], value_y: [...]},
                ...
            }

        **kwargs : Any
            Optional configuration parameters such as weighting, normalization,
            stabilization constants, or ranking strategies.

        Returns
        -------
        MetricResult
            Structured result containing computed feature importance scores.
        """
        raise NotImplementedError(
            "Subclasses must implement 'compute()' to evaluate feature importance."
        )
