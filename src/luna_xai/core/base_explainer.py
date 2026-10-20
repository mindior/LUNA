"""
BaseExplainer
=============

Unified abstract base class for all LUNA explainers.

This class coordinates the high-level explainability workflow, including:
- Domain-driven data preparation.
- Sensitivity method execution.
- Metric computation and validation.
- Optional rendering of explanation results.

All subclasses should implement local and global explanation routines while
leveraging the standardized pipeline helpers provided here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from luna_xai.core.results import MetricResult
from luna_xai.utils.logging_utils import get_luna_logger
from luna_xai.utils.debug_mixin import DebugMixin


class BaseExplainer(DebugMixin, ABC):
    """
    Abstract base class defining the unified interface for all LUNA XAI explainers.

    Responsibilities
    ----------------
    - Coordinate domain preprocessing and hybrid neighborhood construction.
    - Invoke the explanation method to generate sensitivity or intermediate data.
    - Evaluate results through a metric component.
    - Optionally render visual outputs via a renderer.

    This class provides a robust, extensible foundation for both local and
    global explainability techniques.
    """

    def __init__(
        self,
        model: Any,
        domain: Any,
        method: Any,
        metric: Any,
        renderer: Optional[Any] = None,
        output_dir: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize a BaseExplainer with all core components.

        Parameters
        ----------
        model : Any
            Predictive model implementing at least a `predict()` method.
        domain : Any
            Domain instance responsible for preparing inputs and neighborhoods.
        method : Any
            Sensitivity or explanation method.
        metric : Any
            Metric responsible for computing final explanation outputs.
        renderer : Any, optional
            Optional component that renders explanation results.
        output_dir : str, optional
            Optional output directory for rendered artifacts.
        debug : bool, default False
            If True, enables debug-level logging for this explainer.
        """
        self.model = model
        self.domain = domain
        self.method = method
        self.metric = metric
        self.renderer = renderer
        self.output_dir = output_dir

        # Logger for all explainer implementations.
        # By default, handlers should be configured to log to a file.
        # If you prefer console output, you can configure the "luna_xai"
        # logger accordingly (e.g. via `configure_luna_logging`) or attach
        # a StreamHandler manually.
        self._logger = get_luna_logger("explainer")
        self._debug_enabled: bool = bool(debug)

    # ------------------------------------------------------------------
    # Protected local explanation pipeline
    # ------------------------------------------------------------------
    def _explain_local_pipeline(self, instance: Any, **kwargs: Any) -> MetricResult:
        """
        Execute the complete local explanation workflow.

        Steps
        -----
        1. Validate domain and model compatibility.
        2. Ask the domain to prepare a hybrid neighborhood.
        3. Generate sensitivity data via the explanation method.
        4. Compute feature importances using the metric.
        5. Optionally render results.

        Parameters
        ----------
        instance : Any
            Instance to be explained.

        Returns
        -------
        MetricResult
            Final explanation object.
        """
        self._debug(
            f"Starting local explanation for instance of type "
            f"{type(instance).__name__} with kwargs={list(kwargs.keys())}."
        )

        self._validate_domain()
        self._validate_model()

        # Domain preparation
        self._debug("Preparing neighborhood data via domain.")
        neighborhood_data, context = self.domain.prepare_neighborhood(
            instance,
            **kwargs,
        )

        # Sensitivity generation
        self._debug("Generating sensitivity data using the method component.")
        sensitivity_data = self.method.generate(
            self.domain,
            neighborhood_data,
            context=context,
        )

        # Metric computation
        self._debug("Computing metric result from sensitivity data.")
        result = self.metric.compute(sensitivity_data)

        if not isinstance(result, MetricResult):
            raise TypeError(
                f"Metric must return a MetricResult, got {type(result).__name__}."
            )

        # Rendering
        if self.renderer:
            self._debug("Rendering explanation results with renderer component.")
            self.renderer.render(
                result,
                context={"instance": instance, **context},
            )

        self._debug("Local explanation workflow completed successfully.")
        return result

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_domain(self) -> None:
        """
        Ensure that the domain instance exposes the required APIs.

        Raises
        ------
        RuntimeError
            If no valid domain or missing required methods.
        """
        if self.domain is None or not hasattr(self.domain, "prepare_neighborhood"):
            raise RuntimeError(
                "Explainer domain is not properly initialized. Ensure the factory "
                "or constructor provided a valid domain instance."
            )

    def _validate_model(self) -> None:
        """
        Validate that the model exposes the expected prediction interface.

        Requirements
        ------------
        - Must contain a callable `predict()` method.
        - Classification models may optionally expose `predict_proba()`.

        Raises
        ------
        TypeError
            If the model does not implement a valid prediction API.
        """
        predict = getattr(self.model, "predict", None)
        if not callable(predict):
            raise TypeError(
                "Model must implement a callable 'predict()' method."
            )

        # Classification-like models
        is_classification = any(
            hasattr(self.model, attr) for attr in ("classes_", "predict_proba")
        )

        if is_classification and not hasattr(self.model, "predict_proba"):
            raise TypeError(
                "This explainer requires classification models that implement "
                "'predict_proba()'. Probabilistic outputs are mandatory for the "
                "explanation method."
            )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={type(self.model).__name__}, "
            f"domain={type(self.domain).__name__}, "
            f"method={type(self.method).__name__}, "
            f"metric={type(self.metric).__name__}, "
            f"renderer={type(self.renderer).__name__ if self.renderer else 'None'})"
        )
