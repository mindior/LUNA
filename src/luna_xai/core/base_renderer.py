"""
BaseRenderer
============

Abstract base class for all LUNA XAI renderers.

This interface standardizes visualization behavior across explainers by
providing a unified rendering entry point and a dispatch mechanism for
multiple plot types (e.g., importance bars, line plots, boxplots).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from luna_xai.core.results import MetricResult
from luna_xai.utils.logging_utils import get_luna_logger
from luna_xai.utils.debug_mixin import DebugMixin


class BaseRenderer(DebugMixin, ABC):
    """
    Abstract renderer supporting multiple visualization types.

    Responsibilities
    -----------------
    - Provide a unified `render()` method for explanation output.
    - Dispatch visualization requests to specialized rendering methods.
    - Offer extensibility for custom plot types in concrete subclasses.

    Notes
    -----
    Subclasses should implement at least `render_importance()`. Additional
    rendering styles (line plots, boxplots, etc.) remain optional but must be
    implemented if requested via `plot_type`.
    """

    def __init__(self, debug: bool = False) -> None:
        """
        Initialize the base renderer with an optional debug flag.

        Parameters
        ----------
        debug : bool, default False
            If True, enables debug-level logging for this renderer instance.
        """
        # Logger for all renderer implementations.
        # By default, logging handlers should be configured to write into a log file
        # (e.g., via a FileHandler attached to the "luna_xai" logger). If you prefer
        # console output instead, you can either:
        #
        # - configure the "luna_xai" logger with a StreamHandler, or
        # - use `configure_luna_logging` in your application or demos.
        self._logger = get_luna_logger("renderer")
        self._debug_enabled: bool = bool(debug)

    # ------------------------------------------------------------------
    @abstractmethod
    def render(
        self,
        result: MetricResult,
        context: Optional[Dict[str, Any]] = None,
        plot_type: str = "importance",
    ) -> None:
        """
        Unified entry point for rendering explanation results.

        Parameters
        ----------
        result : MetricResult
            Computed explanation object containing feature importances or
            sensitivity data.
        context : dict, optional
            Additional metadata such as the original instance, neighborhood
            statistics, or domain-specific artifacts.
        plot_type : str, default="importance"
            Type of visualization to generate. Supported options include:
            - "importance" : Bar chart of feature importances.
            - "line"       : Trend or sensitivity line plot.
            - "boxplot"    : Distribution of predictions or scores.
        """
        raise NotImplementedError(
            "Subclasses must implement 'render()' as the public entry point."
        )

    # ------------------------------------------------------------------
    def _dispatch_render(
        self,
        plot_type: str,
        result: MetricResult,
        context: Dict[str, Any],
    ) -> None:
        """
        Internal dispatcher for plot-type-specific rendering.

        Parameters
        ----------
        plot_type : str
            Requested visualization type.
        result : MetricResult
            Computed metric results.
        context : dict
            Additional auxiliary metadata.

        Raises
        ------
        ValueError
            If an unsupported `plot_type` is requested.
        """
        normalized = plot_type.lower()

        if normalized == "importance":
            return self.render_importance(result, context)
        if normalized == "line":
            return self.render_line(result, context)
        if normalized == "boxplot":
            return self.render_boxplot(result, context)

        raise ValueError(
            f"Unknown plot_type '{plot_type}'. Expected one of: 'importance', 'line', 'boxplot'."
        )

    # ------------------------------------------------------------------
    # Rendering stubs — must be implemented by subclasses
    # ------------------------------------------------------------------
    def render_importance(self, result: MetricResult, context: Dict[str, Any]) -> None:
        """
        Render a bar chart of feature importances.
        """
        raise NotImplementedError(
            "Subclasses must implement 'render_importance()'."
        )

    def render_line(self, result: MetricResult, context: Dict[str, Any]) -> None:
        """
        Render a line plot representing sensitivity patterns.
        """
        raise NotImplementedError(
            "Subclasses must implement 'render_line()' to support line plots."
        )

    def render_boxplot(self, result: MetricResult, context: Dict[str, Any]) -> None:
        """
        Render a boxplot visualization of prediction distributions.
        """
        raise NotImplementedError(
            "Subclasses must implement 'render_boxplot()' to support boxplots."
        )
