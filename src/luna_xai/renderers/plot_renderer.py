# luna_xai/renderers/plot_renderer.py

from __future__ import annotations

import os
from pyexpat import features
from typing import Any, Dict, Optional, List

from luna_xai.core.base_renderer import BaseRenderer
from luna_xai.core.results import MetricResult
from luna_xai.core.plot_exporter import (
    MatplotlibPdfExporter,
    MatplotlibShowExporter,
    CompositePlotExporter,
    PlotExportStrategy,
)
from luna_xai.utils.plotting import (
    plot_importances,
    plot_feature_trends,
    plot_feature_boxplots,
)
from luna_xai.utils.logging_utils import get_luna_logger


class PlotRenderer(BaseRenderer):
    """
    Renderer for LUNA XAI plots.

    This renderer delegates the actual plotting to utility functions in
    ``luna_xai.utils.plotting`` and uses a ``PlotExportStrategy`` to
    persist the generated figures (Matplotlib-based by default).

    Parameters
    ----------
    plot_type : {"importance", "line", "boxplot"}, default "importance"
        Default type of plot to render when calling :meth:`render`.
    output_dir : str, default "results"
        Base directory where plots will be saved. Subdirectories per
        instance (``idx``) are created automatically.
    exporter : PlotExportStrategy, optional
        Strategy object responsible for saving Matplotlib figures.
        If None, :class:`MatplotlibPdfExporter` is used.
    features : list of str, optional
        Subset of feature names to be plotted in line/boxplot modes.
        If None, all available features in the sensitivity data are used.
    top_k_importances : int, default 5
        Number of top features to highlight in the importance plot.
    debug : bool, default False
        If True, enables lightweight debug messages via ``_debug()``.
    show : bool, default False
        If True, display the graph

    Notes
    -----
    Logging for this renderer is integrated with the global LUNA logging
    infrastructure. By default, handlers are configured to log to a file
    (see ``configure_luna_logging``). If you prefer console logging, attach
    a ``logging.StreamHandler`` to the ``"luna_xai"`` logger instead.
    """

    VALID_PLOT_TYPES = {"importance", "line", "boxplot"}

    def __init__(
        self,
        plot_type: str = "importance",
        output_dir: str = "results",
        exporter: Optional[PlotExportStrategy] = None,
        features: Optional[List[str]] = None,
        top_k_importances: int = 5,
        debug: bool = False,
        show: bool = False, 
    ) -> None:
        plot_type_normalized = plot_type.lower()
        if plot_type_normalized not in self.VALID_PLOT_TYPES:
            raise ValueError(
                f"[PlotRenderer] Invalid plot_type '{plot_type}'. "
                f"Expected one of {sorted(self.VALID_PLOT_TYPES)}."
            )

        if top_k_importances <= 0:
            raise ValueError(
                "[PlotRenderer] 'top_k_importances' must be a positive integer."
            )

        # Initialize BaseRenderer / DebugMixin
        super().__init__(debug=debug)

        self.show: bool = show
        self.plot_type: str = plot_type_normalized
        self.output_dir: str = output_dir

        if self.show:
            self.exporter = CompositePlotExporter(
                [
                    exporter or MatplotlibPdfExporter(),
                    MatplotlibShowExporter(),
                ]
            )
        
        else:
            
            self.exporter = exporter or MatplotlibPdfExporter()

        self.features: Optional[List[str]] = features
        self.top_k_importances: int = int(top_k_importances)

        # Use a more specific logger name than the base "renderer"
        self._logger = get_luna_logger("renderer.plot")

        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def render(self, result: MetricResult, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Render a plot from a :class:`MetricResult` object.

        Parameters
        ----------
        result : MetricResult
            Result object containing importances and/or sensitivity data.
        context : dict, optional
            Extra metadata used to customize the plot, typically including:

            - ``scope`` : {"local", "global"}, default "local"
            - ``instance`` : object, optional
            - ``idx`` : str, default "instance"
            - ``instance_prediction`` : float, optional

        Raises
        ------
        TypeError
            If ``result`` is not a :class:`MetricResult` instance.
        ValueError
            If the configured ``plot_type`` is invalid.
        """
        if not isinstance(result, MetricResult):
            raise TypeError(
                f"[PlotRenderer] Expected MetricResult, got {type(result).__name__}."
            )

        context = context or {}
        scope = context.get("scope", "local")
        instance = context.get("instance")
        idx = context.get("idx", "instance")
        instance_pred = context.get("instance_prediction")

        self._logger.info(
            "Rendering %s explanation as '%s' plot (idx=%s).",
            scope.upper(),
            self.plot_type,
            idx,
        )

        self._debug(
            f"Rendering {scope.upper()} explanation as '{self.plot_type}' plot "
            f"(idx={idx})."
        )

        if self.plot_type == "importance":
            self._render_importances(result, idx)
        elif self.plot_type == "line":
            self._render_lineplots(result.sensitivity_data, instance, idx)
        elif self.plot_type == "boxplot":
            self._render_boxplots(result.sensitivity_data, instance, idx, instance_pred)
        else:
            # Should be unreachable due to __init__ validation
            raise ValueError(f"[PlotRenderer] Invalid plot_type '{self.plot_type}'.")

    # ------------------------------------------------------------------
    def render_from_data(
        self,
        sensitivity_data: Dict[str, Dict[Any, List[float]]],
        plot_type: Optional[str] = None,
        instance: Optional[Any] = None,
        idx: str = "instance",
        instance_prediction: Optional[float] = None,
    ) -> None:
        """
        Render plots directly from raw sensitivity data.

        Parameters
        ----------
        sensitivity_data : dict
            Nested dictionary of sensitivity values, typically in the same
            format produced by the sensitivity methods (e.g. DSA).
        plot_type : {"line", "boxplot"}, optional
            Plot mode to use. If None, falls back to the renderer's
            default ``plot_type``.
        instance : object, optional
            Instance associated with the sensitivity data.
        idx : str, default "instance"
            Identifier used to name files/directories for this instance.
        instance_prediction : float, optional
            Model prediction for the instance, used in boxplots.

        Notes
        -----
        - This method does **not** support ``"importance"``; that mode
          is only available via :meth:`render` using a :class:`MetricResult`.
        """
        if not sensitivity_data:
            self._debug("[PlotRenderer] No sensitivity data to render.")
            return

        effective_plot_type = (plot_type or self.plot_type).lower()
        self._logger.info(
            "Rendering raw sensitivity data as '%s' (idx=%s).",
            effective_plot_type,
            idx,
        )
        self._debug(
            f"Rendering raw sensitivity data as '{effective_plot_type}' "
            f"(idx={idx})."
        )

        if effective_plot_type == "line":
            self._render_lineplots(sensitivity_data, instance, idx)
        elif effective_plot_type == "boxplot":
            self._render_boxplots(sensitivity_data, instance, idx, instance_prediction)
        else:
            raise ValueError(
                f"[PlotRenderer] Unsupported plot_type '{effective_plot_type}' "
                "for render_from_data (only 'line' and 'boxplot' are allowed)."
            )

    # ------------------------------------------------------------------
    def _render_importances(self, result: MetricResult, idx: str, show: bool = False) -> None:
        """Internal helper to render importance plots."""
        importances = result.importances

        if not importances:
            self._debug("[PlotRenderer] No importances to plot.")
            return

        self._debug(
            f"Rendering importance plot for idx={idx} with "
            f"{len(importances)} features (top_k={self.top_k_importances})."
        )

        fig = plot_importances(
            idx=idx,
            importances_dict=importances,
            type_="local",
            top_k=self.top_k_importances,
        )

        save_path = f"{self.output_dir}/{idx}/importances"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.exporter.save(fig, save_path)

    # ------------------------------------------------------------------
    def _render_lineplots(
        self,
        sensitivity_data: Dict[str, Dict[Any, List[float]]],
        instance: Optional[Any],
        idx: str,
        show: bool = False,
    ) -> None:
        """Internal helper to render line plots of feature trends."""
        self._debug(
            f"Rendering line plots for idx={idx}, "
            f"features={self.features if self.features is not None else 'ALL'}."
        )

        figs = plot_feature_trends(
            idx=idx,
            sensitivity_data=sensitivity_data,
            instance=instance,
            features=self.features,
        )

        save_dir = f"{self.output_dir}/{idx}"
        os.makedirs(save_dir, exist_ok=True)

        for feature, fig in figs.items():
            self.exporter.save(fig, f"{save_dir}/lineplot_{feature}")

    # ------------------------------------------------------------------
    def _render_boxplots(
        self,
        sensitivity_data: Dict[str, Dict[Any, List[float]]],
        instance: Optional[Any],
        idx: str,
        instance_prediction: Optional[float],
        show: bool = False,
    ) -> None:
        """Internal helper to render boxplots of feature effects."""
        self._debug(
            f"Rendering boxplots for idx={idx}, "
            f"features={self.features if self.features is not None else 'ALL'}, "
            f"instance_prediction={instance_prediction}."
        )

        figs = plot_feature_boxplots(
            idx=idx,
            predictions=sensitivity_data,
            instance_prediction=instance_prediction,
            instance=instance,
            features=self.features,
        )

        save_dir = f"{self.output_dir}/{idx}"
        os.makedirs(save_dir, exist_ok=True)

        for feature, fig in figs.items():
            self.exporter.save(fig, f"{save_dir}/boxplot_{feature}")


# Auto-register
from luna_xai.core.registry import register_component

register_component("renderer", "plot", PlotRenderer)
