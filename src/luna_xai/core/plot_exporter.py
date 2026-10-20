# luna_xai/core/plot_exporter.py

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from luna_xai.utils.logging_utils import get_luna_logger

LOGGER = get_luna_logger("plot.exporter")


class PlotExportStrategy(ABC):
    @abstractmethod
    def save(self, fig: Any, filepath: str) -> None:
        """
        Save the given matplotlib figure to a file or otherwise handle it.
        """
        raise NotImplementedError


class _MatplotlibExporterBase(PlotExportStrategy):
    extension: str = ""
    format_name: str = ""

    def save(self, fig: Any, filepath: str) -> None:
        if fig is None:
            raise ValueError("Cannot save figure: 'fig' is None.")

        if not isinstance(filepath, str):
            raise TypeError(f"'filepath' must be a string, got {type(filepath).__name__}")

        path = Path(filepath).with_suffix(f".{self.extension}")

        LOGGER.debug(
            "Saving figure using %s exporter to '%s' (format=%s).",
            self.__class__.__name__,
            str(path),
            self.format_name,
        )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                path,
                format=self.format_name,
                dpi=300,
                bbox_inches="tight",
            )
            LOGGER.debug("Figure successfully saved to '%s'.", str(path))
        except Exception as exc:
            LOGGER.error("Failed to save figure to '%s': %s", str(path), exc)
            raise RuntimeError(f"Failed to save figure to '{path}': {exc}") from exc
        finally:
            plt.close(fig)

class MatplotlibPdfExporter(_MatplotlibExporterBase):
    extension = "pdf"
    format_name = "pdf"


class MatplotlibPngExporter(_MatplotlibExporterBase):
    extension = "png"
    format_name = "png"


class MatplotlibEpsExporter(_MatplotlibExporterBase):
    extension = "eps"
    format_name = "eps"


class MatplotlibShowExporter(PlotExportStrategy):
    """
    Exporter that only shows the figure (does not save).
    """

    def save(self, fig: Any, filepath: str | None = None) -> None:
        if fig is None:
            raise ValueError("Cannot show figure: 'fig' is None.")

        try:
            try:
                from IPython.display import display  # type: ignore
                display(fig)
            except Exception:
                fig.show()
        finally:
            plt.close(fig)

class CompositePlotExporter(PlotExportStrategy):
    def __init__(self, exporters: list[PlotExportStrategy]):
        self.exporters = exporters

    def save(self, fig: Any, filepath: str) -> None:
        for exporter in self.exporters:
            exporter.save(fig, filepath)

# -------------------------------------------------------------------------
# Inline exporter (Jupyter)
# -------------------------------------------------------------------------


class MatplotlibInlineExporter(PlotExportStrategy):
    """
    Exporter that does NOT save to a file, only displays the image inline.
    """

    extension: str | None = None  # just for consistency, it is not used

    def save(self, fig: Any, filepath: str | None = None) -> None:  # type: ignore[override]
        if fig is None:
            raise ValueError("Cannot display figure inline: 'fig' is None.")

        try:
            # Ideal use on notebooks
            try:
                from IPython.display import display  # type: ignore

                display(fig)
                LOGGER.debug(
                    "MatplotlibInlineExporter: figure displayed inline "
                    "(ignored filepath='%s').",
                    filepath,
                )
            except Exception:
                # Fallback if not in Jupyter
                fig.show()
                LOGGER.debug(
                    "MatplotlibInlineExporter: figure shown via fig.show() "
                    "(ignored filepath='%s').",
                    filepath,
                )
        except Exception as exc:
            LOGGER.error("Failed to display figure inline: %s", exc)
            raise RuntimeError(f"Failed to display figure inline: {exc}") from exc

