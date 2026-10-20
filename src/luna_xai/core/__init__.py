"""
LUNA XAI — Core Package
=======================

Provides the fundamental abstractions and the global registry powering
the entire LUNA XAI architecture.

This module exposes:
- Base abstractions (Domain, Method, Metric, Renderer, Explainer)
- Unified explainer interface (LunaExplainer)
- Global component registry (discoverable by namespace.name)
- Plot exporting strategies (PDF/PNG/EPS)
- Optional utilities (normalization functions)

The core package must remain lightweight, stable, and free of
side effects that could interfere with user applications or IDEs.
"""

# ---------------------------------------------------------------------
# Base abstractions
# ---------------------------------------------------------------------
from luna_xai.core.base_domain import BaseDomain
from luna_xai.core.base_explainer import BaseExplainer
from luna_xai.core.base_method import BaseMethod
from luna_xai.core.base_metric import BaseMetric
from luna_xai.core.base_renderer import BaseRenderer
from luna_xai.core.luna_explainer import LunaExplainer

# ---------------------------------------------------------------------
# Plot exporting strategies
# ---------------------------------------------------------------------
from luna_xai.core.plot_exporter import (
    PlotExportStrategy,
    MatplotlibPdfExporter,
    MatplotlibPngExporter,
    MatplotlibEpsExporter,
)

# ---------------------------------------------------------------------
# Global registry system
# ---------------------------------------------------------------------
from luna_xai.core.registry import (
    REGISTRY,
    register_component,
    get_component,
    auto_discover_core_components,
)

# ---------------------------------------------------------------------
# Optional utilities (normalization)
# ---------------------------------------------------------------------
try:
    from luna_xai.utils.normalization import arctan_normalize
except Exception:
    # Optional dependency — safely ignore if missing
    arctan_normalize = None

# ---------------------------------------------------------------------
# Internal auto-discovery
# ---------------------------------------------------------------------
# NOTE:
# Auto-discovery is intentionally non-blocking and should never raise.
# If a module fails to import (e.g., missing dependency), discovery is skipped.
try:
    auto_discover_core_components(verbose=False)
except Exception:
    pass

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
__all__ = [
    # Base abstractions
    "BaseDomain",
    "BaseExplainer",
    "BaseMethod",
    "BaseMetric",
    "BaseRenderer",
    "LunaExplainer",

    # Plot exporters
    "PlotExportStrategy",
    "MatplotlibPdfExporter",
    "MatplotlibPngExporter",
    "MatplotlibEpsExporter",

    # Registry system
    "REGISTRY",
    "register_component",
    "get_component",
    "auto_discover_core_components",

    # Optional utilities
    "arctan_normalize",
]
