"""
Renderers Subpackage for LUNA XAI
=================================

Responsible for visualization and rendering of explainability results.

Includes
--------
- PlotRenderer: default visual renderer for feature importances, trends,
  and boxplots.

Notes
-----
All renderers automatically register themselves in the global LUNA registry
upon import, ensuring discovery through the `auto_discover_core_components()`
mechanism in `luna_xai.core`.
"""

from .plot_renderer import PlotRenderer
# Importing the registry module ensures that the global registration
# system is initialized before any auto-register calls run.
from luna_xai.core import registry  # noqa: F401  (used for side-effect)

__all__ = [
    "PlotRenderer",
]
