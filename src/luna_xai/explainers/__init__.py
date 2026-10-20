"""
LUNA XAI — Explainability Subpackage
====================================

Provides the high-level explainer interfaces used across the LUNA system.
An explainer orchestrates the complete explainability pipeline:

    Domain → Method → Metric → Renderer

This subpackage exposes:
    - TabularRegressionExplainer
    - TabularClassificationExplainer
    - ExplainerFactory (registry-based dynamic constructor)

Every explainer registers itself automatically under:
    explainer.<task>

Currently implemented explainers:
    • explainer.regression    → TabularRegressionExplainer
    • explainer.classification → TabularClassificationExplainer

"""

# ---------------------------------------------------------------------
#  Base APIs
# ---------------------------------------------------------------------
from luna_xai.core.base_explainer import BaseExplainer

# ---------------------------------------------------------------------
#  Concrete explainers (auto-registered on import)
# ---------------------------------------------------------------------
from luna_xai.explainers.tabular_regression import TabularRegressionExplainer
from luna_xai.explainers.tabular_classification import TabularClassificationExplainer

# ---------------------------------------------------------------------
#  Factory for dynamic construction
# ---------------------------------------------------------------------
from luna_xai.explainers.factory import ExplainerFactory

# ---------------------------------------------------------------------
#  Ensure registry system is imported (auto-discovery handled in core)
# ---------------------------------------------------------------------
from luna_xai.core import registry

# ---------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------
__all__ = [
    "BaseExplainer",
    "TabularRegressionExplainer",
    "TabularClassificationExplainer",
    "ExplainerFactory",
]