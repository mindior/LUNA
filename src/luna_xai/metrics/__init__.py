"""
Metrics Subpackage for LUNA XAI
===============================

Provides all feature-importance metrics used across LUNA XAI.

Purpose
-------
This subpackage contains all algorithms responsible for converting
local sensitivity structures into quantitative feature importances.

Key Responsibilities
--------------------
- Offer standardized metric APIs (via BaseMetric).
- Register all built-in metrics automatically in LUNA's global registry.
- Enable explainers to dynamically select metrics by name using
  the registry mechanism.

Currently Implemented Metrics
-----------------------------
- CHMetric:
    Computes Calinski–Harabasz (CH) separability scores across sensitivity
    groups, then applies directional sign correction based on monotonic
    prediction-level trends.

Notes
-----
Metrics do *not* perform normalization internally. Optional normalization
(e.g., via arctan-based compaction) is available under:
    ``luna_xai.utils.normalization``

All metrics auto-register themselves upon import and become available under:
    registry["metric.<name>"]
"""

# Import metrics to trigger auto-registration through decorators
from luna_xai.metrics.ch_metric import CHMetric

# Ensures registry system is initialized
from luna_xai.core import registry

__all__ = [
    "CHMetric",
]
