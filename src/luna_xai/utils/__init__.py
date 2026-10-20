"""
Utility Subpackage for LUNA XAI
===============================

Provides shared helper modules used across the LUNA ecosystem:

- I/O operations (saving/loading models and datasets, safe formatting)
- Preprocessing utilities (HybridPreprocessor: CANE + Scaler)
- Mathematical/statistical metrics (Calinski–Harabasz separability)
- Normalization transforms (e.g., arctan-based compacting)
- Sampling utilities (empirical or Latin Hypercube)
- Plotting and rendering utilities for explainability
- Logging utilities (timing and profiling of code execution)

Notes
-----
These are *infrastructure-level* tools. They are **not** registered
in the LUNA registry but are used internally by Domains, Methods,
Metrics, and Explainers.
"""

# ---------------------------------------------------------------------
# Absolute Imports — Stable and Explicit
# ---------------------------------------------------------------------
from luna_xai.utils.io_utils import (
    ensure_directory,
    load_model,
    load_data,
    save_dataframe_to_csv,
    save_dataframe_to_excel,
    safe_format,
    auto_convert_numeric,
)

from luna_xai.utils.preprocessing import HybridPreprocessor

from luna_xai.utils.metrics import calculate_ch_index
from luna_xai.utils.normalization import arctan_normalize

from luna_xai.utils.sampling import create_synthetic_samples

from luna_xai.utils.plotting import (
    plot_importances,
    plot_feature_boxplots,
    plot_feature_trends,
)

from luna_xai.utils.logging_utils import log_execution_segment


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
__all__ = [
    # --- Core I/O ---
    "ensure_directory",
    "load_model",
    "load_data",
    "save_dataframe_to_csv",
    "save_dataframe_to_excel",
    "safe_format",
    "auto_convert_numeric",

    # --- Preprocessing ---
    "HybridPreprocessor",

    # --- Metrics & Normalization ---
    "calculate_ch_index",
    "arctan_normalize",

    # --- Sampling ---
    "create_synthetic_samples",

    # --- Plotting ---
    "plot_importances",
    "plot_feature_boxplots",
    "plot_feature_trends",

    # --- Logging ---
    "log_execution_segment",
]
