# luna_xai/utils/plotting.py

from __future__ import annotations

from itertools import cycle
from typing import Any, Dict, List, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from luna_xai.utils.logging_utils import get_luna_logger

# Module-level logger for plotting utilities. Handlers and log levels are
# configured centrally via `configure_luna_logging`. By default logs are
# written to a file; to switch to console output, attach a StreamHandler
# to the "luna_xai" logger in your application.
_LOGGER = get_luna_logger("utils.plotting")


def format_if_number(value: Any) -> str:
    """
    Format a value as a float with 4 decimal places if possible.

    If the value cannot be safely cast to float, its string representation
    is returned instead.
    """
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


# -------------------------------------------------------------------------
# IMPORTANCES
# -------------------------------------------------------------------------
def plot_importances(
    idx: str,
    importances_dict: Mapping[str, float],
    type_: str,
    title: str = None,
    top_k: int = 5,
) -> Figure:
    """
    Plot the top-k feature importances plus an optional "Others" group.

    The function sorts features by absolute importance, keeps the `top_k`
    most relevant features and, if additional features remain, aggregates
    them into an "Others" bar.

    Parameters
    ----------
    idx : str
        Identifier of the instance / explanation.
    importances_dict : Mapping[str, float]
        Mapping from feature name to its importance value.
    type_ : str
        Scope of the explanation ("local", "global", etc.). Currently
        does not change the behavior but is kept for compatibility.
    title : str
        Title of the plot.
    top_k : int, default 5
        Number of individual features to display before grouping remaining
        ones into "Others".

    Returns
    -------
    matplotlib.figure.Figure
        The generated Matplotlib figure.
    """
    if type_ == "local":
        title = "Local Feature Contributions"
            
    if not importances_dict:
        _LOGGER.debug(
            "plot_importances called for idx=%s with empty importances_dict.",
            idx,
        )
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(
            0.5,
            0.5,
            "No importances to display",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.axis("off")
        fig.tight_layout()
        return fig

    if top_k <= 0:
        raise ValueError("'top_k' must be a positive integer.")

    _LOGGER.debug(
        "Plotting importances for idx=%s (n_features=%d, top_k=%d, type=%s).",
        idx,
        len(importances_dict),
        top_k,
        type_,
    )

    # Sort by absolute importance (descending)
    sorted_items = sorted(
        importances_dict.items(), key=lambda x: abs(x[1]), reverse=True
    )

    # Separate top_k and remaining ("others")
    top_items = sorted_items[:top_k]
    others = sorted_items[top_k:]

    final_features: List[str] = [feat for feat, _ in top_items]
    final_values: List[float] = [float(val) for _, val in top_items]

    # Add "Others" if there are more features beyond top_k
    if len(sorted_items) > top_k:
        others_sum = float(sum(v for _, v in others))
        final_features.append("Others")
        final_values.append(others_sum)
        _LOGGER.debug(
            "Aggregated %d remaining features into 'Others' (sum=%f).",
            len(others),
            others_sum,
        )

    # Colors: positive = red, negative = blue (original tones)
    colors = ["#D32F2F" if v > 0 else "#1E88E5" for v in final_values]

    # Figure height depends on number of features
    height = max(4, len(final_features) * 0.4)
    fig, ax = plt.subplots(figsize=(10, height))

    bars = ax.barh(
        final_features,
        final_values,
        color=colors,
        edgecolor="#424242",
        height=0.5,
    )

    # Vertical line at zero for reference
    ax.axvline(0, color="#616161", linewidth=0.8)

    # Add labels on bars
    for bar, val in zip(bars, final_values):
        text_x = bar.get_width() / 2 if val != 0 else 0
        ax.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.8f}",
            ha="center",
            va="center",
            fontsize=9,
            color="#212121",
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Feature Contribution", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()
    _LOGGER.debug("Importance plot created successfully for idx=%s.", idx)
    return fig


# -------------------------------------------------------------------------
# FEATURE TRENDS (ICE)
# -------------------------------------------------------------------------
def plot_feature_trends(
    idx: str,
    sensitivity_data: Dict[str, Dict[Any, List[float]]],
    instance: Optional[pd.Series],
    features: Optional[List[str]] = None,
) -> Dict[str, Figure]:
    """
    Generate per-feature trend (ICE) plots.

    All individual curves are drawn in shades of gray and the mean curve
    is highlighted in black. The x-axis tick corresponding to the
    instance's feature value is highlighted in bold black.

    Parameters
    ----------
    idx : str
        Identifier of the instance / explanation.
    sensitivity_data : dict
        Mapping from feature name to a dictionary of
        `level -> list(predictions_at_that_level)`.
    instance : pandas.Series or None
        Original instance used to highlight the corresponding level on
        the x-axis, if available.
    features : list of str, optional
        Subset of features to plot. If None, all keys from
        `sensitivity_data` are used.

    Returns
    -------
    dict
        Mapping from feature name to the corresponding Matplotlib figure.
    """
    figs: Dict[str, Figure] = {}

    selected = sensitivity_data.keys() if features is None else features

    _LOGGER.debug(
        "Plotting feature trends for idx=%s (n_features=%d, filtered=%s).",
        idx,
        len(sensitivity_data),
        "ALL" if features is None else ",".join(features),
    )

    # Gray palette for individual curves
    grey_palette = cycle(
        ["#B0B0B0", "#9E9E9E", "#8D8D8D", "#7B7B7B", "#6A6A6A", "#5C5C5C"]
    )

    for feature in selected:
        if feature not in sensitivity_data:
            # Feature not present in sensitivity_data; skip silently
            _LOGGER.debug(
                "Feature '%s' not found in sensitivity_data for idx=%s; skipping.",
                feature,
                idx,
            )
            continue

        level_map = sensitivity_data[feature]

        # Build DataFrame: columns = levels, rows = samples
        try:
            df = pd.DataFrame.from_dict(level_map, orient="index").T
        except Exception:
            # Could not structure data into a DataFrame; skip this feature
            _LOGGER.debug(
                "Could not build DataFrame for feature '%s' (idx=%s); skipping.",
                feature,
                idx,
            )
            continue

        if df.empty:
            _LOGGER.debug(
                "Empty sensitivity data for feature '%s' (idx=%s); skipping.",
                feature,
                idx,
            )
            continue

        # Sort columns by numeric value when possible
        try:
            ordered_cols = sorted(df.columns, key=lambda x: float(x))
        except Exception:
            ordered_cols = sorted(df.columns)

        df = df[ordered_cols]
        levels = [format_if_number(c) for c in ordered_cols]

        fig, ax = plt.subplots(figsize=(12, 4))

        # Individual curves in shades of gray
        for ix in df.index:
            ax.plot(
                levels,
                df.loc[ix],
                color=next(grey_palette),
                alpha=0.5,
                linewidth=1.0,
            )

        # Mean curve in black
        mean_vals = df.mean()
        ax.plot(
            levels,
            mean_vals,
            color="#000000",
            linewidth=2.5,
            label="Mean",
            zorder=3,
        )

        # Highlight instance value on the x-axis ticks (bold, black)
        if instance is not None and feature in instance.index:
            try:
                val = instance[feature]
                formatted = format_if_number(val)
                for tick in ax.get_xticklabels():
                    if tick.get_text() == formatted:
                        tick.set_color("#000000")
                        tick.set_fontweight("bold")
            except Exception:
                # Highlighting is a best-effort step; ignore failures
                _LOGGER.debug(
                    "Failed to highlight instance tick for feature '%s' (idx=%s).",
                    feature,
                    idx,
                )

        ax.set_title(f"Trends for '{feature}'", fontsize=12)
        ax.set_ylabel("Prediction", fontsize=10)
        ax.set_xlabel(feature, fontsize=10)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Improve readability of x-axis labels
        ax.tick_params(axis="x", labelrotation=30)

        fig.tight_layout()
        figs[feature] = fig

    _LOGGER.debug(
        "Created %d trend plots for idx=%s.",
        len(figs),
        idx,
    )
    return figs


# -------------------------------------------------------------------------
# FEATURE BOXPLOTS
# -------------------------------------------------------------------------
def plot_feature_boxplots(
    idx: str,
    predictions: Dict[str, Dict[Any, List[float]]],
    instance_prediction: Optional[float],
    instance: Optional[pd.Series],
    features: Optional[List[str]] = None,
) -> Dict[str, Figure]:
    """
    Generate per-feature boxplots with transparent boxes and a mean line.

    Boxes are drawn with transparent face and a visible gray border. A
    mean prediction line per level is added, along with an optional
    horizontal line for the instance's prediction. The x-axis tick
    corresponding to the instance's feature value is highlighted in
    bold black.

    Parameters
    ----------
    idx : str
        Identifier of the instance / explanation.
    predictions : dict
        Mapping from feature name to a dictionary of
        `level -> list(predictions_at_that_level)`.
    instance_prediction : float or None
        Model prediction for the instance, used as a horizontal reference
        line.
    instance : pandas.Series or None
        Original instance used to highlight the corresponding level on
        the x-axis, if available.
    features : list of str, optional
        Subset of features to plot. If None, all keys from `predictions`
        are used.

    Returns
    -------
    dict
        Mapping from feature name to the corresponding Matplotlib figure.
    """
    figs: Dict[str, Figure] = {}

    selected = predictions.keys() if features is None else features

    _LOGGER.debug(
        "Plotting feature boxplots for idx=%s (n_features=%d, filtered=%s).",
        idx,
        len(predictions),
        "ALL" if features is None else ",".join(features),
    )

    for feature in selected:
        if feature not in predictions:
            # Feature not present in predictions; skip silently
            _LOGGER.debug(
                "Feature '%s' not found in predictions for idx=%s; skipping.",
                feature,
                idx,
            )
            continue

        data = predictions[feature]

        # Sort levels by numeric value when possible
        try:
            sorted_levels = sorted(data.keys(), key=lambda x: float(x))
        except Exception:
            sorted_levels = sorted(data.keys())

        labels = [format_if_number(l) for l in sorted_levels if len(data[l]) > 0]
        values = [data[l] for l in sorted_levels if len(data[l]) > 0]

        if not values:
            # No non-empty levels; nothing to plot
            _LOGGER.debug(
                "No non-empty levels for feature '%s' (idx=%s); skipping.",
                feature,
                idx,
            )
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        box = ax.boxplot(
            values,
            patch_artist=True,
            tick_labels=labels,
            showmeans=True,
            meanprops=dict(marker="o", markerfacecolor="#FF0000", markersize=5),
        )

        # Boxes: transparent face with a clearly visible gray border
        for patch in box["boxes"]:
            patch.set(facecolor="none", edgecolor="#424242", linewidth=1.5)

        # Whiskers, caps, and medians in gray tones
        for whisker in box["whiskers"]:
            whisker.set(color="#757575", linewidth=1.0)
        for cap in box["caps"]:
            cap.set(color="#757575", linewidth=1.0)
        for median in box["medians"]:
            median.set(color="#424242", linewidth=1.2)

        # Fliers (outliers) styling
        for flier in box.get("fliers", []):
            flier.set(
                marker="o",
                markerfacecolor="#9E9E9E",
                markeredgecolor="#9E9E9E",
                alpha=0.7,
                markersize=4,
            )

        # Mean line per level (dark red)
        means = [np.mean(vals) if len(vals) > 0 else np.nan for vals in values]
        xs = np.arange(1, len(means) + 1)

        ax.plot(
            xs,
            means,
            marker="o",
            linestyle="-",
            color="#FF0000",
            linewidth=2,
            label="Mean Prediction",
        )

        # Horizontal line for the instance's prediction
        if instance_prediction is not None:
            ax.axhline(
                instance_prediction,
                color="#FF13F4",
                linestyle="--",
                linewidth=1.5,
                label="Instance Prediction",
                zorder=2,
            )

        # Highlight instance level on the x-axis (bold, black)
        if instance is not None and feature in instance.index:
            try:
                val = instance[feature]
                formatted = format_if_number(val)
                for tick in ax.get_xticklabels():
                    if tick.get_text() == formatted:
                        tick.set_color("#000000")
                        tick.set_fontweight("bold")
            except Exception:
                # Highlighting is a best-effort step; ignore failures
                _LOGGER.debug(
                    "Failed to highlight instance tick for feature '%s' (idx=%s).",
                    feature,
                    idx,
                )

        ax.set_title(f"Boxplot for '{feature}'", fontsize=12)
        ax.set_ylabel("Predicted Value", fontsize=10)
        ax.set_xlabel(feature, fontsize=10)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        fig.tight_layout()
        figs[feature] = fig

    _LOGGER.debug(
        "Created %d boxplots for idx=%s.",
        len(figs),
        idx,
    )
    return figs
