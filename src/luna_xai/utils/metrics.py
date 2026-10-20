from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import math

from sklearn.isotonic import spearmanr

from luna_xai.core import results
from luna_xai.utils.logging_utils import get_luna_logger

# Module-level logger for metric utilities. Handlers and log levels are
# configured centrally via `configure_luna_logging`. By default the log
# goes to a file; to switch to console output, attach a StreamHandler to
# the "luna_xai" logger in your application.
_LOGGER = get_luna_logger("utils.metrics")

def calculate_ch_index(
    feature_prediction_dict: Dict[str, Dict[float, List[float]]],
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """
    Compute a per-feature Calinski–Harabasz-like index without discarding features.

    For each feature, predictions are grouped by discretized "levels" (e.g., bins).
    We compute a CH-style ratio between between-level and within-level variance
    of predictions. Features that are degenerate (e.g. 1 level, or invalid
    variance) receive a score of 0.0.

    Parameters
    ----------
    feature_prediction_dict : dict
        Mapping
        ``feature_name -> {level_value -> list[predictions_at_that_level]}``.
        Example:
        ``{"age": {20.0: [0.1, 0.2], 30.0: [0.4, 0.5]}}``.

    Returns
    -------
    results : dict
        Mapping ``feature_name -> ch_score (float)``. Features with
        insufficient or degenerate spread receive 0.0.
    means_map : dict
        Mapping ``feature_name -> list[mean_prediction_per_level]``, where
        levels are sorted by their numeric value.

    Notes
    -----
    - Features with fewer than 2 levels get CH = 0.0.
    - If ``N - m <= 0`` (total samples - number of levels) or the within-cluster
      variance is too small/NaN, CH = 0.0.
    """
    _LOGGER.debug(
        "Starting CH index computation for %d features.",
        len(feature_prediction_dict),
    )

    results: Dict[str, float] = {}
    means_map: Dict[str, List[float]] = {}

    for feature, level_map in feature_prediction_dict.items():
        # Sort levels to enforce a deterministic order
        sorted_levels = dict(sorted(level_map.items(), key=lambda kv: kv[0]))

        # Always store means (even if only 1 level)
        means: List[float] = []
        for preds in sorted_levels.values():
            if len(preds) > 0:
                means.append(float(np.mean(preds)))
            else:
                means.append(0.0)

        means_map[feature] = means

        # If less than 2 levels → CH = 0
        if len(sorted_levels) < 2:
            results[feature] = 0.0
            continue

        m = len(sorted_levels)
        N = sum(len(preds) for preds in sorted_levels.values())

        # Need at least one degree of freedom in the denominator
        if (N - m) <= 0:
            results[feature] = 0.0
            continue

        global_mean = float(np.mean(means))

        # Between-level sum of squares
        S_B = 0.0
        for preds, mu in zip(sorted_levels.values(), means):
            S_B += len(preds) * (mu - global_mean) ** 2

        # Within-level sum of squares
        S_W = 0.0
        for preds, mu in zip(sorted_levels.values(), means):
            arr = np.asarray(preds, dtype=float)
            if arr.size > 0:
                S_W += float(np.sum((arr - mu) ** 2))

        if S_W <= 0.0 or not math.isfinite(S_W):
            results[feature] = 0.0
            continue

        ch_score = (S_B / (m - 1)) / (S_W / (N - m))

        results[feature] = (
            float(ch_score)
            if math.isfinite(ch_score)
            else 0.0
        )

    _LOGGER.debug(
        "Finished CH computation. %d features scored.",
        len(results),
    )

    return results, means_map

def adjust_signal(
    means: Dict[str, List[float]],
    ch_values: Dict[str, float],
) -> Dict[str, float]:
    """
    Adjust the sign of CH scores based on the direction of the feature effect.

    For each feature, the sequence of mean model predictions across its
    ordered levels is analyzed. The direction of the effect is inferred
    using the Spearman rank correlation between the level indices
    (0, 1, ..., n-1) and the corresponding mean predictions.

    The sign of the correlation is then used to adjust the CH score:
    - Positive or zero correlation: the CH score keeps a positive sign.
    - Negative correlation: the CH score sign is inverted.

    In degenerate cases (constant values leading to undefined correlation)
    or when a feature has a single level, the CH score remains positive.

    Parameters
    ----------
    means : Dict[str, List[float]]
        Mapping ``feature_name -> list of mean predictions per level``.
        The list is assumed to be ordered according to the feature levels.

    ch_values : Dict[str, float]
        Mapping ``feature_name -> CH score`` previously computed.

    Returns
    -------
    Dict[str, float]
        Mapping ``feature_name -> sign-adjusted CH score``.
        The absolute value of the CH score is preserved; only the sign
        may be flipped to reflect the direction of the feature effect.
    """
    _LOGGER.debug(
        "Adjusting CH scores with signal direction for %d features.",
        len(means),
    )

    adjusted: Dict[str, float] = {}

    for feature, values in means.items():
        if len(values) >= 2:
            x = np.arange(len(values), dtype=float)
            rho, _ = spearmanr(x, values)

            # Em casos degenerados (valores constantes), spearmanr pode retornar nan
            if np.isnan(rho):
                sign = 1
            else:
                sign = 1.0 if rho >= 0.0 else -1.0
        else:
            # With a single level, we cannot infer direction; keep positive.
            sign = 1.0

        base_value = float(ch_values.get(feature, 0.0))
        adjusted_value = base_value * sign
        adjusted[feature] = adjusted_value

    _LOGGER.debug("Signal adjustment complete.")

    return adjusted