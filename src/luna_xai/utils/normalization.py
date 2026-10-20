from __future__ import annotations

from typing import Dict

import numpy as np

from luna_xai.utils.logging_utils import get_luna_logger

# Module-level logger for normalization utilities. Handlers and log levels are
# configured centrally via `configure_luna_logging`. By default, logs go to a
# file; to switch to console output you can attach a StreamHandler to the
# "luna_xai" logger in your application.
_LOGGER = get_luna_logger("utils.normalization")


def arctan_normalize(
    importances: Dict[str, float],
    scale_factor: float = 1000.0,
) -> Dict[str, float]:
    """
    Apply arctangent-based normalization with a scaling factor.

    This transformation smoothly compresses large-magnitude importance
    values into the interval [-1, 1], preserving sign and improving
    stability/interpretability. Values close to zero remain almost
    unchanged; very large magnitudes asymptotically approach -1 or 1.

    The transformation is defined as:

        norm(x) = arctan(x / scale_factor) / (π / 2)

    Parameters
    ----------
    importances : dict of str to float
        Raw importance scores from the metric. Values must be numeric
        (castable to ``float``).
    scale_factor : float, default 1000.0
        Controls curvature — larger values result in gentler compression
        (i.e., the arctan transition becomes slower).

    Returns
    -------
    dict of str to float
        Normalized importance scores in the range [-1, 1].

    Raises
    ------
    TypeError
        If ``importances`` is not a dict or if any value is not numeric.
    ValueError
        If ``scale_factor`` is not a positive float.
    """
    if not isinstance(importances, dict):
        raise TypeError(
            "arctan_normalize expects a dict[str, float] as 'importances', "
            f"got {type(importances).__name__}."
        )

    if scale_factor <= 0.0:
        raise ValueError(
            "Parameter 'scale_factor' must be a positive float; "
            f"received {scale_factor}."
        )

    _LOGGER.debug(
        "Starting arctan normalization for %d features (scale_factor=%g).",
        len(importances),
        scale_factor,
    )

    normalized: Dict[str, float] = {}

    for feature, value in importances.items():
        try:
            x = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Importance value for feature '{feature}' is not numeric: {value!r}."
            ) from exc

        # Normalized arctan with curvature control
        scaled = x / scale_factor
        atan_value = np.arctan(scaled)
        normalized_value = float(atan_value / (np.pi / 2))

        normalized[feature] = normalized_value

    _LOGGER.debug("Arctan normalization complete.")

    return normalized
