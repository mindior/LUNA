from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Mapping
import copy

import pandas as pd

from luna_xai.utils.logging_utils import get_luna_logger

# Module-level logger for metric result utilities.
# By default, application code is expected to configure handlers so that
# these logs are written to a file (e.g., via a FileHandler attached to
# the "luna_xai" logger). If you prefer console output instead, you can
# configure the "luna_xai" logger with a StreamHandler or via
# `configure_luna_logging`.
LOGGER = get_luna_logger("metric.result")


@dataclass
class MetricResult:
    """
    Unified container for metric computation results in LUNA XAI.

    Stores
    ------
    - Feature importances (optional)
    - Raw sensitivity data used during metric computation
    - Arbitrary metadata (runtime info, normalization stats, etc.)

    This object is the standard output of all Metric classes.
    """

    # Computed feature importances (may be absent for some metrics)
    importances: Optional[Mapping[str, float]] = field(default=None)

    # Raw sensitivity data: feature → level → list[predictions]
    sensitivity_data: Mapping[str, Mapping[Any, list]] = field(default_factory=dict)

    # Arbitrary metadata (sampling info, timing, normalization, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Feature availability checks
    # ------------------------------------------------------------------
    def has_importances(self) -> bool:
        """Return True if the metric produced feature importances."""
        return bool(self.importances)

    def has_sensitivity(self) -> bool:
        """Return True if sensitivity information is available."""
        return bool(self.sensitivity_data)

    # ------------------------------------------------------------------
    # DataFrame conversion
    # ------------------------------------------------------------------
    def as_dataframe(self) -> pd.DataFrame:
        """
        Convert feature importances into a Pandas DataFrame
        for visualization, logging, or export.
        """
        if not self.has_importances():
            raise ValueError("Cannot convert to DataFrame: no importances available.")

        LOGGER.debug(
            "Converting MetricResult importances to DataFrame (n_features=%d).",
            len(self.importances),  # type: ignore[arg-type]
        )

        return pd.DataFrame(
            [(feature, score) for feature, score in self.importances.items()],  # type: ignore[union-attr]
            columns=["feature", "importance"],
        )

    # ------------------------------------------------------------------
    # Concise summary
    # ------------------------------------------------------------------
    def summary(self) -> str:
        """
        Produce a compact, human-readable summary of the metric result.

        Useful for console logs and debugging.
        """
        parts = []

        if self.has_importances():
            parts.append(f"{len(self.importances)} features")  # type: ignore[arg-type]

        if self.has_sensitivity():
            parts.append(f"{len(self.sensitivity_data)} sensitivity-sets")

        if self.metadata:
            parts.append(f"metadata={list(self.metadata.keys())}")

        summary_str = " | ".join(parts) if parts else "MetricResult(empty)"
        LOGGER.debug("MetricResult.summary(): %s", summary_str)
        return summary_str

    # ------------------------------------------------------------------
    # Normalization helper (optional)
    # ------------------------------------------------------------------
    def apply_normalization(self, normalizer: Any) -> None:
        """
        Apply a normalization function to the importances.

        Parameters
        ----------
        normalizer : callable
            Function that accepts a dict of {feature: importance}
            and returns a normalized dict.
        """
        if not self.has_importances():
            raise ValueError("Cannot normalize: no importances available.")

        if not callable(normalizer):
            raise TypeError("Normalizer must be a callable.")

        LOGGER.debug(
            "Applying normalization to MetricResult importances (n_features=%d).",
            len(self.importances),  # type: ignore[arg-type]
        )

        normalized = normalizer(self.importances)
        if not isinstance(normalized, Mapping):
            raise TypeError("Normalizer must return a mapping.")

        self.importances = dict(normalized)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result into a JSON-serializable Python dictionary.

        Note: sensitivity_data may contain non-JSON-friendly items,
        so deep copies and conversions are used where appropriate.
        """
        LOGGER.debug(
            "Converting MetricResult to dict (importances=%s, sensitivity_sets=%d, metadata_keys=%d).",
            "present" if self.has_importances() else "absent",
            len(self.sensitivity_data),
            len(self.metadata),
        )

        return {
            "importances": dict(self.importances) if self.importances else None,
            "sensitivity_data": {
                feat: {lvl: list(vals) for lvl, vals in groups.items()}
                for feat, groups in self.sensitivity_data.items()
            },
            "metadata": copy.deepcopy(self.metadata),
        }

    def copy(self) -> "MetricResult":
        """Return a deep copy of this MetricResult."""
        LOGGER.debug(
            "Creating a deep copy of MetricResult (importances=%s, sensitivity_sets=%d, metadata_keys=%d).",
            "present" if self.has_importances() else "absent",
            len(self.sensitivity_data),
            len(self.metadata),
        )
        return MetricResult(
            importances=copy.deepcopy(self.importances),
            sensitivity_data=copy.deepcopy(self.sensitivity_data),
            metadata=copy.deepcopy(self.metadata),
        )

    # ------------------------------------------------------------------
    # Debug-friendly representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        n_imp = len(self.importances) if self.importances else 0
        n_sens = len(self.sensitivity_data) if self.sensitivity_data else 0
        meta_keys = list(self.metadata.keys()) if self.metadata else []

        return (
            f"MetricResult("
            f"importances={n_imp}, "
            f"sensitivity_sets={n_sens}, "
            f"metadata_keys={meta_keys})"
        )
