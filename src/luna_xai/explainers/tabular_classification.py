"""
TabularClassificationExplainer
==============================

LUNA explainer for tabular *classification* models.

This explainer performs local sensitivity analysis based on the
predicted probability of the *most probable class* for the instance.

Pipeline (aligned with regression):
1️⃣ generate_explanation_data() → builds raw sensitivity data + metadata
2️⃣ explain_from_data()        → computes importances, applies normalizer
                                and triggers rendering.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from luna_xai.core.luna_explainer import LunaExplainer
from luna_xai.core.results import MetricResult
from luna_xai.utils.dataframe import ensure_dataframe


class TabularClassificationExplainer(LunaExplainer):
    """
    Classification explainer for tabular models.

    Workflow
    --------
    1. Transform instance → model space
    2. Compute class probabilities (predict_proba)
    3. Select predicted class (argmax over classes)
    4. Generate sensitivity data via `method.generate(...)`
       conditioned on the predicted class (target_class index)
    5. Compute feature importances via `metric.compute(...)`
    6. Optionally apply a normalizer (e.g., arctan_normalize)
    7. Render plots via `renderer.render(...)`
    """

    # ------------------------------------------------------------------
    def generate_explanation_data(
        self,
        instance: pd.Series | pd.DataFrame,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate raw explanation data for a single classification instance.

        Produces
        --------
        - sensitivity_data: nested structure produced by the method
              e.g. DSA → { feature: { level: [probabilities...] }, ... }
          where the values are probabilities of the *predicted class*.
        - metadata: dictionary with trace information (method, metric, etc.)
          including predicted class index and its probability.

        Parameters
        ----------
        instance : pd.Series or pd.DataFrame
            Raw instance to be explained (single sample).
        **kwargs : dict
            Extra parameters forwarded to the method's `generate` routine
            (e.g., n_neighbors, n_samples, seed, ...).

        Returns
        -------
        dict
            {
                "sensitivity_data": ...,
                "metadata": {...},
            }
        """
        self._logger.info(
            "[LUNA] Generating explanation data (TabularClassificationExplainer)..."
        )

        # --------------------------------------------------------------
        # 1) Standardize instance to a single-row DataFrame
        # --------------------------------------------------------------
        instance_df = ensure_dataframe(instance)

        # --------------------------------------------------------------
        # 2) Compute class probabilities in model space
        # --------------------------------------------------------------
        data_interface = getattr(self.domain, "data_interface", None)
        if data_interface is None or not hasattr(data_interface, "transform_input"):
            raise RuntimeError(
                "[TabularClassificationExplainer] Domain DataInterface with "
                "a valid transform_input(...) is required for classification."
            )

        instance_input = data_interface.transform_input(instance_df)

        # Sparse → dense if needed
        if hasattr(instance_input, "toarray"):
            instance_input = instance_input.toarray()

        instance_input = np.asarray(instance_input, dtype=np.float32, order="C")

        model = getattr(self.domain, "model", None)
        if model is None or not hasattr(model, "predict_proba"):
            raise RuntimeError(
                "[TabularClassificationExplainer] The underlying model does not "
                "implement predict_proba, required for classification explanation."
            )

        proba = model.predict_proba(instance_input)

        # Expecting shape (1, n_classes)
        if proba.ndim != 2 or proba.shape[0] != 1:
            raise RuntimeError(
                "[TabularClassificationExplainer] Expected predict_proba output "
                f"with shape (1, n_classes), got {proba.shape} instead."
            )

        # Index of most probable class
        predicted_class_idx = int(np.argmax(proba, axis=1)[0])
        predicted_proba = float(proba[0, predicted_class_idx])

        self._logger.info(
            "[LUNA] Predicted class index: %d (p=%.4f)",
            predicted_class_idx,
            predicted_proba,
        )

        # --------------------------------------------------------------
        # 3) Generate local sensitivity data using the configured method,
        #    conditioned on the predicted class.
        # --------------------------------------------------------------
        sensitivity_data = self.method.generate(
            domain=self.domain,
            instance=instance_df,
            target_class=predicted_class_idx,
            **kwargs,
        )

        # --------------------------------------------------------------
        # 4) Metadata for traceability / logging
        # --------------------------------------------------------------
        metadata = {
            "domain": "tabular",
            "task": "classification",
            "method": type(self.method).__name__,
            "metric": type(self.metric).__name__,
            "neighbors": getattr(self.method, "n_neighbors", None),
            "samples": getattr(self.method, "n_samples", None),
            "predicted_class": predicted_class_idx,
            "predicted_proba": predicted_proba,
        }

        self._debug(
            "Explanation metadata assembled: method=%s, metric=%s, neighbors=%s, "
            "samples=%s, predicted_class=%s, predicted_proba=%.4f",
        )
        self._debug(
            "%s | %s | %s | %s | %s | %.4f"
            % (
                metadata["method"],
                metadata["metric"],
                metadata["neighbors"],
                metadata["samples"],
                metadata["predicted_class"],
                metadata["predicted_proba"],
            )
        )

        return {
            "sensitivity_data": sensitivity_data,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    def explain_instance(
        self,
        instance: pd.Series | pd.DataFrame,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Run local classification explanation on a single instance.

        The sensitivity analysis is performed over the probability of
        the predicted class (argmax of predict_proba).

        Parameters
        ----------
        instance : pd.Series or pd.DataFrame
            Raw instance to be explained (single sample).
        **kwargs : dict
            Extra parameters forwarded to `generate_explanation_data` and
            ultimately to the method's `generate(...)`.

        Returns
        -------
        MetricResult
            Structured result containing:
            - feature importances (possibly normalized)
            - sensitivity data (as provided by the method)
            - metadata (including predicted class and probability)
        """
        self._logger.info("[LUNA] Running TabularClassificationExplainer...")

        # Ensure the first plot generated by this explainer is an importance plot.
        if self.renderer is not None:
            self._debug(
                "Setting renderer.plot_type='importance' for primary explanation plot."
            )
            self.renderer.plot_type = "importance"

        # Stage 1 — Build explanation data (no plotting here)
        explanation_data = self.generate_explanation_data(instance, **kwargs)
        metadata = explanation_data.get("metadata", {}) or {}

        # Stage 2 — Compute importances and render (delegated to base class)
        result = self.explain_from_data(
            explanation_data,
            render=True,
            context={
                "scope": "local",
                "instance": ensure_dataframe(instance),
                "idx": kwargs.get("idx", "instance"),
                "predicted_class": metadata.get("predicted_class"),
                "predicted_proba": metadata.get("predicted_proba"),
            },
        )

        self._debug(
            "TabularClassificationExplainer completed for instance with context idx=%s.",
        )
        self._debug(str(kwargs.get("idx", "instance")))

        return result


# ---------------------------------------------------------------------
# Auto-registration
# ---------------------------------------------------------------------
from luna_xai.core.registry import register_component

register_component("explainer", "classification", TabularClassificationExplainer)
