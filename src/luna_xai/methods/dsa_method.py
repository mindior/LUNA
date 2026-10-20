"""
DSA Method — Deterministic Sensitivity Analysis
==============================================

Local perturbation-based sensitivity method used in LUNA XAI.

This implementation follows a PDP/ICE-style "what-if" analysis using:
    • The local neighborhood (real neighbors)
    • Synthetic perturbations (empirical sampling)
    • Full recomputation of predictions for each feature × level pair

The method is:
    - Deterministic (when seeded)
    - Faithfully aligned with the original design
    - Model-agnostic and fully domain-driven

Outputs a nested mapping:
{
    feature_1: { level_a: [preds...], level_b: [...], ... },
    feature_2: { ... },
}
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from luna_xai.core.base_method import BaseMethod
from luna_xai.utils.sampling import create_synthetic_samples
from luna_xai.utils.dataframe import ensure_dataframe


class DSAMethod(BaseMethod):
    """
    Deterministic Sensitivity Analysis (DSA).

    Generates PDP/ICE-style local sensitivity curves by:
        1. Extracting the hybrid-space neighborhood around an instance
        2. Generating synthetic samples (empirical)
        3. For each feature:
            - Enumerating the real neighbor feature levels
            - Overwriting the feature column with each level
            - Predicting the model response over the full neighborhood

    For classification, when a `target_class` index is provided and the
    underlying model implements `predict_proba`, DSA operates on the
    probability of that class instead of raw predictions.
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(
        self,
        n_neighbors: int = 7,
        n_samples: int = 1000,
        sampler: str = "lhs",
        seed: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        n_neighbors : int, default=7
            Number of real neighbors used to define the local region.
        n_samples : int, default=1000
            Number of synthetic samples drawn from the local neighborhood.
        sampler : {"lhs", ...}, default="lhs"
            Sampling strategy used in `create_synthetic_samples`.
        seed : int, optional
            Random seed for deterministic sampling. If None, sampling
            may be non-deterministic unless a seed is provided via kwargs.
        debug : bool, default=False
            If True, enables debug-level logging via BaseMethod/DebugMixin.
        """
        super().__init__(debug=debug)

        self.n_neighbors = n_neighbors
        self.n_samples = n_samples
        self.sampler = sampler
        self.seed = seed

        # These are bound at runtime (factory or explainer will set domain/neighbor_finder)
        self.domain: Optional[Any] = None
        self.neighbor_finder: Optional[Any] = None

    # ------------------------------------------------------------------
    # Private helper: prediction for explanation
    # ------------------------------------------------------------------
    def _predict_for_explanation(
        self,
        df: pd.DataFrame,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Centralizes prediction logic for DSA.

        - If `target_class` is provided and the underlying model supports
          `predict_proba`, returns the probability for that class.
        - Otherwise, falls back to `domain.predict(df)`.

        Parameters
        ----------
        df : pd.DataFrame
            Perturbed neighborhood data.
        target_class : int or None
            Index of the class of interest for classification explanations.

        Returns
        -------
        np.ndarray
            1D array of scalar responses (predictions or probabilities).
        """
        model = getattr(self.domain, "model", None)
        data_interface = getattr(self.domain, "data_interface", None)

        # Classification path: use predict_proba for the target class
        if target_class is not None and model is not None and hasattr(model, "predict_proba"):
            if data_interface is not None and hasattr(data_interface, "transform_input"):
                X_input = data_interface.transform_input(df)
            else:
                X_input = df

            if hasattr(X_input, "toarray"):
                X_input = X_input.toarray()

            X_input = np.asarray(X_input, dtype=np.float32, order="C")

            proba = model.predict_proba(X_input)

            if proba.ndim != 2 or target_class < 0 or target_class >= proba.shape[1]:
                raise RuntimeError(
                    "[DSA ERROR] Invalid target_class index or predict_proba output "
                    f"shape={proba.shape}, target_class={target_class}."
                )

            return np.asarray(proba[:, target_class], dtype=float)

        # Regression / generic path: delegate to domain.predict
        preds = self.domain.predict(df)
        return np.asarray(preds, dtype=float)

    # ------------------------------------------------------------------
    # Main sensitivity generator
    # ------------------------------------------------------------------
    def generate(
        self,
        domain: Any,
        instance: pd.Series | pd.DataFrame,
        data: Any = None,  # kept for interface compatibility
        **kwargs: Any,
    ) -> Dict[str, Dict[Any, list]]:
        """
        Generate local PDP/ICE-style sensitivity mappings.

        Parameters
        ----------
        domain : BaseDomain
            Provides model prediction (`predict`) and neighbor search
            (via `neighbor_finder`).
        instance : pd.Series or pd.DataFrame
            Query instance. Will be normalized into a single-row DataFrame.
        data : Any, optional
            Unused placeholder, kept only for backward compatibility.
        **kwargs :
            Optional overrides such as:
                - sampler="lhs" | ...
                - seed=123
                - n_neighbors, n_samples (if desired in the future)
                - target_class=<int> (for classification explanations)

        Returns
        -------
        dict
            Nested sensitivity structure: {feature -> {level -> [predictions...]}}
        """
        # Bind domain and neighbor_finder
        self.domain = domain
        self.neighbor_finder = getattr(domain, "neighbor_finder", None)

        if self.neighbor_finder is None:
            raise RuntimeError(
                "[DSA ERROR] No neighbor finder found. "
                "Call domain.build_index() before using DSA."
            )

        # Optional classification target class index
        target_class = kwargs.get("target_class", None)

        # Ensure a single-row DataFrame
        instance_df = ensure_dataframe(instance)

        # ---------------------------------------------------------------
        # Step 1 — Retrieve local neighborhood
        # ---------------------------------------------------------------
        self._logger.info("[LUNA DSA] Retrieving local neighborhood...")

        neighbors = self.neighbor_finder.find_neighbors(
            instance=instance_df,
            n_neighbors=self.n_neighbors,
        )

        if neighbors is None or len(neighbors) == 0:
            raise RuntimeError("[DSA ERROR] No neighbors found for this instance.")

        # ---------------------------------------------------------------
        # Step 2 — Synthetic samples
        # ---------------------------------------------------------------
        # Effective sampler and seed: kwargs override instance attributes
        effective_sampler = kwargs.get("sampler", self.sampler)
        effective_seed = kwargs.get("seed", self.seed)

        self._logger.info(
            "[LUNA DSA] Creating synthetic samples (method='%s', n_samples=%d, seed=%s)",
            effective_sampler,
            self.n_samples,
            str(effective_seed),
        )

        synthetic = create_synthetic_samples(
            neighbors,
            n_samples=self.n_samples,
            method=effective_sampler,
            seed=effective_seed,
        )

        # ---------------------------------------------------------------
        # Step 3 — Cleanup (remove auxiliary columns, ensure DataFrame)
        # ---------------------------------------------------------------
        def clean(df_any: Any) -> pd.DataFrame:
            df_local = df_any
            if isinstance(df_local, pd.DataFrame):
                if "distance" in df_local.columns:
                    df_local = df_local.drop(columns=["distance"]).copy()
            return df_local

        neighbors_clean = clean(neighbors)
        synthetic_clean = clean(synthetic)
        instance_df = ensure_dataframe(instance_df)
        instance_clean = clean(instance_df)

        full_neighborhood = pd.concat(
            [neighbors_clean, synthetic_clean, instance_clean],
            ignore_index=True,
        )

        self._logger.info(
            "[LUNA DSA] Base neighborhood assembled (%d samples).",
            len(full_neighborhood),
        )

        # ---------------------------------------------------------------
        # Step 4 — PDP/ICE-style sensitivity computation
        # ---------------------------------------------------------------
        sensitivity_data: Dict[str, Dict[Any, list]] = {}
        base_df = full_neighborhood.copy()

        for feature in base_df.columns:
            # Possible values = unique values from real neighbors only
            base_levels = neighbors_clean[feature].dropna().unique()

            try:
                feature_values = sorted(base_levels)
            except TypeError:
                # Fallback when values are not directly comparable
                feature_values = sorted(base_levels, key=lambda x: str(x))

            sensitivity_data[feature] = {}

            for val in feature_values:
                temp_df = base_df.copy()
                temp_df[feature] = val

                preds = self._predict_for_explanation(
                    temp_df,
                    target_class=target_class,
                )
                preds = np.asarray(preds).ravel()

                sensitivity_data[feature][val] = preds.tolist()

                self._debug(
                    f"[DSA PDP] feature={feature}, value={val} "
                    f"→ {len(preds)} predictions"
                )

        self._logger.info("[LUNA DSA] Sensitivity data generated successfully.")
        return sensitivity_data


# ---------------------------------------------------------------------
# Auto-registration
# ---------------------------------------------------------------------
from luna_xai.core.registry import register_component

register_component("method", "dsa", DSAMethod)
