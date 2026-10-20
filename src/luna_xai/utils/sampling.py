from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import qmc

from luna_xai.utils.logging_utils import get_luna_logger
from luna_xai.utils.debug_mixin import DebugMixin


_LOGGER = get_luna_logger("utils.sampling")


# ---------------------------------------------------------------------
# Internal debug helper using DebugMixin (STATIC)
# ---------------------------------------------------------------------
def _debug_sampling(msg: str, verbose: bool) -> None:
    """
    Lightweight debug helper routed through DebugMixin.
    """
    DebugMixin._debug_static(msg, logger=_LOGGER, enabled=verbose)


# ---------------------------------------------------------------------
# Utility: ensure numeric conversion where possible
# ---------------------------------------------------------------------
def _auto_convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    return df


# ---------------------------------------------------------------------
# 1. Latin Hypercube Sampling
# ---------------------------------------------------------------------
def create_lhs_dataset_empirical(
    neighbors: pd.DataFrame,
    n_samples: int,
    seed_neighbors: Optional[int] = None,
    *,
    verbose: bool = False,
) -> pd.DataFrame:

    if n_samples <= 0:
        raise ValueError("Parameter 'n_samples' must be a positive integer.")

    if neighbors.empty:
        raise ValueError("'neighbors' must not be an empty DataFrame.")

    df = neighbors.copy()

    # Make categorical explicit
    for col in df.columns:
        if df[col].dtype == object or isinstance(df[col].iloc[0], str):
            df[col] = df[col].astype(str)

    _debug_sampling("Running auto-conversion to numeric where possible.", verbose)
    df = _auto_convert_numeric(df)

    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    _debug_sampling(
        f"Numeric columns: {numeric_cols}, categorical columns: {categorical_cols}.",
        verbose,
    )

    synthetic = pd.DataFrame(columns=df.columns)

    # Numeric sampling via LHS
    if numeric_cols:
        sampler = qmc.LatinHypercube(d=len(numeric_cols), seed=seed_neighbors)
        lhs = sampler.random(n=n_samples)
        transformed = np.zeros_like(lhs)

        for i, col in enumerate(numeric_cols):
            values = df[col].dropna().values
            if len(values) == 0:
                continue
            values = np.sort(values)
            percentiles = np.linspace(0, 1, len(values), endpoint=False)
            transformed[:, i] = np.interp(lhs[:, i], percentiles, values)

        synthetic[numeric_cols] = pd.DataFrame(transformed, columns=numeric_cols)

    # Categorical sampling
    if categorical_cols:
        rng = np.random.default_rng(seed_neighbors)
        for col in categorical_cols:
            levels = df[col].dropna().unique()
            if len(levels) == 0:
                synthetic[col] = np.nan
            else:
                synthetic[col] = rng.choice(levels, size=n_samples, replace=True)

    return synthetic[df.columns]


# ---------------------------------------------------------------------
# 2. CTGAN sampling
# ---------------------------------------------------------------------
def create_ctgan_dataset(
    neighbors: pd.DataFrame,
    n_samples: int,
    seed_neighbors: Optional[int] = None,
    *,
    verbose: bool = False,
) -> pd.DataFrame:

    if n_samples <= 0:
        raise ValueError("Parameter 'n_samples' must be positive.")
    if neighbors.empty:
        raise ValueError("'neighbors' must not be empty.")

    try:
        from ctgan import CTGAN
    except ImportError as exc:
        raise ImportError("Install with `pip install ctgan`.") from exc

    categorical_cols = neighbors.select_dtypes(include=["object", "category"]).columns.tolist()

    _debug_sampling(
        f"Training CTGAN on {neighbors.shape[0]} rows, {len(categorical_cols)} categorical.",
        verbose,
    )

    if seed_neighbors is not None:
        np.random.seed(seed_neighbors)

    try:
        ctgan = CTGAN(epochs=300, verbose=verbose, random_state=seed_neighbors)
    except TypeError:
        ctgan = CTGAN(epochs=300, verbose=verbose)

    ctgan.fit(neighbors, categorical_cols)
    return ctgan.sample(n_samples)


# ---------------------------------------------------------------------
# 3. Unified interface
# ---------------------------------------------------------------------
def create_synthetic_samples(
    neighbors: pd.DataFrame,
    n_samples: int,
    method: Literal["lhs", "ctgan"] = "lhs",
    seed: Optional[int] = None,
    *,
    verbose: bool = False,
) -> pd.DataFrame:

    if method.lower() == "lhs":
        return create_lhs_dataset_empirical(
            neighbors, n_samples, seed_neighbors=seed, verbose=verbose
        )
    elif method.lower() == "ctgan":
        return create_ctgan_dataset(
            neighbors, n_samples, seed_neighbors=seed, verbose=verbose
        )

    raise ValueError("Unsupported sampling method. Use 'lhs' or 'ctgan'.")
