"""
HybridPreprocessor
==================

Unified preprocessing module for dual-space explainability in LUNA XAI.

Responsibilities
----------------
- Handle both numeric and categorical variables
- Apply CANE-based IDF encoding for categorical features
- Scale all features numerically for HNSW / hybrid neighbor spaces
- Maintain consistent feature metadata for downstream methods
- Persist and reload its full state (CANE mappings + scaler)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from luna_xai.utils.logging_utils import get_luna_logger
from luna_xai.utils.debug_mixin import DebugMixin

PathLike = Union[str, Path]


class HybridPreprocessor(DebugMixin):
    """
    Hybrid preprocessor for hybrid-space neighbor search and explainability.

    Processing steps
    ----------------
    1. Optionally apply CANE-IDF encoding to categorical columns.
    2. Convert all columns to numeric types (coercing invalid values to NaN).
    3. Fill NaN with zeros (for a fully numeric representation).
    4. Apply a global scaler (default: StandardScaler).

    Parameters
    ----------
    categorical_features : list of str, optional
        Names of categorical columns to be encoded via CANE. If None,
        they are automatically inferred.
    use_cane : bool, default True
        Whether to apply CANE-IDF encoding to categorical features.
    scaler : object, optional
        A scikit-learn-compatible scaler implementing `fit` and `transform`.
        If None, StandardScaler is used.
    verbose : bool, default True
        Enable lightweight debug logging inside the preprocessor.
    """

    def __init__(
        self,
        categorical_features: Optional[List[str]] = None,
        use_cane: bool = True,
        scaler: Optional[Any] = None,
        verbose: bool = True,
    ) -> None:

        # DebugMixin setup
        self._logger = get_luna_logger("utils.preprocessing.hybrid_preprocessor")
        self._debug_enabled = bool(verbose)

        self.categorical_features = categorical_features
        self.use_cane = use_cane
        self.scaler_ = scaler or StandardScaler()

        # Internal fitted state
        self.idf_dict_: Dict[str, Dict[Any, float]] = {}
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Fit / Transform
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> "HybridPreprocessor":
        """
        Fit CANE-IDF (if enabled) and global scaler.

        Parameters
        ----------
        X : pd.DataFrame
            Training data containing mixed numeric + categorical features.

        Returns
        -------
        HybridPreprocessor
            The fitted instance.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HybridPreprocessor.fit() expects a pandas DataFrame.")

        self._debug(
            f"Starting fit on DataFrame with shape={X.shape}, use_cane={self.use_cane}."
        )

        df = X.copy()
        self.feature_names_ = list(df.columns)

        # Detect categorical features if not provided
        if self.categorical_features is None:
            self.categorical_features = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            self._debug(
                f"Auto-detected categorical features: {self.categorical_features}."
            )
        else:
            self._debug(
                f"Using provided categorical features: {self.categorical_features}."
            )

        # Step 1 – CANE-IDF encoding (vetorizado, como no protótipo)
        if self.use_cane and self.categorical_features:
            try:
                import cane  # type: ignore
            except ImportError as exc:
                self._logger.error("CANE library not installed while use_cane=True.")
                raise ImportError(
                    "CANE library not found. Install via `pip install cane`."
                ) from exc

            cat_cols = self.categorical_features
            self._debug(
                f"Applying CANE IDF to {len(cat_cols)} categorical columns (vectorized call)..."
            )

            data_idf = cane.idf(df, columns_use=cat_cols)

            idf_dict_all = cane.idfDictionary(
                Original=df,
                Transformed=data_idf,
                columns_use=cat_cols,
            )

            df[cat_cols] = data_idf[cat_cols]

            #self.idf_dict_ = {col: idf_dict_all[col] for col in cat_cols}
            self.idf_dict_ = idf_dict_all

            self._debug("CANE IDF encoding completed (vectorized).")
        else:
            self._debug(
                "Skipping CANE IDF step (disabled or no categorical features detected)."
            )
            self.idf_dict_ = {}

        # Step 2 – Global numeric scaling
        df_numeric = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        self.scaler_.fit(df_numeric)
        self.is_fitted_ = True

        self._debug(
            f"Universal scaling complete. Shape after numeric conversion: {df_numeric.shape}."
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply stored CANE-IDF mappings and scaling to new data.

        Expected columns must match those seen during `fit`.
        """
        if not self.is_fitted_:
            raise RuntimeError("HybridPreprocessor must be fitted before transform().")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("HybridPreprocessor.transform() expects a pandas DataFrame.")

        self._debug(f"Starting transform on DataFrame with shape={X.shape}.")

        df = X.copy()

        # Column consistency check
        input_cols = set(df.columns)
        expected_cols = set(self.feature_names_)

        missing = expected_cols - input_cols
        extra = input_cols - expected_cols

        if missing or extra:
            self._logger.error(
                "Mismatch in DataFrame columns during transform. Missing=%s | Extra=%s",
                sorted(missing),
                sorted(extra),
            )
            raise ValueError(
                f"Input DataFrame columns do not match training set. "
                f"Missing: {sorted(missing)}; Extra: {sorted(extra)}."
            )

        # Reorder columns
        df = df[self.feature_names_]

        # Apply CANE-IDF mappings
        if self.idf_dict_:
            for col, mapping in self.idf_dict_.items():
                if col in df.columns:
                    max_val = max(mapping.values()) if mapping else 1.0
                    df[col] = df[col].map(lambda v: mapping.get(v, max_val))

        # Numeric conversion + scaling
        df_numeric = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        transformed = self.scaler_.transform(df_numeric)

        self._debug(f"Transform completed. Output shape={df_numeric.shape}.")

        return pd.DataFrame(transformed, columns=self.feature_names_)

    def fit_transform(
        self, X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[Any, float]], Any]:
        """
        Fit and transform data in a single step.
        """
        self._debug(f"Running fit_transform on DataFrame with shape={X.shape}.")
        transformed = self.fit(X).transform(X)
        return transformed, self.idf_dict_, self.scaler_

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: PathLike) -> None:
        """
        Persist preprocessing state to disk.

        The following fields are serialized:
        - categorical_features
        - use_cane
        - idf_dict_
        - scaler_
        - feature_names_
        - is_fitted_
        """
        path = Path(path)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "categorical_features": self.categorical_features,
            "use_cane": self.use_cane,
            "idf_dict": self.idf_dict_,
            "scaler": self.scaler_,
            "feature_names": self.feature_names_,
            "is_fitted": self.is_fitted_,
        }

        with path.open("wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._debug(f"HybridPreprocessor state saved to {path}.")

    @classmethod
    def load(cls, path: PathLike) -> "HybridPreprocessor":
        """
        Restore a HybridPreprocessor from a pickle file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"HybridPreprocessor file not found: {path}")

        with path.open("rb") as f:
            state = pickle.load(f)

        obj = cls(
            categorical_features=state.get("categorical_features"),
            use_cane=state.get("use_cane", True),
            scaler=state.get("scaler"),
            verbose=False,
        )

        obj.idf_dict_ = state.get("idf_dict", {})
        obj.feature_names_ = state.get("feature_names", [])
        obj.is_fitted_ = state.get("is_fitted", False)

        obj._logger.debug(
            "HybridPreprocessor state loaded from %s (n_features=%d, cane=%s).",
            path,
            len(obj.feature_names_),
            obj.use_cane,
        )

        return obj

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def output_dim_(self) -> int:
        """Return the dimensionality of the transformed space."""
        return len(self.feature_names_)

    def summary(self) -> str:
        """Human-readable configuration summary."""
        n_features = len(self.feature_names_)
        n_categorical = len(self.categorical_features) if self.categorical_features else 0

        return (
            "HybridPreprocessor("
            f"n_features={n_features}, "
            f"categorical={n_categorical}, "
            f"cane={self.use_cane}, "
            f"scaler={type(self.scaler_).__name__})"
        )

    def __repr__(self) -> str:
        return self.summary()
