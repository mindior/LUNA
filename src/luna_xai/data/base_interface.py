"""
DataInterface (Enhanced)
========================

Adds full persistence:
- save(path)  → saves preprocessor + label preprocessor
- load(path)  → restores everything for prediction & inference

Used by TabularDomain.save_full_state() and load_full_state().
"""

from __future__ import annotations

from typing import Any, Optional, Union, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from luna_xai.utils.logging_utils import get_luna_logger
from luna_xai.utils.debug_mixin import DebugMixin

# Module-level logger for DataInterface operations.
# Usado por métodos de classe/estáticos (ex: load) e logs mais gerais.
LOGGER = get_luna_logger("data.interface")


class DataInterface(DebugMixin):
    """
    Standardizes preprocessing and metadata handling for tabular or structured data.
    Now includes full persistence (save/load).
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        preprocessor: Optional[Any] = None,
        label_preprocessor: Optional[Any] = None,
        debug: bool = False,
    ) -> None:

        # Logger per instance, following the standard of other databases
        self._logger = LOGGER
        self._debug_enabled: bool = bool(debug)

        self.model: Optional[Any] = None
        self.preprocessor: Optional[Any] = None
        self.label_preprocessor = label_preprocessor

        # Detect sklearn Pipeline vs simple transformer
        if isinstance(preprocessor, Pipeline):
            self.model = preprocessor.named_steps.get("model", None)
            self.preprocessor = preprocessor.named_steps.get(
                "preprocessor", preprocessor
            )
        else:
            self.preprocessor = preprocessor

        self._debug(
            "Initialized DataInterface(preprocessor=%s, label_preprocessor=%s, model=%s)",
        )
        self._debug(
            "preprocessor=%s, label_preprocessor=%s, model=%s"
            % (
                type(self.preprocessor).__name__ if self.preprocessor is not None else "None",
                type(self.label_preprocessor).__name__ if self.label_preprocessor is not None else "None",
                type(self.model).__name__ if self.model is not None else "None",
            )
        )

    # ------------------------------------------------------------------
    # Persistence API
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Persist the entire DataInterface:
            - preprocessor
            - label_preprocessor
            - model (if inside pipeline)
        Saves to: path.joblib
        """
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "preprocessor": self.preprocessor,
            "label_preprocessor": self.label_preprocessor,
            "model": self.model,
        }

        target = base.with_suffix(".joblib")
        joblib.dump(data, target)
        self._logger.info("Saved DataInterface to '%s'.", str(target))

        # Example (commented) configuration for console logging instead of file:
        # import logging
        # from luna_xai.utils.logging_utils import get_luna_logger
        # logger = get_luna_logger("data.interface")
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.INFO)
        # logger.addHandler(console_handler)

    @classmethod
    def load(cls, path: str) -> "DataInterface":
        """
        Load a DataInterface saved via save().
        """
        base = Path(path)
        target = base.with_suffix(".joblib")
        obj = joblib.load(target)

        di = cls(
            preprocessor=obj.get("preprocessor"),
            label_preprocessor=obj.get("label_preprocessor"),
        )
        di.model = obj.get("model")

        LOGGER.info("Loaded DataInterface from '%s'.", str(target))
        return di

    # ------------------------------------------------------------------
    # Feature preprocessing
    # ------------------------------------------------------------------
    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "DataInterface":
        if hasattr(self.preprocessor, "fit"):
            self._debug(
                "Fitting preprocessor %s on data with shape %s.",
            )
            self._debug(
                "%s / shape=%s"
                % (
                    type(self.preprocessor).__name__,
                    getattr(X, "shape", "unknown"),
                )
            )
            self.preprocessor.fit(X)
        return self

    def transform_input(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray, Sequence[Any]],
    ) -> Any:

        if self.preprocessor is None:
            self._debug(
                "No preprocessor set. Returning input as-is from transform_input."
            )
            return X

        if isinstance(X, pd.Series):
            X = X.to_frame().T

        if hasattr(self.preprocessor, "feature_names_in_") and isinstance(
            X, pd.DataFrame
        ):
            expected = list(self.preprocessor.feature_names_in_)
            X = X[[c for c in expected if c in X.columns]]

        self._debug(
            "Transforming input using preprocessor %s.",
        )
        self._debug(
            "%s" % (type(self.preprocessor).__name__,)
        )

        if hasattr(self.preprocessor, "transform"):
            return self.preprocessor.transform(X)

        return X

    # ------------------------------------------------------------------
    def inverse_transform_input(self, X_transformed: Any) -> Any:
        if hasattr(self.preprocessor, "inverse_transform"):
            try:
                self._debug(
                    "Inverse transforming input using preprocessor %s.",
                )
                self._debug(
                    "%s" % (type(self.preprocessor).__name__,)
                )
                return self.preprocessor.inverse_transform(X_transformed)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "inverse_transform failed for preprocessor %s: %s. "
                    "Returning original transformed input.",
                    type(self.preprocessor).__name__,
                    exc,
                )
                return X_transformed
        return X_transformed

    # ------------------------------------------------------------------
    # Label preprocessing
    # ------------------------------------------------------------------
    def transform_output(
        self,
        y: Union[pd.Series, np.ndarray, Sequence[Any]],
    ) -> np.ndarray:

        y_arr = np.asarray(y)

        if hasattr(self.label_preprocessor, "transform"):
            self._debug(
                "Transforming output using label_preprocessor %s.",
            )
            self._debug(
                "%s" % (type(self.label_preprocessor).__name__,)
            )
            y_arr = y_arr.reshape(-1, 1)
            return self.label_preprocessor.transform(y_arr).ravel()

        return y_arr

    def inverse_transform_output(
        self,
        y_pred: Union[np.ndarray, Sequence[Any]],
    ) -> np.ndarray:

        y_arr = np.asarray(y_pred)

        if hasattr(self.label_preprocessor, "inverse_transform"):
            try:
                self._debug(
                    "Inverse transforming output using label_preprocessor %s.",
                )
                self._debug(
                    "%s" % (type(self.label_preprocessor).__name__,)
                )
                return self.label_preprocessor.inverse_transform(
                    y_arr.reshape(-1, 1)
                ).ravel()
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "inverse_transform failed for label_preprocessor %s: %s. "
                    "Returning original predictions.",
                    type(self.label_preprocessor).__name__,
                    exc,
                )
                return y_arr

        return y_arr

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def get_feature_names(
        self, X: Optional[pd.DataFrame] = None
    ) -> Optional[Union[list, np.ndarray]]:

        prep = self.preprocessor

        if hasattr(prep, "get_feature_names_out"):
            return prep.get_feature_names_out()

        if hasattr(prep, "feature_names_in_"):
            return prep.feature_names_in_

        if X is not None and hasattr(X, "columns"):
            return X.columns

        return None

    def has_pipeline(self) -> bool:
        return isinstance(self.preprocessor, Pipeline)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        p_name = type(self.preprocessor).__name__ if self.preprocessor else "None"
        l_name = (
            type(self.label_preprocessor).__name__
            if self.label_preprocessor
            else "None"
        )
        m_name = type(self.model).__name__ if self.model else "None"

        n_features = (
            len(self.preprocessor.feature_names_in_)  # type: ignore[arg-type]
            if hasattr(self.preprocessor, "feature_names_in_")
            else "unknown"
        )

        summary_str = (
            f"DataInterface(preprocessor={p_name}, model={m_name}, "
            f"label_preproc={l_name}, n_features={n_features})"
        )
        self._debug("DataInterface.summary(): %s",)
        self._debug(summary_str)
        return summary_str

    def __repr__(self) -> str:
        return self.summary()
