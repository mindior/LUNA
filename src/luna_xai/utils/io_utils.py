"""
I/O Utilities for LUNA XAI
==========================

Provides safe and standardized file operations for:
- Model loading (joblib)
- Dataset loading (CSV, Parquet, Excel)
- Directory management
- Data export (CSV, Excel)
- Safe formatting and type conversions
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Union

import joblib
import pandas as pd

from luna_xai.utils.logging_utils import get_luna_logger
from luna_xai.utils.debug_mixin import DebugMixin  # NEW

PathLike = Union[str, Path]

# Module-level logger for I/O operations.
_LOGGER = get_luna_logger("utils.io")


# ---------------------------------------------------------------------
# 🔹 Internal debug helper (now routed through DebugMixin)
# ---------------------------------------------------------------------
def _debug_io(msg: str, verbose: bool) -> None:
    """
    Lightweight debug helper routed through DebugMixin.

    Parameters
    ----------
    msg : str
        Debug message.
    verbose : bool
        If True, emits the message through logger.debug().
    """
    DebugMixin._debug_static(msg, logger=_LOGGER, enabled=verbose)


# ---------------------------------------------------------------------
# 🔹 Model and Data Loading
# ---------------------------------------------------------------------
def load_model(model_path: PathLike, *, verbose: bool = False) -> Any:
    """
    Load a serialized model from disk using joblib.

    Parameters
    ----------
    model_path : str or Path
        Path to the serialized model file (typically .pkl or .joblib).
    verbose : bool, default False
        If True, emits debugging information.

    Returns
    -------
    Any
        The deserialized model.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    _debug_io(f"Loading model from: {model_path}", verbose)
    return joblib.load(model_path)


def load_data(data_path: PathLike, *, verbose: bool = False) -> pd.DataFrame:
    """
    Load a dataset from CSV, Parquet, or Excel formats.

    Parameters
    ----------
    data_path : str or Path
        Path to the data file.
    verbose : bool, default False
        If True, emits debugging information.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is unsupported.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    ext = data_path.suffix.lower()
    _debug_io(f"Reading dataset: {data_path}", verbose)

    if ext == ".csv":
        df = pd.read_csv(data_path)
    elif ext == ".parquet":
        df = pd.read_parquet(data_path)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(data_path)
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            "Supported: .csv, .parquet, .xlsx, .xls."
        )

    _debug_io(f"Dataset loaded successfully. Shape: {df.shape}", verbose)
    return df


# ---------------------------------------------------------------------
# 🔹 Directory and File Utilities
# ---------------------------------------------------------------------
def ensure_directory(path: PathLike, *, verbose: bool = False) -> Path:
    """
    Ensure a directory exists, creating it if needed.

    Parameters
    ----------
    path : str or Path
        Directory path.
    verbose : bool, default False
        If True, emits debugging information.

    Returns
    -------
    Path
        The ensured directory path.

    Raises
    ------
    NotADirectoryError
        If the path exists but is not a directory.
    """
    path = Path(path)

    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Path exists and is not a directory: {path}")

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        _debug_io(f"Created directory: {path}", verbose)

    return path


def save_dataframe_to_excel(
    df: pd.DataFrame,
    file_path: PathLike,
    *,
    index: bool = False,
    sheet_name: str = "Sheet1",
    verbose: bool = False,
) -> None:
    """Save a DataFrame to Excel."""
    file_path = Path(file_path)
    ensure_directory(file_path.parent, verbose=verbose)
    df.to_excel(file_path, index=index, sheet_name=sheet_name)
    _debug_io(f"DataFrame exported to Excel: {file_path}", verbose)


def save_dataframe_to_csv(
    df: pd.DataFrame,
    file_path: PathLike,
    *,
    index: bool = False,
    sep: str = ",",
    verbose: bool = False,
) -> None:
    """Save a DataFrame to CSV."""
    file_path = Path(file_path)
    ensure_directory(file_path.parent, verbose=verbose)
    df.to_csv(file_path, index=index, sep=sep)
    _debug_io(f"DataFrame exported to CSV: {file_path}", verbose)


# ---------------------------------------------------------------------
# 🔹 Safe Formatting & Conversion
# ---------------------------------------------------------------------
def safe_format(value: Any, precision: int = 3) -> str:
    """Safely format numeric or string values."""
    try:
        if isinstance(value, (float, int)):
            return f"{value:.{precision}g}"
        return str(value)
    except Exception:
        return str(value)


def auto_convert_numeric(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Convert object columns to numeric when most values are numeric.

    Parameters
    ----------
    df : pandas.DataFrame
    threshold : float in [0,1]

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        If threshold is outside [0,1].
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            "Parameter 'threshold' must be between 0 and 1 inclusive."
        )

    df_converted = df.copy()

    for col in df_converted.select_dtypes(include="object").columns:
        try:
            numeric_col = pd.to_numeric(df_converted[col], errors="coerce")
            if numeric_col.notna().mean() > threshold:
                df_converted[col] = numeric_col
        except Exception:
            continue

    return df_converted
