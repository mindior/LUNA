"""
Logging utilities for LUNA XAI
==============================

This module provides a thin, library-friendly logging layer for LUNA XAI.

Goals
-----
- Avoid configuring global logging implicitly inside the library.
- Provide a consistent logger naming convention: "luna_xai" and
  "luna_xai.<subsystem>".
- Offer a single entry point to configure logging when desired
  (e.g. from CLI, demos, or user applications).

Usage
-----

1) Inside core classes
----------------------

Classes can create a logger and delegate debug messages to it:

    from luna_xai.utils.logging_utils import get_luna_logger

    class HybridPreprocessor:
        def __init__(..., debug: bool = False) -> None:
            self._logger = get_luna_logger("preprocessing.hybrid")
            self._debug_enabled = bool(debug)

        def _debug(self, msg: str) -> None:
            if self._debug_enabled:
                self._logger.debug(msg)

2) Module-level helpers
-----------------------

For helpers like `_debug_sampling`, use a module-level logger:

    from luna_xai.utils.logging_utils import get_luna_logger

    LOGGER = get_luna_logger("sampling")

    def _debug_sampling(msg: str, verbose: bool) -> None:
        if verbose:
            LOGGER.debug(msg)

3) Configuring logging from the outside (CLI / demos / user code)
-----------------------------------------------------------------

The library does not configure global logging by default. To enable
console logs for LUNA, call:

    from luna_xai.utils.logging_utils import configure_luna_logging

    configure_luna_logging(level="INFO")

By default this configures a StreamHandler (console). If you prefer
file-based logging, you can either pass a custom stream or attach your
own FileHandler to the "luna_xai" logger.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Optional, TextIO, Union

LogLevel = Union[int, str]

# Base logger name for all LUNA components
_BASE_LOGGER_NAME = "luna_xai"

# Internal flag to avoid re-initializing the base logger
_BASE_LOGGER_INITIALIZED = False


def _ensure_base_logger() -> logging.Logger:
    """
    Ensure that the base LUNA logger exists and has a NullHandler.

    The base logger is named "luna_xai". A NullHandler is attached to
    avoid "No handler could be found" warnings when the application
    has not configured logging yet. User code may override this
    configuration via :func:`configure_luna_logging`.
    """
    global _BASE_LOGGER_INITIALIZED

    logger = logging.getLogger(_BASE_LOGGER_NAME)

    if not _BASE_LOGGER_INITIALIZED:
        # Attach a NullHandler only once, if there are no handlers yet.
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.NOTSET)
        _BASE_LOGGER_INITIALIZED = True

    return logger


def _coerce_log_level(level: LogLevel) -> int:
    """
    Normalize a log level value to an integer constant.

    Parameters
    ----------
    level : int or str
        Logging level. If a string is provided, it is case-insensitive
        and must match one of the standard logging level names
        (e.g. "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").

    Returns
    -------
    int
        Numeric logging level.

    Raises
    ------
    ValueError
        If the level string is not a valid logging level.
    """
    if isinstance(level, int):
        return level

    if not isinstance(level, str):
        raise ValueError(
            f"Invalid log level type {type(level)!r}; expected int or str."
        )

    level_upper = level.upper()
    if level_upper not in logging._nameToLevel:  # type: ignore[attr-defined]
        raise ValueError(
            f"Unknown log level '{level}'. "
            f"Valid names are: {sorted(logging._nameToLevel.keys())}."
        )
    return logging._nameToLevel[level_upper]  # type: ignore[attr-defined]


def get_luna_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger scoped under the LUNA base logger.

    Parameters
    ----------
    name : str, optional
        Sub-logger name. If provided, the resulting logger will be
        named "luna_xai.<name>". If None, the base "luna_xai" logger
        is returned.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> logger = get_luna_logger("sampling")
    >>> logger.debug("Sampling started")
    """
    _ensure_base_logger()

    if not name:
        return logging.getLogger(_BASE_LOGGER_NAME)

    full_name = f"{_BASE_LOGGER_NAME}.{name}"
    return logging.getLogger(full_name)


def configure_luna_logging(
    level: LogLevel = "INFO",
    *,
    stream: Optional[TextIO] = None,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
) -> None:
    """
    Configure console logging for the LUNA XAI loggers.

    This is a convenience function intended to be called explicitly
    by the CLI, demos, or user applications. It will:

    - Ensure the base "luna_xai" logger exists.
    - Remove any existing handlers from the base logger.
    - Attach a new `StreamHandler` with a configurable format.
    - Set the desired log level on the base logger.
    - Disable propagation to the root logger to avoid duplicate logs.

    Parameters
    ----------
    level : int or str, default "INFO"
        Desired log level for LUNA loggers. Accepts either numeric
        logging levels (e.g. ``logging.DEBUG``) or standard level
        names (e.g. "DEBUG", "INFO", "WARNING").
    stream : file-like object, optional
        Stream to which logs will be written. If None, defaults to
        ``sys.stderr`` (as in the default `StreamHandler`).
    fmt : str, optional
        Logging format string. If None, a simple default format is used:
        ``"[%(levelname)s] %(name)s - %(message)s"``.
    datefmt : str, optional
        Date format string used by the formatter. If None, the default
        logging date format is used.

    Notes
    -----
    By default this configures logging to a stream (typically the console).
    If you prefer a file-based handler instead, you can:

    - pass a custom stream that writes to a file, or
    - use :func:`get_luna_logger` and attach a `FileHandler` manually.
    """
    base_logger = _ensure_base_logger()

    level_num = _coerce_log_level(level)
    base_logger.setLevel(level_num)

    # Remove existing handlers (including the NullHandler)
    base_logger.handlers.clear()

    handler = logging.StreamHandler(stream)
    formatter = logging.Formatter(
        fmt or "[%(levelname)s] %(name)s - %(message)s",
        datefmt=datefmt,
    )
    handler.setFormatter(formatter)
    handler.setLevel(level_num)

    base_logger.addHandler(handler)

    # Avoid propagating to the root logger by default
    base_logger.propagate = False


def enable_debug_for_logger(name: str) -> None:
    """
    Convenience helper to set a specific LUNA logger to DEBUG level.

    This is useful in interactive sessions or debugging scenarios where
    you only want verbose output from a particular subsystem.

    Parameters
    ----------
    name : str
        Sub-logger name (without the "luna_xai." prefix). For example,
        "sampling" or "preprocessing.hybrid".
    """
    logger = get_luna_logger(name)
    logger.setLevel(logging.DEBUG)


@contextmanager
def log_execution_segment(
    segment_name: str,
    *,
    logger: Optional[logging.Logger] = None,
    level: LogLevel = "INFO",
) -> None:
    """
    Context manager to log the start, end and duration of a code segment.

    This is intended for demos, CLI tools and optional instrumentation in
    core components. It does not configure handlers; it only emits log
    messages to the provided or default LUNA logger.

    Parameters
    ----------
    segment_name : str
        Human-readable name of the segment, e.g. "Load model",
        "Build domain", "Explain instance #42".
    logger : logging.Logger, optional
        Logger instance to use. If None, the base "luna_xai" logger
        is used.
    level : int or str, default "INFO"
        Logging level used for the start/end messages.

    Examples
    --------
    >>> from luna_xai.utils.logging_utils import log_execution_segment
    >>> with log_execution_segment("Build domain"):
    ...     domain = TabularDomainManager.create_and_save(...)

    The log output will look roughly like:
    [INFO] luna_xai - [START] Build domain
    [INFO] luna_xai - [END]   Build domain (0.123 s)
    """
    log = logger or get_luna_logger()
    lvl = _coerce_log_level(level)

    start_msg = f"[START] {segment_name}"
    end_template = "[END]   {name} ({elapsed:.3f} s)"

    start_time = time.perf_counter()
    log.log(lvl, start_msg)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        end_msg = end_template.format(name=segment_name, elapsed=elapsed)
        log.log(lvl, end_msg)
