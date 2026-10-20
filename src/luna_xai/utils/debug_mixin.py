from __future__ import annotations

from typing import Optional
from luna_xai.utils.logging_utils import get_luna_logger


class DebugMixin:
    """
    Reusable debug mixin for all LUNA components.

    Supports both:
    - instance-level debugging (self._debug_enabled + self._logger)
    - static debugging for utility modules (DebugMixin._debug_static)
    """

    # ------------------------------------------------------------------
    # Instance-level debug
    # ------------------------------------------------------------------
    def __init__(self, debug: bool = False, logger_name: Optional[str] = None):
        self._debug_enabled = bool(debug)
        name = logger_name or self.__class__.__name__.lower()
        self._logger = get_luna_logger(name)

    def _debug(self, msg: str) -> None:
        """Instance-level debug message."""
        if getattr(self, "_debug_enabled", False):
            self._logger.debug(msg)

    # ------------------------------------------------------------------
    # Static debug for module-level functions (used by io_utils)
    # ------------------------------------------------------------------
    @staticmethod
    def _debug_static(msg: str, *, logger, enabled: bool) -> None:
        """
        Static debug method for procedural modules.

        Parameters
        ----------
        msg : str
            Debug message.
        logger : logging.Logger
            Logger instance to emit the debug message.
        enabled : bool
            If True, emits the debug message; otherwise does nothing.
        """
        if enabled:
            logger.debug(msg)
