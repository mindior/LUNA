# tests/test_logging_utils.py
import io
import pytest

from luna_xai.utils.logging_utils import (
    get_luna_logger,
    configure_luna_logging,
    enable_debug_for_logger,
    log_execution_segment,
)


def test_get_luna_logger_hierarchy():
    base = get_luna_logger()
    child = get_luna_logger("sampling")

    assert child.name.startswith(base.name)
    assert child.name == "luna_xai.sampling"


def test_configure_luna_logging_to_stream():
    stream = io.StringIO()
    configure_luna_logging(level="INFO", stream=stream)
    logger = get_luna_logger("test")

    logger.info("hello")
    out = stream.getvalue()
    assert "hello" in out
    assert "luna_xai.test" in out


def test_enable_debug_for_logger():
    logger = get_luna_logger("debug_test")
    enable_debug_for_logger("debug_test")
    assert logger.level <= 10  # DEBUG


def test_log_execution_segment_context_manager():
    stream = io.StringIO()
    configure_luna_logging(level="INFO", stream=stream)
    logger = get_luna_logger("segment")

    with log_execution_segment("MySegment", logger=logger, level="INFO"):
        logger.info("inside")

    text = stream.getvalue()
    assert "[START] MySegment" in text
    assert "[END]   MySegment" in text
    assert "inside" in text
