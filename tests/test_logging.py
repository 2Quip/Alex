"""Tests for app/core/logging.py."""

import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.core.logging import JsonLogFormatter, logger_hook, setup_logging


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Reset root logger after each test to avoid cross-test pollution."""
    yield
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)


def test_setup_logging_creates_handlers(tmp_path):
    log_file = str(tmp_path / "test.log")
    setup_logging(log_level="INFO", log_file=log_file)

    root = logging.getLogger()
    handler_types = [type(h) for h in root.handlers]
    assert logging.StreamHandler in handler_types
    assert RotatingFileHandler in handler_types


def test_setup_logging_sets_level(tmp_path):
    log_file = str(tmp_path / "test.log")
    setup_logging(log_level="DEBUG", log_file=log_file)

    root = logging.getLogger()
    assert root.level == logging.DEBUG


def test_setup_logging_creates_log_directory(tmp_path):
    log_dir = tmp_path / "nested" / "logs"
    log_file = str(log_dir / "test.log")
    setup_logging(log_file=log_file)

    assert log_dir.exists()


def test_setup_logging_suppresses_noisy_loggers(tmp_path):
    log_file = str(tmp_path / "test.log")
    setup_logging(log_file=log_file)

    for name in ("httpx", "httpcore", "urllib3"):
        assert logging.getLogger(name).level == logging.WARNING


def test_setup_logging_clears_previous_handlers(tmp_path):
    log_file = str(tmp_path / "test.log")
    setup_logging(log_file=log_file)
    setup_logging(log_file=log_file)

    root = logging.getLogger()
    # Should have exactly 2 handlers (console + file), not 4
    assert len(root.handlers) == 2


def test_setup_logging_invalid_level_defaults_to_info(tmp_path):
    log_file = str(tmp_path / "test.log")
    setup_logging(log_level="INVALID", log_file=log_file)

    root = logging.getLogger()
    assert root.level == logging.INFO


# --- logger_hook ---


def test_logger_hook_returns_result():
    fn = MagicMock(return_value=42)
    result = logger_hook("my_func", fn, {"x": 1})
    assert result == 42
    fn.assert_called_once_with(x=1)


def test_logger_hook_measures_duration(caplog):
    def slow_fn(**kwargs):
        time.sleep(0.05)
        return "done"

    with caplog.at_level(logging.INFO, logger="app.tools"):
        logger_hook("slow_func", slow_fn, {})

    assert any("slow_func" in r.message and "executed in" in r.message for r in caplog.records)


def test_logger_hook_truncates_long_args(caplog):
    long_args = {"data": "x" * 1000}

    with caplog.at_level(logging.INFO, logger="app.tools"):
        logger_hook("my_func", MagicMock(return_value=None), long_args)

    for record in caplog.records:
        if "my_func" in record.message and "args=" in record.message:
            # The args portion should be truncated to 500 chars
            assert len(record.message) < 600


# --- JSON formatter ---


def test_json_formatter_produces_valid_json():
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Hello %s",
        args=("world",),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test.logger"
    assert parsed["message"] == "Hello world"
    assert "timestamp" in parsed
    assert "exception" not in parsed


def test_json_formatter_includes_exception():
    formatter = JsonLogFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="test.py",
        lineno=1,
        msg="something failed",
        args=(),
        exc_info=exc_info,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert "exception" in parsed
    assert any("ValueError" in line for line in parsed["exception"])


def test_json_formatter_includes_extra_fields():
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name="app.tools",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="tool ran",
        args=(),
        exc_info=None,
    )
    record.tool_name = "search"
    record.duration_s = 1.234
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["tool_name"] == "search"
    assert parsed["duration_s"] == 1.234


def test_setup_logging_json_format_uses_json_formatter(tmp_path):
    log_file = str(tmp_path / "test.log")
    setup_logging(log_level="INFO", log_file=log_file, log_format="json")

    root = logging.getLogger()
    for handler in root.handlers:
        assert isinstance(handler.formatter, JsonLogFormatter)


def test_setup_logging_text_format_uses_standard_formatter(tmp_path):
    log_file = str(tmp_path / "test.log")
    setup_logging(log_level="INFO", log_file=log_file, log_format="text")

    root = logging.getLogger()
    for handler in root.handlers:
        assert isinstance(handler.formatter, logging.Formatter)
        assert not isinstance(handler.formatter, JsonLogFormatter)


def test_logger_hook_passes_extra_fields(caplog):
    fn = MagicMock(return_value="ok")

    with caplog.at_level(logging.INFO, logger="app.tools"):
        logger_hook("search_tool", fn, {"q": "test"})

    info_records = [r for r in caplog.records if r.levelno == logging.INFO and "search_tool" in r.message]
    assert len(info_records) == 1
    record = info_records[0]
    assert record.tool_name == "search_tool"
    assert isinstance(record.duration_s, float)
