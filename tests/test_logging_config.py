"""Unit tests for logging_config module."""
from logging_config import JSONFormatter, get_logger
import pytest
import json
import logging
import os
import sys
from io import StringIO
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestJSONFormatter:
    """Test JSONFormatter functionality."""

    def test_json_formatter_basic_record(self):
        """Test that JSONFormatter produces valid JSON output."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=123,
            msg="Test message",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed['level'] == 'INFO'
        assert parsed['module'] == 'test'
        assert parsed['message'] == 'Test message'
        assert 'timestamp' in parsed
        # Check timezone is UTC (either Z or +00:00 format)
        assert parsed['timestamp'].endswith(
            'Z') or parsed['timestamp'].endswith('+00:00')

    def test_json_formatter_with_extra_data(self):
        """Test that extra_data is included in JSON output."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=123,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.extra_data = {"request_id": "123", "latency_ms": 45.6}

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed['request_id'] == "123"
        assert parsed['latency_ms'] == 45.6

    def test_json_formatter_with_exception(self):
        """Test that exceptions are properly formatted."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test exception")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=123,
            msg="Error occurred",
            args=(),
            exc_info=exc_info
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed['level'] == 'ERROR'
        assert 'exception' in parsed
        assert 'ValueError: Test exception' in parsed['exception']

    def test_json_formatter_output_is_valid_json(self):
        """Test that formatter always produces valid JSON."""
        formatter = JSONFormatter()

        # Test with various message types
        test_messages = [
            "Simple message",
            "Message with 'quotes' and \"double quotes\"",
            "Message with\nnewlines",
            "Message with special chars: éñ中文"
        ]

        for msg in test_messages:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg=msg,
                args=(),
                exc_info=None
            )
            output = formatter.format(record)
            # Should not raise exception
            json.loads(output)


class TestGetLogger:
    """Test get_logger functionality."""

    def test_get_logger_returns_logger_instance(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_sets_json_formatter(self):
        """Test that get_logger sets up JSONFormatter."""
        logger_name = "test_json_logger"
        logger = get_logger(logger_name)

        # Check that handler exists and has JSONFormatter
        assert len(logger.handlers) > 0
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)

    def test_json_formatter_with_different_log_levels(self):
        """Test that JSONFormatter works with different log levels."""
        formatter = JSONFormatter()

        test_cases = [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
        ]

        for level_int, level_str in test_cases:
            record = logging.LogRecord(
                name="test",
                level=level_int,
                pathname="test.py",
                lineno=1,
                msg="Test message",
                args=(),
                exc_info=None
            )

            output = formatter.format(record)
            parsed = json.loads(output)
            assert parsed['level'] == level_str

    def test_get_logger_default_level_is_info(self):
        """Test that get_logger uses INFO as default level."""
        logger = get_logger("test_default_level_logger")
        # Default should be INFO (20) since that's what's in .env.example
        assert logger.level == logging.INFO

    def test_get_logger_same_name_returns_same_instance(self):
        """Test that calling get_logger with same name returns same instance."""
        logger1 = get_logger("same_name_logger")
        logger2 = get_logger("same_name_logger")
        assert logger1 is logger2

    def test_logger_output_format(self):
        """Test that logger produces correctly formatted JSON output."""
        # Capture log output
        log_stream = StringIO()

        # Create logger with custom handler to capture output
        logger = logging.getLogger("test_output_logger")
        logger.handlers.clear()  # Remove any existing handlers

        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Log a message
        logger.info(
            "Test log message", extra={
                "extra_data": {
                    "test_key": "test_value"}})

        # Parse output
        output = log_stream.getvalue().strip()
        parsed = json.loads(output)

        assert parsed['message'] == "Test log message"
        assert parsed['level'] == "INFO"
        assert parsed['test_key'] == "test_value"
