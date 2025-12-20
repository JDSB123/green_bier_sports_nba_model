"""Tests for logging utilities."""
import logging
import json
import pytest

from src.utils.logging import setup_logger, get_logger, JSONFormatter


class TestJSONFormatter:
    """Tests for JSON log formatter."""

    def test_json_formatter_basic(self):
        """Test that JSONFormatter produces valid JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        output = formatter.format(record)
        data = json.loads(output)
        
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "Test message"
        assert data["line"] == 10
        assert "timestamp" in data

    def test_json_formatter_with_exception(self):
        """Test that JSONFormatter includes exception info."""
        formatter = JSONFormatter()
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=20,
                msg="Error occurred",
                args=(),
                exc_info=exc_info,
            )
            
            output = formatter.format(record)
            data = json.loads(output)
            
            assert "exception" in data
            assert "ValueError: Test error" in data["exception"]


class TestSetupLogger:
    """Tests for logger setup."""

    def test_setup_logger_default(self):
        """Test default logger setup."""
        logger = setup_logger("test_logger_1")
        assert logger.name == "test_logger_1"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_setup_logger_custom_level(self):
        """Test logger setup with custom level."""
        logger = setup_logger("test_logger_2", level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_setup_logger_no_duplicates(self):
        """Test that calling setup_logger twice doesn't create duplicate handlers."""
        logger1 = setup_logger("test_logger_3")
        handler_count1 = len(logger1.handlers)
        
        logger2 = setup_logger("test_logger_3")
        handler_count2 = len(logger2.handlers)
        
        assert handler_count1 == handler_count2
        assert logger1 is logger2

    def test_get_logger(self):
        """Test get_logger convenience function."""
        logger = get_logger("test_logger_4")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger_4"

