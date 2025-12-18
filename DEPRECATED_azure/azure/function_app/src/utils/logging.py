"""Centralized logging configuration for the NBA prediction system.

This module provides a consistent logging setup across all modules.
Logs are output in JSON format for easy parsing by logging aggregators.
"""
import logging
import sys
from typing import Optional
import json
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add any extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


def setup_logger(
    name: str,
    level: Optional[str] = None,
    json_format: bool = True,
) -> logging.Logger:
    """Set up a logger with consistent formatting.

    Args:
        name: Logger name (usually __name__ of the module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, uses LOG_LEVEL env var or defaults to INFO.
        json_format: If True, use JSON formatting. If False, use human-readable format.

    Returns:
        Configured logger instance
    """
    import os

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Determine log level
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    logger.setLevel(getattr(logging, level, logging.INFO))

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level, logging.INFO))

    # Set formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the standard configuration.

    This is a convenience function for getting a logger.

    Args:
        name: Logger name (usually __name__ of the module)

    Returns:
        Configured logger instance
    """
    return setup_logger(name)

