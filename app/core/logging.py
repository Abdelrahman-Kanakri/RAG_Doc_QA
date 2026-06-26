"""Structured JSON logging configuration using structlog."""

import os
import sys
import logging
import structlog
from structlog.types import FilteringBoundLogger

# ── Configuration ───────────────────────────────────────────────────────────
logging.basicConfig(level = logging.INFO,
            filename = "log.log",
            filemode = "a",)

structlog.configure(
    processors=[
        # Merges context variables (useful for tracking web requests)
        structlog.contextvars.merge_contextvars,
        # Adds the log level (e.g., "info", "error") to the JSON
        structlog.processors.add_log_level,
        # Adds the logger name to the JSON payload automatically
        structlog.stdlib.add_logger_name,
        # Captures stack traces if an exception occurs
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        # Standard ISO 8601 timestamp format is preferred in production (e.g., for ELK/Splunk)
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Renders the final output as a JSON string
        structlog.processors.JSONRenderer()
    ],
    # Correct type hint binding for modern structlog versions
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True
)
# ── Logger factory ──────────────────────────────────────────────────────────

def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Get a structured JSON logger instance with the specified name.

    Args:
        name: The name of the logger (usually __name__).

    Returns:
        A bound logger instance configured for JSON output.
    """
    # Using kwargs forces the name into the structlog processor pipeline
    return structlog.get_logger(logger_name=name)
