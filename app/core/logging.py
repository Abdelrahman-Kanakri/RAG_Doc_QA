"""Structured JSON logging configuration using structlog."""

import structlog

structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            structlog.processors.JSONRenderer()
            ],
        
        wrapper_class=structlog.make_filtering_bound_logger(min_level="INFO"),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True
        )

def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    Get a logger instance with the specified name.

    Args:
        name (str): The name of the logger. If None, the root logger is used.

    Returns:
        structlog.BoundLogger: A logger instance.
    """
    
    
    return structlog.get_logger(name)