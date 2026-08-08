"""
Structured logging setup.

Why not print()?
    print() gives you unstructured text with no timestamp, no severity level,
    no way to filter, and no way to search in a log aggregator (e.g. Azure
    Monitor, Datadog, ELK). In production, logs are DATA, not decoration.

    structlog gives every log line as a JSON object, e.g.:
        {"event": "document_loaded", "source": "local", "file": "a.pdf",
         "level": "info", "timestamp": "..."}

    That means later, when we add monitoring, we can filter logs by field
    (e.g. "show me every log where source=azure_blob and level=error")
    instead of grepping raw text.
"""

import logging
import sys

import structlog

from app.config import settings


def configure_logging() -> None:
    """Call this once, at app startup, before anything else logs."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=settings.log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    """Every module calls this to get its own named logger."""
    return structlog.get_logger(name)
