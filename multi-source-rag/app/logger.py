"""
Structured logging setup.

Why not print()?
    print() gives you unstructured text with no timestamp, no severity level,
    no way to filter, and no way to search in a log aggregator. In
    production, logs are DATA, not decoration.

Why write to a FILE now, not just stdout:
    Everything logged so far only lived in your terminal window -- once
    closed, gone. A real system needs logs to persist, so you (or a
    monitoring tool) can look back at what happened earlier. Writing
    structured JSON lines to a file is the simplest version of what log
    aggregators (Datadog, ELK, CloudWatch) do at scale. This file is what
    scripts/metrics_summary.py reads to compute aggregate stats -- the
    bridge from "logging" (individual events) to "monitoring" (trends).
"""

import logging
import sys
from pathlib import Path

import structlog

from app.config import settings

LOG_FILE_PATH = Path("logs/app.jsonl")


def configure_logging() -> None:
    """Call this once, at app startup, before anything else logs."""
    LOG_FILE_PATH.parent.mkdir(exist_ok=True)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=settings.log_level,
    )

    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    logging.getLogger().addHandler(file_handler)

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
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    """Every module calls this to get its own named logger."""
    return structlog.get_logger(name)