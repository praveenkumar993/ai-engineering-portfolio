"""
Structured logging -- identical pattern to multi-source-rag.

One addition: we also set LangSmith environment variables here if
they are configured, so tracing starts before the first LangGraph
node runs. LangSmith reads these from the environment automatically
-- no other setup needed.
"""

import logging
import os
import sys
from pathlib import Path

import structlog

from app.config import settings

LOG_FILE_PATH = Path("logs/app.jsonl")


def configure_logging() -> None:
    LOG_FILE_PATH.parent.mkdir(exist_ok=True)

    # Set LangSmith env vars if configured -- LangGraph picks these
    # up automatically and sends traces to smith.langchain.com
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

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
    return structlog.get_logger(name)