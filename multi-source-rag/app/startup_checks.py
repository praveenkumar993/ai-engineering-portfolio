"""
Startup validation — check required config BEFORE doing any real work.

Why this matters:
    Without this, a missing GROQ_API_KEY wouldn't surface until after we've
    already loaded files, downloaded an embedding model, embedded chunks,
    and written to Qdrant -- all wasted work, and a confusing error deep in
    a stack trace. Failing fast at startup, with a clear message pointing
    at exactly what's missing, is a small thing that saves a lot of
    confused debugging time.
"""

from app.config import settings
from app.logger import get_logger

log = get_logger(__name__)


class ConfigError(Exception):
    """Raised when required configuration is missing at startup."""


def validate_startup_config() -> None:
    errors = []

    if not settings.groq_api_key.strip():
        errors.append(
            "GROQ_API_KEY is not set. Add it to your .env file. "
            "Get a free key at https://console.groq.com"
        )

    if errors:
        for e in errors:
            log.error("startup_config_error", detail=e)
        raise ConfigError(
            "Startup validation failed:\n- " + "\n- ".join(errors)
        )

    log.info("startup_config_valid")