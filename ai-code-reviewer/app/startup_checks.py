"""
Startup validation -- fail fast if required config is missing.

Same pattern as multi-source-rag. Check everything required before
any graph node runs, so failures surface immediately with a clear
message rather than deep inside a LangGraph node.
"""

from app.config import settings
from app.logger import get_logger

log = get_logger(__name__)


class ConfigError(Exception):
    pass


def validate_startup_config() -> None:
    errors = []

    if not settings.github_token.strip():
        errors.append(
            "GITHUB_TOKEN is not set. Create one at: "
            "github.com -> Settings -> Developer settings -> "
            "Personal access tokens -> Tokens (classic)"
        )

    if not settings.github_repo.strip():
        errors.append(
            "GITHUB_REPO is not set. Format: 'owner/repo-name' "
            "e.g. 'praveenkumar993/sample-python-api'"
        )

    if not settings.groq_api_key.strip():
        errors.append(
            "GROQ_API_KEY is not set. "
            "Get a free key at: console.groq.com"
        )

    if errors:
        for e in errors:
            log.error("startup_config_error", detail=e)
        raise ConfigError(
            "Startup validation failed:\n- " + "\n- ".join(errors)
        )

    log.info(
        "startup_config_valid",
        repo=settings.github_repo,
        tracing="enabled" if settings.langsmith_api_key else "disabled",
    )