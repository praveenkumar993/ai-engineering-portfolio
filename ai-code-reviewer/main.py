"""
Entry point -- Step 1: config + logging spine.

Same pattern as multi-source-rag/main.py. This file grows one
step at a time. Right now it just confirms the spine works.
"""

from app.config import settings
from app.logger import configure_logging, get_logger
from app.startup_checks import validate_startup_config, ConfigError

configure_logging()
log = get_logger(__name__)


def main() -> None:
    log.info(
        "app_starting",
        app_name=settings.app_name,
        environment=settings.environment,
    )

    try:
        validate_startup_config()
    except ConfigError as e:
        print(f"\nStartup failed:\n{e}\n")
        return

    log.info("app_ready")


if __name__ == "__main__":
    main()