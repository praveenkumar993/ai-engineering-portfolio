"""
Temporary entry point for Step 1: prove config + logging work.

This file will evolve into the real app entry point (FastAPI startup) in a
later step. Right now it does ONE job: confirm the spine is solid before we
build anything on top of it.
"""

from app.config import settings
from app.logger import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)


def main() -> None:
    log.info(
        "app_starting",
        app_name=settings.app_name,
        environment=settings.environment,
        log_level=settings.log_level,
    )
    log.warning("this_is_a_warning_example")
    log.info("app_ready")


if __name__ == "__main__":
    main()
