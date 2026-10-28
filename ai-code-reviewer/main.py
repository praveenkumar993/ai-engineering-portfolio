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
            # --- Step 2: verify state definition works ---
        from app.state import PRReviewState, CodeIssue

        # create a sample state the way the graph will at startup
        initial_state = PRReviewState(
            repo=settings.github_repo,
            pr_number=1,
        )
        log.info(
            "state_initialized",
            repo=initial_state.repo,
            pr_number=initial_state.pr_number,
            fields=list(PRReviewState.model_fields.keys()),
        )

        # prove the Annotated[list, add] reducer works --
        # two agents adding issues should APPEND not OVERWRITE
        from operator import add as list_add
        issues_from_security = [CodeIssue(
            file="auth_service.py",
            line_reference="line 12",
            severity="critical",
            issue_type="hardcoded_credential",
            description="API key hardcoded in source",
            suggested_fix="Use os.getenv('API_KEY')",
        )]
        issues_from_logic = [CodeIssue(
            file="user_service.py",
            line_reference="line 34",
            severity="medium",
            issue_type="missing_null_check",
            description="user could be None before accessing .email",
            suggested_fix="Add: if user is None: raise UserNotFoundError()",
        )]
        combined = list_add(issues_from_security, issues_from_logic)
        log.info(
            "state_reducer_verified",
            total_issues=len(combined),
            files=[i.file for i in combined],
        )

        # --- Step 3: verify GitHub client can connect ---
        from app.mcp_client.github_client import get_github_client

        try:
            client = get_github_client()
            log.info("github_client_verified", repo=settings.github_repo)
        except Exception as e:
            log.error("github_client_verification_failed", error=str(e))
            return

    except ConfigError as e:
        print(f"\nStartup failed:\n{e}\n")
        return

    log.info("app_ready")


if __name__ == "__main__":
    main()