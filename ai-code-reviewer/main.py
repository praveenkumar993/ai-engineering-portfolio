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

            # --- Step 4: PR Fetcher node ---
        from app.nodes.pr_fetcher import pr_fetcher_node
        from app.state import PRReviewState

        state = PRReviewState(
            repo=settings.github_repo,
            pr_number=1,
        )

        log.info("pr_fetcher_running", pr_number=state.pr_number)
        updates = pr_fetcher_node(state, client)

        if updates.get("error"):
            log.error("pr_fetcher_error", detail=updates["error"])
            return

        log.info(
            "pr_fetcher_result",
            pr_title=updates["pr_title"],
            changed_files=updates["changed_files"],
            files_read=list(updates["file_contents"].keys()),
            diff_preview=updates["diff"][:200],
        )

            # --- Step 5: Summarizer node ---
        from app.llm import get_llm
        from app.nodes.summarizer import summarizer_node

        # LangGraph will do this merge for us in Step 6. Until then we
        # do it by hand, so the Summarizer sees what the Fetcher wrote.
        state = state.model_copy(update=updates)

        llm = get_llm()
        summary_updates = summarizer_node(state, llm)

        if summary_updates.get("error"):
            log.error("summarizer_error", detail=summary_updates["error"])
            return

        print("\n" + "=" * 60)
        print("PR SUMMARY")
        print("=" * 60)
        print(summary_updates["summary"])
        print("=" * 60 + "\n")

    except ConfigError as e:
        print(f"\nStartup failed:\n{e}\n")
        return

    log.info("app_ready")


if __name__ == "__main__":
    main()