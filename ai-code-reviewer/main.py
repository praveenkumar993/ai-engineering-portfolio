"""
Entry point.

Step 1:  Config + logging spine
Step 2:  PRReviewState verified
Step 3:  GitHub client verified
Step 4:  PR Fetcher node
Step 5:  Summarizer node
Step 6:  First working LangGraph graph
Step 7:  Supervisor node
Step 8:  Specialist agents (Security, Logic, Design)
Step 9:  Writer node
Step 10: Publisher node
Step 11: Human-in-the-loop approval gate

The graph runs in two phases when require_human_approval=True:

Phase 1: pr_fetcher -> summarizer -> supervisor -> agents
         -> writer -> human_approval [PAUSE]
         Operator reads the draft review and decides.

Phase 2: graph resumes from checkpoint with operator decision.
         'approve' -> publisher -> review posted to GitHub
         'reject'  -> graph ends, review discarded
"""

from app.config import settings
from app.logger import configure_logging, get_logger
from app.startup_checks import validate_startup_config, ConfigError
from app.state import PRReviewState

configure_logging()
log = get_logger(__name__)

THREAD_CONFIG = {"configurable": {"thread_id": "pr-1"}}


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

    from app.mcp_client.github_client import get_github_client
    from app.graph import build_graph

    github_client = get_github_client()
    graph = build_graph(github_client)

    log.info("graph_run_started", pr_number=1)

    # --- Phase 1: run until interrupt or completion ---
    list(graph.stream(
        PRReviewState(
            repo=settings.github_repo,
            pr_number=1,
        ),
        config=THREAD_CONFIG,
        stream_mode="values",
    ))

    state_snapshot = graph.get_state(THREAD_CONFIG)

    if state_snapshot.next == ("human_approval",):
        # Graph paused at the approval gate
        review = state_snapshot.values.get("review_comment", "")

        print("\n" + "=" * 60)
        print("DRAFT REVIEW — WAITING FOR APPROVAL")
        print("=" * 60)
        print(review)
        print("=" * 60)
        print("\nOptions:")
        print("  approve / yes / y  → post the review to GitHub")
        print("  reject  / no  / n  → discard, do not post")
        print()

        decision = input("Your decision: ").strip().lower()
        log.info("graph_resuming", decision=decision)

        # --- Phase 2: resume with the operator's decision ---
        from langgraph.types import Command
        list(graph.stream(
            Command(resume=decision),
            config=THREAD_CONFIG,
            stream_mode="values",
        ))

        final_state = graph.get_state(THREAD_CONFIG).values

    else:
        # Graph completed without pausing
        # (require_human_approval=False)
        final_state = state_snapshot.values

    # --- Show result ---
    posted = final_state.get("review_posted", False)
    error = final_state.get("error", "")

    if error:
        print(f"\n❌ {error}")
    elif posted:
        print(f"\n✅ Review posted to GitHub PR #1")
        print(f"   https://github.com/{settings.github_repo}/pull/1")
    else:
        print("\n⚠️  Review was produced but not posted.")

    log.info(
        "graph_run_completed",
        review_posted=posted,
        error=error or None,
    )


if __name__ == "__main__":
    main()