"""
Entry point.

Step 11: Human-in-the-loop approval gate added between Writer
         and Publisher. The graph now runs in two phases:

Phase 1: pr_fetcher -> summarizer -> supervisor -> agents -> writer
         -> human_approval [PAUSE]
         The review is shown to the operator. Graph is frozen.

Phase 2: operator types 'approve' or 'reject'
         Graph resumes from the checkpoint.
         'approve' -> publisher -> review posted to GitHub
         'reject'  -> graph ends, review discarded
"""

from app.config import settings
from app.logger import configure_logging, get_logger
from app.startup_checks import validate_startup_config, ConfigError
from app.state import PRReviewState

configure_logging()
log = get_logger(__name__)

# A thread_id ties both phases of a run together.
# LangGraph uses it to find the right checkpoint when resuming.
# Using the PR number means each PR has its own checkpoint thread.
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

    # --- Phase 1: run the graph until it pauses ---
    log.info("graph_run_started", pr_number=1)

    initial_state = PRReviewState(
        repo=settings.github_repo,
        pr_number=1,
    )

    # stream() lets us see the interrupt event as it happens.
    # invoke() would work too but would silently swallow the pause.
    events = list(graph.stream(
        initial_state,
        config=THREAD_CONFIG,
        stream_mode="values",
    ))

    # Check if the graph paused at the interrupt
    state_snapshot = graph.get_state(THREAD_CONFIG)

    if state_snapshot.next == ("human_approval",):
        # Graph is paused inside human_approval_node waiting for input
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

        # --- Phase 2: resume the graph with the operator's decision ---
        log.info("graph_resuming", decision=decision)

        final_events = list(graph.stream(
            # None means "resume from checkpoint, don't restart"
            # The Command with resume passes our decision to interrupt()
            {"type": "resume", "value": decision} if False else None,
            config=THREAD_CONFIG,
            stream_mode="values",
        ))

        # Pass the decision back through the resume mechanism
        from langgraph.types import Command
        final_events = list(graph.stream(
            Command(resume=decision),
            config=THREAD_CONFIG,
            stream_mode="values",
        ))

        final_state = graph.get_state(THREAD_CONFIG).values

    else:
        # Graph completed without pausing (require_human_approval=False)
        final_state = state_snapshot.values

    # --- Show result ---
    review = final_state.get("review_comment", "")
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