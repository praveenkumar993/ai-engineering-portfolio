"""
Entry point.

Step 1: config + logging spine.
Step 2: PRReviewState definition verified.
Step 3: GitHub client verified.
Step 4: PR Fetcher node verified.
Step 5: Summarizer node verified.
Step 6: first working LangGraph graph -- PR Fetcher -> Summarizer.
        LangGraph now manages state transitions instead of us doing
        it manually.
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

    # --- Build graph ---
    from app.mcp_client.github_client import get_github_client
    from app.graph import build_graph

    github_client = get_github_client()
    graph = build_graph(github_client)

    # --- Run the graph ---
    # graph.invoke() takes the initial state values and returns the
    # final state after all nodes have run. LangGraph handles every
    # state merge, every conditional edge, and every node call.
    log.info("graph_run_started", pr_number=1)

    final_state = graph.invoke(
        PRReviewState(
            repo=settings.github_repo,
            pr_number=1,
        )
    )

    # --- Show results ---
    if final_state.get("error"):
        log.error("graph_run_failed", error=final_state["error"])
        return

    print("\n" + "=" * 60)
    print("PR SUMMARY")
    print("=" * 60)
    print(final_state["summary"])
    print("=" * 60)

    # --- Show supervisor decision ---
    if final_state.get("agent_decision"):
        decision = final_state["agent_decision"]
        print("\n" + "=" * 60)
        print("SUPERVISOR DECISION")
        print("=" * 60)
        print(f"Run security agent: {decision.run_security}")
        print(f"Run logic agent:    {decision.run_logic}")
        print(f"Run design agent:   {decision.run_design}")
        print(f"Reason: {decision.reason}")
        print("=" * 60 + "\n")

    # --- Show specialist agent results ---
    security_issues = final_state.get("security_issues", [])
    logic_issues = final_state.get("logic_issues", [])
    design_issues = final_state.get("design_issues", [])

    all_issues = security_issues + logic_issues + design_issues

    print("\n" + "=" * 60)
    print(f"ISSUES FOUND: {len(all_issues)} total")
    print(f"  Security: {len(security_issues)}")
    print(f"  Logic:    {len(logic_issues)}")
    print(f"  Design:   {len(design_issues)}")
    print("=" * 60)

    for issue in all_issues:
        print(f"\n[{issue.severity.upper()}] {issue.issue_type}")
        print(f"  File: {issue.file}")
        print(f"  Where: {issue.line_reference}")
        print(f"  What: {issue.description}")
        print(f"  Fix: {issue.suggested_fix}")

    log.info(
        "graph_run_completed",
        summary_chars=len(final_state["summary"]),
        changed_files=final_state["changed_files"],
    )


# import here so the invoke call above can reference it
from app.state import PRReviewState

if __name__ == "__main__":
    main()