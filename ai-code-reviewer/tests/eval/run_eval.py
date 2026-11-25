"""
run_eval.py -- measures whether the agent catches known issues.

Runs the full graph against PR #1 (our deliberately buggy test PR)
and checks the findings against the golden issues list.

Usage:
    cd ai-code-reviewer
    python tests/eval/run_eval.py

What it checks:
    For each golden issue, it searches the agent's findings for
    at least one issue whose description or suggested_fix contains
    at least one of the golden issue's keywords (case-insensitive).
    If found: PASS. If not found: FAIL.

It also checks the review_comment produced by the writer,
since some issues might appear there even if the structured
issue list missed them.

Output format mirrors Project 1's eval harness exactly:
    [PASS] (category=security) hardcoded_api_key
    [FAIL] (category=logic)    delete_endpoint_not_deleting
    ...
    EVAL RESULTS: 9/10 passed
"""

import sys
from pathlib import Path

# Make sure we can import from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import settings
from app.logger import configure_logging, get_logger
from app.mcp_client.github_client import get_github_client
from app.graph import build_graph
from app.state import PRReviewState, CodeIssue
from tests.eval.golden_prs import GOLDEN_ISSUES, TEST_PR

configure_logging()
log = get_logger(__name__)


def _keywords_match(issue: CodeIssue, keywords: list[str]) -> bool:
    """
    Check if any keyword appears in the issue's description
    or suggested_fix (case-insensitive).
    """
    haystack = (
        (issue.description or "").lower()
        + " "
        + (issue.suggested_fix or "").lower()
        + " "
        + (issue.issue_type or "").lower()
    )
    return any(kw.lower() in haystack for kw in keywords)


def _keywords_in_review(review_comment: str, keywords: list[str]) -> bool:
    """
    Fallback: check if any keyword appears in the full review comment.
    Some issues surface in the writer's synthesis even if the
    structured issue list missed them.
    """
    haystack = review_comment.lower()
    return any(kw.lower() in haystack for kw in keywords)


def run_eval():
    log.info("eval_started", total_golden_issues=len(GOLDEN_ISSUES))

    # --- Build the graph ---
    github_client = get_github_client()
    graph = build_graph(github_client)

    # Use a unique thread_id for the eval run so it doesn't
    # collide with any in-progress review sessions
    thread_config = {"configurable": {"thread_id": "eval-pr-1"}}

    # --- Run the graph ---
    # For eval, we always skip human approval -- we want automated
    # results, not an interactive prompt. We temporarily override
    # the config flag by passing require_human_approval=False.
    # We do this by running the graph directly and catching the
    # interrupt if it fires, then immediately resuming with approve.
    log.info("eval_graph_running", repo=TEST_PR["repo"], pr_number=TEST_PR["pr_number"])

    list(graph.stream(
        PRReviewState(
            repo=TEST_PR["repo"],
            pr_number=TEST_PR["pr_number"],
        ),
        config=thread_config,
        stream_mode="values",
    ))

    state_snapshot = graph.get_state(thread_config)

    # If the graph paused for human approval, resume automatically
    if state_snapshot.next == ("human_approval",):
        log.info("eval_auto_approving")
        from langgraph.types import Command
        list(graph.stream(
            Command(resume="approve"),
            config=thread_config,
            stream_mode="values",
        ))
        state_snapshot = graph.get_state(thread_config)

    final_state = state_snapshot.values

    # --- Extract all issues ---
    security_issues = final_state.get("security_issues", [])
    logic_issues = final_state.get("logic_issues", [])
    design_issues = final_state.get("design_issues", [])
    all_issues = security_issues + logic_issues + design_issues
    review_comment = final_state.get("review_comment", "")

    log.info(
        "eval_issues_collected",
        security=len(security_issues),
        logic=len(logic_issues),
        design=len(design_issues),
        total=len(all_issues),
    )

    # --- Check each golden issue ---
    results = []

    for golden in GOLDEN_ISSUES:
        keywords = golden["keywords"]

        # First check the structured issue list
        matched_issue = next(
            (issue for issue in all_issues if _keywords_match(issue, keywords)),
            None,
        )

        # Fallback: check the full review comment
        matched_in_review = _keywords_in_review(review_comment, keywords)

        passed = matched_issue is not None or matched_in_review
        match_source = (
            "structured_issue" if matched_issue
            else "review_comment" if matched_in_review
            else "not_found"
        )

        results.append({
            "id": golden["id"],
            "category": golden["category"],
            "severity": golden["severity"],
            "passed": passed,
            "match_source": match_source,
            "matched_issue_type": matched_issue.issue_type if matched_issue else None,
        })

    # --- Print report ---
    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"\n{'=' * 60}")
    print(f"EVAL RESULTS: {passed_count}/{total} passed")
    print(f"{'=' * 60}\n")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        source = f"via={r['match_source']}" if r["passed"] else "NOT FOUND"
        print(
            f"[{status}] (category={r['category']}, severity={r['severity']}) "
            f"{r['id']} | {source}"
        )

    print(f"\n{'=' * 60}")
    print(f"Security issues found:  {len(security_issues)}")
    print(f"Logic issues found:     {len(logic_issues)}")
    print(f"Design issues found:    {len(design_issues)}")
    print(f"Review posted:          {final_state.get('review_posted', False)}")
    print(f"{'=' * 60}\n")

    log.info(
        "eval_completed",
        passed=passed_count,
        total=total,
        pass_rate=passed_count / total,
    )

    return passed_count, total


if __name__ == "__main__":
    passed, total = run_eval()
    # Exit with non-zero code if any checks failed
    # This is what CI uses to mark the build as failed
    sys.exit(0 if passed == total else 1)