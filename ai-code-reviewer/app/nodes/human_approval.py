"""
Human Approval Node -- pauses the graph for human review.

Job: show the draft review to the operator and pause the graph.
The graph resumes only when the operator explicitly approves.

Why this exists:
    The publisher is the only node that changes the real world --
    it posts a comment that cannot be un-posted. Human-in-the-loop
    is the safety gate before that boundary crossing. Early in
    deployment, when you're calibrating agent quality, you want to
    read every review before it goes out. Later, once you trust
    the quality, you turn it off with require_human_approval=False
    in config.

How LangGraph interrupt() works:
    interrupt() is not a sleep or a wait -- it raises a special
    exception that LangGraph catches. LangGraph saves the full
    graph state to the checkpointer at that exact point, then
    surfaces the interrupt to the caller (your main.py). The graph
    is frozen. When you call graph.invoke() again with the same
    thread_id and an approval command, LangGraph restores the saved
    state and continues from exactly where it stopped -- the
    publisher node runs with the same review_comment the writer
    produced.

What it writes to state:
    Nothing. It either pauses (interrupt) or passes through
    (when require_human_approval is False). The state is unchanged
    either way.
"""

from langgraph.types import interrupt

from app.config import settings
from app.logger import get_logger
from app.state import PRReviewState

log = get_logger(__name__)


def human_approval_node(state: PRReviewState) -> dict:
    """
    Pauses the graph if human approval is required.
    Returns an empty dict -- this node only controls flow,
    it never changes state fields.
    """
    if not settings.require_human_approval:
        log.info("human_approval_skipped", reason="require_human_approval=False")
        return {}

    log.info(
        "human_approval_waiting",
        pr_number=state.pr_number,
        review_chars=len(state.review_comment),
    )

    # interrupt() pauses the graph here and surfaces the review
    # to the caller. The value passed to interrupt() is what
    # main.py receives -- we pass the review so the operator
    # can read it before deciding.
    decision = interrupt({
        "message": "Review the draft comment above. Approve to post or reject to discard.",
        "review_preview": state.review_comment[:500] + "..." if len(state.review_comment) > 500 else state.review_comment,
        "pr_number": state.pr_number,
        "repo": state.repo,
    })

    # When the graph resumes, decision contains whatever value
    # main.py passed to graph.invoke() in the second call.
    # We check for explicit rejection -- anything else is approval.
    if isinstance(decision, str) and decision.lower() in ("reject", "no", "n"):
        log.info("human_approval_rejected", pr_number=state.pr_number)
        return {"error": "Review rejected by operator -- not posted to GitHub."}

    log.info("human_approval_approved", pr_number=state.pr_number)
    return {}