"""
Publisher Node -- the only node that writes to the real world.

Job: post the review comment to the GitHub PR.

Why this deserves its own node instead of being inside the writer:
    The writer's job is producing the review. The publisher's job
    is sending it. These are different concerns -- the writer
    should work correctly even if the publisher is replaced with
    "save to a file" or "send via email" or "post to Slack."
    Separating them means you can test the writer's output without
    actually posting to GitHub, and you can swap the delivery
    mechanism without touching the synthesis logic.

Why it never raises:
    By the time this node runs, every agent has done its work and
    the writer has produced the review. If posting fails, the
    review still exists in state.review_comment -- it can be
    retrieved, retried, or posted manually. Raising here would
    make the graph crash after all the analysis work is done,
    which is worse than logging the failure and returning cleanly.

What it writes to state:
    review_posted: bool
"""

from app.mcp_client.github_client import GitHubClient
from app.logger import get_logger
from app.state import PRReviewState

log = get_logger(__name__)


def publisher_node(state: PRReviewState, client: GitHubClient) -> dict:
    """
    Posts the review comment to GitHub.
    Returns {"review_posted": True/False}.
    """
    log.info("publisher_started", pr_number=state.pr_number)

    if not state.review_comment:
        log.error(
            "publisher_no_review_comment",
            pr_number=state.pr_number,
        )
        return {"review_posted": False}

    success = client.post_review_comment(
        pr_number=state.pr_number,
        body=state.review_comment,
    )

    if success:
        log.info(
            "publisher_completed",
            pr_number=state.pr_number,
            chars_posted=len(state.review_comment),
        )
    else:
        log.error(
            "publisher_failed",
            pr_number=state.pr_number,
        )

    return {"review_posted": success}