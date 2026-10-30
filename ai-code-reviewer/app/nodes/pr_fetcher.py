"""
PR Fetcher Node -- the first node in the graph.

Job: fetch everything downstream nodes need from GitHub and write
it all into state. This is the ONLY node that talks to GitHub for
data. Every other node reads from state instead of making its own
API calls.

Why isolate all data fetching here?
    If GitHub is down or the token is invalid, we want to know
    immediately, at this node, before any LLM work starts. Failing
    fast here is cheap -- no tokens spent, no analysis wasted.
    If data fetching were spread across nodes, a GitHub failure at
    node 6 would waste everything nodes 1-5 did.

What it writes to state:
    pr_title, pr_description, diff, changed_files, file_contents
"""

from app.mcp_client.github_client import GitHubClient
from app.state import PRReviewState
from app.logger import get_logger

log = get_logger(__name__)


def pr_fetcher_node(state: PRReviewState, client: GitHubClient) -> dict:
    """
    Reads PR data from GitHub and returns state updates.

    Returns a dict of only the fields it changed -- LangGraph
    merges this into the full state automatically.
    """
    repo = state.repo
    pr_number = state.pr_number

    log.info("pr_fetcher_started", repo=repo, pr_number=pr_number)

    try:
        # --- Fetch PR metadata ---
        pr_info = client.get_pr_info(pr_number)

        # --- Fetch the diff ---
        diff = client.get_pr_diff(pr_number)

        # --- Fetch list of changed files ---
        changed_files = client.get_changed_files(pr_number)

        # --- Fetch full content of each changed file ---
        # We read the file at the PR's head branch so we see the
        # version AS IT WILL BE after merging, not the current main.
        # This gives agents full context beyond just the diff lines.
        head_ref = pr_info["head_branch"]
        file_contents = {}
        for file_path in changed_files:
            # Only fetch Python files -- binary files, images etc
            # will return empty string from the client anyway, but
            # skipping them here avoids unnecessary API calls.
            if file_path.endswith((".py", ".txt", ".md", ".yml", ".yaml")):
                content = client.get_file_contents(
                    file_path,
                    ref=head_ref
                )
                if content:
                    file_contents[file_path] = content

        log.info(
            "pr_fetcher_completed",
            pr_number=pr_number,
            changed_files=len(changed_files),
            files_read=len(file_contents),
            diff_chars=len(diff),
        )

        # Return ONLY the fields this node changed.
        # LangGraph merges this partial dict into the full state.
        return {
            "pr_title": pr_info["title"],
            "pr_description": pr_info["description"],
            "diff": diff,
            "changed_files": changed_files,
            "file_contents": file_contents,
        }

    except Exception as e:
        log.error("pr_fetcher_failed", pr_number=pr_number, error=str(e))
        # Write the error to state so the graph can end cleanly
        # instead of crashing with an unhandled exception.
        return {"error": f"PR Fetcher failed: {str(e)}"}