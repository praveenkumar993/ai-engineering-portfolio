"""
Tools -- functions the specialist agents can call during their
ReAct loops when they need more context than the diff provides.

Each tool is a plain Python function decorated with @tool from
LangChain. The decorator does two things: it generates a JSON
schema describing the function's arguments (which is what the LLM
sees when it decides whether to call the tool), and it makes the
function compatible with LangGraph's tool execution system.

Why these two tools specifically:
    get_file_contents -- lets an agent read a full file, not just
    the diff lines. The security agent needs this to check whether
    a SQL query pattern is consistent with the rest of the codebase.

    search_code -- lets an agent find all places a pattern appears.
    The design agent needs this to check for duplication -- "is
    this logic already written somewhere else?"

Both tools close over the github_client, so they always operate
on the correct repository without the agent needing to know the
repo name.
"""

from langchain_core.tools import tool

from app.mcp_client.github_client import GitHubClient
from app.logger import get_logger

log = get_logger(__name__)


def make_tools(github_client: GitHubClient, head_ref: str = "feature/user-updates"):
    """
    Factory that returns the tool list bound to a specific client
    and branch. Called once per graph run.

    head_ref is the PR's branch -- tools read files at this ref
    so agents see the version of the code being reviewed, not main.
    """

    @tool("read_file")
    def get_file_contents(file_path: str) -> str:
        """
        Read the full contents of a file in the repository.
        Use this when you need more context than the diff provides,
        for example to check how a function is used elsewhere, or
        to see the full implementation of a helper that was called
        but not defined in the diff.

        Args:
            file_path: path to the file, e.g. app/services/auth_service.py
        """
        log.info("tool_get_file_contents", path=file_path)
        content = github_client.get_file_contents(file_path, ref=head_ref)
        if not content:
            return f"Could not read {file_path} -- file may not exist or may be binary."
        return content

    @tool("search")
    def search_code(query: str) -> str:
        """
        Search the repository for a code pattern.
        Use this to find other places in the codebase that use a
        similar pattern, to check for duplication or to understand
        the established convention before flagging an inconsistency.

        Args:
            query: the pattern to search for, e.g. "db.execute" or "hash_password"
        """
        log.info("tool_search_code", query=query)
        results = github_client.search_code(query)
        if not results:
            return f"No results found for: {query}"
        lines = [f"- {r['path']}: {r['url']}" for r in results]
        return f"Found {len(results)} result(s):\n" + "\n".join(lines)

    return [get_file_contents, search_code]