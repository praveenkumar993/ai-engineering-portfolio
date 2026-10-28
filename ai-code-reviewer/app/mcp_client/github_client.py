"""
GitHub MCP client -- wraps the official GitHub MCP server and exposes
the tools our nodes need as simple Python async functions.

Why wrap it instead of calling MCP directly from each node?
    Each node would need to know how to start the MCP server, manage
    the subprocess, handle connection errors, and format MCP requests.
    That's infrastructure, not agent logic. This wrapper handles all
    of that once, and nodes just call clean Python functions:
    get_pr_diff(repo, pr_number) instead of mcp_client.call_tool(...)

Why async?
    The MCP server communicates over stdio -- reading and writing to a
    subprocess. Async lets us wait for responses without blocking the
    whole process. LangGraph nodes can be async, so this fits naturally.

Fallback to PyGitHub:
    If the MCP server isn't available (npm not installed, Node.js
    missing), we fall back to calling the GitHub REST API directly
    via PyGitHub. Same data, different transport. This means the
    project works even without Node.js, which is important for CI.
"""

import os
import subprocess
from typing import Optional

import requests
from github import Github
from github.GithubException import GithubException

from app.config import settings
from app.logger import get_logger

log = get_logger(__name__)


class GitHubClient:
    """
    Wraps GitHub API access for our agent nodes.

    Uses PyGitHub directly (simpler, more reliable for our use case
    than managing a Node.js subprocess in a learning project).
    The MCP server adds value when the LLM is deciding dynamically
    which tools to call -- for our PR Fetcher node, we know exactly
    what data we need upfront, so a clean Python wrapper is the
    right choice here.
    """

    def __init__(self, token: str, repo_name: str):
        self.github = Github(token)
        self.repo_name = repo_name
        try:
            self.repo = self.github.get_repo(repo_name)
            log.info("github_client_ready", repo=repo_name)
        except GithubException as e:
            log.error("github_client_failed", repo=repo_name, error=str(e))
            raise

    def get_pr_info(self, pr_number: int) -> dict:
        """Get basic PR metadata -- title, description, author."""
        try:
            pr = self.repo.get_pull(pr_number)
            result = {
                "title": pr.title,
                "description": pr.body or "",
                "author": pr.user.login,
                "base_branch": pr.base.ref,
                "head_branch": pr.head.ref,
                "state": pr.state,
            }
            log.info("pr_info_fetched", pr_number=pr_number, title=pr.title)
            return result
        except GithubException as e:
            log.error("pr_info_failed", pr_number=pr_number, error=str(e))
            raise

    def get_pr_diff(self, pr_number: int) -> str:
        """Get the raw unified diff of a PR."""
        try:
            pr = self.repo.get_pull(pr_number)
            # GitHub API returns diff via a special Accept header
            headers = {
                "Authorization": f"token {settings.github_token}",
                "Accept": "application/vnd.github.v3.diff",
            }
            url = f"https://api.github.com/repos/{self.repo_name}/pulls/{pr_number}"
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            log.info("pr_diff_fetched", pr_number=pr_number, chars=len(response.text))
            return response.text
        except Exception as e:
            log.error("pr_diff_failed", pr_number=pr_number, error=str(e))
            raise

    def get_changed_files(self, pr_number: int) -> list[str]:
        """Get list of file paths changed in this PR."""
        try:
            pr = self.repo.get_pull(pr_number)
            files = [f.filename for f in pr.get_files()]
            log.info("changed_files_fetched", pr_number=pr_number, count=len(files))
            return files
        except GithubException as e:
            log.error("changed_files_failed", pr_number=pr_number, error=str(e))
            raise

    def get_file_contents(self, file_path: str, ref: Optional[str] = None) -> str:
        """
        Get the full contents of a file at a specific commit/branch.

        ref defaults to the repo's default branch if not specified.
        In practice our nodes will pass the PR's head branch ref so
        they read the version of the file AS IT WILL BE after the PR
        merges, not the current main branch version.
        """
        try:
            content_file = self.repo.get_contents(file_path, ref=ref)
            # get_contents returns bytes -- decode to string
            if isinstance(content_file, list):
                # path is a directory, not a file
                log.warning("file_path_is_directory", path=file_path)
                return ""
            content = content_file.decoded_content.decode("utf-8")
            log.info("file_contents_fetched", path=file_path, chars=len(content))
            return content
        except GithubException as e:
            log.error("file_contents_failed", path=file_path, error=str(e))
            return ""  # return empty string -- one missing file shouldn't crash the run

    def post_review_comment(self, pr_number: int, body: str) -> bool:
        """
        Post a review comment to the PR.

        Returns True if successful, False if not -- never raises,
        because a posting failure shouldn't crash the graph after
        all the analysis work is done.
        """
        try:
            pr = self.repo.get_pull(pr_number)
            pr.create_issue_comment(body)
            log.info("review_posted", pr_number=pr_number, chars=len(body))
            return True
        except GithubException as e:
            log.error("review_post_failed", pr_number=pr_number, error=str(e))
            return False

    def search_code(self, query: str, file_extension: str = ".py") -> list[dict]:
        """
        Search the repository for a code pattern.

        Used by specialist agents during their ReAct loop when they
        need to check how similar patterns are handled elsewhere in
        the codebase. For example: "are there other database queries?
        Let me search for db.query to compare patterns."
        """
        try:
            # GitHub code search requires "repo:owner/name" qualifier
            full_query = f"{query} repo:{self.repo_name}"
            results = self.github.search_code(full_query)
            matches = []
            for item in results[:10]:  # cap at 10 to avoid rate limits
                if item.path.endswith(file_extension):
                    matches.append({
                        "path": item.path,
                        "url": item.html_url,
                    })
            log.info("code_search_completed", query=query, matches=len(matches))
            return matches
        except GithubException as e:
            log.error("code_search_failed", query=query, error=str(e))
            return []


def get_github_client() -> GitHubClient:
    """
    Factory function -- creates one GitHubClient using settings.
    Called once during graph startup and stored in graph state or
    passed to nodes that need it.
    """
    return GitHubClient(
        token=settings.github_token,
        repo_name=settings.github_repo,
    )