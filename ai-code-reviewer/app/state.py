"""
PRReviewState -- the shared whiteboard every node in the graph
reads from and writes to.

Design decisions worth understanding:

Why one flat state instead of nested objects?
    Nested state (state.analysis.security_issues) looks cleaner
    but LangGraph's reducer system works at the top level. Keeping
    everything flat makes merging results from parallel agents
    straightforward -- each agent just returns {"security_issues": [...]}
    and LangGraph appends it to the existing list automatically.

Why Annotated[list, add] for issue lists?
    Default LangGraph behaviour for any field is "overwrite" -- the
    new value replaces the old one. That's correct for strings like
    summary and review_comment (each node sets it once).
    For lists like security_issues, logic_issues, design_issues we
    want "append" -- the security agent adds its findings, then the
    logic agent adds its findings, without wiping out what the
    security agent found. Annotated[list, add] tells LangGraph to
    use list concatenation instead of overwrite.

Why Optional fields with None defaults?
    Not every node runs on every PR. A README-only change won't
    trigger the security agent, so security_issues stays as its
    default empty list. A None on review_posted means "not attempted
    yet" versus False meaning "attempted but failed". These
    distinctions matter for debugging.
"""

from operator import add
from typing import Annotated, Optional
from pydantic import BaseModel


# --- Typed models for individual issues ---
# Using Pydantic here (not just dicts) for the same reason we used
# Document and Chunk in Project 1: typos get caught immediately,
# editors autocomplete, and the eval harness can check fields
# programmatically without string matching.

class CodeIssue(BaseModel):
    file: str
    line_reference: str
    severity: str           # "critical" | "high" | "medium" | "low"
    issue_type: str         # "sql_injection" | "hardcoded_credential" etc
    description: str
    suggested_fix: str


class AgentDecision(BaseModel):
    run_security: bool = False
    run_logic: bool = False
    run_design: bool = False
    reason: str = ""


# --- The main state ---

class PRReviewState(BaseModel):
    # --- Input: set once at the start, never changed ---
    repo: str = ""
    pr_number: int = 0

    # --- PR data: filled by PR Fetcher node ---
    pr_title: str = ""
    pr_description: str = ""
    diff: str = ""
    changed_files: list[str] = []
    # filename -> full file content
    # e.g. {"app/auth_service.py": "def login(): ..."}
    file_contents: dict[str, str] = {}

    # --- Summary: filled by Summarizer node ---
    summary: str = ""

    # --- Routing decision: filled by Supervisor node ---
    agent_decision: Optional[AgentDecision] = None

    # --- Issues: each specialist agent appends to these ---
    # Annotated[list, add] means LangGraph APPENDS new items
    # instead of OVERWRITING the existing list.
    security_issues: Annotated[list[CodeIssue], add] = []
    logic_issues: Annotated[list[CodeIssue], add] = []
    design_issues: Annotated[list[CodeIssue], add] = []
    style_issues: Annotated[list[CodeIssue], add] = []

    # --- Final output: filled by Writer node ---
    review_comment: str = ""

    # --- Control ---
    # True once Publisher successfully posts to GitHub
    review_posted: bool = False
    # If anything fails badly enough to stop the graph,
    # the node writes a message here and the graph ends cleanly
    error: str = ""