"""
Specialist Agents -- Security, Logic, and Design.

Each agent is a LangGraph node that runs a ReAct loop:
    1. Reads the PR diff, file contents, and summary from state
    2. Reasons about its specific concern
    3. Calls tools if it needs more context
    4. Produces a final structured list of CodeIssue objects

Why one file for all three agents instead of three separate files?
    The structure is identical across all three: system prompt,
    create_react_agent call, response parsing, state update return.
    Putting them together makes the differences obvious -- the only
    meaningful difference between agents is their system prompt and
    which state field they write to. If they were in separate files,
    that similarity would be hidden.

How create_react_agent works:
    It takes an LLM and a list of tools and returns a runnable graph
    that handles the full ReAct loop automatically. The LLM decides
    when to call tools and when to produce a final answer. You just
    define the prompt and the tools.

Structured output:
    We use with_structured_output() on the LLM so the final answer
    is always a valid JSON object matching our schema. This is more
    reliable than asking the model to "reply in JSON" and then
    parsing the string -- the LangChain framework enforces the
    schema at the API level.
"""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from app.logger import get_logger
from app.state import CodeIssue, PRReviewState

log = get_logger(__name__)


# --- Output schema for all agents ---
class IssueList(BaseModel):
    """The structured output every specialist agent must return."""
    issues: list[CodeIssue]
    files_checked: list[str]


# --- System prompts ---

SECURITY_SYSTEM_PROMPT = """You are a security engineer reviewing a pull request for vulnerabilities.

Focus ONLY on security issues. Do not comment on logic correctness or code style.

Look for:
- Hardcoded credentials, API keys, tokens, or secrets
- SQL injection (string formatting or concatenation in queries)
- Insecure cryptographic algorithms (MD5 or SHA1 for passwords)
- Missing authentication or authorisation checks on endpoints
- Sensitive data exposed in API responses
- Input validation missing on user-supplied data

For each issue found, provide:
- file: the filename where the issue occurs
- line_reference: a short description of where in the file (e.g. "authenticate_user function, line 8")
- severity: "critical", "high", "medium", or "low"
- issue_type: a short label (e.g. "sql_injection", "hardcoded_credential")
- description: what the issue is and why it is dangerous
- suggested_fix: a concrete code-level suggestion

If you need to read a full file or search for a pattern to confirm your finding, use your tools.
When you have finished your analysis, reply with a JSON object matching this schema:
{"issues": [...], "files_checked": [...]}"""

LOGIC_SYSTEM_PROMPT = """You are a senior software engineer reviewing a pull request for logic bugs and correctness issues.

Focus ONLY on whether the code does what it is supposed to do. Do not comment on security or style.

Look for:
- Missing null or None checks before accessing attributes
- Functions that do not return a value in all code paths
- Incorrect or reversed conditions
- Operations on wrong types
- Edge cases not handled (empty list, zero, negative number)
- Functions that claim to do one thing but do another (based on their name or docstring)
- DELETE endpoint that does not actually delete

For each issue found, provide:
- file: the filename where the issue occurs
- line_reference: a short description of where in the file
- severity: "critical", "high", "medium", or "low"
- issue_type: a short label (e.g. "missing_null_check", "incorrect_behaviour")
- description: what the bug is and what it would cause at runtime
- suggested_fix: a concrete code-level suggestion

If you need to read a full file to understand the context, use your tools.
When you have finished your analysis, reply with a JSON object matching this schema:
{"issues": [...], "files_checked": [...]}"""

DESIGN_SYSTEM_PROMPT = """You are a software architect reviewing a pull request for structural and design problems.

Focus ONLY on code structure, design patterns, and consistency. Do not comment on security or logic bugs.

Look for:
- Functions doing more than one thing (violating single responsibility)
- Logic duplicated elsewhere in the codebase (use search_code to check)
- Inconsistency with how the rest of the codebase handles similar problems
- Missing type hints where the rest of the file uses them
- Public functions missing docstrings
- Endpoints added without proper error handling patterns

For each issue found, provide:
- file: the filename where the issue occurs
- line_reference: a short description of where in the file
- severity: "critical", "high", "medium", or "low"
- issue_type: a short label (e.g. "missing_type_hints", "code_duplication")
- description: what the design problem is
- suggested_fix: a concrete suggestion

Use search_code to check for existing patterns before flagging inconsistency.
Prefer get_file_contents over search_code when you need to check a specific file, since search results may not be available for recently pushed code.
When you have finished your analysis, reply with a JSON object matching this schema:
{"issues": [...], "files_checked": [...]}"""


# --- Shared parsing helper ---

def _parse_agent_output(messages: list, agent_name: str) -> list[CodeIssue]:
    """
    Extract the final answer from the agent's message list and
    parse it into CodeIssue objects.

    The last message in a ReAct loop is always the agent's final
    answer -- everything before it is thoughts and tool results.
    We try to parse it as JSON. If it fails we return an empty
    list and log the error rather than crashing the graph.
    """
    if not messages:
        log.error(f"{agent_name}_no_messages")
        return []

    last = messages[-1]
    content = last.content if hasattr(last, "content") else str(last)

    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    content = content.strip()

    # Strip markdown fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(
            line for line in lines
            if not line.startswith("```")
        ).strip()

    try:
        data = json.loads(content)
        issues = [CodeIssue(**issue) for issue in data.get("issues", [])]
        log.info(
            f"{agent_name}_parsed",
            issues_found=len(issues),
            files_checked=data.get("files_checked", []),
        )
        return issues
    except Exception as e:
        log.error(f"{agent_name}_parse_failed", error=str(e), content_preview=content[:200])
        return []


# --- Context builder ---

def _build_context(state: PRReviewState) -> str:
    """
    Builds the user message every agent receives.
    Contains the summary, diff, and full file contents.
    Same context for all agents -- they differ only in their
    system prompt, not in what they can see.
    """
    file_sections = []
    for path, content in state.file_contents.items():
        file_sections.append(f"=== {path} ===\n{content}")

    return (
        f"PR Summary:\n{state.summary}\n\n"
        f"Diff:\n{state.diff}\n\n"
        f"Full file contents:\n"
        + "\n\n".join(file_sections)
    )


# --- The three agent nodes ---

def security_agent_node(state: PRReviewState, llm, tools: list) -> dict:
    log.info("security_agent_started", pr_number=state.pr_number)

    agent = create_react_agent(llm, tools)
    context = _build_context(state)

    result = agent.invoke({
        "messages": [
            SystemMessage(content=SECURITY_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
    }, config={"recursion_limit": 8})

    issues = _parse_agent_output(result["messages"], "security_agent")
    log.info("security_agent_completed", issues_found=len(issues))
    return {"security_issues": issues}


def logic_agent_node(state: PRReviewState, llm, tools: list) -> dict:
    log.info("logic_agent_started", pr_number=state.pr_number)

    agent = create_react_agent(llm, tools)
    context = _build_context(state)

    result = agent.invoke({
        "messages": [
            SystemMessage(content=LOGIC_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
    }, config={"recursion_limit": 8})

    issues = _parse_agent_output(result["messages"], "logic_agent")
    log.info("logic_agent_completed", issues_found=len(issues))
    return {"logic_issues": issues}


def design_agent_node(state: PRReviewState, llm, tools: list) -> dict:
    log.info("design_agent_started", pr_number=state.pr_number)

    agent = create_react_agent(llm, tools)
    context = _build_context(state)

    result = agent.invoke({
        "messages": [
            SystemMessage(content=DESIGN_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
    }, config={"recursion_limit": 8})

    issues = _parse_agent_output(result["messages"], "design_agent")
    log.info("design_agent_completed", issues_found=len(issues))
    return {"design_issues": issues}