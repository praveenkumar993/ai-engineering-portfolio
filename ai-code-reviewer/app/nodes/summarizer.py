"""
Summarizer Node -- the first node that calls an LLM.

Job: read the PR diff and metadata from state and write a short,
factual summary of what the PR actually does. Every specialist agent
reads state.summary before starting, so this reasoning is done once
instead of three times.

Two deliberate design choices:

1. DESCRIBE, DON'T JUDGE.
   The summarizer is told not to list bugs or security problems. If
   it did, the specialist agents would read "this PR has a SQL
   injection" in their context and anchor on it -- they'd confirm
   what they were told instead of investigating independently. Keeping
   the summary neutral keeps each specialist's analysis independent.

2. THE PR DESCRIPTION IS A CLAIM, NOT A FACT.
   Our test PR says "updates authentication to improve performance",
   but the diff replaces a safe query with raw string formatting.
   Authors (and sometimes attackers) write descriptions that
   understate or misdescribe a change. The summarizer treats the
   description as context only, and reports what the code does.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.logger import get_logger
from app.state import PRReviewState

log = get_logger(__name__)

SUMMARIZER_SYSTEM_PROMPT = """You are a senior software engineer preparing a briefing for a code review team.

Describe what a pull request actually does, based on its code diff.

Rules:
- Be specific and factual. Name the files, functions and API endpoints that changed, and say what behaviour is added, removed or altered.
- Describe, do not judge. Do not list bugs, security problems or style opinions -- other reviewers handle that.
- The PR title and description are the author's claims. Use them as context, but base your summary on the diff. If the diff does something the description does not mention, say so plainly.
- Write 4 to 6 sentences of plain prose. No headings, no bullet points."""


def _extract_text(content) -> str:
    """Chat models usually return a str, but some return a list of
    content parts. Normalise both to a plain string."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts).strip()
    return ""


def summarizer_node(state: PRReviewState, llm) -> dict:
    """Returns only the field it changed: {"summary": "..."}."""
    log.info("summarizer_started", pr_number=state.pr_number)

    # Guard against huge diffs blowing the model's context window.
    # We keep the start of the diff and tell the model it was cut, so
    # it doesn't pretend to have seen the whole change.
    diff = state.diff
    if len(diff) > settings.max_diff_chars:
        log.warning(
            "diff_truncated",
            original_chars=len(diff),
            kept_chars=settings.max_diff_chars,
        )
        diff = diff[: settings.max_diff_chars]
        diff += "\n\n[... diff truncated: the remaining changes are not shown ...]"

    user_prompt = (
        f"PR title: {state.pr_title}\n\n"
        f"PR description (author's claim):\n{state.pr_description or '(none provided)'}\n\n"
        f"Changed files: {', '.join(state.changed_files)}\n\n"
        f"Diff:\n{diff}"
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
    except Exception as e:
        log.error("summarizer_failed", pr_number=state.pr_number, error=str(e))
        return {"error": f"Summarizer failed: {e}"}

    summary = _extract_text(response.content)
    if not summary:
        log.error("summarizer_empty_response", pr_number=state.pr_number)
        return {"error": "Summarizer failed: the model returned an empty summary"}

    # Token usage is how you reason about cost per review later.
    usage = getattr(response, "usage_metadata", None) or {}
    log.info(
        "summarizer_completed",
        pr_number=state.pr_number,
        summary_chars=len(summary),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
    )
    return {"summary": summary}