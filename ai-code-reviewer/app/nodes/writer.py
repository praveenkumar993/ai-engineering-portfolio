"""
Writer Node -- synthesizes all specialist findings into one review.

Job: take the accumulated issues from all three specialist agents
and produce a single, well-structured markdown review comment.

Why a separate node instead of each agent writing its own section?
    Three agents writing independently would produce three separate
    review styles, potentially contradicting each other, with no
    overall prioritization. The writer sees everything at once and
    can:
    - Group related issues across agents (a SQL injection flagged
      by security and a logic bug in the same function can be
      addressed together)
    - Prioritize -- critical issues first, LOW docstring warnings
      last
    - Control tone and length -- a 27-item dump is not a review,
      it's a list. A review explains, groups, and guides.
    - Add positive observations -- good code deserves acknowledgment

What it writes to state:
    review_comment: str (the full markdown review)
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.logger import get_logger
from app.state import CodeIssue, PRReviewState

log = get_logger(__name__)

WRITER_SYSTEM_PROMPT = """You are a senior engineer writing a pull request review.

You have received findings from three specialist reviewers (security, logic, design).
Your job is to synthesize these into one clear, helpful, actionable review comment in GitHub markdown.

Rules:
- Lead with the most important issues. Critical and High severity issues come first.
- Group related issues together when they share a root cause or the same file/function.
- Be specific: name the file, the function, and what exactly is wrong.
- Be constructive: every issue must have a concrete suggested fix.
- Keep the tone professional and helpful, not harsh.
- If there are more than 3 LOW severity issues, summarize them briefly in a final section rather than listing every one individually.
- End with a short "What looks good" section acknowledging anything done well.
- Use GitHub markdown: ## for sections, **bold** for file names, `code` for inline code.
- Maximum length: a thorough review, but not overwhelming. Aim for quality over quantity."""


def _format_issues_for_prompt(
    security_issues: list[CodeIssue],
    logic_issues: list[CodeIssue],
    design_issues: list[CodeIssue],
) -> str:
    """
    Format all issues into a structured text the LLM can read.
    Groups by agent so the writer knows which specialist flagged what.
    """
    sections = []

    if security_issues:
        lines = ["SECURITY FINDINGS:"]
        for i in security_issues:
            lines.append(
                f"  [{i.severity.upper()}] {i.issue_type} in {i.file} "
                f"({i.line_reference}): {i.description} "
                f"| Fix: {i.suggested_fix}"
            )
        sections.append("\n".join(lines))

    if logic_issues:
        lines = ["LOGIC FINDINGS:"]
        for i in logic_issues:
            lines.append(
                f"  [{i.severity.upper()}] {i.issue_type} in {i.file} "
                f"({i.line_reference}): {i.description} "
                f"| Fix: {i.suggested_fix}"
            )
        sections.append("\n".join(lines))

    if design_issues:
        lines = ["DESIGN FINDINGS:"]
        for i in design_issues:
            lines.append(
                f"  [{i.severity.upper()}] {i.issue_type} in {i.file} "
                f"({i.line_reference}): {i.description} "
                f"| Fix: {i.suggested_fix}"
            )
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def writer_node(state: PRReviewState, llm) -> dict:
    """
    Reads all issue lists from state.
    Returns {"review_comment": "..."}.
    """
    log.info(
        "writer_started",
        pr_number=state.pr_number,
        security_issues=len(state.security_issues),
        logic_issues=len(state.logic_issues),
        design_issues=len(state.design_issues),
    )

    total_issues = (
        len(state.security_issues)
        + len(state.logic_issues)
        + len(state.design_issues)
    )

    # If no issues found by any agent, write a clean pass review
    if total_issues == 0:
        log.info("writer_no_issues_found", pr_number=state.pr_number)
        review = (
            "## AI Code Review\n\n"
            "✅ No significant issues found.\n\n"
            f"**PR:** {state.pr_title}\n\n"
            "The changes look good across security, logic, and design dimensions."
        )
        return {"review_comment": review}

    issues_text = _format_issues_for_prompt(
        state.security_issues,
        state.logic_issues,
        state.design_issues,
    )

    user_prompt = (
        f"PR title: {state.pr_title}\n\n"
        f"PR summary: {state.summary}\n\n"
        f"Findings from specialist reviewers:\n\n{issues_text}\n\n"
        f"Write a complete GitHub PR review comment in markdown."
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=WRITER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )

        content = response.content
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            ).strip()

        review_comment = content.strip()

        usage = getattr(response, "usage_metadata", None) or {}
        log.info(
            "writer_completed",
            pr_number=state.pr_number,
            review_chars=len(review_comment),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )
        return {"review_comment": review_comment}

    except Exception as e:
        log.error("writer_failed", pr_number=state.pr_number, error=str(e))
        # Fallback: build a basic review directly from the raw issues
        # without LLM synthesis. Not pretty, but always works.
        lines = ["## AI Code Review\n"]
        lines.append(f"*{total_issues} issues found*\n")
        for issue in state.security_issues + state.logic_issues + state.design_issues:
            lines.append(
                f"**[{issue.severity.upper()}]** `{issue.file}` — "
                f"{issue.description}\n"
            )
        return {"review_comment": "\n".join(lines)}