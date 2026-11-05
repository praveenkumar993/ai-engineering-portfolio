"""
Supervisor Node -- decides which specialist agents to run.

Job: look at the PR summary and changed files, then decide which
specialist agents are needed. Writes an AgentDecision to state.
The graph's conditional edges read that decision to fan out using
Send.

Why a separate node instead of hardcoding "run all agents always"?
    Running all agents on every PR wastes tokens and time. A README
    change doesn't need a security agent. A change to auth code
    absolutely does. The supervisor makes this call once, cleanly,
    before any specialist work starts.

Why use the LLM for routing instead of keyword matching?
    Pure keyword matching ("if 'password' in diff: run security")
    is brittle. It misses synonyms, misses context, and produces
    false positives. The LLM reads the summary and understands
    intent -- it can tell the difference between a comment
    mentioning "token" and actual authentication code changing.
    For routing decisions that need semantic understanding, the LLM
    is the right tool.

What it writes to state:
    agent_decision: AgentDecision
"""

from langchain_core.messages import HumanMessage, SystemMessage

from app.logger import get_logger
from app.state import AgentDecision, PRReviewState

log = get_logger(__name__)

SUPERVISOR_SYSTEM_PROMPT = """You are a code review coordinator.

Given a summary of a pull request, decide which specialist review agents should run.

You have three specialist agents available:
- security: checks for vulnerabilities, hardcoded credentials, injection flaws, insecure algorithms
- logic: checks for bugs, missing null checks, incorrect behaviour, edge cases not handled
- design: checks for structural problems, code duplication, wrong abstractions, inconsistency with the rest of the codebase

Rules:
- Always run logic. Every code change can have logic issues.
- Run security if the PR touches authentication, passwords, tokens, API keys, database queries, user input handling, or permissions.
- Run design if the PR changes more than two files, introduces new functions or classes, or restructures existing code.
- Reply with ONLY a JSON object in this exact format, no explanation:
{"run_security": true, "run_logic": true, "run_design": false, "reason": "one sentence explaining the decision"}"""


def supervisor_node(state: PRReviewState, llm) -> dict:
    """
    Reads state.summary and state.changed_files.
    Returns {"agent_decision": AgentDecision}.
    """
    log.info("supervisor_started", pr_number=state.pr_number)

    user_prompt = (
        f"PR summary:\n{state.summary}\n\n"
        f"Files changed: {', '.join(state.changed_files)}\n\n"
        f"Number of files changed: {len(state.changed_files)}"
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )

        # Parse the JSON response
        import json
        content = response.content.strip()

        # Strip markdown code fences if the model wrapped the JSON
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        data = json.loads(content)
        decision = AgentDecision(
            run_security=bool(data.get("run_security", False)),
            run_logic=bool(data.get("run_logic", True)),
            run_design=bool(data.get("run_design", False)),
            reason=str(data.get("reason", "")),
        )

    except Exception as e:
        # Fallback: if routing fails, run all agents. A wrong routing
        # decision is worse than running one agent too many.
        log.warning(
            "supervisor_parse_failed",
            error=str(e),
            fallback="running all agents",
        )
        decision = AgentDecision(
            run_security=True,
            run_logic=True,
            run_design=True,
            reason="Fallback: supervisor could not parse response, running all agents",
        )

    log.info(
        "supervisor_completed",
        pr_number=state.pr_number,
        run_security=decision.run_security,
        run_logic=decision.run_logic,
        run_design=decision.run_design,
        reason=decision.reason,
    )
    return {"agent_decision": decision}