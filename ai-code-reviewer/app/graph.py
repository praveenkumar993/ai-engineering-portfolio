"""
graph.py -- builds and compiles the LangGraph review graph.

Step 6: PR Fetcher -> Summarizer
Step 7: + Supervisor (routing decision)
Step 8: + Security, Logic, Design agents (parallel via Send)

The Send pattern fans out to multiple agents simultaneously.
LangGraph runs all selected agents in parallel, waits for all
to finish, then merges their results into shared state using
the list reducers defined on PRReviewState.
"""

import functools

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.llm import get_llm
from app.logger import get_logger
from app.mcp_client.github_client import GitHubClient
from app.nodes.pr_fetcher import pr_fetcher_node
from app.nodes.summarizer import summarizer_node
from app.nodes.supervisor import supervisor_node
from app.nodes.agents import (
    security_agent_node,
    logic_agent_node,
    design_agent_node,
)
from app.nodes.tools import make_tools
from app.state import PRReviewState

log = get_logger(__name__)


def _should_continue(state: PRReviewState) -> str:
    if state.error:
        log.warning("graph_stopping_on_error", error=state.error)
        return END
    return "continue"


def _route_after_supervisor(state: PRReviewState) -> str:
    """
    After supervisor, go to whichever agent should run first.
    Agents run sequentially: security -> logic -> design -> END.
    Each agent's edge checks the decision to skip if not needed.
    """
    if state.error or not state.agent_decision:
        return END
    decision = state.agent_decision
    if decision.run_security:
        return "security_agent"
    if decision.run_logic:
        return "logic_agent"
    if decision.run_design:
        return "design_agent"
    return END


def _route_after_security(state: PRReviewState) -> str:
    if state.error:
        return END
    decision = state.agent_decision
    if decision and decision.run_logic:
        return "logic_agent"
    if decision and decision.run_design:
        return "design_agent"
    return END


def _route_after_logic(state: PRReviewState) -> str:
    if state.error:
        return END
    decision = state.agent_decision
    if decision and decision.run_design:
        return "design_agent"
    return END

def build_graph(github_client: GitHubClient):
    llm = get_llm()
    tools = make_tools(github_client)

    fetcher = functools.partial(pr_fetcher_node, client=github_client)
    summarizer = functools.partial(summarizer_node, llm=llm)
    supervisor = functools.partial(supervisor_node, llm=llm)
    security = functools.partial(security_agent_node, llm=llm, tools=tools)
    logic = functools.partial(logic_agent_node, llm=llm, tools=tools)
    design = functools.partial(design_agent_node, llm=llm, tools=tools)

    builder = StateGraph(PRReviewState)

    # --- Register nodes ---
    builder.add_node("pr_fetcher", fetcher)
    builder.add_node("summarizer", summarizer)
    builder.add_node("supervisor", supervisor)
    builder.add_node("security_agent", security)
    builder.add_node("logic_agent", logic)
    builder.add_node("design_agent", design)

    # --- Wire edges ---
    builder.add_edge(START, "pr_fetcher")

    builder.add_conditional_edges(
        "pr_fetcher",
        _should_continue,
        {"continue": "summarizer", END: END},
    )
    builder.add_conditional_edges(
        "summarizer",
        _should_continue,
        {"continue": "supervisor", END: END},
    )

        # Supervisor -> first agent (sequential, not parallel)
    builder.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "security_agent": "security_agent",
            "logic_agent": "logic_agent",
            "design_agent": "design_agent",
            END: END,
        },
    )

    # security -> logic or design or END
    builder.add_conditional_edges(
        "security_agent",
        _route_after_security,
        {
            "logic_agent": "logic_agent",
            "design_agent": "design_agent",
            END: END,
        },
    )

    # logic -> design or END
    builder.add_conditional_edges(
        "logic_agent",
        _route_after_logic,
        {
            "design_agent": "design_agent",
            END: END,
        },
    )

    # design always goes to END
    # Step 9 replaces END with the Writer node
    builder.add_edge("design_agent", END)

    graph = builder.compile()
    log.info("graph_compiled", nodes=list(builder.nodes))
    return graph