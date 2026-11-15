"""
graph.py -- builds and compiles the LangGraph review graph.

Step 6: PR Fetcher -> Summarizer
Step 7: + Supervisor
Step 8: + Security, Logic, Design agents (sequential)
Step 9: + Writer (synthesizes all findings into one review comment)
"""

import functools

from langgraph.graph import END, START, StateGraph

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
from app.nodes.writer import writer_node
from app.state import PRReviewState

log = get_logger(__name__)


def _should_continue(state: PRReviewState) -> str:
    if state.error:
        log.warning("graph_stopping_on_error", error=state.error)
        return END
    return "continue"


def _route_after_supervisor(state: PRReviewState) -> str:
    if state.error or not state.agent_decision:
        return "writer"
    decision = state.agent_decision
    if decision.run_security:
        return "security_agent"
    if decision.run_logic:
        return "logic_agent"
    if decision.run_design:
        return "design_agent"
    return "writer"


def _route_after_security(state: PRReviewState) -> str:
    if state.error:
        return "writer"
    decision = state.agent_decision
    if decision and decision.run_logic:
        return "logic_agent"
    if decision and decision.run_design:
        return "design_agent"
    return "writer"


def _route_after_logic(state: PRReviewState) -> str:
    if state.error:
        return "writer"
    decision = state.agent_decision
    if decision and decision.run_design:
        return "design_agent"
    return "writer"


def build_graph(github_client: GitHubClient):
    llm = get_llm()
    tools = make_tools(github_client)

    fetcher = functools.partial(pr_fetcher_node, client=github_client)
    summarizer = functools.partial(summarizer_node, llm=llm)
    supervisor = functools.partial(supervisor_node, llm=llm)
    security = functools.partial(security_agent_node, llm=llm, tools=tools)
    logic = functools.partial(logic_agent_node, llm=llm, tools=tools)
    design = functools.partial(design_agent_node, llm=llm, tools=tools)
    writer = functools.partial(writer_node, llm=llm)

    builder = StateGraph(PRReviewState)

    # --- Register nodes ---
    builder.add_node("pr_fetcher", fetcher)
    builder.add_node("summarizer", summarizer)
    builder.add_node("supervisor", supervisor)
    builder.add_node("security_agent", security)
    builder.add_node("logic_agent", logic)
    builder.add_node("design_agent", design)
    builder.add_node("writer", writer)

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
    builder.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "security_agent": "security_agent",
            "logic_agent": "logic_agent",
            "design_agent": "design_agent",
            "writer": "writer",
        },
    )
    builder.add_conditional_edges(
        "security_agent",
        _route_after_security,
        {
            "logic_agent": "logic_agent",
            "design_agent": "design_agent",
            "writer": "writer",
        },
    )
    builder.add_conditional_edges(
        "logic_agent",
        _route_after_logic,
        {
            "design_agent": "design_agent",
            "writer": "writer",
        },
    )

    # design always goes to writer
    builder.add_edge("design_agent", "writer")

    # writer goes to END
    # Step 10 adds the Publisher node here
    builder.add_conditional_edges(
        "writer",
        _should_continue,
        {"continue": END, END: END},
    )

    graph = builder.compile()
    log.info("graph_compiled", nodes=list(builder.nodes))
    return graph