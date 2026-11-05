"""
graph.py -- builds and compiles the LangGraph review graph.

Step 6: PR Fetcher -> Summarizer
Step 7: PR Fetcher -> Summarizer -> Supervisor
        Supervisor decides which specialist agents to run.
        (Specialist agents added in Step 8.)
"""

import functools

from langgraph.graph import END, START, StateGraph

from app.llm import get_llm
from app.logger import get_logger
from app.mcp_client.github_client import GitHubClient
from app.nodes.pr_fetcher import pr_fetcher_node
from app.nodes.summarizer import summarizer_node
from app.nodes.supervisor import supervisor_node
from app.state import PRReviewState

log = get_logger(__name__)


def _should_continue(state: PRReviewState) -> str:
    if state.error:
        log.warning("graph_stopping_on_error", error=state.error)
        return END
    return "continue"


def build_graph(github_client: GitHubClient):
    llm = get_llm()

    fetcher = functools.partial(pr_fetcher_node, client=github_client)
    summarizer = functools.partial(summarizer_node, llm=llm)
    supervisor = functools.partial(supervisor_node, llm=llm)

    builder = StateGraph(PRReviewState)

    # --- Register nodes ---
    builder.add_node("pr_fetcher", fetcher)
    builder.add_node("summarizer", summarizer)
    builder.add_node("supervisor", supervisor)

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

    # Supervisor -> END for now.
    # Step 8 replaces END with the specialist agents using Send.
    builder.add_conditional_edges(
        "supervisor",
        _should_continue,
        {"continue": END, END: END},
    )

    graph = builder.compile()
    log.info("graph_compiled", nodes=list(builder.nodes))
    return graph