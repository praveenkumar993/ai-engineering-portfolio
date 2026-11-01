"""
graph.py -- builds and compiles the LangGraph review graph.

This is the central file of the whole project. Every node, every
edge, and every routing decision is defined here. Reading this file
top to bottom should tell you exactly what the agent does and in
what order.

Step 6: PR Fetcher -> Summarizer only.
We build the minimal working graph first and confirm it runs before
adding specialist agents. Same philosophy as every step before this
-- one concern at a time, prove it works, then add the next.

Why compile() matters:
    graph_builder.compile() validates the graph -- checks that every
    node referenced in an edge actually exists, that START and END
    are reachable, and that the state schema is consistent. If
    anything is wired wrong, compile() raises immediately instead of
    failing at runtime mid-review. Think of it as the type-checker
    for your graph topology.
"""

import functools

from langgraph.graph import END, START, StateGraph

from app.llm import get_llm
from app.logger import get_logger
from app.mcp_client.github_client import GitHubClient
from app.nodes.pr_fetcher import pr_fetcher_node
from app.nodes.summarizer import summarizer_node
from app.state import PRReviewState

log = get_logger(__name__)


def _should_continue(state: PRReviewState) -> str:
    """
    Shared routing function used after every node.

    If any node wrote to state.error, the graph ends immediately
    and cleanly instead of passing bad state to the next node.
    This is the graph-level equivalent of the fallback chain in
    Project 1: fail fast, fail clearly, never pass broken state
    forward.

    Returns the name of the next node, or END.
    """
    if state.error:
        log.warning("graph_stopping_on_error", error=state.error)
        return END
    return "continue"


def build_graph(github_client: GitHubClient):
    """
    Builds and compiles the review graph.

    Takes a GitHubClient so nodes that need GitHub access get it
    via functools.partial -- binding the client to the node function
    so LangGraph can call it with just the state, which is the only
    argument LangGraph passes to a node.

    Why functools.partial instead of a global client?
        Global state makes testing hard -- you can't swap the client
        for a fake one without patching globals. partial keeps the
        dependency explicit and injectable.
    """
    llm = get_llm()

    # Bind the client and llm to each node that needs them.
    # LangGraph calls every node as node(state) -> dict.
    # partial turns node(state, client) into node(state) by
    # pre-filling the client argument.
    fetcher = functools.partial(pr_fetcher_node, client=github_client)
    summarizer = functools.partial(summarizer_node, llm=llm)

    # --- Build the graph ---
    builder = StateGraph(PRReviewState)

    # Register nodes
    builder.add_node("pr_fetcher", fetcher)
    builder.add_node("summarizer", summarizer)

    # Wire edges
    # START -> pr_fetcher (always)
    builder.add_edge(START, "pr_fetcher")

    # pr_fetcher -> check error -> summarizer or END
    builder.add_conditional_edges(
        "pr_fetcher",
        _should_continue,
        {
            "continue": "summarizer",
            END: END,
        },
    )

    # summarizer -> check error -> END
    # (more nodes will be inserted before END in later steps)
    builder.add_conditional_edges(
        "summarizer",
        _should_continue,
        {
            "continue": END,
            END: END,
        },
    )

    graph = builder.compile()
    log.info("graph_compiled", nodes=list(builder.nodes))
    return graph