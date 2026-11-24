"""
api.py -- FastAPI wrapper around the LangGraph review graph.

Exposes three endpoints:

POST /review
    Starts a new review run (Phase 1).
    Runs the full graph up to the human_approval interrupt.
    Returns the draft review comment for the caller to read.
    If require_human_approval=False in config, the graph runs all
    the way through and posts automatically -- response status
    will be "posted" instead of "awaiting_approval".

POST /review/approve
    Resumes a paused graph run with approval (Phase 2).
    The graph continues from the interrupt to the publisher.
    Returns confirmation that the review was posted to GitHub.

POST /review/reject
    Resumes a paused graph run with rejection (Phase 2).
    The graph ends cleanly without posting.
    Returns confirmation of rejection.

GET /health
    Returns service health. Checks whether the graph was built
    successfully at startup. Load balancers and monitoring tools
    call this to confirm the service is alive and ready.

Design decisions:

Why three endpoints instead of one?
    The human-in-the-loop pattern is inherently two-phase: run,
    pause, decide, resume. HTTP is request-response: one request,
    one response, done. The only clean way to map a two-phase
    graph onto HTTP is two separate requests. The thread_id ties
    them together -- the caller gets it from /review and passes
    it back to /review/approve or /review/reject.

Why thread_id in the response instead of a cookie or session?
    The caller (a CI script, a webhook handler, a frontend) might
    not support cookies. A plain string in the JSON response works
    everywhere. The caller is responsible for storing it and
    sending it back -- the same way a payment system gives you a
    transaction ID and expects you to reference it later.

Why store graph on app.state instead of a module-level global?
    Module-level globals make testing hard -- you can't swap in
    a fake graph without patching the module. app.state is
    injectable: in tests you set app.state.graph = FakeGraph()
    before the test and the route handler uses it automatically.
    Same philosophy as functools.partial in graph.py.
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.logger import configure_logging, get_logger
from app.startup_checks import validate_startup_config, ConfigError
from app.state import PRReviewState

configure_logging()
log = get_logger(__name__)


# --- Request and Response Models ---

class ReviewRequest(BaseModel):
    """
    What the caller sends to POST /review.
    repo and pr_number uniquely identify a PR.
    thread_id is optional -- if not provided we generate one
    from the repo and pr_number. Callers can provide their own
    if they want deterministic IDs (useful for idempotency).
    """
    repo: str
    pr_number: int
    thread_id: Optional[str] = None


class ApprovalRequest(BaseModel):
    """
    What the caller sends to POST /review/approve or /review/reject.
    thread_id must match the one returned by POST /review.
    """
    thread_id: str


class ReviewResponse(BaseModel):
    """
    What POST /review returns.
    status is one of:
        "awaiting_approval" -- graph paused, review ready to read
        "posted"            -- review posted (human approval skipped)
        "error"             -- something failed
    """
    status: str
    thread_id: str
    pr_number: int
    repo: str
    review_comment: str = ""
    review_posted: bool = False
    error: str = ""


class ApprovalResponse(BaseModel):
    """
    What POST /review/approve and /review/reject return.
    """
    status: str
    thread_id: str
    review_posted: bool
    error: str = ""


class HealthResponse(BaseModel):
    status: str
    graph_ready: bool
    repo: str
    require_human_approval: bool


# --- Lifespan: build once at startup, reuse forever ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Everything before yield runs once at startup.
    Everything after yield runs once at shutdown.

    This is where expensive initialization happens:
    - GitHub client (makes a network call to validate the token)
    - LangGraph graph (loads LLM clients, compiles the graph)

    Both are stored on app.state so every route handler can
    access them without rebuilding on every request.
    """
    log.info(
        "api_starting",
        app_name=settings.app_name,
        environment=settings.environment,
    )

    app.state.ready = False
    app.state.graph = None
    app.state.github_client = None

    try:
        validate_startup_config()
    except ConfigError as e:
        log.error("startup_config_failed", error=str(e))
        # Don't raise -- the server starts but marks itself not ready.
        # /health will return 503 so load balancers know to not route
        # traffic here. This is better than the server refusing to
        # start at all, because it gives you a running process you
        # can inspect and debug.
        yield
        return

    try:
        from app.mcp_client.github_client import get_github_client
        from app.graph import build_graph

        github_client = get_github_client()
        graph = build_graph(github_client)

        app.state.github_client = github_client
        app.state.graph = graph
        app.state.ready = True

        log.info(
            "api_ready",
            repo=settings.github_repo,
            require_human_approval=settings.require_human_approval,
        )

    except Exception as e:
        log.error("startup_graph_failed", error=str(e))
        yield
        return

    yield

    # Shutdown
    log.info("api_shutting_down")


# --- FastAPI app ---

app = FastAPI(
    title="AI Code Reviewer",
    description="Multi-agent code review system using LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Helper: build thread_id ---

def _thread_id(repo: str, pr_number: int) -> str:
    """
    Generates a deterministic thread_id from repo and PR number.
    Using repo+pr_number means re-running a review for the same PR
    resumes the same checkpoint thread rather than starting fresh.
    Replace the / with - so it's URL-safe.
    """
    return f"{repo.replace('/', '-')}-pr-{pr_number}"


# --- Helper: get graph or raise 503 ---

def _get_graph(request_app):
    if not getattr(request_app.state, "ready", False):
        raise HTTPException(
            status_code=503,
            detail="Service is not ready. Check logs for startup errors.",
        )
    return request_app.state.graph


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    Returns 200 if the service started successfully and the graph
    is ready to handle review requests.
    Returns 503 if startup failed.

    Load balancers and monitoring tools call this endpoint to
    decide whether to route traffic to this instance.
    """
    ready = getattr(app.state, "ready", False)

    if not ready:
        raise HTTPException(
            status_code=503,
            detail="Service not ready",
        )

    return HealthResponse(
        status="ok",
        graph_ready=ready,
        repo=settings.github_repo,
        require_human_approval=settings.require_human_approval,
    )


@app.post("/review", response_model=ReviewResponse)
async def start_review(request: ReviewRequest):
    """
    Start a new code review run (Phase 1).

    Runs the full graph from PR Fetcher through to the human
    approval gate. If require_human_approval=True (default),
    the graph pauses here and returns the draft review for the
    caller to read and decide on.

    If require_human_approval=False, the graph runs all the way
    through and posts automatically. The response status will be
    "posted" in that case.

    The thread_id in the response must be passed back to
    POST /review/approve or POST /review/reject.
    """
    graph = _get_graph(app)

    thread_id = request.thread_id or _thread_id(
        request.repo, request.pr_number
    )
    thread_config = {"configurable": {"thread_id": thread_id}}

    log.info(
        "review_started",
        repo=request.repo,
        pr_number=request.pr_number,
        thread_id=thread_id,
    )

    try:
        # Phase 1: run until interrupt or completion
        list(graph.stream(
            PRReviewState(
                repo=request.repo,
                pr_number=request.pr_number,
            ),
            config=thread_config,
            stream_mode="values",
        ))

        state_snapshot = graph.get_state(thread_config)
        state = state_snapshot.values

        # Check if the graph paused at the approval interrupt
        if state_snapshot.next == ("human_approval",):
            log.info(
                "review_awaiting_approval",
                thread_id=thread_id,
                review_chars=len(state.get("review_comment", "")),
            )
            return ReviewResponse(
                status="awaiting_approval",
                thread_id=thread_id,
                pr_number=request.pr_number,
                repo=request.repo,
                review_comment=state.get("review_comment", ""),
                review_posted=False,
            )

        # Graph completed without pausing
        # (require_human_approval=False)
        error = state.get("error", "")
        posted = state.get("review_posted", False)

        log.info(
            "review_completed_auto",
            thread_id=thread_id,
            posted=posted,
            error=error or None,
        )

        return ReviewResponse(
            status="posted" if posted else "error",
            thread_id=thread_id,
            pr_number=request.pr_number,
            repo=request.repo,
            review_comment=state.get("review_comment", ""),
            review_posted=posted,
            error=error,
        )

    except Exception as e:
        log.error("review_failed", thread_id=thread_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Review failed: {str(e)}",
        )


@app.post("/review/approve", response_model=ApprovalResponse)
async def approve_review(request: ApprovalRequest):
    """
    Resume a paused review run with approval (Phase 2).

    The graph resumes from the human_approval interrupt and
    continues to the Publisher node, which posts the review
    to the GitHub PR.

    thread_id must match the one returned by POST /review.
    """
    graph = _get_graph(app)

    thread_config = {"configurable": {"thread_id": request.thread_id}}

    # Confirm the graph is actually paused at approval
    state_snapshot = graph.get_state(thread_config)
    if state_snapshot.next != ("human_approval",):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Thread '{request.thread_id}' is not awaiting approval. "
                f"Next nodes: {state_snapshot.next}"
            ),
        )

    log.info("review_approving", thread_id=request.thread_id)

    try:
        from langgraph.types import Command

        list(graph.stream(
            Command(resume="approve"),
            config=thread_config,
            stream_mode="values",
        ))

        final_state = graph.get_state(thread_config).values
        posted = final_state.get("review_posted", False)
        error = final_state.get("error", "")

        log.info(
            "review_approved",
            thread_id=request.thread_id,
            posted=posted,
            error=error or None,
        )

        return ApprovalResponse(
            status="posted" if posted else "error",
            thread_id=request.thread_id,
            review_posted=posted,
            error=error,
        )

    except Exception as e:
        log.error(
            "approve_failed",
            thread_id=request.thread_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Approval failed: {str(e)}",
        )


@app.post("/review/reject", response_model=ApprovalResponse)
async def reject_review(request: ApprovalRequest):
    """
    Resume a paused review run with rejection (Phase 2).

    The graph resumes from the human_approval interrupt and
    ends cleanly without posting to GitHub.

    thread_id must match the one returned by POST /review.
    """
    graph = _get_graph(app)

    thread_config = {"configurable": {"thread_id": request.thread_id}}

    state_snapshot = graph.get_state(thread_config)
    if state_snapshot.next != ("human_approval",):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Thread '{request.thread_id}' is not awaiting approval. "
                f"Next nodes: {state_snapshot.next}"
            ),
        )

    log.info("review_rejecting", thread_id=request.thread_id)

    try:
        from langgraph.types import Command

        list(graph.stream(
            Command(resume="reject"),
            config=thread_config,
            stream_mode="values",
        ))

        final_state = graph.get_state(thread_config).values
        error = final_state.get("error", "")

        log.info("review_rejected", thread_id=request.thread_id)

        return ApprovalResponse(
            status="rejected",
            thread_id=request.thread_id,
            review_posted=False,
            error=error,
        )

    except Exception as e:
        log.error(
            "reject_failed",
            thread_id=request.thread_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Rejection failed: {str(e)}",
        )