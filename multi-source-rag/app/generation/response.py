"""
RAGResponse — a typed result for the whole ask-a-question flow.

Why this matters:
    Up to now, both a real LLM answer and a fallback message ("I don't have
    enough info", "system issue, try again") were just plain strings. That
    means calling code (or a future API endpoint) can't tell the difference
    between "the model genuinely answered" and "something went wrong and
    this is a canned message" without string-matching, which is fragile.

    A typed response makes the distinction explicit and machine-checkable:
    `response.status` tells you exactly what happened, and a UI/API layer
    can react differently (e.g. show a retry button only on "llm_error",
    not on "no_relevant_context").
"""

from enum import Enum

from pydantic import BaseModel

from app.ingestion.models import Chunk


class ResponseStatus(str, Enum):
    OK = "ok"
    NO_CONTEXT = "no_relevant_context"
    LLM_ERROR = "llm_error"


class RAGResponse(BaseModel):
    status: ResponseStatus
    answer: str
    sources: list[Chunk] = []