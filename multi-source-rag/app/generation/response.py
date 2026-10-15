
"""
RAGResponse — a typed result for the whole ask-a-question flow.

status tells callers WHAT happened (real answer, nothing relevant found,
LLM failure) without string-matching on message text. top_score is the
reranker's confidence in the best chunk -- useful for debugging and for
calibrating the relevance threshold against real data.
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
    top_score: float | None = None