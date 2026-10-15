"""
BaseReranker — the contract every reranking model must follow.

The contract is rerank_with_scores(), not rerank(): the pipeline needs the
actual scores to decide whether the best match is relevant ENOUGH to answer
from, not just the order. rerank() stays as a convenience wrapper for callers
that only want the chunks.
"""

from abc import ABC, abstractmethod

from app.ingestion.models import Chunk


class BaseReranker(ABC):
    @abstractmethod
    def rerank_with_scores(
        self, query: str, chunks: list[Chunk], top_k: int
    ) -> list[tuple[Chunk, float]]:
        """Return the top_k chunks as (chunk, score) pairs, best first."""
        raise NotImplementedError

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        """Same ranking, scores dropped."""
        return [chunk for chunk, _ in self.rerank_with_scores(query, chunks, top_k)]