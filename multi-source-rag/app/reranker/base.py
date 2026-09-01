"""
BaseReranker — the contract every reranking model must follow.

Same pattern as every other Base* file in this project. Today it's a local
cross-encoder model. Later you could swap in a hosted reranker API (Cohere
Rerank, Azure AI Search's semantic ranker) behind this same contract.
"""

from abc import ABC, abstractmethod

from app.ingestion.models import Chunk


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        """Re-score chunks against the query and return the top_k best, in order."""
        raise NotImplementedError