"""
BaseEmbedder — the contract every embedding model must follow.

Same reasoning as BaseLoader in Step 2: today we use one small local model
(all-MiniLM-L6-v2). Later we'll add a second implementation wrapping a large
model (Azure OpenAI / Cohere embeddings). Both will implement this same
contract, so the vector store and retrieval code never need to know which
embedding model produced the vectors -- they just call `.embed()`.
"""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Convert a list of text strings into a list of embedding vectors."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The size of the vectors this embedder produces (e.g. 384)."""
        raise NotImplementedError