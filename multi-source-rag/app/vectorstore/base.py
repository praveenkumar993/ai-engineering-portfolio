"""
BaseVectorStore — the contract every vector database must follow.

Today it's Qdrant running locally on disk. In Step 11 we containerize it as
a proper Qdrant server. As long as every implementation follows this
contract (add chunks in, search by vector, get matches back), the retrieval
logic built on top never has to change.
"""

from abc import ABC, abstractmethod

from app.ingestion.models import Chunk


class BaseVectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Store chunks alongside their embedding vectors."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int = 5) -> list[Chunk]:
        """Return the top_k most similar chunks to the query vector."""
        raise NotImplementedError