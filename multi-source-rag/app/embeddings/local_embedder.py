"""
LocalEmbedder — small, free, local embedding model.

Why all-MiniLM-L6-v2?
    It's a well-known "small" embedding model: 384-dimensional vectors, runs
    fast on CPU, no API key or cost. Once we plug in a "large" model later
    (e.g. Azure OpenAI text-embedding-3-large, 3072 dimensions), you'll be
    able to compare retrieval quality between the two on the exact same
    documents.
"""

from sentence_transformers import SentenceTransformer

from app.embeddings.base import BaseEmbedder
from app.logger import get_logger

log = get_logger(__name__)


class LocalEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        log.info("embedder_loading", model=model_name)
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
        log.info("embedder_loaded", model=model_name, dimension=self._dimension)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, show_progress_bar=False)
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension