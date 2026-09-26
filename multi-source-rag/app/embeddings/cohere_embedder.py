"""
CohereEmbedder — a "large" hosted embedding model, used to compare against
our local "small" MiniLM embedder.

Why Cohere specifically:
    Cohere's embed-english-v3.0 has a generous free tier, and it produces
    1024-dimensional vectors versus MiniLM's 384 -- a real, concrete
    difference in scale. Comparing the two side by side builds genuine
    intuition for the small-vs-large embedding tradeoff.

What "large" actually buys you, and what it costs:
    Larger embedding models are generally trained on more data and capture
    finer-grained semantic distinctions. The cost: every embedding call is
    now a network request, versus MiniLM running instantly, locally, free.
"""

import cohere

from app.embeddings.base import BaseEmbedder
from app.logger import get_logger

log = get_logger(__name__)


class CohereEmbedder(BaseEmbedder):
    def __init__(self, api_key: str, model_name: str = "embed-english-v3.0"):
        self.client = cohere.Client(api_key)
        self.model_name = model_name
        self._dimension = 1024
        log.info("cohere_embedder_ready", model=model_name, dimension=self._dimension)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(
            texts=texts,
            model=self.model_name,
            input_type="search_document",
        )
        return response.embeddings

    @property
    def dimension(self) -> int:
        return self._dimension