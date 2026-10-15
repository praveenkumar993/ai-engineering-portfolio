"""
CrossEncoderReranker — re-scores retrieved chunks using a cross-encoder
model for more accurate relevance ranking than raw vector similarity alone.

The cross-encoder reads the query and one chunk TOGETHER and outputs a
single relevance score. That is more accurate than comparing embeddings,
but it can't be precomputed, so it only runs on the small candidate set
that retrieval already narrowed down (e.g. top 10).

Note: this class holds NO per-request state. Scores are returned to the
caller instead of being stashed on the object, because one reranker
instance is shared by every concurrent API request.
"""

from sentence_transformers import CrossEncoder

from app.ingestion.models import Chunk
from app.logger import get_logger
from app.reranker.base import BaseReranker

log = get_logger(__name__)


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        log.info("reranker_loading", model=model_name)
        self.model = CrossEncoder(model_name)
        log.info("reranker_loaded", model=model_name)

    def rerank_with_scores(
        self, query: str, chunks: list[Chunk], top_k: int
    ) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []

        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self.model.predict(pairs)

        scored_chunks = sorted(
            zip(chunks, scores), key=lambda pair: pair[1], reverse=True
        )

        log.info(
            "reranking_completed",
            input_count=len(chunks),
            output_count=min(top_k, len(scored_chunks)),
            top_score=float(scored_chunks[0][1]),
        )
        return [(chunk, float(score)) for chunk, score in scored_chunks[:top_k]]