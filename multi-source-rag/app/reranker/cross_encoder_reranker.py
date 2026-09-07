"""
CrossEncoderReranker — re-scores retrieved chunks using a cross-encoder
model for more accurate relevance ranking than raw vector similarity alone.

Why this is a SEPARATE step from vector search, not a replacement for it:
    Cross-encoders are much more accurate at judging "is this chunk
    actually relevant to this query" because they look at the query and
    the chunk TOGETHER, letting the model directly compare them word by
    word. But that accuracy has a cost: you can't precompute a cross-encoder
    score ahead of time the way we precompute embeddings, because it needs
    both the query AND the chunk at scoring time. Running it over every
    chunk in the database, for every query, would be far too slow.

    The standard production pattern (this file included) is:
        1. Vector search casts a wide net fast (e.g. top 10)
        2. Cross-encoder re-scores just those 10 (slow but now it's cheap,
           since there are only 10 of them)
        3. Take the new top_k after reranking -- these are the ones that
           actually go to the LLM

Model choice:
    cross-encoder/ms-marco-MiniLM-L-6-v2 is a small, well-known reranking
    model, trained specifically on query-passage relevance (MS MARCO is a
    search-relevance dataset). It's the standard "starter" cross-encoder,
    the same way MiniLM was our starter embedding model.
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
        self.last_top_score: float | None = None
        log.info("reranker_loaded", model=model_name)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        if not chunks:
            return []

        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self.model.predict(pairs)

        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        reranked = [chunk for chunk, score in scored_chunks[:top_k]]

        # Stash the top score so the pipeline can check relevance afterward.
        self.last_top_score = float(scored_chunks[0][1]) if scored_chunks else None

        log.info(
            "reranking_completed",
            input_count=len(chunks),
            output_count=len(reranked),
            top_score=self.last_top_score,

        )
        return reranked