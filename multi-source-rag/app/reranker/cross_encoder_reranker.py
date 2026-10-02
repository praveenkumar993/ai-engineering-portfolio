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
        1. Retrieval casts a wide net fast (e.g. top 10)
        2. Cross-encoder re-scores just those 10 (slow but cheap now,
           since there are only 10 of them)
        3. Take the new top_k after reranking -- these are the ones that
           actually go to the LLM

Model choice:
    cross-encoder/ms-marco-MiniLM-L-6-v2 is a small, well-known reranking
    model, trained specifically on query-passage relevance (MS MARCO is a
    search-relevance dataset). It's the standard "starter" cross-encoder,
    the same way MiniLM was our starter embedding model.

Two methods, one job:
    rerank_with_scores() does the real work and returns (chunk, score)
    pairs, in ranked order -- useful for debugging and for
    scripts/compare_reranking.py, which needs to SEE the actual scores,
    not just the final chunk order. rerank() is a thin wrapper around it
    that drops the scores, so every existing caller (RAGPipeline) keeps
    working exactly as before.
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
        # Stashed after every call so the pipeline can check relevance
        # (Step 8's MIN_RELEVANCE_SCORE check) without us returning scores
        # through every single call site.
        self.last_top_score: float | None = None
        log.info("reranker_loaded", model=model_name)

    def rerank_with_scores(
        self, query: str, chunks: list[Chunk], top_k: int
    ) -> list[tuple[Chunk, float]]:
        """Re-score every chunk against the query, return the top_k as
        (chunk, score) pairs, ranked highest score first."""
        if not chunks:
            self.last_top_score = None
            return []

        # Cross-encoder needs (query, chunk_text) pairs -- one pair per chunk
        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self.model.predict(pairs)

        # Sort chunks by their new relevance score, highest first
        scored_chunks = sorted(
            zip(chunks, scores), key=lambda pair: pair[1], reverse=True
        )

        self.last_top_score = float(scored_chunks[0][1])

        log.info(
            "reranking_completed",
            input_count=len(chunks),
            output_count=min(top_k, len(scored_chunks)),
            top_score=self.last_top_score,
        )
        return [(chunk, float(score)) for chunk, score in scored_chunks[:top_k]]

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        """Same as rerank_with_scores(), but returns just the chunks --
        this is what RAGPipeline and everything else in the app calls."""
        return [chunk for chunk, _ in self.rerank_with_scores(query, chunks, top_k)]