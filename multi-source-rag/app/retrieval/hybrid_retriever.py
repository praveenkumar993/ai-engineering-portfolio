"""
HybridRetriever — combines BM25 (keyword) and semantic (vector) search
results using Reciprocal Rank Fusion (RRF).

Why not just average the two scores?
    BM25 scores and cosine similarity scores live on completely different
    scales. Averaging them directly would let whichever score happens to
    have a bigger numeric range dominate, for no principled reason.

    RRF sidesteps this by using only RANK POSITION, not raw scores:
        score = 1 / (k + rank)
    summed across both result lists. k=60 is the standard constant from
    the original RRF paper. A chunk that ranks well in BOTH lists gets a
    high combined score; a chunk that ranks well in only one still gets a
    moderate boost.
"""

from app.ingestion.models import Chunk
from app.logger import get_logger
from app.retrieval.bm25_retriever import BM25Retriever
from app.vectorstore.base import BaseVectorStore
from app.embeddings.base import BaseEmbedder

log = get_logger(__name__)

RRF_K = 60


class HybridRetriever:
    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        bm25_retriever: BM25Retriever,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

    def search(self, query: str, top_k: int = 10) -> list[Chunk]:
        query_vector = self.embedder.embed([query])[0]
        semantic_results = self.vector_store.search(query_vector, top_k=top_k)
        keyword_results = self.bm25_retriever.search(query, top_k=top_k)

        rrf_scores: dict[str, float] = {}
        chunk_lookup: dict[str, Chunk] = {}

        for rank, chunk in enumerate(semantic_results):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + 1 / (RRF_K + rank)
            chunk_lookup[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(keyword_results):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + 1 / (RRF_K + rank)
            chunk_lookup[chunk.chunk_id] = chunk

        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        fused_results = [chunk_lookup[cid] for cid in ranked_ids[:top_k]]

        log.info(
            "hybrid_search_completed",
            query=query,
            semantic_count=len(semantic_results),
            keyword_count=len(keyword_results),
            fused_count=len(fused_results),
        )
        return fused_results