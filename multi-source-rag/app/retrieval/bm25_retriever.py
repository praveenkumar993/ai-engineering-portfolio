"""
BM25Retriever — keyword-based retrieval using the BM25 algorithm.

Why this is a DIFFERENT kind of component than QdrantStore:
    QdrantStore does semantic search -- it needs precomputed embeddings and
    a real database. BM25 works purely on the actual words in the text, so
    it doesn't need embeddings or a database at all -- just the chunk texts
    themselves, held in memory, indexed by word statistics.

How BM25 scoring works, briefly:
    For a given query, BM25 scores each document based on how many query
    terms it contains, weighted so that rare terms count for more than
    common terms, and repeating a term has diminishing returns. It has no
    idea that "car" and "automobile" mean the same thing -- that's the
    trade-off vs semantic search.
"""

from rank_bm25 import BM25Okapi

from app.ingestion.models import Chunk
from app.logger import get_logger

log = get_logger(__name__)


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        tokenized_corpus = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        log.info("bm25_index_built", chunk_count=len(chunks))

    def search(self, query: str, top_k: int = 10) -> list[Chunk]:
        if not self.chunks:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        scored = sorted(
            zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True
        )
        top_chunks = [chunk for chunk, score in scored[:top_k]]

        log.info("bm25_search_completed", query=query, results=len(top_chunks))
        return top_chunks