"""
compare_reranking.py — shows what the reranker actually CHANGES.

For each question, we print two orderings of the SAME candidate chunks:
    BEFORE: the order hybrid retrieval (BM25 + semantic, fused with RRF)
            produced
    AFTER:  the order after the cross-encoder re-scored every chunk

Then we mark how far each chunk moved. This is a learning tool: it turns
"the reranker improves precision" from a claim you repeat into something
you have watched happen.

What to look for when you run this:
    - Does the #1 chunk change? If reranking never changes the top result,
      it isn't earning its latency cost on this data.
    - Which chunks jump up? Usually ones that directly answer the question
      but share few exact words with it (hard for retrieval, easier for a
      cross-encoder that reads query and chunk together).
    - Do the scores separate clearly? A big gap between #1 and #2 means the
      model is confident; a flat spread means nothing stands out.

Uses smaller chunks than the main pipeline (200 chars vs 500) purely so a
short test PDF yields enough candidates to reorder. It uses its own Qdrant
folder so it never collides with anything else holding a lock on
./qdrant_data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.logger import configure_logging
from app.ingestion.local_loader import LocalLoader
from app.ingestion.chunker import chunk_documents
from app.embeddings.local_embedder import LocalEmbedder
from app.vectorstore.qdrant_store import QdrantStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.reranker.cross_encoder_reranker import CrossEncoderReranker

configure_logging()

QUESTIONS = [
    "What are the stages of a RAG pipeline?",
    "How do production systems handle failures?",
    "What happens to a document before it is stored?",
]


def snippet(text: str, width: int = 70) -> str:
    return " ".join(text.split())[:width]


def compare(question, retriever, reranker) -> None:
    candidates = retriever.search(question, top_k=10)
    reranked = reranker.rerank_with_scores(question, candidates, top_k=len(candidates))

    before_rank = {c.chunk_id: i + 1 for i, c in enumerate(candidates)}

    print(f"\n{'=' * 78}\nQUESTION: {question}\n{'=' * 78}")
    print(f"{'AFTER':<7}{'BEFORE':<9}{'MOVE':<8}{'SCORE':<9}CHUNK")
    print("-" * 78)

    for after_pos, (chunk, score) in enumerate(reranked, start=1):
        before_pos = before_rank[chunk.chunk_id]
        delta = before_pos - after_pos
        move = f"+{delta}" if delta > 0 else (str(delta) if delta < 0 else "=")
        print(f"#{after_pos:<6}#{before_pos:<8}{move:<8}{score:<9.2f}{snippet(chunk.text)}")

    top_before = candidates[0].chunk_id
    top_after = reranked[0][0].chunk_id
    changed = "YES" if top_before != top_after else "NO"
    print(f"\nDid reranking change the #1 result? {changed}")


def main() -> None:
    documents = LocalLoader(folder_path="docs").load()
    chunks = chunk_documents(documents, chunk_size=200, chunk_overlap=40)

    if not chunks:
        print("No documents in docs/ -- nothing to compare.")
        return
    print(f"Working with {len(chunks)} chunks (200-char chunks, for more to reorder).")

    embedder = LocalEmbedder()
    vectors = embedder.embed([c.text for c in chunks])

    store = QdrantStore(
        collection_name="rerank_compare",
        dimension=embedder.dimension,
        path="./qdrant_data_rerank",
    )
    store.add_chunks(chunks, vectors)

    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=store,
        bm25_retriever=BM25Retriever(chunks),
    )
    reranker = CrossEncoderReranker()

    for question in QUESTIONS:
        compare(question, retriever, reranker)


if __name__ == "__main__":
    main()