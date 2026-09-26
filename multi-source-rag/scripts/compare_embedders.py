"""
compare_embedders.py — runs the SAME question through both the small
(MiniLM) and large (Cohere) embedders, against the same documents, and
shows what each one retrieves as most relevant.

This is a learning tool, not a permanent part of the pipeline.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.logger import configure_logging, get_logger
from app.ingestion.local_loader import LocalLoader
from app.ingestion.chunker import chunk_documents
from app.embeddings.local_embedder import LocalEmbedder
from app.embeddings.cohere_embedder import CohereEmbedder
from app.vectorstore.qdrant_store import QdrantStore

configure_logging()
log = get_logger(__name__)


def run_comparison(question: str) -> None:
    loader = LocalLoader(folder_path="docs")
    documents = loader.load()
    chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=80)

    if not chunks:
        print("No documents found in docs/ -- nothing to compare.")
        return

    print(f"\n{'='*70}\nQUESTION: {question}\n{'='*70}")

    small_embedder = LocalEmbedder()
    start = time.time()
    small_vectors = small_embedder.embed([c.text for c in chunks])
    small_ingest_time = time.time() - start

    small_store = QdrantStore(collection_name="compare_small", dimension=small_embedder.dimension, path="./qdrant_data_small")
    small_store.add_chunks(chunks, small_vectors)

    start = time.time()
    small_query_vector = small_embedder.embed([question])[0]
    small_results = small_store.search(small_query_vector, top_k=1)
    small_query_time = time.time() - start

    print(f"\n--- SMALL (MiniLM, {small_embedder.dimension} dims) ---")
    print(f"Ingest time: {small_ingest_time:.3f}s | Query time: {small_query_time:.3f}s")
    print(f"Top match: {small_results[0].text[:200] if small_results else 'none'}")

    if not settings.cohere_api_key.strip():
        print("\n--- LARGE (Cohere) SKIPPED: COHERE_API_KEY not set in .env ---")
        print("Get a free key at https://dashboard.cohere.com/api-keys")
        return

    large_embedder = CohereEmbedder(api_key=settings.cohere_api_key)
    start = time.time()
    large_vectors = large_embedder.embed([c.text for c in chunks])
    large_ingest_time = time.time() - start

    large_store = QdrantStore(collection_name="compare_large", dimension=large_embedder.dimension, path="./qdrant_data_large")
    large_store.add_chunks(chunks, large_vectors)

    start = time.time()
    large_query_vector = large_embedder.embed([question])[0]
    large_results = large_store.search(large_query_vector, top_k=1)
    large_query_time = time.time() - start

    print(f"\n--- LARGE (Cohere, {large_embedder.dimension} dims) ---")
    print(f"Ingest time: {large_ingest_time:.3f}s | Query time: {large_query_time:.3f}s")
    print(f"Top match: {large_results[0].text[:200] if large_results else 'none'}")

    same_result = (
        small_results and large_results
        and small_results[0].chunk_id == large_results[0].chunk_id
    )
    print(f"\nSame top chunk retrieved by both? {'YES' if same_result else 'NO'}")


if __name__ == "__main__":
    run_comparison("What are the stages of a RAG pipeline?")