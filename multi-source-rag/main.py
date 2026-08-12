"""
Entry point.

Step 1: config + logging spine.
Step 2: load local documents and chunk them.
Step 3: embed chunks and store them in Qdrant, then run a test search.
"""

from app.config import settings
from app.logger import configure_logging, get_logger
from app.ingestion.local_loader import LocalLoader
from app.ingestion.chunker import chunk_documents
from app.embeddings.local_embedder import LocalEmbedder
from app.vectorstore.qdrant_store import QdrantStore

configure_logging()
log = get_logger(__name__)


def main() -> None:
    log.info(
        "app_starting",
        app_name=settings.app_name,
        environment=settings.environment,
        log_level=settings.log_level,
    )

    loader = LocalLoader(folder_path="docs")
    documents = loader.load()
    chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=80)
    log.info("pipeline_summary", documents=len(documents), chunks=len(chunks))

    if not chunks:
        log.warning("no_chunks_to_embed_stopping")
        return

    embedder = LocalEmbedder()
    vectors = embedder.embed([c.text for c in chunks])

    store = QdrantStore(
        collection_name="rag_chunks",
        dimension=embedder.dimension,
    )
    store.add_chunks(chunks, vectors)

    test_query = "What are the stages of a RAG pipeline?"
    query_vector = embedder.embed([test_query])[0]
    results = store.search(query_vector, top_k=2)

    print(f"\nQuery: {test_query}")
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} (from {r.source_path}) ---")
        print(r.text[:200])

    log.info("app_ready")


if __name__ == "__main__":
    main()