"""
Entry point.

Step 1: config + logging spine.
Step 2: load local documents and chunk them.
Step 3: embed chunks and store them in Qdrant.
Step 4: retrieve relevant chunks for a question and generate a grounded
        answer with an LLM -- this completes the first working RAG loop.
"""

from app.config import settings
from app.logger import configure_logging, get_logger
from app.ingestion.local_loader import LocalLoader
from app.ingestion.chunker import chunk_documents
from app.embeddings.local_embedder import LocalEmbedder
from app.vectorstore.qdrant_store import QdrantStore
from app.generation.groq_llm import GroqLLM
from app.generation.prompt import build_rag_prompt

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

    question = "What are the stages of a RAG pipeline?"
    query_vector = embedder.embed([question])[0]
    retrieved_chunks = store.search(query_vector, top_k=3)

    prompt = build_rag_prompt(question, retrieved_chunks)
    llm = GroqLLM()
    answer = llm.generate(prompt)

    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{answer}")
    print(f"\n(Grounded in {len(retrieved_chunks)} retrieved chunks)")

    log.info("app_ready")


if __name__ == "__main__":
    main()