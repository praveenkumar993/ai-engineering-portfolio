"""
Entry point.

Step 1: config + logging spine.
Step 2: load local documents and chunk them.
Step 3: embed chunks and store them in Qdrant.
Step 4: retrieve relevant chunks and generate a grounded answer.
Step 5: add Azure Blob Storage as a second source, merged with local --
        this is the multi-source part of "multi-source RAG."
"""

from app.config import settings
from app.logger import configure_logging, get_logger
from app.ingestion.local_loader import LocalLoader
from app.ingestion.azure_blob_loader import AzureBlobLoader
from app.ingestion.chunker import chunk_documents
from app.embeddings.local_embedder import LocalEmbedder
from app.vectorstore.qdrant_store import QdrantStore
from app.generation.groq_llm import GroqLLM
from app.generation.prompt import build_rag_prompt
from app.ingestion.api_loader import APILoader
from app.ingestion.s3_loader import S3Loader
from app.reranker.cross_encoder_reranker import CrossEncoderReranker

configure_logging()
log = get_logger(__name__)


def main() -> None:
    log.info(
        "app_starting",
        app_name=settings.app_name,
        environment=settings.environment,
        log_level=settings.log_level,
    )

    # --- Step 2 + Step 5: multi-source ingestion ---
    # Every loader implements the same BaseLoader contract, so we just
    # call .load() on each and merge the results into one list.
    all_documents = []

    local_loader = LocalLoader(folder_path="docs")
    all_documents.extend(local_loader.load())

    if (settings.azure_storage_connection_string.strip() and settings.azure_container_name.strip() 
        and "your_connection_string" not in settings.azure_storage_connection_string):
        azure_loader = AzureBlobLoader(
            connection_string=settings.azure_storage_connection_string,
            container_name=settings.azure_container_name,
        )
        all_documents.extend(azure_loader.load())
    else:
        log.warning("azure_source_skipped_not_configured")
        api_loader = APILoader(search_query="retrieval augmented generation", max_results=3)
        all_documents.extend(api_loader.load())

    # --- Step 5: S3 ingestion ---
    if (settings.aws_s3_bucket_name.strip() and settings.aws_access_key_id.strip() 
        and "your_access_key_id" not in settings.aws_access_key_id):
        s3_loader = S3Loader(
            bucket_name=settings.aws_s3_bucket_name,
            access_key_id=settings.aws_access_key_id,
            secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region_name,
        )
        all_documents.extend(s3_loader.load())
    else:
        log.warning("s3_source_skipped_not_configured")

    chunks = chunk_documents(all_documents, chunk_size=500, chunk_overlap=80)
    log.info("pipeline_summary", documents=len(all_documents), chunks=len(chunks))

    if not chunks:
        log.warning("no_chunks_to_embed_stopping")
        return

    # --- Step 3: embed + store ---
    embedder = LocalEmbedder()
    vectors = embedder.embed([c.text for c in chunks])

    store = QdrantStore(
        collection_name="rag_chunks",
        dimension=embedder.dimension,
    )
    store.add_chunks(chunks, vectors)

    # --- Step 4: retrieve + generate ---

    question = "What are the stages of a RAG pipeline?"
    query_vector = embedder.embed([question])[0]
    retrieved_chunks = store.search(query_vector, top_k=10)

    # --- Step 5: rerank ---
    reranker = CrossEncoderReranker()
    reranked_chunks = reranker.rerank(question, retrieved_chunks, top_k=3)

    prompt = build_rag_prompt(question, reranked_chunks)
    llm = GroqLLM()
    answer = llm.generate(prompt)

    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{answer}")
    print(f"\n(Grounded in {len(reranked_chunks)} reranked chunks)")
    for c in reranked_chunks:
        print(f"  - {c.source_type}: {c.source_path}")

    log.info("app_ready")


if __name__ == "__main__":
    main()