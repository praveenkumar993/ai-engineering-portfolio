"""
Entry point.

Step 1: config + logging spine.
Step 2: load local documents and chunk them.
Step 3: embed chunks and store them in Qdrant.
Step 4: retrieve relevant chunks and generate a grounded answer.
Step 5: Azure Blob as a second source (graceful skip if unconfigured).
Step 6: API (arXiv) as a third source.
Step 6.5: S3 as a fourth source (graceful skip if unconfigured).
Step 7: reranker for better retrieval precision.
Step 8: formalized fallback handling via RAGPipeline + startup validation.
"""

from app.config import settings
from app.logger import configure_logging, get_logger
from app.startup_checks import validate_startup_config, ConfigError
from app.ingestion.local_loader import LocalLoader
from app.ingestion.azure_blob_loader import AzureBlobLoader
from app.ingestion.api_loader import APILoader
from app.ingestion.s3_loader import S3Loader
from app.ingestion.chunker import chunk_documents
from app.embeddings.local_embedder import LocalEmbedder
from app.vectorstore.qdrant_store import QdrantStore
from app.reranker.cross_encoder_reranker import CrossEncoderReranker
from app.generation.groq_llm import GroqLLM
from app.generation.pipeline import RAGPipeline
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever

configure_logging()
log = get_logger(__name__)


def main() -> None:
    log.info(
        "app_starting",
        app_name=settings.app_name,
        environment=settings.environment,
        log_level=settings.log_level,
    )

    # --- Step 8: fail fast if required config is missing ---
    try:
        validate_startup_config()
    except ConfigError as e:
        log.error("startup_aborted")
        print(f"\nStartup failed:\n{e}\n")
        return

    # --- Multi-source ingestion ---
    all_documents = []

    local_loader = LocalLoader(folder_path="docs")
    all_documents.extend(local_loader.load())

    if settings.azure_storage_connection_string.strip() and settings.azure_container_name.strip():
        azure_loader = AzureBlobLoader(
            connection_string=settings.azure_storage_connection_string,
            container_name=settings.azure_container_name,
        )
        all_documents.extend(azure_loader.load())
    else:
        log.warning("azure_source_skipped_not_configured")

    api_loader = APILoader(search_query="retrieval augmented generation", max_results=3)
    all_documents.extend(api_loader.load())

    if settings.aws_s3_bucket_name.strip() and settings.aws_access_key_id.strip():
        s3_loader = S3Loader(
            bucket_name=settings.aws_s3_bucket_name,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
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

    # --- Embed + store ---
    embedder = LocalEmbedder()
    vectors = embedder.embed([c.text for c in chunks])

    store = QdrantStore(collection_name="rag_chunks", dimension=embedder.dimension)
    store.add_chunks(chunks, vectors)

    # --- Step 8: build the pipeline and ask through it ---
    bm25 = BM25Retriever(chunks)
    hybrid_retriever = HybridRetriever(embedder=embedder, vector_store=store, bm25_retriever=bm25)

    reranker = CrossEncoderReranker()
    llm = GroqLLM()
    pipeline = RAGPipeline(retriever=hybrid_retriever, reranker=reranker, llm=llm)

    question = "What are the stages of a RAG pipeline?"
    response = pipeline.ask(question)

    print(f"\nQuestion: {question}")
    print(f"\nStatus: {response.status.value}")
    print(f"\nAnswer:\n{response.answer}")
    print(f"\n(Grounded in {len(response.sources)} chunks)")
    for c in response.sources:
        print(f"  - {c.source_type}: {c.source_path}")

    log.info("app_ready")


if __name__ == "__main__":
    main()