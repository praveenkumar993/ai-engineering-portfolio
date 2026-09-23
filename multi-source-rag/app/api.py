"""
FastAPI app — exposes the RAG pipeline as a real HTTP service.

Design decision: build everything ONCE at startup, not per-request.
    Loading the embedding model, connecting to Qdrant, ingesting documents --
    all of this is expensive. If we did it inside the /ask endpoint, every
    single request would re-load models and re-embed the same documents,
    which would make each request take 10+ seconds instead of a fraction of
    a second. FastAPI's `lifespan` context manager runs setup code once when
    the server starts, and teardown code once when it stops. We build the
    RAGPipeline there and stash it in `app.state`, so every request just
    reuses the already-built pipeline.

Design decision: /health endpoint.
    A production service always needs a cheap endpoint that says "I'm up
    and able to serve requests" -- used by load balancers, container
    orchestrators (Kubernetes readiness probes), and uptime monitors. It
    should NOT do real work (no embedding, no LLM calls) -- just confirm
    the process is alive and the pipeline was built successfully.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

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


def build_pipeline() -> RAGPipeline:
    """Everything from main.py's setup, packaged as one reusable function."""
    validate_startup_config()

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

    embedder = LocalEmbedder()
    vectors = embedder.embed([c.text for c in chunks]) if chunks else []

    store = QdrantStore(collection_name="rag_chunks", dimension=embedder.dimension, url=settings.qdrant_url)
    if chunks:
        store.add_chunks(chunks, vectors)
        
    bm25 = BM25Retriever(chunks)
    hybrid_retriever = HybridRetriever(embedder=embedder, vector_store=store, bm25_retriever=bm25)

    reranker = CrossEncoderReranker()
    llm = GroqLLM()
    pipeline = RAGPipeline(retriever=hybrid_retriever, reranker=reranker, llm=llm)
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("api_startup_begin")
    try:
        app.state.pipeline = build_pipeline()
        app.state.ready = True
        log.info("api_startup_complete")
    except ConfigError as e:
        log.error("api_startup_failed", error=str(e))
        app.state.pipeline = None
        app.state.ready = False

    yield

    log.info("api_shutdown")


app = FastAPI(
    title="Multi-Source RAG API",
    description="Production-style RAG pipeline over local, Azure Blob, S3, and API sources.",
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    status: str
    answer: str
    source_count: int


@app.get("/health")
def health_check():
    ready = getattr(app.state, "ready", False)
    return {"status": "ok" if ready else "not_ready", "ready": ready}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not getattr(app.state, "ready", False):
        return AskResponse(
            status="service_unavailable",
            answer="The service is not ready. Check server startup logs.",
            source_count=0,
        )

    response = app.state.pipeline.ask(request.question)
    return AskResponse(
        status=response.status.value,
        answer=response.answer,
        source_count=len(response.sources),
    )