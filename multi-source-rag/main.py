"""
Entry point.

Step 1: config + logging spine.
Step 2: load local documents and chunk them -- no embeddings yet.
"""

from app.config import settings
from app.logger import configure_logging, get_logger
from app.ingestion.local_loader import LocalLoader
from app.ingestion.chunker import chunk_documents

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

    for c in chunks[:3]:
        print(f"\n--- Chunk {c.chunk_index} from {c.source_path} ---")
        print(c.text[:200])

    log.info("app_ready")


if __name__ == "__main__":
    main()