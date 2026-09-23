"""
QdrantStore — stores and searches chunk embeddings using Qdrant.

Why local disk mode for now?
    QdrantClient(path="...") runs Qdrant embedded, writing to a local folder
    -- no server, no Docker needed yet. In Step 11 we'll switch to
    QdrantClient(url="http://localhost:6333") pointing at a real Qdrant
    server running in Docker -- a one-line change, everything else stays
    the same.
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.ingestion.models import Chunk
from app.logger import get_logger
from app.vectorstore.base import BaseVectorStore

log = get_logger(__name__)


class QdrantStore(BaseVectorStore):
    def __init__(self, collection_name: str, dimension: int, path: str = "./qdrant_data", url: str = ""):
        self.collection_name = collection_name
        # If a URL is configured (Docker/production), connect to a real
        # Qdrant server. Otherwise fall back to local-disk mode (local dev
        # without Docker) -- exactly the "one-line change" we set up for
        # back in Step 3.
        if url:
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantClient(path=path)

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            log.info("collection_created", name=collection_name, dimension=dimension)
        else:
            log.info("collection_exists", name=collection_name)

    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=chunk.model_dump(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        log.info("chunks_stored", count=len(points), collection=self.collection_name)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[Chunk]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        ).points

        chunks = [Chunk(**r.payload) for r in results]
        log.info("search_completed", results_found=len(chunks), top_k=top_k)
        return chunks