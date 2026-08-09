"""
Chunker — splits a Document's text into smaller Chunks.

Why chunk at all?
    LLMs and embedding models have limited context windows, and retrieval
    works better on focused pieces of text than on entire documents.

Why fixed-size with overlap, for v1?
    There are fancier strategies (semantic chunking, sentence-boundary aware
    splitting) -- we will get there. For now we use the simplest strategy
    that actually works: split text into chunks of `chunk_size` characters,
    each overlapping the previous by `chunk_overlap` characters so we don't
    lose context right at a chunk boundary.
"""

from app.ingestion.models import Chunk, Document
from app.logger import get_logger

log = get_logger(__name__)


def chunk_document(
    document: Document,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    text = document.content
    chunks: list[Chunk] = []

    start = 0
    index = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()

        if piece:
            chunks.append(
                Chunk(
                    chunk_id=f"{document.source_path}::{index}",
                    text=piece,
                    source_type=document.source_type,
                    source_path=document.source_path,
                    file_type=document.file_type,
                    chunk_index=index,
                )
            )
            index += 1

        start += chunk_size - chunk_overlap

    log.info("document_chunked", source=document.source_path, chunk_count=len(chunks))
    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return all_chunks