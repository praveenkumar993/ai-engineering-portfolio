"""
Data models for the ingestion layer.

Why typed models instead of plain dicts?
    If a "document" is just a dict, every function that touches it has to
    guess what keys exist. A typo like doc["souce"] instead of doc["source"]
    fails silently at runtime, often deep in a pipeline.

    With Pydantic models, that typo is caught immediately, editors autocomplete
    fields, and every function's signature tells you exactly what shape of
    data it expects.
"""

from pydantic import BaseModel


class Document(BaseModel):
    """One raw file, fully loaded, before chunking."""

    content: str          # full extracted text
    source_type: str      # "local" | "azure_blob" | "api" (more added later)
    source_path: str      # file path, blob name, or API resource id
    file_type: str        # "pdf" | "docx" | "pptx" | "csv"


class Chunk(BaseModel):
    """A smaller piece of a Document, ready for embedding."""

    chunk_id: str
    text: str
    source_type: str
    source_path: str
    file_type: str
    chunk_index: int      # position of this chunk within its source document