"""
BaseLoader — the contract every data source must follow.

Why this file matters most in Step 2:
    Right now we only have ONE loader (local filesystem). It would be
    tempting to skip this abstraction and just write a `load_local_files()`
    function. That works fine... until Step 5, when we add Azure Blob, and
    Step 6, when we add an API source. Without a shared contract, we'd end
    up with three unrelated functions with different signatures, and every
    piece of code downstream (chunker, embedder) would need to know which
    one it's talking to.

    With a BaseLoader, downstream code only ever calls `.load()` and gets
    back `list[Document]` — it never needs to know or care if the source
    was local disk, Azure, or an API.
"""

from abc import ABC, abstractmethod

from app.ingestion.models import Document


class BaseLoader(ABC):
    """Every concrete loader (Local, AzureBlob, API...) implements this."""

    @abstractmethod
    def load(self) -> list[Document]:
        """Read from the source and return parsed Documents."""
        raise NotImplementedError