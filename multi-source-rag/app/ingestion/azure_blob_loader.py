"""
AzureBlobLoader — reads PDF/DOCX/PPTX/CSV/TXT files from an Azure Blob
Storage container.

This is the payoff of Step 2's BaseLoader abstraction. Notice this class
implements the exact same contract as LocalLoader: a `.load()` method that
returns `list[Document]`. Nothing in main.py, chunker.py, or anywhere else
needs to change to support this new source -- we just point at it.

Design note -- why download to a temp file instead of parsing bytes directly:
    `unstructured.partition()` is built to accept a file path. Blob storage
    gives us raw bytes over the network. The reliable approach is: download
    the blob's bytes into a temporary local file, parse that file the same
    way LocalLoader does, then delete the temp file. This is a common
    real-world pattern -- cloud sources are usually "materialize locally,
    then parse."

Design note -- failures:
    Same pattern as LocalLoader: one bad blob should not kill the whole
    ingestion run. We log and continue.
"""

import tempfile
from pathlib import Path

from azure.storage.blob import ContainerClient
from unstructured.partition.auto import partition

from app.ingestion.base import BaseLoader
from app.ingestion.models import Document
from app.logger import get_logger

log = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".csv", ".txt"}


class AzureBlobLoader(BaseLoader):
    def __init__(self, connection_string: str, container_name: str):
        self.container_name = container_name
        self.container_client = ContainerClient.from_connection_string(
            conn_str=connection_string,
            container_name=container_name,
        )

    def load(self) -> list[Document]:
        documents: list[Document] = []

        try:
            blobs = list(self.container_client.list_blobs())
        except Exception as e:
            log.error("azure_container_list_failed", container=self.container_name, error=str(e))
            return []

        relevant_blobs = [
            b for b in blobs if Path(b.name).suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        log.info("azure_load_started", container=self.container_name, blob_count=len(relevant_blobs))

        for blob in relevant_blobs:
            try:
                blob_client = self.container_client.get_blob_client(blob.name)
                blob_bytes = blob_client.download_blob().readall()

                suffix = Path(blob.name).suffix.lower()
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(blob_bytes)
                    tmp_path = tmp.name

                elements = partition(filename=tmp_path)
                content = "\n".join(str(el) for el in elements)

                Path(tmp_path).unlink(missing_ok=True)

                if not content.strip():
                    log.warning("empty_blob_skipped", blob=blob.name)
                    continue

                documents.append(
                    Document(
                        content=content,
                        source_type="azure_blob",
                        source_path=f"{self.container_name}/{blob.name}",
                        file_type=suffix.lstrip("."),
                    )
                )
                log.info("blob_loaded", blob=blob.name, chars=len(content))

            except Exception as e:
                log.error("blob_load_failed", blob=blob.name, error=str(e))
                continue

        log.info("azure_load_finished", documents_loaded=len(documents))
        return documents