"""
LocalLoader — reads PDF/DOCX/PPTX/CSV/TXT files from a local folder.

Design note:
    We use the `unstructured` library because it gives one consistent
    function (`partition`) for many file types, instead of us writing a
    separate parser for PDF (PyMuPDF), DOCX (python-docx), PPTX
    (python-pptx), and CSV (pandas) by hand.

Design note on failures:
    A single corrupt or unsupported file should NOT crash the whole
    ingestion run. We log the failure and continue with the rest. This is
    a "partial fallback" -- one of the production patterns we called out
    in the roadmap.
"""

from pathlib import Path

from unstructured.partition.auto import partition

from app.ingestion.base import BaseLoader
from app.ingestion.models import Document
from app.logger import get_logger

log = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".csv", ".txt"}


class LocalLoader(BaseLoader):
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)

    def load(self) -> list[Document]:
        if not self.folder_path.exists():
            log.error("local_folder_missing", path=str(self.folder_path))
            return []

        documents: list[Document] = []
        files = [
            f for f in self.folder_path.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        log.info("local_load_started", file_count=len(files))

        for file_path in files:
            try:
                elements = partition(filename=str(file_path))
                content = "\n".join(str(el) for el in elements)

                if not content.strip():
                    log.warning("empty_file_skipped", file=file_path.name)
                    continue

                documents.append(
                    Document(
                        content=content,
                        source_type="local",
                        source_path=str(file_path),
                        file_type=file_path.suffix.lstrip(".").lower(),
                    )
                )
                log.info("file_loaded", file=file_path.name, chars=len(content))

            except Exception as e:
                # One bad file should never kill the whole ingestion run.
                log.error("file_load_failed", file=file_path.name, error=str(e))
                continue

        log.info("local_load_finished", documents_loaded=len(documents))
        return documents