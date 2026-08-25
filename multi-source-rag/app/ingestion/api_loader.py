"""
APILoader — fetches content from a REST API (arXiv, in this example) and
wraps each result as a Document.

Why arXiv, specifically:
    It's a free, public, no-auth-required API that returns real research
    paper abstracts as XML/Atom feed entries. This mirrors a very common
    real pattern: pulling structured records from an internal API (a
    ticketing system, a CRM, a knowledge base) and treating each record as
    a retrievable "document."

Why this loader looks different from LocalLoader/AzureBlobLoader:
    Files need parsing (PDF -> text). API responses are usually already
    text/JSON -- there's no "partition" step. But there's a new kind of
    failure to handle here that files don't have: the network itself can
    fail (timeout, rate limit, malformed response), so this loader's error
    handling is built around `requests` exceptions instead of file I/O
    exceptions.
"""

import xml.etree.ElementTree as ET

import requests

from app.ingestion.base import BaseLoader
from app.ingestion.models import Document
from app.logger import get_logger

log = get_logger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"


class APILoader(BaseLoader):
    def __init__(self, search_query: str, max_results: int = 5):
        self.search_query = search_query
        self.max_results = max_results

    def load(self) -> list[Document]:
        log.info("api_load_started", query=self.search_query, max_results=self.max_results)

        try:
            response = requests.get(
                ARXIV_API_URL,
                params={
                    "search_query": f"all:{self.search_query}",
                    "max_results": self.max_results,
                },
                timeout=10,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            log.error("api_request_failed", query=self.search_query, error=str(e))
            return []

        documents = self._parse_atom_feed(response.text)
        log.info("api_load_finished", documents_loaded=len(documents))
        return documents

    def _parse_atom_feed(self, xml_text: str) -> list[Document]:
        documents: list[Document] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            log.error("api_response_parse_failed", error=str(e))
            return []

        for entry in root.findall(f"{ATOM_NS}entry"):
            title_el = entry.find(f"{ATOM_NS}title")
            summary_el = entry.find(f"{ATOM_NS}summary")
            id_el = entry.find(f"{ATOM_NS}id")

            if summary_el is None or not summary_el.text:
                continue

            title = (title_el.text or "").strip()
            summary = summary_el.text.strip()
            paper_id = (id_el.text or "unknown").strip()

            documents.append(
                Document(
                    content=f"{title}\n\n{summary}",
                    source_type="api",
                    source_path=paper_id,
                    file_type="api_json",
                )
            )

        return documents