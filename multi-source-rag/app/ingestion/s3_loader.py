"""
S3Loader — reads PDF/DOCX/PPTX/CSV/TXT files from an AWS S3 bucket.

This is nearly identical to AzureBlobLoader in structure -- same
"download to temp file, parse, delete" pattern -- just using boto3 (AWS's
SDK) instead of azure-storage-blob. Seeing both side by side is useful:
it shows you that "add a new cloud storage source" is a repeatable recipe
once you've done it once, not a from-scratch problem each time.

Differences worth noting vs Azure:
    - AWS auth is typically via access key + secret key (or IAM role, in
      real deployments) rather than a single connection string.
    - S3 calls it a "bucket" instead of a "container," and "object" instead
      of "blob" -- same concept, different vocabulary.
"""

import tempfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from unstructured.partition.auto import partition

from app.ingestion.base import BaseLoader
from app.ingestion.models import Document
from app.logger import get_logger

log = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".csv", ".txt"}


class S3Loader(BaseLoader):
    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str = "us-east-1",
    ):
        self.bucket_name = bucket_name
        self.client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )

    def load(self) -> list[Document]:
        documents: list[Document] = []

        try:
            response = self.client.list_objects_v2(Bucket=self.bucket_name)
            objects = response.get("Contents", [])
        except ClientError as e:
            log.error("s3_bucket_list_failed", bucket=self.bucket_name, error=str(e))
            return []

        relevant_objects = [
            o for o in objects if Path(o["Key"]).suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        log.info("s3_load_started", bucket=self.bucket_name, object_count=len(relevant_objects))

        for obj in relevant_objects:
            key = obj["Key"]
            try:
                suffix = Path(key).suffix.lower()
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    self.client.download_fileobj(self.bucket_name, key, tmp)
                    tmp_path = tmp.name

                elements = partition(filename=tmp_path)
                content = "\n".join(str(el) for el in elements)

                Path(tmp_path).unlink(missing_ok=True)

                if not content.strip():
                    log.warning("empty_object_skipped", key=key)
                    continue

                documents.append(
                    Document(
                        content=content,
                        source_type="s3",
                        source_path=f"{self.bucket_name}/{key}",
                        file_type=suffix.lstrip("."),
                    )
                )
                log.info("object_loaded", key=key, chars=len(content))

            except Exception as e:
                log.error("object_load_failed", key=key, error=str(e))
                continue

        log.info("s3_load_finished", documents_loaded=len(documents))
        return documents