import logging
import mimetypes
from typing import Optional
from urllib.parse import urlparse

import boto3
import httpx
from botocore.exceptions import BotoCoreError, ClientError
from agno.tools.toolkit import Toolkit

logger = logging.getLogger(__name__)

MAX_RESULTS = 20


class S3SearchTool(Toolkit):
    """Agno tool that searches for documents in an S3 bucket and generates presigned URLs."""

    def __init__(
        self,
        bucket_name: str,
        region: str = "us-east-1",
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        presigned_url_expiry: int = 3600,
    ):
        super().__init__(name="s3_search")
        self.bucket_name = bucket_name
        self.presigned_url_expiry = presigned_url_expiry

        # Build client kwargs; omit credentials to fall back to IAM/env
        client_kwargs = {"region_name": region}
        if access_key_id and secret_access_key:
            client_kwargs["aws_access_key_id"] = access_key_id
            client_kwargs["aws_secret_access_key"] = secret_access_key

        self._s3 = boto3.client("s3", **client_kwargs)

        self.register(self.search_documents)
        self.register(self.get_document_url)
        self.register(self.save_document)

    def search_documents(self, prefix: str) -> str:
        """Search for documents in the S3 bucket by filename prefix.

        Use this tool to find OEM manuals, repair guides, parts catalogs, work order
        attachments, or any other documents stored in the company document store.

        Args:
            prefix: The filename prefix or path to search for (e.g., "kubota/", "manuals/SVL97", "parts-catalog")

        Returns:
            A formatted list of matching document keys, or a message if no results found.
        """
        try:
            paginator = self._s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

            keys = []
            for page in pages:
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
                    if len(keys) >= MAX_RESULTS:
                        break
                if len(keys) >= MAX_RESULTS:
                    break

            if not keys:
                return f"No documents found matching prefix '{prefix}' in the document store."

            result_lines = [f"Found {len(keys)} document(s) matching '{prefix}':"]
            for key in keys:
                result_lines.append(f"  - {key}")
            if len(keys) == MAX_RESULTS:
                result_lines.append("  (results limited to 20 — try a more specific prefix)")
            return "\n".join(result_lines)

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            logger.error("S3 search ClientError (%s): %s", error_code, e)
            return f"Failed to search documents: AWS error {error_code}."
        except BotoCoreError as e:
            logger.error("S3 search BotoCoreError: %s", e)
            return "Failed to search documents: could not reach the document store."

    def get_document_url(self, key: str) -> str:
        """Generate a temporary download URL for a specific document.

        Use this tool after finding a document with search_documents, or when you
        already know the exact document key/path.

        Args:
            key: The exact S3 object key (e.g., "manuals/kubota/SVL97-2_repair_guide.pdf")

        Returns:
            A presigned download URL valid for the configured expiry time, or an error message.
        """
        try:
            # Verify the object exists first
            self._s3.head_object(Bucket=self.bucket_name, Key=key)

            url = self._s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": key},
                ExpiresIn=self.presigned_url_expiry,
            )
            expiry_minutes = self.presigned_url_expiry // 60
            logger.info("Generated presigned URL for %s (expires in %dm)", key, expiry_minutes)
            return f"Download URL for '{key}' (valid for {expiry_minutes} minutes):\n{url}"

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ("404", "NoSuchKey"):
                logger.warning("S3 object not found: %s", key)
                return f"Document '{key}' was not found in the document store."
            logger.error("S3 get_document_url ClientError (%s): %s", error_code, e)
            return f"Failed to get document URL: AWS error {error_code}."
        except BotoCoreError as e:
            logger.error("S3 get_document_url BotoCoreError: %s", e)
            return "Failed to get document URL: could not reach the document store."

    def save_document(self, url: str, key: str) -> str:
        """Download a document from a URL and save it to the S3 document store.

        Use this tool after finding a document on the web that should be stored for
        future access. This avoids repeated web searches for the same document.

        Args:
            url: The source URL to download the document from
            key: The S3 key/path to store it under (e.g., "manuals/kubota/SVL97-2_repair_guide.pdf")

        Returns:
            A confirmation message with the stored key, or an error message.
        """
        try:
            response = httpx.get(url, timeout=30.0, follow_redirects=True)
            response.raise_for_status()
        except httpx.TimeoutException:
            logger.error("Timeout downloading document from %s", url)
            return f"Failed to save document: timed out downloading from the source URL."
        except httpx.HTTPError as e:
            logger.error("HTTP error downloading document from %s: %s", url, e)
            return f"Failed to save document: could not download from the source URL."

        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if not content_type:
            content_type, _ = mimetypes.guess_type(key)
            content_type = content_type or "application/octet-stream"

        try:
            self._s3.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=response.content,
                ContentType=content_type,
            )
            size_kb = len(response.content) / 1024
            logger.info("Saved document to S3: %s (%.1f KB)", key, size_kb)
            return f"Document saved to the document store as '{key}' ({size_kb:.1f} KB). It will be available for future searches."
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            logger.error("S3 save_document ClientError (%s): %s", error_code, e)
            return f"Failed to save document: AWS error {error_code}."
        except BotoCoreError as e:
            logger.error("S3 save_document BotoCoreError: %s", e)
            return "Failed to save document: could not reach the document store."
