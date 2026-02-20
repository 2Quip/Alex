import logging
from typing import Optional

import boto3
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
