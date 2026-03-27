"""Preventive Maintenance schedule extraction from PDF documents.

Downloads a PDF from S3, extracts text/tables with pdfplumber, and uses
an LLM to return structured PM schedule rows as a JSON array.
"""

import io
import json
import logging

import boto3
import httpx
import pdfplumber
from botocore.exceptions import BotoCoreError, ClientError

from app.config.settings import settings

logger = logging.getLogger(__name__)

# Maximum characters of extracted text to send to the LLM
_MAX_TEXT_CHARS = 100_000

_EXTRACTION_PROMPT = """\
You are a data extraction specialist. The following text was extracted from a \
preventive maintenance schedule PDF for heavy equipment.

Extract ALL maintenance schedule items into a JSON object with a single key \
"schedule" containing an array of objects. Each object represents one \
maintenance task. Use these column names when possible: \
"Equipment", "Interval (Hours)", "Description", "Parts", "Frequency", \
"Tools / Procedures", "Estimated Labor Hours", "Notes". \
If the PDF uses different column names, include those as well. \
Preserve all original values as strings. If a field is empty, use an empty string.

Return valid JSON in this exact format:
{"schedule": [
  {"Equipment": "CAT 320", "Interval (Hours)": "250", "Description": "Oil change", "Parts": "Oil filter 1R-0750", "Frequency": "Every 250 hours", "Tools / Procedures": "", "Estimated Labor Hours": "", "Notes": ""},
  {"Equipment": "CAT 320", "Interval (Hours)": "500", "Description": "Filter replace", "Parts": "Hydraulic filter", "Frequency": "Every 500 hours", "Tools / Procedures": "", "Estimated Labor Hours": "", "Notes": ""}
]}

Text:
"""


class PMScheduleService:
    """Service for extracting PM schedules from PDF documents stored in S3."""

    def __init__(self):
        self._s3 = None
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return
        if not settings.S3_BUCKET_NAME:
            logger.warning("PM schedule service disabled: S3_BUCKET_NAME not set")
            return

        client_kwargs = {"region_name": settings.S3_REGION}
        if settings.S3_ACCESS_KEY_ID and settings.S3_SECRET_ACCESS_KEY:
            client_kwargs["aws_access_key_id"] = settings.S3_ACCESS_KEY_ID
            client_kwargs["aws_secret_access_key"] = settings.S3_SECRET_ACCESS_KEY

        self._s3 = boto3.client("s3", **client_kwargs)
        self._initialized = True
        logger.info("PM schedule service initialized")

    async def cleanup(self):
        self._s3 = None
        self._initialized = False
        logger.info("PM schedule service cleaned up")

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    async def extract_schedule(self, s3_key: str) -> list[dict[str, str]]:
        """Download a PDF from S3 and extract PM schedule rows.

        Returns:
            List of dicts where keys are PDF column names and values are strings.

        Raises:
            FileNotFoundError: S3 key does not exist.
            ValueError: PDF contains no extractable text.
            RuntimeError: LLM or S3 errors.
        """
        if not self._initialized or self._s3 is None:
            await self.initialize()
            if self._s3 is None:
                raise RuntimeError("PM schedule service is not configured (S3_BUCKET_NAME missing)")

        pdf_bytes = self._download_pdf(s3_key)
        text = self._extract_text(pdf_bytes)

        if not text.strip():
            raise ValueError(
                "Could not extract text from this PDF. It may be a scanned image."
            )

        return await self._llm_extract(text)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _download_pdf(self, s3_key: str) -> bytes:
        """Download PDF bytes from S3."""
        try:
            resp = self._s3.get_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key)
            return resp["Body"].read()
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("404", "NoSuchKey"):
                raise FileNotFoundError(f"Document not found in S3: {s3_key}")
            raise RuntimeError(f"S3 error ({code}) downloading {s3_key}") from e
        except BotoCoreError as e:
            raise RuntimeError(f"S3 connection error downloading {s3_key}") from e

    def _extract_text(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes using pdfplumber.

        Prefers table extraction; falls back to plain text per page.
        """
        parts: list[str] = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        # First row is typically the header
                        for row in table:
                            cells = [str(c).strip() if c else "" for c in row]
                            parts.append(" | ".join(cells))
                        parts.append("")  # blank line between tables
                else:
                    text = page.extract_text()
                    if text:
                        parts.append(text)

        full = "\n".join(parts)
        if len(full) > _MAX_TEXT_CHARS:
            logger.warning(
                "PDF text truncated from %d to %d chars", len(full), _MAX_TEXT_CHARS
            )
            full = full[:_MAX_TEXT_CHARS]
        return full

    async def _llm_extract(self, text: str) -> list[dict[str, str]]:
        """Send extracted text to the LLM and parse the JSON array response."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                json={
                    "model": "gpt-4.1-mini",
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "user", "content": _EXTRACTION_PROMPT + text},
                    ],
                },
            )
            resp.raise_for_status()

        content = resp.json()["choices"][0]["message"]["content"]
        logger.debug("LLM raw response: %s", content[:500])
        return self._parse_llm_json(content)

    @staticmethod
    def _parse_llm_json(raw: str) -> list[dict[str, str]]:
        """Parse LLM output into a list of dicts.

        Handles both a raw JSON array and an object wrapping one.
        """
        data = json.loads(raw)

        if isinstance(data, list):
            return data

        # Object wrapper — take the first array-valued field
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
            # Single row returned as a flat dict — wrap it
            if any(isinstance(v, str) for v in data.values()):
                return [data]

        raise ValueError("LLM returned unexpected JSON format — expected an array of objects")


# Singleton
pm_schedule_service = PMScheduleService()
