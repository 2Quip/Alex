"""Tests for PMScheduleService in app/services/pm_schedule_service.py."""

import json
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.pm_schedule_service import PMScheduleService


SAMPLE_LLM_ARRAY = [
    {"Service Hours": "250", "Task": "Oil change", "Parts": "Oil filter 1R-0750"},
    {"Service Hours": "500", "Task": "Filter replace", "Parts": "Hydraulic filter"},
]

SAMPLE_LLM_WRAPPED = {"schedule": SAMPLE_LLM_ARRAY}


@pytest.fixture
def service():
    svc = PMScheduleService()
    svc._s3 = MagicMock()
    svc._initialized = True
    return svc


# --- _parse_llm_json ---


def test_parse_llm_json_array():
    result = PMScheduleService._parse_llm_json(json.dumps(SAMPLE_LLM_ARRAY))
    assert len(result) == 2
    assert result[0]["Service Hours"] == "250"


def test_parse_llm_json_wrapped_object():
    result = PMScheduleService._parse_llm_json(json.dumps(SAMPLE_LLM_WRAPPED))
    assert len(result) == 2
    assert result[1]["Task"] == "Filter replace"


def test_parse_llm_json_single_dict_as_one_row():
    """A flat dict with string values is treated as a single row."""
    result = PMScheduleService._parse_llm_json(json.dumps({"Task": "Oil change", "Hours": "250"}))
    assert len(result) == 1
    assert result[0]["Task"] == "Oil change"


def test_parse_llm_json_unexpected_format():
    with pytest.raises(ValueError, match="unexpected JSON format"):
        PMScheduleService._parse_llm_json(json.dumps(42))


# --- _extract_text ---


def test_extract_text_with_tables(service):
    """pdfplumber table extraction produces pipe-delimited rows."""
    mock_page = MagicMock()
    mock_page.extract_tables.return_value = [
        [["Hours", "Task"], ["250", "Oil change"], ["500", "Filter replace"]]
    ]

    with patch("app.services.pm_schedule_service.pdfplumber.open") as mock_open:
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_pdf

        result = service._extract_text(b"fake pdf bytes")

    assert "Hours | Task" in result
    assert "250 | Oil change" in result


def test_extract_text_fallback_to_plain_text(service):
    """When no tables found, falls back to page.extract_text()."""
    mock_page = MagicMock()
    mock_page.extract_tables.return_value = []
    mock_page.extract_text.return_value = "Maintenance at 250 hours: change oil."

    with patch("app.services.pm_schedule_service.pdfplumber.open") as mock_open:
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_pdf

        result = service._extract_text(b"fake pdf bytes")

    assert "Maintenance at 250 hours" in result


# --- _download_pdf ---


def test_download_pdf_success(service):
    body_mock = MagicMock()
    body_mock.read.return_value = b"pdf content"
    service._s3.get_object.return_value = {"Body": body_mock}

    result = service._download_pdf("manuals/test.pdf")
    assert result == b"pdf content"
    service._s3.get_object.assert_called_once()


def test_download_pdf_not_found(service):
    from botocore.exceptions import ClientError

    service._s3.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject"
    )
    with pytest.raises(FileNotFoundError, match="not found"):
        service._download_pdf("missing.pdf")


# --- extract_schedule (integration) ---


@pytest.mark.asyncio
async def test_extract_schedule_happy_path(service):
    """Full flow: download → extract text → LLM → parsed array."""
    body_mock = MagicMock()
    body_mock.read.return_value = b"fake pdf"
    service._s3.get_object.return_value = {"Body": body_mock}

    with patch.object(service, "_extract_text", return_value="Hours | Task\n250 | Oil change"):
        with patch.object(service, "_llm_extract", new_callable=AsyncMock, return_value=SAMPLE_LLM_ARRAY):
            result = await service.extract_schedule("manuals/test.pdf")

    assert len(result) == 2
    assert result[0]["Service Hours"] == "250"


@pytest.mark.asyncio
async def test_extract_schedule_empty_text(service):
    body_mock = MagicMock()
    body_mock.read.return_value = b"fake pdf"
    service._s3.get_object.return_value = {"Body": body_mock}

    with patch.object(service, "_extract_text", return_value="   "):
        with pytest.raises(ValueError, match="scanned image"):
            await service.extract_schedule("manuals/scan.pdf")


@pytest.mark.asyncio
async def test_extract_schedule_s3_not_found(service):
    from botocore.exceptions import ClientError

    service._s3.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject"
    )
    with pytest.raises(FileNotFoundError):
        await service.extract_schedule("missing.pdf")
