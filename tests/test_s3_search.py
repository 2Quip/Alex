"""Tests for S3SearchTool in app/tools/s3_search.py."""

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import BotoCoreError, ClientError

from app.tools.s3_search import S3SearchTool


@pytest.fixture
def tool():
    with patch("app.tools.s3_search.boto3.client") as mock_client_factory:
        mock_s3 = MagicMock()
        mock_client_factory.return_value = mock_s3
        t = S3SearchTool(bucket_name="test-bucket")
        t._s3 = mock_s3
        yield t


# --- search_documents tests ---

def test_search_documents_returns_results(tool):
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Contents": [{"Key": "manuals/kubota/SVL97-2.pdf"}, {"Key": "manuals/kubota/M7060.pdf"}]}
    ]
    tool._s3.get_paginator.return_value = paginator

    result = tool.search_documents(prefix="manuals/kubota/")

    assert "Found 2 document(s)" in result
    assert "manuals/kubota/SVL97-2.pdf" in result
    assert "manuals/kubota/M7060.pdf" in result
    paginator.paginate.assert_called_once_with(Bucket="test-bucket", Prefix="manuals/kubota/")


def test_search_documents_no_results(tool):
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": []}]
    tool._s3.get_paginator.return_value = paginator

    result = tool.search_documents(prefix="nonexistent/")

    assert "No documents found" in result


def test_search_documents_empty_page(tool):
    """Pages with no Contents key are handled gracefully."""
    paginator = MagicMock()
    paginator.paginate.return_value = [{}]
    tool._s3.get_paginator.return_value = paginator

    result = tool.search_documents(prefix="empty/")

    assert "No documents found" in result


def test_search_documents_limits_to_20(tool):
    keys = [{"Key": f"docs/file{i}.pdf"} for i in range(30)]
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": keys}]
    tool._s3.get_paginator.return_value = paginator

    result = tool.search_documents(prefix="docs/")

    assert "Found 20 document(s)" in result
    assert "results limited to 20" in result


def test_search_documents_client_error(tool):
    paginator = MagicMock()
    paginator.paginate.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}, "ListObjectsV2"
    )
    tool._s3.get_paginator.return_value = paginator

    result = tool.search_documents(prefix="secret/")

    assert "Failed to search documents" in result
    assert "AccessDenied" in result


def test_search_documents_botocore_error(tool):
    paginator = MagicMock()
    paginator.paginate.side_effect = BotoCoreError()
    tool._s3.get_paginator.return_value = paginator

    result = tool.search_documents(prefix="broken/")

    assert "could not reach the document store" in result


# --- get_document_url tests ---

def test_get_document_url_success(tool):
    tool._s3.head_object.return_value = {}
    tool._s3.generate_presigned_url.return_value = "https://s3.amazonaws.com/test-bucket/doc.pdf?signed"

    result = tool.get_document_url(key="doc.pdf")

    assert "https://s3.amazonaws.com/test-bucket/doc.pdf?signed" in result
    assert "valid for 60 minutes" in result
    tool._s3.head_object.assert_called_once_with(Bucket="test-bucket", Key="doc.pdf")
    tool._s3.generate_presigned_url.assert_called_once_with(
        "get_object",
        Params={"Bucket": "test-bucket", "Key": "doc.pdf"},
        ExpiresIn=3600,
    )


def test_get_document_url_not_found(tool):
    tool._s3.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
    )

    result = tool.get_document_url(key="missing.pdf")

    assert "not found" in result


def test_get_document_url_no_such_key(tool):
    tool._s3.head_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "HeadObject"
    )

    result = tool.get_document_url(key="gone.pdf")

    assert "not found" in result


def test_get_document_url_client_error(tool):
    tool._s3.head_object.side_effect = ClientError(
        {"Error": {"Code": "InternalError", "Message": "Something broke"}}, "HeadObject"
    )

    result = tool.get_document_url(key="broken.pdf")

    assert "Failed to get document URL" in result
    assert "InternalError" in result


def test_get_document_url_botocore_error(tool):
    tool._s3.head_object.side_effect = BotoCoreError()

    result = tool.get_document_url(key="unreachable.pdf")

    assert "could not reach the document store" in result


# --- init / registration tests ---

def test_tool_registers_both_methods():
    with patch("app.tools.s3_search.boto3.client"):
        tool = S3SearchTool(bucket_name="test-bucket")

    func_names = [f.name for f in tool.functions.values()]
    assert "search_documents" in func_names
    assert "get_document_url" in func_names


def test_init_with_explicit_credentials():
    with patch("app.tools.s3_search.boto3.client") as mock_client_factory:
        S3SearchTool(
            bucket_name="my-bucket",
            region="eu-west-1",
            access_key_id="AKID",
            secret_access_key="SECRET",
            presigned_url_expiry=7200,
        )

    mock_client_factory.assert_called_once_with(
        "s3",
        region_name="eu-west-1",
        aws_access_key_id="AKID",
        aws_secret_access_key="SECRET",
    )


def test_init_without_credentials_uses_defaults():
    with patch("app.tools.s3_search.boto3.client") as mock_client_factory:
        S3SearchTool(bucket_name="my-bucket", region="us-west-2")

    mock_client_factory.assert_called_once_with(
        "s3",
        region_name="us-west-2",
    )
