"""Tests for SendDocumentTool in app/tools/send_document.py."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.tools.send_document import SendDocumentTool


@pytest.fixture
def tool():
    return SendDocumentTool(webhook_url="https://example.com/webhook")


def test_send_document_success(tool):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    with patch("app.tools.send_document.httpx.post", return_value=mock_response) as mock_post:
        result = tool.send_document(
            title="Kubota SVL97-2 Guide",
            url="https://example.com/guide.pdf",
            recipient="user1",
        )

    assert "sent successfully" in result
    mock_post.assert_called_once()
    payload = mock_post.call_args.kwargs["json"]
    assert payload["title"] == "Kubota SVL97-2 Guide"
    assert payload["url"] == "https://example.com/guide.pdf"
    assert payload["recipient"] == "user1"
    assert "timestamp" in payload


def test_send_document_payload_format(tool):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    with patch("app.tools.send_document.httpx.post", return_value=mock_response) as mock_post:
        tool.send_document(title="Test Doc", url="https://example.com/doc.pdf")

    call_kwargs = mock_post.call_args
    assert call_kwargs.args[0] == "https://example.com/webhook"
    assert call_kwargs.kwargs["timeout"] == 10.0
    payload = call_kwargs.kwargs["json"]
    assert set(payload.keys()) == {"title", "url", "recipient", "timestamp"}


def test_send_document_timeout(tool):
    with patch("app.tools.send_document.time.sleep"), patch(
        "app.tools.send_document.httpx.post",
        side_effect=httpx.TimeoutException("timeout"),
    ) as mock_post:
        result = tool.send_document(title="Guide", url="https://example.com/g.pdf")

    assert "timed out" in result
    assert mock_post.call_count == 3  # retried 3 times


def test_send_document_http_error(tool):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.request = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=MagicMock(), response=mock_response
    )

    with patch("app.tools.send_document.time.sleep"), patch(
        "app.tools.send_document.httpx.post", return_value=mock_response
    ) as mock_post:
        result = tool.send_document(title="Guide", url="https://example.com/g.pdf")

    assert "status 500" in result
    assert mock_post.call_count == 3  # retried 3 times on 5xx


def test_send_document_connection_error(tool):
    with patch("app.tools.send_document.time.sleep"), patch(
        "app.tools.send_document.httpx.post",
        side_effect=httpx.ConnectError("refused"),
    ) as mock_post:
        result = tool.send_document(title="Guide", url="https://example.com/g.pdf")

    assert "could not reach" in result
    assert mock_post.call_count == 3  # retried 3 times


def test_retry_succeeds_on_second_attempt(tool):
    """Server error on first attempt, success on second."""
    fail_response = MagicMock()
    fail_response.status_code = 502
    fail_response.request = MagicMock()
    ok_response = MagicMock()
    ok_response.status_code = 200
    ok_response.raise_for_status = MagicMock()

    with patch("app.tools.send_document.time.sleep"), patch(
        "app.tools.send_document.httpx.post",
        side_effect=[fail_response, ok_response],
    ) as mock_post:
        result = tool.send_document(title="Guide", url="https://example.com/g.pdf")

    assert "sent successfully" in result
    assert mock_post.call_count == 2


def test_no_retry_on_client_error(tool):
    """4xx errors should not be retried."""
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "bad request", request=MagicMock(), response=mock_response
    )

    with patch("app.tools.send_document.time.sleep") as mock_sleep, patch(
        "app.tools.send_document.httpx.post", return_value=mock_response
    ) as mock_post:
        result = tool.send_document(title="Guide", url="https://example.com/g.pdf")

    assert "status 400" in result
    assert mock_post.call_count == 1  # no retry
    mock_sleep.assert_not_called()


def test_tool_registers_send_document():
    tool = SendDocumentTool(webhook_url="https://example.com/webhook")
    func_names = [f.name for f in tool.functions.values()]
    assert "send_document" in func_names


def test_auth_header_sent_when_secret_provided():
    tool = SendDocumentTool(webhook_url="https://example.com/webhook", webhook_secret="my-secret-key")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    with patch("app.tools.send_document.httpx.post", return_value=mock_response) as mock_post:
        tool.send_document(title="Doc", url="https://example.com/doc.pdf")

    headers = mock_post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer my-secret-key"


def test_no_auth_header_when_no_secret(tool):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    with patch("app.tools.send_document.httpx.post", return_value=mock_response) as mock_post:
        tool.send_document(title="Doc", url="https://example.com/doc.pdf")

    headers = mock_post.call_args.kwargs["headers"]
    assert "Authorization" not in headers


def test_default_recipient_is_empty(tool):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    with patch("app.tools.send_document.httpx.post", return_value=mock_response) as mock_post:
        tool.send_document(title="Doc", url="https://example.com/doc.pdf")

    payload = mock_post.call_args.kwargs["json"]
    assert payload["recipient"] == ""
