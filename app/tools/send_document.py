import logging
import time
from datetime import datetime, timezone

import httpx
from agno.tools.toolkit import Toolkit

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]  # seconds between retries


class SendDocumentTool(Toolkit):
    """Agno tool that sends document URLs to users via a webhook."""

    def __init__(self, webhook_url: str, webhook_secret: str | None = None):
        super().__init__(name="send_document")
        self.webhook_url = webhook_url
        self.webhook_secret = webhook_secret
        self.register(self.send_document)

    def _post(self, payload: dict, headers: dict) -> httpx.Response:
        """POST with retry logic for transient failures."""
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                response = httpx.post(
                    self.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=10.0,
                )
                # Don't retry client errors (4xx) — only server errors (5xx)
                if response.status_code < 500:
                    return response
                last_exception = httpx.HTTPStatusError(
                    f"Server error {response.status_code}",
                    request=response.request,
                    response=response,
                )
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e

            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF[attempt]
                logger.warning(
                    "Webhook attempt %d/%d failed, retrying in %ds",
                    attempt + 1, MAX_RETRIES, wait,
                )
                time.sleep(wait)

        # All retries exhausted — raise the last exception
        raise last_exception

    def send_document(self, title: str, url: str, recipient: str = "") -> str:
        """Send a document URL to the user via webhook.

        Use this tool when the user asks you to send, share, or deliver a document,
        PDF, manual, repair guide, or any file link.

        Args:
            title: Document title (e.g., "Kubota SVL97-2 Repair Guide")
            url: Full URL to the document
            recipient: Optional recipient identifier (email, phone, or user ID)

        Returns:
            A message confirming whether the document was sent successfully.
        """
        payload = {
            "title": title,
            "url": url,
            "recipient": recipient,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        headers = {}
        if self.webhook_secret:
            headers["Authorization"] = f"Bearer {self.webhook_secret}"

        try:
            response = self._post(payload, headers)
            response.raise_for_status()
            logger.info("Document sent via webhook: %s -> %s", title, url)
            return f"Document '{title}' has been sent successfully."
        except httpx.TimeoutException:
            logger.error("Webhook timeout sending document after %d attempts: %s", MAX_RETRIES, title)
            return f"Failed to send document '{title}': the request timed out after {MAX_RETRIES} attempts."
        except httpx.HTTPStatusError as e:
            logger.error("Webhook error %s sending document: %s", e.response.status_code, title)
            return f"Failed to send document '{title}': received status {e.response.status_code}."
        except httpx.HTTPError as e:
            logger.error("Webhook request failed for document %s after %d attempts: %s", title, MAX_RETRIES, e)
            return f"Failed to send document '{title}': could not reach the delivery service after {MAX_RETRIES} attempts."
