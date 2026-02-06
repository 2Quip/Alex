# Document Webhook Integration

The 2Quip Agent can send document URLs (PDFs, manuals, guides) to users during chat or voice calls via a webhook POST to your backend.

## How It Works

1. User asks the agent for a document (e.g., "Send me the Kubota SVL97-2 repair guide")
2. Agent searches for the document URL (via web search or database)
3. Agent calls the `send_document` tool, which POSTs the document info to your webhook
4. Your backend receives the payload and delivers the document to the user (push notification, in-app message, SMS, etc.)
5. Agent confirms to the user that the document was sent

## Configuration

Add these to your `.env` file:

```env
# Required - your backend endpoint that receives document payloads
DOCUMENT_WEBHOOK_URL=https://your-backend.com/api/document-webhook

# Optional - bearer token for authenticating webhook requests
DOCUMENT_WEBHOOK_SECRET=your-secret-key-here
```

If `DOCUMENT_WEBHOOK_URL` is not set, the send document tool is not loaded and the agent will not attempt to send documents.

## Webhook Payload

Your endpoint will receive a `POST` request with the following JSON body:

```json
{
  "title": "Kubota SVL97-2 Repair Guide",
  "url": "https://example.com/kubota-svl97-2-guide.pdf",
  "recipient": "tech-user-123",
  "timestamp": "2026-02-06T16:30:00+00:00"
}
```

### Fields

| Field       | Type   | Description                                                                 |
|-------------|--------|-----------------------------------------------------------------------------|
| `title`     | string | Document title as described by the agent                                    |
| `url`       | string | Full URL to the document (PDF, web page, etc.)                              |
| `recipient` | string | Optional recipient identifier extracted from conversation (may be empty)    |
| `timestamp` | string | ISO 8601 UTC timestamp of when the request was made                         |

## Authentication

When `DOCUMENT_WEBHOOK_SECRET` is configured, every request includes a bearer token header:

```
Authorization: Bearer your-secret-key-here
```

Your endpoint should validate this header and reject requests without a valid token.

### Example validation (Python/FastAPI)

```python
from fastapi import Header, HTTPException

@app.post("/api/document-webhook")
async def receive_document(
    payload: dict,
    authorization: str = Header(...),
):
    expected = "Bearer your-secret-key-here"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    title = payload["title"]
    url = payload["url"]
    recipient = payload["recipient"]

    # Deliver the document to the user
    # e.g., send push notification, in-app message, SMS, etc.

    return {"status": "ok"}
```

### Example validation (Node.js/Express)

```javascript
app.post('/api/document-webhook', (req, res) => {
  const token = req.headers.authorization;
  if (token !== 'Bearer your-secret-key-here') {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const { title, url, recipient, timestamp } = req.body;

  // Deliver the document to the user

  res.json({ status: 'ok' });
});
```

## Error Handling

The agent handles failures gracefully and tells the user what happened:

| Scenario             | Agent response to user                                      |
|----------------------|-------------------------------------------------------------|
| Success (2xx)        | "Document 'Kubota SVL97-2 Repair Guide' has been sent successfully." |
| Timeout (10s)        | "Failed to send document: the request timed out."           |
| HTTP error (4xx/5xx) | "Failed to send document: received status 500."             |
| Connection refused   | "Failed to send document: could not reach the delivery service." |

## Testing

### Local testing with webhook.site

1. Go to https://webhook.site and copy your unique URL
2. Set it in `.env`:
   ```env
   DOCUMENT_WEBHOOK_URL=https://webhook.site/your-unique-id
   ```
3. Start the server: `uv run uvicorn app.main:app --reload`
4. Send a test request:
   ```bash
   curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message": "Send me the repair guide for the Kubota SVL97-2", "user_id": "tech1"}'
   ```
5. Check webhook.site to see the incoming payload

### Unit tests

```bash
uv run pytest tests/test_send_document.py -v
```

Covers: success, payload format, timeout, HTTP errors, connection errors, auth header with/without secret, tool registration, and default recipient.

## Recipient Field

The `recipient` field is populated from what the user says in conversation. For example:

- "Send it to John" -> `recipient: "John"`
- "Send me the guide" -> `recipient: ""`  (empty, your backend should use the session context)

Your backend may need to map these natural language identifiers to actual user IDs, emails, or phone numbers based on your application logic.

## Which Agents Support This

The `SendDocumentTool` is loaded on all three agents when `DOCUMENT_WEBHOOK_URL` is set:

- Chat API (`POST /chat` and `POST /chat/stream`)
- Diagnostics API (`POST /diagnostics`)
- Voice Agent (LiveKit)
