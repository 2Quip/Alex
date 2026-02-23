# LiveKit Token Generation

The 2Quip Agent API provides a token endpoint that your frontend uses to connect users to the LiveKit voice agent. The frontend requests a token, then uses it to join a LiveKit room where Alex (the voice AI) is listening.

## How It Works

1. User clicks "Start Voice Call" in the frontend
2. Frontend sends a `POST /livekit/token` request with the user's identity and a room name
3. API generates a signed JWT with room access grants
4. Frontend receives the token and LiveKit server URL
5. Frontend connects to the LiveKit room using the LiveKit client SDK
6. Alex (voice agent) automatically joins the room and begins listening

## Configuration

The API requires these environment variables to be set for token generation to work:

```env
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your-livekit-api-key
LIVEKIT_API_SECRET=your-livekit-api-secret
```

If any of these are missing, the endpoint returns `503 Service Unavailable`.

## Endpoint

```
POST /livekit/token
Content-Type: application/json
```

### Request Body

```json
{
  "identity": "user-123",
  "room": "support-room-456",
  "name": "John Smith"
}
```

| Field      | Type   | Required | Description                                                              |
|------------|--------|----------|--------------------------------------------------------------------------|
| `identity` | string | Yes      | Unique identifier for the participant (user ID, email, etc.)             |
| `room`     | string | Yes      | Room name to join. Use a unique name per session (e.g., UUID or user-specific) |
| `name`     | string | No       | Display name for the participant (shown in LiveKit dashboard/logs)        |

### Response (200 OK)

```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "url": "wss://your-project.livekit.cloud"
}
```

| Field   | Type   | Description                                      |
|---------|--------|--------------------------------------------------|
| `token` | string | Signed JWT for the participant to join the room   |
| `url`   | string | LiveKit server WebSocket URL to connect to        |

### Error Responses

| Status | Condition                         | Body                                          |
|--------|-----------------------------------|-----------------------------------------------|
| 422    | Missing `identity` or `room`      | Validation error details                      |
| 503    | LiveKit not configured on server  | `{"detail": "LiveKit is not configured"}`     |

## Token Grants

The generated token includes these permissions:

- `room_join: true` — allows the participant to join the specified room
- Room is scoped to the exact room name provided in the request
- Token is signed with the LiveKit API secret

The token does not grant admin, publish, or room creation permissions beyond joining.

## Frontend Integration (Next.js)

### Installation

```bash
npm install livekit-client @livekit/components-react
```

### Requesting a Token

```ts
// lib/livekit.ts

export async function getLivekitToken(identity: string, room: string, name?: string) {
  const response = await fetch('/api/livekit/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ identity, room, name }),
  })

  if (!response.ok) {
    throw new Error(`Token request failed: ${response.status}`)
  }

  return response.json() as Promise<{ token: string; url: string }>
}
```

### Connecting to a Room

```tsx
'use client'

import { useState } from 'react'
import { LiveKitRoom, AudioConference } from '@livekit/components-react'
import { getLivekitToken } from '@/lib/livekit'

export function VoiceCall({ userId, userName }: { userId: string; userName?: string }) {
  const [connection, setConnection] = useState<{ token: string; url: string } | null>(null)

  async function startCall() {
    const roomName = `support-${userId}-${Date.now()}`
    const result = await getLivekitToken(userId, roomName, userName)
    setConnection(result)
  }

  function endCall() {
    setConnection(null)
  }

  if (!connection) {
    return <button onClick={startCall}>Start Voice Call</button>
  }

  return (
    <LiveKitRoom
      token={connection.token}
      serverUrl={connection.url}
      connect={true}
      onDisconnected={endCall}
    >
      <AudioConference />
      <button onClick={endCall}>End Call</button>
    </LiveKitRoom>
  )
}
```

### Minimal Audio-Only Example

If you don't need the full `AudioConference` UI and want to build a custom interface:

```tsx
'use client'

import { useEffect, useState } from 'react'
import { Room, RoomEvent } from 'livekit-client'
import { getLivekitToken } from '@/lib/livekit'

export function VoiceAgent({ userId }: { userId: string }) {
  const [room, setRoom] = useState<Room | null>(null)
  const [connected, setConnected] = useState(false)

  async function connect() {
    const { token, url } = await getLivekitToken(userId, `support-${userId}-${Date.now()}`)
    const newRoom = new Room()

    newRoom.on(RoomEvent.Connected, () => setConnected(true))
    newRoom.on(RoomEvent.Disconnected, () => {
      setConnected(false)
      setRoom(null)
    })

    await newRoom.connect(url, token)
    await newRoom.localParticipant.setMicrophoneEnabled(true)
    setRoom(newRoom)
  }

  function disconnect() {
    room?.disconnect()
  }

  return (
    <div>
      {connected ? (
        <button onClick={disconnect}>End Call</button>
      ) : (
        <button onClick={connect}>Call Alex</button>
      )}
    </div>
  )
}
```

## Room Naming

Use a consistent naming convention for rooms so the voice agent can associate sessions:

| Pattern                          | Example                          | Use Case                      |
|----------------------------------|----------------------------------|-------------------------------|
| `support-{userId}-{timestamp}`   | `support-user123-1708700000`     | One-off support calls         |
| `wo-{workOrderId}`               | `wo-WO-2024-0456`               | Work order specific calls     |
| `diag-{listingId}-{timestamp}`   | `diag-EQ-789-1708700000`        | Diagnostic sessions           |

The room name is passed to the voice agent as `ctx.room.name` and used for session tracking.

## Voice Agent Capabilities

Once connected, Alex (the voice agent) can:

- Answer questions about equipment, troubleshooting, and work orders
- Search the web for OEM documentation and part numbers
- Query the database for work order history and equipment metrics
- Search the company document store (S3) for manuals and guides
- Send documents to the user via webhook
- Save web-found documents to the document store for future access

The voice agent uses plain conversational speech (no markdown or tables) since responses are spoken aloud via text-to-speech.

## Testing

### Quick test with curl

```bash
curl -X POST http://localhost:8000/livekit/token \
  -H "Content-Type: application/json" \
  -d '{"identity": "test-user", "room": "test-room", "name": "Test User"}'
```

Expected response:

```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "url": "wss://your-project.livekit.cloud"
}
```

### Unit tests

```bash
uv run pytest tests/test_endpoints.py -k livekit -v
```

Covers: successful token generation, missing identity validation, LiveKit not configured (503), and optional name field.
