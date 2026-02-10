"""Lightweight health endpoint for the LiveKit voice agent."""

import json
import logging
import threading
import time
from dataclasses import dataclass, field

from aiohttp import web

logger = logging.getLogger(__name__)


@dataclass
class VoiceAgentHealth:
    """Thread-safe health state tracker for the voice agent."""

    _status: str = "starting"
    _error: str | None = None
    _start_time: float = field(default_factory=time.monotonic)
    _active_sessions: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def mark_running(self) -> None:
        with self._lock:
            self._status = "running"
            self._error = None

    def mark_error(self, error: str) -> None:
        with self._lock:
            self._status = "error"
            self._error = error

    def session_started(self) -> None:
        with self._lock:
            self._active_sessions += 1

    def session_ended(self) -> None:
        with self._lock:
            if self._active_sessions > 0:
                self._active_sessions -= 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self._status,
                "uptime_seconds": round(time.monotonic() - self._start_time, 1),
                "active_sessions": self._active_sessions,
                "error": self._error,
            }


health = VoiceAgentHealth()


async def _health_handler(request: web.Request) -> web.Response:
    snap = health.snapshot()
    status_code = 200 if snap["status"] == "running" else 503
    return web.Response(
        text=json.dumps(snap),
        content_type="application/json",
        status=status_code,
    )


async def start_health_server(port: int = 8092) -> web.AppRunner:
    """Start an aiohttp server exposing GET /health."""
    app = web.Application()
    app.router.add_get("/health", _health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info("Voice health server listening on port %d", port)
    return runner
