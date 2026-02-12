"""Async retry utility for transient failures in external API calls."""

import asyncio
import logging

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]  # seconds between retries


def _is_retryable(exc: Exception) -> bool:
    """Return False for client errors that will never succeed on retry.

    Checks for a ``status_code`` attribute (used by the OpenAI SDK,
    httpx, and most HTTP client libraries).  4xx errors are not retried
    except for 429 (rate-limit), which is transient.
    """
    status = getattr(exc, "status_code", None)
    if status is not None and 400 <= status < 500 and status != 429:
        return False
    return True


async def with_retry(
    coro_func,
    *args,
    max_retries=MAX_RETRIES,
    backoff=RETRY_BACKOFF,
    **kwargs,
):
    """Call an async function with retry logic for transient failures.

    Args:
        coro_func: An async callable to invoke.
        *args: Positional arguments forwarded to *coro_func*.
        max_retries: Total number of attempts (default 3).
        backoff: List of sleep durations between retries.
        **kwargs: Keyword arguments forwarded to *coro_func*.

    Returns:
        The return value of a successful call.

    Raises:
        The exception from the final failed attempt, or immediately
        for non-retryable errors (4xx except 429).
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            return await coro_func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if not _is_retryable(e):
                logger.warning(
                    "Non-retryable error (%s: %s), not retrying",
                    type(e).__name__,
                    str(e)[:200],
                )
                raise
            if attempt < max_retries - 1:
                wait = backoff[attempt] if attempt < len(backoff) else backoff[-1]
                logger.warning(
                    "Attempt %d/%d failed (%s: %s), retrying in %ds",
                    attempt + 1,
                    max_retries,
                    type(e).__name__,
                    str(e)[:200],
                    wait,
                )
                await asyncio.sleep(wait)
    raise last_exception
