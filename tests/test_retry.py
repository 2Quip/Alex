"""Tests for app/core/retry.py."""

import pytest

from app.core.retry import _is_retryable, with_retry


@pytest.mark.asyncio
async def test_with_retry_succeeds_first_attempt():
    calls = []

    async def succeed():
        calls.append(1)
        return "ok"

    result = await with_retry(succeed)
    assert result == "ok"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_with_retry_succeeds_after_transient_failure():
    calls = []

    async def fail_then_succeed():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("transient")
        return "recovered"

    result = await with_retry(fail_then_succeed, backoff=[0, 0, 0])
    assert result == "recovered"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_with_retry_raises_after_all_retries_exhausted():
    calls = []

    async def always_fail():
        calls.append(1)
        raise RuntimeError("permanent")

    with pytest.raises(RuntimeError, match="permanent"):
        await with_retry(always_fail, max_retries=3, backoff=[0, 0, 0])
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_with_retry_forwards_args_and_kwargs():
    async def echo(a, b, key=None):
        return (a, b, key)

    result = await with_retry(echo, "x", "y", key="z")
    assert result == ("x", "y", "z")


@pytest.mark.asyncio
async def test_with_retry_raises_last_exception():
    """The exception from the final attempt is the one raised."""
    attempt = [0]

    async def different_errors():
        attempt[0] += 1
        raise ValueError(f"error-{attempt[0]}")

    with pytest.raises(ValueError, match="error-3"):
        await with_retry(different_errors, max_retries=3, backoff=[0, 0, 0])


@pytest.mark.asyncio
async def test_with_retry_respects_max_retries():
    calls = []

    async def fail():
        calls.append(1)
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        await with_retry(fail, max_retries=5, backoff=[0, 0, 0, 0, 0])
    assert len(calls) == 5


# --- _is_retryable ---


def test_is_retryable_plain_exception():
    assert _is_retryable(RuntimeError("boom")) is True


def test_is_retryable_server_error():
    exc = Exception("server error")
    exc.status_code = 500
    assert _is_retryable(exc) is True


def test_is_retryable_rate_limit():
    exc = Exception("rate limited")
    exc.status_code = 429
    assert _is_retryable(exc) is True


def test_is_not_retryable_client_error():
    exc = Exception("payment required")
    exc.status_code = 402
    assert _is_retryable(exc) is False


def test_is_not_retryable_auth_error():
    exc = Exception("unauthorized")
    exc.status_code = 401
    assert _is_retryable(exc) is False


@pytest.mark.asyncio
async def test_with_retry_skips_non_retryable():
    """4xx errors (except 429) should fail immediately without retrying."""
    calls = []

    async def client_error():
        calls.append(1)
        exc = Exception("payment required")
        exc.status_code = 402
        raise exc

    with pytest.raises(Exception, match="payment required"):
        await with_retry(client_error, max_retries=3, backoff=[0, 0, 0])
    assert len(calls) == 1  # no retries


@pytest.mark.asyncio
async def test_with_retry_retries_429():
    """429 rate-limit errors should be retried."""
    calls = []

    async def rate_limited_then_ok():
        calls.append(1)
        if len(calls) < 2:
            exc = Exception("rate limited")
            exc.status_code = 429
            raise exc
        return "ok"

    result = await with_retry(rate_limited_then_ok, backoff=[0, 0, 0])
    assert result == "ok"
    assert len(calls) == 2
