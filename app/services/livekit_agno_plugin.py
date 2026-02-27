# Copyright 2025
# Licensed under the Apache License, Version 2.0

"""Agno plugin for LiveKit Agents - wraps Agno Agents as LiveKit LLMs."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

from agno.agent import Agent
from agno.run.agent import RunContentEvent, RunOutput

from app.core.retry import MAX_RETRIES, RETRY_BACKOFF, _is_retryable

from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.agents import llm
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

__version__ = "0.1.0"

__all__ = ["__version__", "LLMAdapter", "AgnoStream"]


class LLMAdapter(llm.LLM):
    """Wraps an Agno Agent as a LiveKit-compatible LLM."""

    def __init__(
        self,
        agent: Agent,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        super().__init__()
        self._agent = agent
        self._session_id = session_id
        self._user_id = user_id

    @property
    def model(self) -> str:
        return self._agent.model.id if self._agent.model else "agno"

    @property
    def provider(self) -> str:
        return "agno"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        # these are unused, since tool execution takes place in agno
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> "AgnoStream":
        return AgnoStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            agent=self._agent,
            session_id=self._session_id,
            user_id=self._user_id,
        )


class AgnoStream(llm.LLMStream):
    """Streams responses from an Agno Agent."""

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
        agent: Agent,
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        super().__init__(
            llm_adapter, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options
        )
        self._agent = agent
        self._session_id = session_id
        self._user_id = user_id

    async def _run(self) -> None:
        # Convert chat context to the last user message for Agno
        user_input = self._get_user_input()
        if not user_input:
            logger.debug("No user input found in chat context")
            return

        logger.debug("AgnoStream running with input: %s", user_input[:200])

        # Retry loop for transient LLM failures during voice streaming.
        # Only retry if no content has been sent yet.
        content_sent = False
        for attempt in range(MAX_RETRIES):
            try:
                response_stream = self._agent.arun(
                    input=user_input,
                    stream=True,
                    session_id=self._session_id,
                    user_id=self._user_id,
                )

                # Sentence buffer: accumulate chunks and release on sentence
                # boundaries so _sanitize_for_tts can match multi-token
                # patterns (reasoning sentences, role tokens, tool routing).
                buffer = ""

                async for event in response_stream:
                    raw = _extract_content(event)
                    if not raw:
                        continue

                    buffer += raw

                    # Flush complete sentences from the buffer
                    while True:
                        idx = _sentence_boundary(buffer)
                        if idx == -1:
                            break
                        sentence = buffer[:idx]
                        buffer = buffer[idx:]
                        cleaned = _sanitize_for_tts(sentence)
                        if cleaned and cleaned.strip():
                            content_sent = True
                            self._event_ch.send_nowait(
                                llm.ChatChunk(
                                    id="agno",
                                    delta=llm.ChoiceDelta(
                                        role="assistant", content=cleaned
                                    ),
                                )
                            )

                # Flush remaining buffer
                if buffer.strip():
                    cleaned = _sanitize_for_tts(buffer)
                    if cleaned and cleaned.strip():
                        content_sent = True
                        self._event_ch.send_nowait(
                            llm.ChatChunk(
                                id="agno",
                                delta=llm.ChoiceDelta(
                                    role="assistant", content=cleaned
                                ),
                            )
                        )

                # Stream completed successfully
                return

            except Exception as e:
                if content_sent or not _is_retryable(e) or attempt >= MAX_RETRIES - 1:
                    raise
                wait = RETRY_BACKOFF[attempt] if attempt < len(RETRY_BACKOFF) else RETRY_BACKOFF[-1]
                logger.warning(
                    "Voice stream attempt %d/%d failed (%s: %s), retrying in %ds",
                    attempt + 1, MAX_RETRIES, type(e).__name__, str(e)[:200], wait,
                )
                await asyncio.sleep(wait)

    def _get_user_input(self) -> str | None:
        """Extract the last actionable message from chat context.

        Priority order:
        1. Last user message (normal conversation)
        2. Last developer message (generate_reply instructions)
        3. Last non-assistant, non-system message (catch-all)
        """
        items = self._chat_ctx.items

        # Log what's in the context for debugging
        if items:
            roles = [
                f"{type(m).__name__}(role={m.role})"
                if isinstance(m, ChatMessage) else type(m).__name__
                for m in items
            ]
            logger.debug("Chat context items: %s", roles)

        # 1. Normal case: last user utterance
        for msg in reversed(items):
            if isinstance(msg, ChatMessage) and msg.role == "user":
                return msg.text_content

        # 2. generate_reply(instructions=...) — may use developer role
        for msg in reversed(items):
            if isinstance(msg, ChatMessage) and msg.role == "developer":
                return msg.text_content

        # 3. Fallback: any non-system, non-assistant message with content
        for msg in reversed(items):
            if isinstance(msg, ChatMessage) and msg.role not in ("assistant", "system"):
                content = msg.text_content
                if content:
                    logger.debug("Fallback input from role=%s: %s", msg.role, content[:100])
                    return content

        return None


# =============================================================================
# Sentence boundary detection
# =============================================================================

def _sentence_boundary(text: str) -> int:
    """Return index of the first sentence boundary, or -1 if none found.

    A sentence boundary is after '.', '!', '?', or a newline, provided there
    is at least one character of content before it.
    """
    for i, ch in enumerate(text):
        if ch in ".!?\n" and i > 0:
            # Return position right after the boundary character
            return i + 1
    return -1


# =============================================================================
# Content extraction
# =============================================================================

def _extract_content(event: Any) -> str | None:
    """Pull raw text content from an Agno event, or None."""
    if isinstance(event, RunContentEvent):
        return event.content
    if isinstance(event, RunOutput) and event.content:
        return str(event.content) if not isinstance(event.content, str) else event.content
    return None


# =============================================================================
# TTS sanitization
# =============================================================================

# Repeated role tokens the model sometimes emits (e.g. "assistantassistantassistant")
_ROLE_TOKEN_RE = re.compile(r"(?:assistant|user|system){2,}", re.IGNORECASE)
# Single standalone role token that is the entire chunk
_LONE_ROLE_RE = re.compile(r"^(?:assistant|user|system)$", re.IGNORECASE)

# Model reasoning / internal tags (opening, closing, or self-closing)
_REASONING_TAG_RE = re.compile(
    r"</?(?:analysis|assistantcommentary|assistantfinal|thinking|thought|reasoning|internal|scratchpad)[^>]*>",
    re.IGNORECASE,
)

# Content between reasoning tags (multiline)
_REASONING_BLOCK_RE = re.compile(
    r"<(?:analysis|assistantcommentary|assistantfinal|thinking|thought|reasoning|internal|scratchpad)>"
    r"[\s\S]*?"
    r"</(?:analysis|assistantcommentary|assistantfinal|thinking|thought|reasoning|internal|scratchpad)>",
    re.IGNORECASE,
)

# Tool-call routing fragments (e.g. "to=functions.run_sql_query", "functions.run_sql_query to=assistant")
_TOOL_ROUTING_RE = re.compile(
    r"(?:to=functions\.\S+|functions\.\S+\s*to=\S*)", re.IGNORECASE
)

# "json" immediately before a brace (e.g. 'json{"query":...')
_JSON_PREFIX_RE = re.compile(r"json\s*\{[^}]{0,500}\}", re.IGNORECASE)

# Raw JSON blobs
_JSON_BLOB_RE = re.compile(r"\{[^}]{0,500}\}")

# Internal reasoning sentences — lines that are clearly the model thinking,
# not talking to the user. Matched case-insensitively.
_REASONING_SENTENCE_PATTERNS = [
    r"(?:^|\. )We (?:need|still|haven't|should|must|could) ",
    r"(?:^|\. )Let'?s (?:try|describe|check|see|look|query|get|use|handle)",
    r"(?:^|\. )Probably ",
    r"(?:^|\. )(?:It )?might (?:have|be|return|need)",
    r"(?:^|\. )Error running ",
    r"(?:^|\. )(?:The )?(?:stream|query|function|tool|result|schema|table|column)s? (?:error|return|fail|might|maybe)",
    r"(?:^|\. )Not shown\b",
    r"(?:^|\. )Not captured\b",
    r"(?:^|\. )Need to handle ",
]
_REASONING_SENTENCE_RE = re.compile(
    "|".join(_REASONING_SENTENCE_PATTERNS), re.IGNORECASE
)

# SQL query fragments that leak
_SQL_LEAK_RE = re.compile(
    r"SELECT\s+\w+.*?FROM\s+\w+", re.IGNORECASE
)


def _sanitize_for_tts(text: str) -> str:
    """Strip markdown, reasoning tokens, tool metadata, and special characters
    so TTS reads natural speech only."""

    # --- Phase 1: Strip model internals ---

    # Remove repeated role tokens (assistantassistantassistant...)
    text = _ROLE_TOKEN_RE.sub("", text)
    # Remove lone role tokens
    if _LONE_ROLE_RE.match(text.strip()):
        return ""

    # Remove reasoning blocks (content between tags)
    text = _REASONING_BLOCK_RE.sub("", text)
    # Remove remaining reasoning tags
    text = _REASONING_TAG_RE.sub("", text)

    # Remove tool-call routing
    text = _TOOL_ROUTING_RE.sub("", text)

    # Remove "json{...}" patterns
    text = _JSON_PREFIX_RE.sub("", text)
    # Remove raw JSON blobs
    text = _JSON_BLOB_RE.sub("", text)

    # Remove SQL query fragments
    text = _SQL_LEAK_RE.sub("", text)

    # Remove reasoning sentences (model thinking aloud)
    text = _REASONING_SENTENCE_RE.sub("", text)

    # --- Phase 2: Strip markdown formatting ---

    # Remove markdown headers (### Header)
    text = re.sub(r"#{1,6}\s*", "", text)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
    # Remove inline code backticks
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Remove code fence markers
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Convert markdown links [text](url) to just the text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove bullet point markers
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    # Remove numbered list markers
    text = re.sub(r"(?m)^\s*\d+\.\s+", "", text)
    # Remove horizontal rules
    text = re.sub(r"(?m)^[-*_]{3,}\s*$", "", text)

    # --- Phase 3: Whitespace cleanup ---

    text = re.sub(r"\n{2,}", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()
