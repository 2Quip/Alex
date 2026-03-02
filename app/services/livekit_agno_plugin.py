# Copyright 2025
# Licensed under the Apache License, Version 2.0

"""Agno plugin for LiveKit Agents - wraps Agno Agents as LiveKit LLMs."""

from __future__ import annotations

import asyncio
import json as _json
import logging
import re
from collections.abc import Callable
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
        send_link: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self._agent = agent
        self._session_id = session_id
        self._user_id = user_id
        self._send_link = send_link

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
            send_link=self._send_link,
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
        send_link: Callable[[str], None] | None = None,
    ):
        super().__init__(
            llm_adapter, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options
        )
        self._agent = agent
        self._session_id = session_id
        self._user_id = user_id
        self._send_link = send_link

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
                sent_urls: set[str] = set()

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

                        # Extract and send URLs before TTS strips them
                        if self._send_link:
                            for url in _extract_urls(sentence):
                                if url not in sent_urls:
                                    sent_urls.add(url)
                                    self._send_link(url)

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
                    if self._send_link:
                        for url in _extract_urls(buffer):
                            if url not in sent_urls:
                                sent_urls.add(url)
                                self._send_link(url)

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
    is at least one character of content before it.  '.' and '?' are only
    treated as boundaries when followed by whitespace or end-of-text so they
    don't split inside URLs like 's3.amazonaws.com/key?AWSAccessKeyId=...'.
    """
    for i, ch in enumerate(text):
        if i == 0:
            continue
        if ch in "!\n":
            return i + 1
        if ch in ".?":
            next_ch = text[i + 1] if i + 1 < len(text) else " "
            if next_ch in " \t\n\r":
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

# ---------------------------------------------------------------------------
# Reasoning-prefix tokens (WITHOUT angle brackets)
# Models like gpt-oss-120b emit these as bare prefixes:
#   "analysisWe have trouble..."  "assistantcommentaryWe got an error..."
# ---------------------------------------------------------------------------
_REASONING_PREFIXES = (
    "analysis", "assistantcommentary", "assistantfinal", "assistantanalysis",
    "thinking", "thought", "reasoning", "internal", "scratchpad",
)

# Bare prefix at the start of text: "analysisWe have..." → strip the prefix
_BARE_PREFIX_RE = re.compile(
    r"^(?:" + "|".join(_REASONING_PREFIXES) + r")",
    re.IGNORECASE,
)

# Repeated role tokens the model sometimes emits (e.g. "assistantassistantassistant")
_ROLE_TOKEN_RE = re.compile(r"(?:assistant|user|system){2,}", re.IGNORECASE)
# Single standalone role token that is the entire chunk
_LONE_ROLE_RE = re.compile(r"^(?:assistant|user|system)$", re.IGNORECASE)

# Model reasoning / internal tags WITH angle brackets (opening, closing, or self-closing)
_REASONING_TAG_RE = re.compile(
    r"</?(?:" + "|".join(_REASONING_PREFIXES) + r")[^>]*>",
    re.IGNORECASE,
)

# Content between reasoning tags (multiline)
_REASONING_BLOCK_RE = re.compile(
    r"<(?:" + "|".join(_REASONING_PREFIXES) + r")>"
    r"[\s\S]*?"
    r"</(?:" + "|".join(_REASONING_PREFIXES) + r")>",
    re.IGNORECASE,
)

# Tool-call routing fragments (e.g. "to=functions.run_sql_query", "functions.run_sql_query to=assistant")
_TOOL_ROUTING_RE = re.compile(
    r"(?:to=functions\.\S+|functions\.\S+\s*to=\S*)", re.IGNORECASE
)

# "json" immediately before a brace (e.g. 'json{"query":...')
_JSON_PREFIX_RE = re.compile(r"json\s*\{[^}]{0,500}\}", re.IGNORECASE)

# Raw JSON blobs (objects and arrays)
_JSON_BLOB_RE = re.compile(r"\{[^}]{0,2000}\}")
_JSON_ARRAY_RE = re.compile(r"\[[\s]*\{[\s\S]{0,5000}\}[\s]*\]")

# Quoted empty arrays/objects: "[]", "{}"
_QUOTED_EMPTY_RE = re.compile(r'"?\[\]"?|"?\{\}"?')

# ---------------------------------------------------------------------------
# Keyword-based reasoning detector
# If a sentence contains ANY of these keywords, it's internal reasoning.
# This is intentionally aggressive — false positives are better than leaking
# database internals to the user.
# ---------------------------------------------------------------------------
_REASONING_KEYWORDS = re.compile(
    r"(?:"
    # DB / query internals (word boundary where possible)
    r"\bquery\b|\bsql\b|\btable\b|\bcolumn\b|\bschema\b|\bdatabase\b|\bconnection\b|stream expir|"
    r"run_?sql|list_?tables|describe_?table|runsqlquery|"
    # Known table names
    r"\bwork_order\b|\baemp_equipment\b|\bsupporting_document\b|\binvoice\b|\bbooking\b|"
    # Tool internals
    r"\bweb_?search\b|\berror running\b|"
    # Model self-talk
    r"\bre-?run\b|fresh (?:query|call)|previous quer|let me search|"
    # Debugging language
    r"expir(?:ed|ation)|closed after|timed? ?out|truncat|"
    # SQL keywords that should never be spoken
    r"\bSELECT\b|\bWHERE\b|\bINSERT\b|\bDELETE\b|\bFROM\b.*\btable\b|"
    r"LIKE\s+['\"%]|"
    # System prompt echo
    r"^You are Alex, a voice assistant|"
    r"^RULES:|^TOOLS:|^CURRENT CONTEXT:|"
    r"Use list_tables|Use describe_table|search the document store|"
    # Internal search/tool narration
    r"\bno results\b|searches? failed|fallback|internal docs|"
    r"\bcatalog site\b|given inability|respond that|unable to (?:locate|find|retrieve)|"
    r"couldn't (?:find|locate|retrieve)|could not (?:find|locate|retrieve)|"
    r"search(?:ed|ing) (?:for|the)|no (?:matching|relevant) |"
    # Tool result echo patterns
    r"\bresults returned\b|\bsnippet\b|\"url\"|\"title\"|\"results\"|"
    # S3 / presigned URL fragments
    r"AWSAccessKeyId|Signature=|Expires=|\bpresigned\b|valid for \d+ minutes?|"
    r"Download URL for|get_?document_?url|search_?documents|save_?document|"
    # More model self-talk
    r"\bnow we have\b|try again|format problematic|correct key|"
    r"returned earlier|we can try|use (?:get|send|search|save)"
    r")",
    re.IGNORECASE,
)

# Full-sentence reasoning patterns — drop the entire text if it starts with these
_REASONING_SENTENCE_PATTERNS = [
    r"^We (?:need|still|haven't|should|must|could|got|have) ",
    r"^Let'?s ",
    r"^Probably ",
    r"^Possibly ",
    r"^(?:It )?might ",
    r"^(?:Maybe|Perhaps) ",
    r"^Error ",
    r"^Not (?:shown|captured)\b",
    r"^Need to ",
    r"^Could (?:try|search|also|attempt|be|we)",
    r"^Should be (?:okay|fine|good)",
    r"^(?:Search|Query|Check|Try|Fetch) ",
    r"^(?:The )?(?:ID|item|token) (?:is|looks|might|could)",
    r"^Great\.\s*We ",
    # Tool/search narration
    r"^So no ",
    r"^No (?:results|matches|documents|data)",
    r"^All (?:searches|queries|attempts) ",
    r"^But maybe ",
    r"^Given (?:inability|that|the) ",
    r"^(?:Cannot|Can't|Couldn't) (?:find|locate|retrieve|access) ",
    r"^(?:Now we |Use |Anyway )",
    r"^Download URL ",
    r"^The URL ",
]
_REASONING_SENTENCE_RE = re.compile(
    "|".join(_REASONING_SENTENCE_PATTERNS), re.IGNORECASE
)

# URLs — extract before sanitizing, replace with spoken placeholder
_URL_RE = re.compile(r"https?://[^\s)\]\"'>]+")

# SQL query fragments — match the entire statement, not just the keyword
_SQL_LEAK_RE = re.compile(
    r"(?:SELECT\s+.+|WHERE\s+.+|INSERT\s+.+|UPDATE\s+.+|DELETE\s+.+|ALTER\s+.+|DROP\s+.+|CREATE\s+.+)",
    re.IGNORECASE,
)


def _extract_urls(text: str) -> list[str]:
    """Pull all URLs from text."""
    return _URL_RE.findall(text)


def _sanitize_for_tts(text: str) -> str:
    """Strip markdown, reasoning tokens, tool metadata, and special characters
    so TTS reads natural speech only."""

    # --- Phase 0a: Replace URLs with spoken placeholder ---
    text = _URL_RE.sub("I'm sending you a link", text)

    # --- Phase 0b: Strip bare reasoning prefixes (no angle brackets) ---
    # e.g. "analysisWe have trouble..." → "We have trouble..."
    # Then the reasoning sentence filter below will catch the rest.
    text = _BARE_PREFIX_RE.sub("", text)

    # If the entire chunk is just a reasoning prefix, drop it
    if not text.strip() or _LONE_ROLE_RE.match(text.strip()):
        return ""

    # --- Phase 1: Strip model internals ---

    # Remove repeated role tokens (assistantassistantassistant...)
    text = _ROLE_TOKEN_RE.sub("", text)

    # Remove reasoning blocks (content between XML tags)
    text = _REASONING_BLOCK_RE.sub("", text)
    # Remove remaining reasoning tags
    text = _REASONING_TAG_RE.sub("", text)

    # Remove tool-call routing
    text = _TOOL_ROUTING_RE.sub("", text)

    # Remove "json{...}" patterns
    text = _JSON_PREFIX_RE.sub("", text)
    # Remove raw JSON blobs and arrays
    text = _JSON_ARRAY_RE.sub("", text)
    text = _JSON_BLOB_RE.sub("", text)
    # Remove quoted empty results: "[]", "{}"
    text = _QUOTED_EMPTY_RE.sub("", text)

    # Remove SQL query fragments
    text = _SQL_LEAK_RE.sub("", text)

    # Drop entire text if it contains DB/tool/reasoning keywords
    if _REASONING_KEYWORDS.search(text):
        return ""

    # Drop entire text if it starts with a reasoning sentence pattern
    if _REASONING_SENTENCE_RE.match(text.strip()):
        return ""

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
