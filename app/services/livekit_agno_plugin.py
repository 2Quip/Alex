# Copyright 2025
# Licensed under the Apache License, Version 2.0

"""Agno plugin for LiveKit Agents - wraps Agno Agents as LiveKit LLMs."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

from agno.agent import Agent
from agno.run.agent import RunContentEvent, RunOutput

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

        # Run agent with streaming
        response_stream = self._agent.arun(
            input=user_input,
            stream=True,
            session_id=self._session_id,
            user_id=self._user_id,
        )

        async for event in response_stream:
            chunk = _to_chat_chunk(event)
            if chunk:
                self._event_ch.send_nowait(chunk)

    def _get_user_input(self) -> str | None:
        """Extract the last user message from chat context."""

        for msg in reversed(self._chat_ctx.items):
            if isinstance(msg, ChatMessage) and msg.role == "user":
                content = msg.text_content
                return content
        return None


def _sanitize_for_tts(text: str) -> str:
    """Strip markdown and special characters so TTS reads natural speech."""
    # Remove markdown headers (### Header)
    text = re.sub(r"#{1,6}\s*", "", text)
    # Remove bold/italic markers (**bold**, *italic*, __bold__, _italic_)
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
    # Remove inline code backticks
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Remove code fence markers
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Convert markdown links [text](url) to just the text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove bullet point markers (-, *, •) at start of lines
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    # Remove numbered list markers (1. 2. etc.) at start of lines
    text = re.sub(r"(?m)^\s*\d+\.\s+", "", text)
    # Remove horizontal rules (---, ***, ___)
    text = re.sub(r"(?m)^[-*_]{3,}\s*$", "", text)
    # Collapse multiple newlines into a single space
    text = re.sub(r"\n{2,}", " ", text)
    # Replace single newlines with space
    text = re.sub(r"\n", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _to_chat_chunk(event: Any) -> llm.ChatChunk | None:
    """Convert Agno event to LiveKit ChatChunk."""
    content = None

    if isinstance(event, RunContentEvent):
        content = event.content
    elif isinstance(event, RunOutput) and event.content:
        content = (
            str(event.content) if not isinstance(event.content, str) else event.content
        )
    elif hasattr(event, "content") and event.content:
        content = str(event.content)

    if content:
        content = _sanitize_for_tts(content)

    if content:
        return llm.ChatChunk(
            id="agno",
            delta=llm.ChoiceDelta(role="assistant", content=content),
        )
    return None
