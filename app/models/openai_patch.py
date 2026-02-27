"""Patched OpenAI model that truncates tool_call_id to 40 characters.

gpt-5-mini occasionally generates tool_call_id values longer than 40
characters, which causes intermittent validation failures on the OpenAI
API.  This thin subclass intercepts the request body and truncates any
offending IDs before the request is sent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from agno.models.openai import OpenAIChat

logger = logging.getLogger(__name__)

MAX_TOOL_CALL_ID_LEN = 40


class PatchedOpenAIChat(OpenAIChat):
    """OpenAIChat with tool_call_id truncation to stay within API limits."""

    def _truncate_tool_call_ids(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Truncate tool_call_id fields that exceed 40 characters."""
        for msg in messages:
            # Truncate tool_call_id on tool-result messages
            if msg.get("tool_call_id") and len(msg["tool_call_id"]) > MAX_TOOL_CALL_ID_LEN:
                original = msg["tool_call_id"]
                msg["tool_call_id"] = original[:MAX_TOOL_CALL_ID_LEN]
                logger.debug("Truncated tool_call_id: %s -> %s", original, msg["tool_call_id"])

            # Truncate tool_call ids inside assistant tool_calls
            for tc in msg.get("tool_calls", []):
                if tc.get("id") and len(tc["id"]) > MAX_TOOL_CALL_ID_LEN:
                    original = tc["id"]
                    tc["id"] = original[:MAX_TOOL_CALL_ID_LEN]
                    logger.debug("Truncated tool_call id: %s -> %s", original, tc["id"])

        return messages

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        if "messages" in kwargs:
            kwargs["messages"] = self._truncate_tool_call_ids(kwargs["messages"])
        return super().invoke(*args, **kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        if "messages" in kwargs:
            kwargs["messages"] = self._truncate_tool_call_ids(kwargs["messages"])
        return await super().ainvoke(*args, **kwargs)

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        if "messages" in kwargs:
            kwargs["messages"] = self._truncate_tool_call_ids(kwargs["messages"])
        return super().invoke_stream(*args, **kwargs)

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        if "messages" in kwargs:
            kwargs["messages"] = self._truncate_tool_call_ids(kwargs["messages"])
        return await super().ainvoke_stream(*args, **kwargs)
