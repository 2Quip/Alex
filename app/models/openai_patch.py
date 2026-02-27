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

    def _truncate_tool_call_ids(self, messages: list) -> list:
        """Truncate tool_call_id fields that exceed 40 characters.

        Messages may be plain dicts or Agno Message (Pydantic) objects, so
        we use getattr/setattr with a dict fallback.
        """
        for msg in messages:
            # --- tool_call_id on tool-result messages ---
            tool_call_id = (
                msg.get("tool_call_id") if isinstance(msg, dict)
                else getattr(msg, "tool_call_id", None)
            )
            if tool_call_id and len(tool_call_id) > MAX_TOOL_CALL_ID_LEN:
                truncated = tool_call_id[:MAX_TOOL_CALL_ID_LEN]
                if isinstance(msg, dict):
                    msg["tool_call_id"] = truncated
                else:
                    msg.tool_call_id = truncated
                logger.debug("Truncated tool_call_id: %s -> %s", tool_call_id, truncated)

            # --- tool_call ids inside assistant tool_calls ---
            tool_calls = (
                msg.get("tool_calls", []) if isinstance(msg, dict)
                else getattr(msg, "tool_calls", None) or []
            )
            for tc in tool_calls:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id and len(tc_id) > MAX_TOOL_CALL_ID_LEN:
                    truncated = tc_id[:MAX_TOOL_CALL_ID_LEN]
                    if isinstance(tc, dict):
                        tc["id"] = truncated
                    else:
                        tc.id = truncated
                    logger.debug("Truncated tool_call id: %s -> %s", tc_id, truncated)

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
