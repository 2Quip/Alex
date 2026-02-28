"""Markdown-to-HTML conversion for API responses.

Used by chat and diagnostics endpoints so the frontend receives
ready-to-render HTML instead of raw markdown.
"""

import re

import markdown

_md = markdown.Markdown(extensions=["tables", "fenced_code", "nl2br", "sane_lists"])

# Ensure a blank line before list items so the markdown parser picks them up
_LIST_FIX_RE = re.compile(r"([^\n])\n([-*] )")


def md_to_html(text: str) -> str:
    """Convert markdown text to HTML."""
    if not text:
        return text
    # Fix lists that follow immediately after a line (no blank line separator)
    fixed = _LIST_FIX_RE.sub(r"\1\n\n\2", text)
    _md.reset()
    return _md.convert(fixed)
