"""Markdown-to-HTML conversion for API responses.

Used by chat and diagnostics endpoints so the frontend receives
ready-to-render HTML instead of raw markdown.
"""

import re

import markdown

_md = markdown.Markdown(extensions=["tables", "fenced_code", "nl2br", "sane_lists"])

# Ensure a blank line before list items so the markdown parser picks them up
_LIST_FIX_RE = re.compile(r"([^\n])\n([-*] )")

# Match raw URLs that are NOT already inside an <a> tag
_RAW_URL_RE = re.compile(r'(?<!href=["\'])(?<!">)(https?://[^\s<)"\'>]+)')


def md_to_html(text: str) -> str:
    """Convert markdown text to HTML."""
    if not text:
        return text
    # Fix lists that follow immediately after a line (no blank line separator)
    fixed = _LIST_FIX_RE.sub(r"\1\n\n\2", text)
    _md.reset()
    html = _md.convert(fixed)
    # Auto-linkify raw URLs not already wrapped in <a> tags
    html = _autolink(html)
    return html


def _autolink(html: str) -> str:
    """Wrap raw URLs in <a> tags, skipping those already inside href or anchor text."""
    def _replace(match: re.Match) -> str:
        url = match.group(0)
        # Check if this URL is already inside an <a> tag
        start = match.start()
        preceding = html[max(0, start - 10):start]
        if 'href=' in preceding or '">' in preceding:
            return url
        return f'<a href="{url}" target="_blank" rel="noopener">{url}</a>'
    return _RAW_URL_RE.sub(_replace, html)
