"""Web search tool for Agno agents.

Wraps agno.tools.duckduckgo.DuckDuckGoTools with consistent defaults
so all services (chat, diagnostics, voice) share the same configuration.
"""

from agno.tools.duckduckgo import DuckDuckGoTools


def create_search_tools() -> DuckDuckGoTools:
    """Create a DuckDuckGo search tool with standard settings."""
    return DuckDuckGoTools(
        timeout=10,
        fixed_max_results=5,
    )
