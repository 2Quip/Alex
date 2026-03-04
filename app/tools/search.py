"""Web search tool for Agno agents.

Wraps agno.tools.tavily.TavilyTools with consistent defaults
so all services (chat, diagnostics, voice) share the same configuration.
"""

from agno.tools.tavily import TavilyTools

from app.config.settings import settings


def create_search_tools() -> TavilyTools:
    """Create a Tavily search tool with standard settings."""
    return TavilyTools(
        api_key=settings.TAVILY_API_KEY,
        search_depth="basic",
        include_answer=True,
        max_tokens=6000,
        format="markdown",
    )
