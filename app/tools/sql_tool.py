"""Read-only SQL tool for Agno agents.

Wraps agno.tools.sql.SQLTools with:
- Hard block on INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, GRANT, REVOKE
- describe_table returns column names and types only (no nullable/default metadata)
- Centralized factory so all three services use the same config
"""

import json
import logging
import re

from agno.tools.sql import SQLTools
from sqlalchemy import Engine

logger = logging.getLogger(__name__)

# Statements that must never run
_WRITE_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|REPLACE|MERGE)\b",
    re.IGNORECASE,
)


class ReadOnlySQLTools(SQLTools):
    """SQLTools subclass that blocks all write operations."""

    def describe_table(self, table_name: str) -> str:
        """Return column names and types only — no nullable/default metadata."""
        try:
            from sqlalchemy import inspect as sa_inspect

            inspector = sa_inspect(self.db_engine)
            columns = inspector.get_columns(table_name, schema=self.schema)
            return json.dumps(
                [{"name": col["name"], "type": str(col["type"])} for col in columns]
            )
        except Exception as e:
            logger.error("Error getting table schema: %s", e)
            return f"Error getting table schema: {e}"

    def run_sql_query(self, query: str, limit: int | None = 10) -> str:
        """Run a SELECT query only. Rejects any write statements."""
        if _WRITE_RE.search(query):
            logger.warning("Blocked write query: %s", query[:200])
            return "Error: Only SELECT queries are allowed. This database is read-only."
        return super().run_sql_query(query=query, limit=limit)

    def run_sql(self, sql: str, limit: int | None = None) -> list[dict]:
        """Run a SELECT query only. Rejects any write statements."""
        if _WRITE_RE.search(sql):
            logger.warning("Blocked write query: %s", sql[:200])
            raise PermissionError("Only SELECT queries are allowed. This database is read-only.")
        return super().run_sql(sql=sql, limit=limit)


def create_sql_tools(db_engine: Engine) -> ReadOnlySQLTools:
    """Create a fresh read-only SQLTools instance.

    Call this per-request to avoid connection expiration with Turso/libSQL.
    """
    return ReadOnlySQLTools(db_engine=db_engine)


def fetch_equipment_summary(db_engine: Engine, listing_id: str) -> dict | None:
    """Pre-fetch equipment details from the listing table.

    Returns a dict with name, make, model, year, serial_number,
    operating_hours, and category — or None on failure.
    Retries once after disposing stale pool connections (Turso/libSQL).
    """
    from sqlalchemy import text

    _query = text(
        "SELECT l.name, l.make, l.model, l.year, l.serial_number, "
        "l.operating_hours, c.name AS category "
        "FROM listing l LEFT JOIN category c ON l.category_id = c.id "
        "WHERE l.id = :lid LIMIT 1"
    )

    for attempt in range(2):
        try:
            with db_engine.connect() as conn:
                row = conn.execute(_query, {"lid": listing_id}).mappings().first()
            if not row:
                logger.info("No listing found for id=%s", listing_id)
                return None
            return dict(row)
        except Exception:
            if attempt == 0:
                logger.debug("Stale connection for pre-fetch, disposing pool and retrying")
                try:
                    db_engine.dispose()
                except Exception:
                    pass
            else:
                logger.warning("Failed to pre-fetch equipment for listing %s", listing_id, exc_info=True)
                return None
    return None
