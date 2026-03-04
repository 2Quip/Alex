"""Tests for app/tools/sql_tool.py — ReadOnlySQLTools and fetch_equipment_summary."""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import Engine

from app.tools.sql_tool import ReadOnlySQLTools, create_sql_tools, fetch_equipment_summary


# --- ReadOnlySQLTools ---


def test_run_sql_query_blocks_insert():
    tool = ReadOnlySQLTools(db_engine=MagicMock(spec=Engine))
    result = tool.run_sql_query("INSERT INTO listing VALUES ('x')")
    assert "read-only" in result.lower()


def test_run_sql_blocks_delete():
    tool = ReadOnlySQLTools(db_engine=MagicMock(spec=Engine))
    with pytest.raises(PermissionError, match="read-only"):
        tool.run_sql("DELETE FROM listing WHERE id = '1'")


def test_create_sql_tools_returns_readonly():
    tools = create_sql_tools(db_engine=MagicMock(spec=Engine))
    assert isinstance(tools, ReadOnlySQLTools)


# --- fetch_equipment_summary ---


def test_fetch_equipment_summary_returns_dict():
    """Happy path: row found, returns dict with all fields."""
    mock_row = {
        "name": "333G Compact Track Loader",
        "make": "John Deere",
        "model": "333G",
        "year": 2021,
        "serial_number": "ABC123",
        "operating_hours": 1500,
        "category": "Compact Track Loaders",
    }

    mock_conn = MagicMock()
    mock_mappings = MagicMock()
    mock_mappings.first.return_value = mock_row
    mock_conn.execute.return_value.mappings.return_value = mock_mappings
    mock_engine = MagicMock(spec=Engine)
    mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    result = fetch_equipment_summary(mock_engine, "listing-123")

    assert result is not None
    assert result["make"] == "John Deere"
    assert result["model"] == "333G"
    assert result["year"] == 2021
    assert result["serial_number"] == "ABC123"
    assert result["operating_hours"] == 1500
    assert result["category"] == "Compact Track Loaders"


def test_fetch_equipment_summary_not_found():
    """No row found returns None."""
    mock_conn = MagicMock()
    mock_mappings = MagicMock()
    mock_mappings.first.return_value = None
    mock_conn.execute.return_value.mappings.return_value = mock_mappings
    mock_engine = MagicMock(spec=Engine)
    mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    result = fetch_equipment_summary(mock_engine, "nonexistent")
    assert result is None


def test_fetch_equipment_summary_exception_returns_none():
    """DB errors are caught and return None."""
    mock_engine = MagicMock(spec=Engine)
    mock_engine.connect.side_effect = RuntimeError("connection refused")

    result = fetch_equipment_summary(mock_engine, "listing-123")
    assert result is None
