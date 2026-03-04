"""Tests for equipment pre-fetch integration in AgnoService._build_context_message."""

import json
from unittest.mock import MagicMock, patch

from app.services.agno_service import AgnoService


SAMPLE_EQUIP = {
    "name": "333G Compact Track Loader",
    "make": "John Deere",
    "model": "333G",
    "year": 2021,
    "serial_number": "ABC123",
    "operating_hours": 1500,
    "category": "Compact Track Loaders",
}


def test_build_context_message_with_equipment_prefetch():
    """When listing_id is present and DB returns data, equipment details appear in context."""
    service = AgnoService()
    metadata = json.dumps({"listing_id": "L1", "equipment_name": "John Deere 333G"})

    with patch("app.services.agno_service.fetch_equipment_summary", return_value=SAMPLE_EQUIP):
        result = service._build_context_message(metadata)

    assert "Make: John Deere" in result
    assert "Model: 333G" in result
    assert "Year: 2021" in result
    assert "Serial: ABC123" in result
    assert "Hours: 1500" in result
    assert "Category: Compact Track Loaders" in result
    assert "re-querying" in result.lower()


def test_build_context_message_no_listing_id():
    """Without listing_id, no pre-fetch occurs."""
    service = AgnoService()
    metadata = json.dumps({"equipment_name": "Something"})

    with patch("app.services.agno_service.fetch_equipment_summary") as mock_fetch:
        result = service._build_context_message(metadata)

    mock_fetch.assert_not_called()
    assert "Equipment from database" not in result


def test_build_context_message_fetch_returns_none():
    """When DB lookup fails/returns None, context still works without equipment."""
    service = AgnoService()
    metadata = json.dumps({"listing_id": "L1"})

    with patch("app.services.agno_service.fetch_equipment_summary", return_value=None):
        result = service._build_context_message(metadata)

    assert "Listing ID: L1" in result
    assert "Equipment from database" not in result


def test_build_context_message_partial_equipment():
    """When some equipment fields are None, only non-null fields appear."""
    service = AgnoService()
    metadata = json.dumps({"listing_id": "L1"})

    partial = {
        "name": "Some Loader",
        "make": "Kubota",
        "model": "SVL97",
        "year": None,
        "serial_number": None,
        "operating_hours": 500,
        "category": None,
    }

    with patch("app.services.agno_service.fetch_equipment_summary", return_value=partial):
        result = service._build_context_message(metadata)

    assert "Make: Kubota" in result
    assert "Model: SVL97" in result
    assert "Hours: 500" in result
    assert "Year:" not in result
    assert "Serial:" not in result
    assert "Category:" not in result


def test_build_context_message_no_metadata():
    """Empty/None metadata returns empty string."""
    service = AgnoService()
    assert service._build_context_message(None) == ""
    assert service._build_context_message("") == ""
