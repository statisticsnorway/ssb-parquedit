import json
import pytest
import pandas as pd
from unittest.mock import MagicMock
from ssb_parquedit.parquedit import ParquEdit
from typing import Any


MOCK_ROW = {
    "change_event_reason": "OTHER",
    "changed_by": "user@ssb.no",
    "table_name": "cities",
    "rowid": 1,
    "unique_key": {"id": 1},
    "change_comment": "Test edit",
    "change_datetime": "2026-01-01T00:00:00+02:00",
    "statistics_name": "test_product",
    "old_values": {"name": "Oslo"},
    "new_values": {"name": "Bergen"},
}

@pytest.fixture
def cities_table(pe: ParquEdit) -> ParquEdit:
    """Helper fixture - creates and populates a standard table."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Oslo", "Bergen", "Tromsø"],
            "population": [700000, 285000, 75000],
        }
    )
    pe.create_table(
        "cities", source=df, product_name="test_product", unique_key=["id"], fill=True
    )
    return pe


@pytest.fixture
def mock_edits_table(pe: ParquEdit, monkeypatch: pytest.MonkeyPatch) -> ParquEdit:
    """Fixture that mocks the snapshots query to return a predictable DataFrame."""
    mock_snapshots_df = pd.DataFrame({"commit_extra_info": [json.dumps(MOCK_ROW)]})
    mock_tables_df = pd.DataFrame({"name": ["cities"]})

    def mock_execute(sql: str, parameters: list[Any] | None = None) -> MagicMock:
        mock_result = MagicMock()
        if "snapshots" in sql:
            mock_result.df.return_value = mock_snapshots_df
        else:
            mock_result.df.return_value = mock_tables_df
        return mock_result

    monkeypatch.setattr(pe._get_connection(), "execute", mock_execute)
    return pe

def test_get_edits_returns_dataframe(mock_edits_table: ParquEdit) -> None:
    """get_edits() should return a DataFrame."""
    result = mock_edits_table.get_edits()
    assert isinstance(result, pd.DataFrame)


def test_get_edits_filtered_by_table_name(mock_edits_table: ParquEdit) -> None:
    """get_edits() should only return rows for the given table name."""
    result = mock_edits_table.get_edits("cities")
    assert isinstance(result, pd.DataFrame)
    assert all(result["table_name"] == "cities")


def test_get_edits_contains_expected_columns(mock_edits_table: ParquEdit) -> None:
    """get_edits() should contain parsed changelog columns."""
    result = mock_edits_table.get_edits("cities")
    expected_cols = {"change_event_reason", "changed_by", "unique_key", "old_values", "new_values", "table_name"}
    assert expected_cols.issubset(result.columns)


def test_get_edits_unknown_table_returns_empty(mock_edits_table: ParquEdit) -> None:
    """get_edits() should raise ValueError for unknown table name."""
    with pytest.raises(ValueError, match="nonexistent_table"):
        mock_edits_table.get_edits("nonexistent_table")


def test_get_edits_no_filter_returns_all_tables(mock_edits_table: ParquEdit) -> None:
    """get_edits() without table_name should return edits for all tables."""
    result = mock_edits_table.get_edits()
    assert isinstance(result, pd.DataFrame)
    assert "table_name" in result.columns
    assert len(result) == 1