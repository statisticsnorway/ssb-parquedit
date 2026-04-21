"""Tests for Query operations."""

from typing import Any
from unittest.mock import MagicMock

import pytest

# Fixtures are imported from conftest.py: stub_external_modules, fake_conn, db_config


@pytest.fixture
def sut_query() -> Any:
    """Import and return the QueryOperations class."""
    import importlib

    module = importlib.import_module("ssb_parquedit.query")
    importlib.reload(module)
    return module.QueryOperations


@pytest.fixture
def query_with_mock_result(
    sut_query: Any, fake_conn: MagicMock
) -> tuple[Any, MagicMock]:
    """Create QueryOperations with mocked connection and result."""
    query_ops: Any = sut_query(fake_conn)

    # Mock the result object
    mock_result = MagicMock()
    mock_df = MagicMock()
    mock_result.df.return_value = mock_df
    fake_conn.execute.return_value = mock_result

    return query_ops, fake_conn


class TestViewBasic:
    """Test basic view functionality."""

    def test_view_simple_query(
        self, query_with_mock_result: tuple[Any, MagicMock]
    ) -> None:
        """Test simple view query execution."""
        query_ops, fake_conn = query_with_mock_result

        query_ops.view("users")

        # Should execute a query
        assert fake_conn.execute.called

    def test_view_validates_table_name(
        self, sut_query: Any, fake_conn: MagicMock
    ) -> None:
        """Test that table name is validated."""
        query_ops = sut_query(fake_conn)

        with pytest.raises(ValueError, match="Invalid table name"):
            query_ops.view("123invalid")

    def test_view_returns_pandas_by_default(
        self, query_with_mock_result: tuple[Any, MagicMock]
    ) -> None:
        """Test that view returns pandas DataFrame by default."""
        query_ops, fake_conn = query_with_mock_result

        result = query_ops.view("users")

        # Should call .df() for pandas
        mock_result = fake_conn.execute.return_value
        mock_result.df.assert_called()
        assert result == mock_result.df.return_value

    def test_view_with_limit(
        self, query_with_mock_result: tuple[Any, MagicMock]
    ) -> None:
        """Test view with LIMIT clause."""
        query_ops, fake_conn = query_with_mock_result

        query_ops.view("users", limit=10)

        # Should include LIMIT in query
        execute_calls = [
            str(call_args[0]) for (call_args, _) in fake_conn.execute.call_args_list
        ]
        assert any("LIMIT" in call for call in execute_calls), "LIMIT clause not found"

    def test_view_with_none_limit(
        self, query_with_mock_result: tuple[Any, MagicMock]
    ) -> None:
        """Test view with limit=None (all rows)."""
        query_ops, fake_conn = query_with_mock_result

        query_ops.view("users", limit=None)

        # Should not include LIMIT if None
        execute_calls = [
            str(call_args[0]) for (call_args, _) in fake_conn.execute.call_args_list
        ]
        # If limit is None, SQL should not have LIMIT clause
        # (This depends on implementation - check if LIMIT is added for None)
        assert all(
            "LIMIT" not in call for call in execute_calls
        ), "LIMIT should not appear when limit=None"


class TestListTables:
    """Test listing all tables in the catalog."""

    def test_list_tables_returns_list(
        self, query_with_mock_result: tuple[Any, MagicMock]
    ) -> None:
        """Test that list_tables returns a list."""
        query_ops, fake_conn = query_with_mock_result

        # Mock DataFrame with name column (from SHOW TABLES)
        mock_result = MagicMock()
        mock_df = MagicMock()
        mock_df_column = MagicMock()
        mock_df_column.tolist.return_value = ["users", "products", "orders"]
        mock_df.__getitem__.return_value = mock_df_column
        mock_result.df.return_value = mock_df
        fake_conn.execute.return_value = mock_result

        result = query_ops.list_tables()

        assert isinstance(result, list)
        assert result == ["users", "products", "orders"]
