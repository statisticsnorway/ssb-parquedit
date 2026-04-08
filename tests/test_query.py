"""Tests for Query operations."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from ssb_parquedit.utils import SQLInjectionError

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

    def test_view_validates_column_names(
        self, sut_query: Any, fake_conn: MagicMock
    ) -> None:
        """Test that column names are validated."""
        query_ops = sut_query(fake_conn)

        with pytest.raises(SQLInjectionError):  # SQLInjectionError
            query_ops.view("users", columns=["id; DROP TABLE users"])


class TestViewOrdering:
    """Test view with ORDER BY clause."""

    def test_view_validates_order_by_clause(
        self, sut_query: Any, fake_conn: MagicMock
    ) -> None:
        """Test that ORDER BY clause is validated."""
        query_ops = sut_query(fake_conn)

        # Order by with dangerous SQL
        with pytest.raises(SQLInjectionError):  # SQLInjectionError
            query_ops.view("users", order_by="id; DROP TABLE users")


class TestCount:
    """Test count operation."""

    def test_count_all_rows(
        self, query_with_mock_result: tuple[Any, MagicMock]
    ) -> None:
        """Test counting all rows."""
        query_ops, fake_conn = query_with_mock_result

        mock_result = MagicMock()
        mock_df = MagicMock()
        mock_df.__getitem__.return_value.iloc = [42]
        mock_result.df.return_value = mock_df
        fake_conn.execute.return_value = mock_result

        # Mock the iloc[0] access
        mock_series = MagicMock()
        mock_series.iloc = MagicMock()
        mock_series.iloc.__getitem__.return_value = 42
        mock_df.__getitem__.return_value = mock_series

        result = query_ops.count("users")

        # Should execute COUNT query
        assert fake_conn.execute.called
        assert result == 42


class TestTableExists:
    """Test table existence check."""

    def test_table_exists_returns_true(
        self, query_with_mock_result: tuple[Any, MagicMock]
    ) -> None:
        """Test that existing table returns True."""
        query_ops, fake_conn = query_with_mock_result

        result = query_ops.table_exists("users")

        # Should return True when query succeeds
        assert fake_conn.execute.called
        assert result is True

    def test_table_exists_returns_false_on_error(
        self, sut_query: Any, fake_conn: MagicMock
    ) -> None:
        """Test that non-existent table returns False."""
        query_ops = sut_query(fake_conn)

        # Mock execute to raise exception
        fake_conn.execute.side_effect = Exception("Table not found")

        result = query_ops.table_exists("nonexistent")

        # Should return False when query fails
        assert result is False


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
