"""Tests for Query operations."""

from unittest.mock import MagicMock

import pytest

from ssb_parquedit.utils import SQLInjectionError

# Fixtures are imported from conftest.py: stub_external_modules, fake_conn, db_config


@pytest.fixture
def sut_query() -> object:
    """Import and return the QueryOperations class."""
    import importlib

    module = importlib.import_module("ssb_parquedit.query")
    importlib.reload(module)
    return module.QueryOperations


@pytest.fixture
def query_with_mock_result(
    sut_query: object, fake_conn: MagicMock
) -> tuple[object, MagicMock]:
    """Create QueryOperations with mocked connection and result."""
    query_ops = sut_query(fake_conn)

    # Mock the result object
    mock_result = MagicMock()
    mock_df = MagicMock()
    mock_result.df.return_value = mock_df
    fake_conn.execute.return_value = mock_result

    return query_ops, fake_conn


class TestViewBasic:
    """Test basic view functionality."""

    def test_view_simple_query(
        self, query_with_mock_result: tuple[object, MagicMock]
    ) -> None:
        """Test simple view query execution."""
        query_ops, fake_conn = query_with_mock_result

        query_ops.view("users")

        # Should execute a query
        assert fake_conn.execute.called

    def test_view_validates_table_name(
        self, sut_query: object, fake_conn: MagicMock
    ) -> None:
        """Test that table name is validated."""
        query_ops = sut_query(fake_conn)

        with pytest.raises(ValueError, match="Invalid table name"):
            query_ops.view("123invalid")

    def test_view_returns_pandas_by_default(
        self, query_with_mock_result: tuple[object, MagicMock]
    ) -> None:
        """Test that view returns pandas DataFrame by default."""
        query_ops, fake_conn = query_with_mock_result

        result = query_ops.view("users")

        # Should call .df() for pandas
        mock_result = fake_conn.execute.return_value
        mock_result.df.assert_called()
        assert result == mock_result.df.return_value

    def test_view_with_limit(
        self, query_with_mock_result: tuple[object, MagicMock]
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
        self, query_with_mock_result: tuple[object, MagicMock]
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

    def test_view_with_offset(
        self, query_with_mock_result: tuple[object, MagicMock]
    ) -> None:
        """Test view with OFFSET clause."""
        query_ops, fake_conn = query_with_mock_result

        query_ops.view("users", offset=20)

        # Should include OFFSET in query
        execute_calls = [
            str(call_args[0]) for (call_args, _) in fake_conn.execute.call_args_list
        ]
        assert any(
            "OFFSET" in call for call in execute_calls
        ), "OFFSET clause not found"

    def test_view_with_columns_select(
        self, query_with_mock_result: tuple[object, MagicMock]
    ) -> None:
        """Test view with specific columns."""
        query_ops, fake_conn = query_with_mock_result

        query_ops.view("users", columns=["id", "name"])

        # Should include specific columns in SELECT
        execute_calls = [
            str(call_args[0]) for (call_args, _) in fake_conn.execute.call_args_list
        ]
        assert any(
            "id" in call and "name" in call for call in execute_calls
        ), "Column names not in SELECT"

    def test_view_validates_column_names(
        self, sut_query: object, fake_conn: MagicMock
    ) -> None:
        """Test that column names are validated."""
        query_ops = sut_query(fake_conn)

        with pytest.raises(SQLInjectionError):  # SQLInjectionError
            query_ops.view("users", columns=["id; DROP TABLE users"])


class TestViewFiltering:
    """Test view with various filters."""

    def test_view_with_simple_filter(
        self, query_with_mock_result: tuple[object, MagicMock]
    ) -> None:
        """Test view with simple filter."""
        query_ops, fake_conn = query_with_mock_result

        filters = {"column": "status", "operator": "=", "value": "active"}
        query_ops.view("users", filters=filters)

        # Should include WHERE clause
        execute_calls = [
            str(call_args[0]) for (call_args, _) in fake_conn.execute.call_args_list
        ]
        assert any("WHERE" in call for call in execute_calls), "WHERE clause not found"

    def test_view_with_multiple_filters_and(
        self, query_with_mock_result: tuple[object, MagicMock]
    ) -> None:
        """Test view with multiple filters (AND logic)."""
        query_ops, fake_conn = query_with_mock_result

        filters = [
            {"column": "age", "operator": ">", "value": 25},
            {"column": "status", "operator": "=", "value": "active"},
        ]
        query_ops.view("users", filters=filters)

        # Should include WHERE clause with AND
        execute_calls = [
            str(call_args[0]) for (call_args, _) in fake_conn.execute.call_args_list
        ]
        assert any("WHERE" in call for call in execute_calls), "WHERE clause not found"

    def test_view_with_or_filters(
        self, query_with_mock_result: tuple[object, MagicMock]
    ) -> None:
        """Test view with OR filters."""
        query_ops, fake_conn = query_with_mock_result

        filters = {
            "or": [
                {"column": "status", "operator": "=", "value": "admin"},
                {"column": "status", "operator": "=", "value": "moderator"},
            ]
        }
        query_ops.view("users", filters=filters)

        # Should include WHERE clause
        assert fake_conn.execute.called

    def test_view_with_like_filter(
        self, query_with_mock_result: tuple[object, MagicMock]
    ) -> None:
        """Test view with LIKE filter."""
        query_ops, fake_conn = query_with_mock_result

        filters = {"column": "name", "operator": "LIKE", "value": "%john%"}
        query_ops.view("users", filters=filters)

        # Should include WHERE clause
        assert fake_conn.execute.called

    def test_view_with_in_filter(
        self, query_with_mock_result: tuple[object, MagicMock]
    ) -> None:
        """Test view with IN filter."""
        query_ops, fake_conn = query_with_mock_result

        filters = {"column": "id", "operator": "IN", "value": [1, 2, 3]}
        query_ops.view("users", filters=filters)

        # Should include WHERE clause
        assert fake_conn.execute.called

    def test_view_with_between_filter(
        self, query_with_mock_result: tuple[object, MagicMock]
    ) -> None:
        """Test view with BETWEEN filter."""
        query_ops, fake_conn = query_with_mock_result

        filters = {"column": "age", "operator": "BETWEEN", "value": [18, 65]}
        query_ops.view("users", filters=filters)

        # Should include WHERE clause
        assert fake_conn.execute.called


class TestViewOrdering:
    """Test view with ORDER BY clause."""

    def test_view_with_order_by(
        self, query_with_mock_result: tuple[object, MagicMock]
    ) -> None:
        """Test view with ORDER BY clause."""
        query_ops, fake_conn = query_with_mock_result

        query_ops.view("users", order_by="created_at DESC")

        # Should include ORDER BY in query
        execute_calls = [
            str(call_args[0]) for (call_args, _) in fake_conn.execute.call_args_list
        ]
        assert any(
            "ORDER BY" in call for call in execute_calls
        ), "ORDER BY clause not found"

    def test_view_with_order_by_multiple_columns(
        self, query_with_mock_result: tuple[object, MagicMock]
    ) -> None:
        """Test view with multiple ORDER BY columns."""
        query_ops, fake_conn = query_with_mock_result

        query_ops.view("users", order_by="category ASC, created_at DESC")

        # Should include ORDER BY in query
        assert fake_conn.execute.called

    def test_view_validates_order_by_clause(
        self, sut_query: object, fake_conn: MagicMock
    ) -> None:
        """Test that ORDER BY clause is validated."""
        query_ops = sut_query(fake_conn)

        # Order by with dangerous SQL
        with pytest.raises(SQLInjectionError):  # SQLInjectionError
            query_ops.view("users", order_by="id; DROP TABLE users")


class TestViewOutputFormats:
    """Test different output formats."""

    def test_view_returns_pandas_format(
        self, query_with_mock_result: tuple[object, MagicMock]
    ) -> None:
        """Test pandas output format."""
        query_ops, fake_conn = query_with_mock_result

        query_ops.view("users", output_format="pandas")

        # Should call .df()
        mock_result = fake_conn.execute.return_value
        mock_result.df.assert_called()

    def test_view_with_invalid_format_raises_error(
        self, sut_query: object, fake_conn: MagicMock
    ) -> None:
        """Test that invalid output format raises ValueError."""
        query_ops = sut_query(fake_conn)

        mock_result = MagicMock()
        fake_conn.execute.return_value = mock_result

        with pytest.raises(ValueError, match="Unknown output_format"):
            query_ops.view("users", output_format="invalid")


class TestCount:
    """Test count operation."""

    def test_count_all_rows(
        self, query_with_mock_result: tuple[object, MagicMock]
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

    def test_count_validates_table_name(
        self, sut_query: object, fake_conn: MagicMock
    ) -> None:
        """Test that table name is validated in count."""
        query_ops = sut_query(fake_conn)

        with pytest.raises(ValueError, match="Invalid table name"):
            query_ops.count("123invalid")

    def test_count_with_filters(
        self, query_with_mock_result: tuple[object, MagicMock]
    ) -> None:
        """Test counting with filters."""
        query_ops, fake_conn = query_with_mock_result

        # Setup mock result
        mock_result = MagicMock()
        mock_df = MagicMock()
        mock_series = MagicMock()
        mock_series.iloc.__getitem__.return_value = 10
        mock_df.__getitem__.return_value = mock_series
        mock_result.df.return_value = mock_df
        fake_conn.execute.return_value = mock_result

        filters = {"column": "status", "operator": "=", "value": "active"}
        query_ops.count("users", filters=filters)

        # Should execute query with WHERE clause
        assert fake_conn.execute.called


class TestTableExists:
    """Test table existence check."""

    def test_table_exists_returns_true(
        self, query_with_mock_result: tuple[object, MagicMock]
    ) -> None:
        """Test that existing table returns True."""
        query_ops, fake_conn = query_with_mock_result

        result = query_ops.table_exists("users")

        # Should return True when query succeeds
        assert fake_conn.execute.called
        assert result is True

    def test_table_exists_returns_false_on_error(
        self, sut_query: object, fake_conn: MagicMock
    ) -> None:
        """Test that non-existent table returns False."""
        query_ops = sut_query(fake_conn)

        # Mock execute to raise exception
        fake_conn.execute.side_effect = Exception("Table not found")

        result = query_ops.table_exists("nonexistent")

        # Should return False when query fails
        assert result is False

    def test_table_exists_validates_table_name(
        self, sut_query: object, fake_conn: MagicMock
    ) -> None:
        """Test that table name is validated in table_exists."""
        query_ops = sut_query(fake_conn)

        with pytest.raises(ValueError, match="Invalid table name"):
            query_ops.table_exists("123invalid")
