"""Tests for Query operations."""

import importlib
from typing import Any
from unittest.mock import MagicMock

import pytest

# Fixtures are imported from conftest.py: stub_external_modules, fake_conn, db_config


@pytest.fixture
def sut_query() -> Any:
    """Import and return QueryOperations with stubs in place."""
    module = importlib.import_module("ssb_parquedit.query")
    importlib.reload(module)
    return module.QueryOperations


@pytest.fixture
def query_ops(sut_query: Any, fake_conn: MagicMock) -> tuple[Any, MagicMock]:
    """Instantiated QueryOperations with a mock result ready for execute."""
    mock_result = MagicMock()
    mock_result.df.return_value = MagicMock()
    fake_conn.execute.return_value = mock_result
    return sut_query(fake_conn), fake_conn


class TestView:
    def test_executes_select(self, query_ops: tuple[Any, MagicMock]) -> None:
        ops, conn = query_ops
        ops.view("users")
        sql = str(conn.execute.call_args[0][0])
        assert "SELECT" in sql and "users" in sql

    def test_with_limit_adds_clause(self, query_ops: tuple[Any, MagicMock]) -> None:
        ops, conn = query_ops
        ops.view("users", limit=5)
        sql = str(conn.execute.call_args[0][0])
        assert "LIMIT" in sql

    def test_invalid_format_raises(self, sut_query: Any, fake_conn: MagicMock) -> None:
        ops = sut_query(fake_conn)
        with pytest.raises(ValueError, match="Unknown output_format"):
            ops.view("users", output_format="csv")

    def test_invalid_table_name_raises(
        self, sut_query: Any, fake_conn: MagicMock
    ) -> None:
        ops = sut_query(fake_conn)
        with pytest.raises(ValueError, match="Invalid table name"):
            ops.view("BAD-name")


class TestCount:
    def test_returns_integer(self, sut_query: Any, fake_conn: MagicMock) -> None:
        mock_series = MagicMock()
        mock_series.iloc.__getitem__.return_value = 42
        mock_df = MagicMock()
        mock_df.__getitem__.return_value = mock_series
        mock_result = MagicMock()
        mock_result.df.return_value = mock_df
        fake_conn.execute.return_value = mock_result

        ops = sut_query(fake_conn)
        result = ops.count("users")
        assert result == 42


class TestTableExists:
    def test_returns_true_when_found(self, query_ops: tuple[Any, MagicMock]) -> None:
        ops, _ = query_ops
        assert ops.table_exists("users") is True

    def test_returns_false_on_error(self, sut_query: Any, fake_conn: MagicMock) -> None:
        fake_conn.execute.side_effect = Exception("not found")
        ops = sut_query(fake_conn)
        assert ops.table_exists("users") is False
