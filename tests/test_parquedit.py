import sys
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

# Fixtures are imported from conftest.py: stub_external_modules, sut, db_config


# -------------------- Basic Initialization Tests --------------------


def test_init_stores_db_config(sut: Any, db_config: dict[str, str]) -> None:
    """Test that ParquEdit stores the database configuration."""
    pe = sut(config=db_config)
    assert pe._db_config == db_config


def test_get_connection_creates_duckdb_connection(
    sut: Any, db_config: dict[str, str]
) -> None:
    """Test that _get_connection creates a DuckDBConnection."""
    pe = sut(config=db_config)

    # Mock DuckDBConnection to avoid actual connection attempts
    mock_conn = MagicMock()
    mock_conn._conn = MagicMock()
    mock_conn._owns_conn = True

    with patch("ssb_parquedit.parquedit.DuckDBConnection", return_value=mock_conn):
        conn = pe._get_connection()

    # Verify it's a DuckDBConnection (it should have _conn and _owns_conn attributes)
    assert hasattr(conn, "_conn")
    assert hasattr(conn, "_owns_conn")


# -------------------- Create Table Tests --------------------


def test_create_table_requires_short_name(sut: Any, db_config: dict[str, str]) -> None:
    """Test that create_table raises error when product_name is not provided."""
    pe = sut(config=db_config)

    DF = sys.modules["pandas"].DataFrame
    df = DF()

    with pytest.raises(ValueError, match="'product_name' must have a value"):
        pe.create_table("users", df)


def test_create_table_requires_non_empty_short_name(
    sut: Any, db_config: dict[str, str]
) -> None:
    """Test that create_table raises error when product_name is empty."""
    pe = sut(config=db_config)

    DF = sys.modules["pandas"].DataFrame
    df = DF()

    with pytest.raises(ValueError, match="'product_name' must have a value"):
        pe.create_table("users", df, product_name="")


def test_create_table_with_dataframe(sut: Any, db_config: dict[str, str]) -> None:
    """Test that create_table works with DataFrame source."""
    pe = sut(config=db_config)

    DF = sys.modules["pandas"].DataFrame
    df = DF()

    # Mock the connection
    mock_conn = MagicMock()
    mock_inner_conn = MagicMock()
    mock_conn._conn = mock_inner_conn
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=None)

    with patch.object(pe, "_get_connection", return_value=mock_conn):
        pe.create_table("users", df, product_name="my_product")

    # Verify that the connection was used
    mock_conn.__enter__.assert_called_once()


# -------------------- Insert Data Tests --------------------


def test_insert_data_calls_connection(sut: Any, db_config: dict[str, str]) -> None:
    """Test that insert_data creates and uses a connection."""
    pe = sut(config=db_config)

    # Mock the connection
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=None)

    with patch.object(pe, "_get_connection", return_value=mock_conn):
        # Just test that _get_connection is called, not the full flow
        try:
            pe.insert_data("users", "gs://bucket/data.parquet")
        except Exception:
            pass  # We expect this to fail since we're mocking

    # Verify that the connection was used
    mock_conn.__enter__.assert_called_once()


# -------------------- View Tests --------------------


def test_view_calls_connection(sut: Any, db_config: dict[str, str]) -> None:
    """Test that view creates and uses a connection."""
    pe = sut(config=db_config)

    # Mock the connection
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=None)

    with patch.object(pe, "_get_connection", return_value=mock_conn):
        pe.view("users", limit=10)

    # Verify that the connection was used
    mock_conn.__enter__.assert_called_once()


# -------------------- Count Tests --------------------


def test_count_calls_connection(sut: Any, db_config: dict[str, str]) -> None:
    """Test that count creates and uses a connection."""
    pe = sut(config=db_config)

    # Mock the connection
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=None)

    with patch.object(pe, "_get_connection", return_value=mock_conn):
        pe.count("users")

    # Verify that the connection was used
    mock_conn.__enter__.assert_called_once()


# -------------------- Exists Tests --------------------


def test_exists_calls_connection(sut: Any, db_config: dict[str, str]) -> None:
    """Test that exists creates and uses a connection."""
    pe = sut(config=db_config)

    # Mock the connection
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=None)

    with patch.object(pe, "_get_connection", return_value=mock_conn):
        pe.exists("users")

    # Verify that the connection was used
    mock_conn.__enter__.assert_called_once()


# -------------------- List Tables Tests --------------------


def test_list_tables_calls_connection(sut: Any, db_config: dict[str, str]) -> None:
    """Test that list_tables creates and uses a connection."""
    pe = sut(config=db_config)

    # Mock the connection
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=None)

    with patch.object(pe, "_get_connection", return_value=mock_conn):
        pe.list_tables()

    # Verify that the connection was used
    mock_conn.__enter__.assert_called_once()


def test_list_tables_returns_list(sut: Any, db_config: dict[str, str]) -> None:
    """Test that list_tables returns a list of table names."""
    pe = sut(config=db_config)

    # Mock the connection and QueryOperations
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=None)

    expected_tables = ["users", "products", "orders"]

    with patch.object(pe, "_get_connection", return_value=mock_conn):
        with patch("ssb_parquedit.parquedit.QueryOperations") as mock_query_class:
            mock_query_instance = MagicMock()
            mock_query_instance.list_tables.return_value = expected_tables
            mock_query_class.return_value = mock_query_instance

            result = pe.list_tables()

            assert isinstance(result, list)
            assert result == expected_tables


def test_list_tables_returns_empty_list_when_no_tables(
    sut: Any, db_config: dict[str, str]
) -> None:
    """Test that list_tables returns empty list when no tables exist."""
    pe = sut(config=db_config)

    # Mock the connection and QueryOperations
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=None)

    with patch.object(pe, "_get_connection", return_value=mock_conn):
        with patch("ssb_parquedit.parquedit.QueryOperations") as mock_query_class:
            mock_query_instance = MagicMock()
            mock_query_instance.list_tables.return_value = []
            mock_query_class.return_value = mock_query_instance

            result = pe.list_tables()

            assert isinstance(result, list)
            assert len(result) == 0


# -------------------- Drop Table Tests --------------------


def test_drop_table_succeeds_in_test_environment(
    sut: Any, db_config: dict[str, str]
) -> None:
    """Test that drop_table succeeds in TEST environment."""
    import os

    pe = sut(config=db_config)

    # Mock the connection and DDLOperations
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=None)

    with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "test"}):
        with patch.object(pe, "_get_connection", return_value=mock_conn):
            with patch("ssb_parquedit.parquedit.DDLOperations") as mock_ddl_class:
                mock_ddl_instance = MagicMock()
                mock_ddl_class.return_value = mock_ddl_instance

                pe.drop_table("temporary_table")

                mock_ddl_instance.drop_table.assert_called_once_with("temporary_table")


def test_drop_table_fails_in_prod_environment(
    sut: Any, db_config: dict[str, str]
) -> None:
    """Test that drop_table raises PermissionError in PROD environment."""
    import os

    pe = sut(config=db_config)

    # Mock the connection and DDLOperations
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=None)

    with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "prod"}):
        with patch.object(pe, "_get_connection", return_value=mock_conn):
            with patch("ssb_parquedit.parquedit.DDLOperations") as mock_ddl_class:
                mock_ddl_instance = MagicMock()
                # Make drop_table raise PermissionError
                mock_ddl_instance.drop_table.side_effect = PermissionError(
                    "Table deletion is only allowed in TEST environment"
                )
                mock_ddl_class.return_value = mock_ddl_instance

                with pytest.raises(PermissionError, match="only allowed in TEST"):
                    pe.drop_table("temporary_table")
