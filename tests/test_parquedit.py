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


# -------------------- Create Table Tests --------------------


def test_create_table_requires_short_name(sut: Any, db_config: dict[str, str]) -> None:
    """Test that create_table raises error when product_name is not provided."""
    pe = sut(config=db_config)

    DF = sys.modules["pandas"].DataFrame
    df = DF()

    with pytest.raises(ValueError, match="'product_name' must have a value"):
        pe.create_table("users", df)

    # -------------------- View Tests --------------------
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

    # -------------------- List Tables Tests --------------------

    # -------------------- Drop Table Tests --------------------
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

                # Should be called with cleanup=True by default
                mock_ddl_instance.drop_table.assert_called_once_with(
                    "temporary_table", cleanup=True
                )

                pe.drop_table("temporary_table")
