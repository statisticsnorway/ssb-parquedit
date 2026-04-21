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
