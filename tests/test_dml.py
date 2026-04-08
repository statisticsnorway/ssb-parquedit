"""Tests for DML (Data Manipulation Language) operations."""

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

# Fixtures are imported from conftest.py: stub_external_modules, fake_conn, db_config


@pytest.fixture
def sut_dml() -> Any:
    """Import and return the DMLOperations class."""
    import importlib

    module = importlib.import_module("ssb_parquedit.dml")
    importlib.reload(module)
    return module.DMLOperations


class TestInsertDataFromDataFrame:
    """Test data insertion from DataFrame."""

    def test_insert_from_dataframe_basic(
        self, sut_dml: Any, fake_conn: MagicMock
    ) -> None:
        """Test basic data insertion from DataFrame."""
        dml_ops: Any = sut_dml(fake_conn)

        DF = sys.modules["pandas"].DataFrame
        df = DF()

        dml_ops.insert_data("users", df)

        # Should call register and execute
        assert fake_conn.register.called
        assert fake_conn.execute.called

    def test_insert_from_dataframe_validates_table_name(
        self, sut_dml: Any, fake_conn: MagicMock
    ) -> None:
        """Test that table name is validated."""
        dml_ops: Any = sut_dml(fake_conn)

        DF = sys.modules["pandas"].DataFrame
        df = DF()

        with pytest.raises(ValueError, match="Invalid table name"):
            dml_ops.insert_data("123invalid", df)


class TestInsertDataFromParquet:
    """Test data insertion from Parquet file."""

    def test_insert_from_parquet_basic(
        self, sut_dml: Any, fake_conn: MagicMock
    ) -> None:
        """Test basic data insertion from Parquet file."""
        dml_ops: Any = sut_dml(fake_conn)

        dml_ops.insert_data("users", "gs://bucket/users.parquet")

        # Should call execute
        assert fake_conn.execute.called


class TestInsertDataTypeErrors:
    """Test error handling for invalid source types."""

    def test_insert_with_invalid_source_type(
        self, sut_dml: Any, fake_conn: MagicMock
    ) -> None:
        """Test that invalid source type raises TypeError."""
        dml_ops: Any = sut_dml(fake_conn)

        with pytest.raises(TypeError, match="source must be a DataFrame"):
            dml_ops.insert_data("users", 12345)
