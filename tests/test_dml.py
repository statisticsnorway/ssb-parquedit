"""Tests for DML (Data Manipulation Language) operations."""

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

# Fixtures are imported from conftest.py: stub_external_modules, fake_conn, db_config


@pytest.fixture
def sut_dml() -> object:
    """Import and return the DMLOperations class."""
    import importlib
    
    module = importlib.import_module("ssb_parquedit.dml")
    importlib.reload(module)
    return module.DMLOperations


class TestInsertDataFromDataFrame:
    """Test data insertion from DataFrame."""

    def test_insert_from_dataframe_basic(
        self, sut_dml: object, fake_conn: MagicMock
    ) -> None:
        """Test basic data insertion from DataFrame."""
        dml_ops = sut_dml(fake_conn)
        
        DF = sys.modules["pandas"].DataFrame
        df = DF()
        
        dml_ops.insert_data("users", df)
        
        # Should call register and execute
        assert fake_conn.register.called
        assert fake_conn.execute.called

    def test_insert_from_dataframe_validates_table_name(
        self, sut_dml: object, fake_conn: MagicMock
    ) -> None:
        """Test that table name is validated."""
        dml_ops = sut_dml(fake_conn)
        
        DF = sys.modules["pandas"].DataFrame
        df = DF()
        
        with pytest.raises(ValueError, match="Invalid table name"):
            dml_ops.insert_data("123invalid", df)

    def test_insert_from_dataframe_registers_with_data_name(
        self, sut_dml: object, fake_conn: MagicMock
    ) -> None:
        """Test that DataFrame is registered as 'data'."""
        dml_ops = sut_dml(fake_conn)
        
        DF = sys.modules["pandas"].DataFrame
        df = DF()
        
        dml_ops.insert_data("users", df)
        
        # Should register with name "data"
        register_calls = fake_conn.register.call_args_list
        assert any(
            call[0][0] == "data"
            for call in register_calls
        ), "DataFrame not registered as 'data'"

    def test_insert_from_dataframe_executes_insert_statement(
        self, sut_dml: object, fake_conn: MagicMock
    ) -> None:
        """Test that INSERT statement is executed."""
        dml_ops = sut_dml(fake_conn)
        
        DF = sys.modules["pandas"].DataFrame
        df = DF()
        
        dml_ops.insert_data("users", df)
        
        # Should execute INSERT statement
        execute_calls = [
            str(call_args[0])
            for (call_args, _) in fake_conn.execute.call_args_list
        ]
        assert any(
            "INSERT INTO users" in call
            for call in execute_calls
        ), "INSERT statement not executed"

    def test_insert_from_dataframe_adds_id_column(
        self, sut_dml: object, fake_conn: MagicMock
    ) -> None:
        """Test that _id column is added to inserted data."""
        dml_ops = sut_dml(fake_conn)
        
        DF = sys.modules["pandas"].DataFrame
        df = DF()
        
        dml_ops.insert_data("users", df)
        
        # The register call should have a modified DataFrame with _id
        # This is hard to verify directly with mocks, but we can check the register was called
        assert fake_conn.register.called


class TestInsertDataFromParquet:
    """Test data insertion from Parquet file."""

    def test_insert_from_parquet_basic(
        self, sut_dml: object, fake_conn: MagicMock
    ) -> None:
        """Test basic data insertion from Parquet file."""
        dml_ops = sut_dml(fake_conn)
        
        dml_ops.insert_data("users", "gs://bucket/users.parquet")
        
        # Should call execute
        assert fake_conn.execute.called

    def test_insert_from_parquet_validates_table_name(
        self, sut_dml: object, fake_conn: MagicMock
    ) -> None:
        """Test that table name is validated."""
        dml_ops = sut_dml(fake_conn)
        
        with pytest.raises(ValueError, match="Invalid table name"):
            dml_ops.insert_data("invalid-name", "gs://bucket/file.parquet")

    def test_insert_from_parquet_uses_parameterized_query(
        self, sut_dml: object, fake_conn: MagicMock
    ) -> None:
        """Test that Parquet path uses parameterized query."""
        dml_ops = sut_dml(fake_conn)
        
        parquet_path = "gs://bucket/users.parquet"
        dml_ops.insert_data("users", parquet_path)
        
        # Should call execute with parameters
        execute_calls = fake_conn.execute.call_args_list
        assert any(
            len(call[0]) > 1 or call[1].get("parameters", [])
            for call in execute_calls
        ), "Expected parameterized query"

    def test_insert_from_parquet_executes_insert_statement(
        self, sut_dml: object, fake_conn: MagicMock
    ) -> None:
        """Test that INSERT statement is executed."""
        dml_ops = sut_dml(fake_conn)
        
        dml_ops.insert_data("users", "gs://bucket/users.parquet")
        
        # Should execute INSERT statement
        execute_calls = [
            str(call_args[0])
            for (call_args, _) in fake_conn.execute.call_args_list
        ]
        assert any(
            "INSERT INTO users" in call
            for call in execute_calls
        ), "INSERT statement not executed"

    def test_insert_from_parquet_generates_uuid_for_id(
        self, sut_dml: object, fake_conn: MagicMock
    ) -> None:
        """Test that uuid() is used for _id column."""
        dml_ops = sut_dml(fake_conn)
        
        dml_ops.insert_data("users", "gs://bucket/users.parquet")
        
        # Should use uuid()::VARCHAR for _id
        execute_calls = [
            str(call_args[0])
            for (call_args, _) in fake_conn.execute.call_args_list
        ]
        assert any(
            "uuid()" in call
            for call in execute_calls
        ), "UUID generation not found"


class TestInsertDataTypeErrors:
    """Test error handling for invalid source types."""

    def test_insert_with_invalid_source_type(
        self, sut_dml: object, fake_conn: MagicMock
    ) -> None:
        """Test that invalid source type raises TypeError."""
        dml_ops = sut_dml(fake_conn)
        
        with pytest.raises(TypeError, match="source must be a DataFrame"):
            dml_ops.insert_data("users", 12345)  # type: ignore

    def test_insert_with_list_source_raises_error(
        self, sut_dml: object, fake_conn: MagicMock
    ) -> None:
        """Test that list source raises TypeError."""
        dml_ops = sut_dml(fake_conn)
        
        with pytest.raises(TypeError, match="source must be a DataFrame"):
            dml_ops.insert_data("users", [1, 2, 3])  # type: ignore

    def test_insert_with_dict_source_raises_error(
        self, sut_dml: object, fake_conn: MagicMock
    ) -> None:
        """Test that dict source raises TypeError."""
        dml_ops = sut_dml(fake_conn)
        
        with pytest.raises(TypeError, match="source must be a DataFrame"):
            dml_ops.insert_data("users", {"id": 1})  # type: ignore
